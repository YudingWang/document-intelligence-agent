"""Bounded LangGraph agent for one question against one indexed document.

Flow (max two retrievals, at most one generation):

    retrieve → evaluate evidence
        ├─ sufficient → generate → END
        └─ not enough → rewrite query → retrieve → generate → END
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.evidence import evidence_is_sufficient
from app.agents.llm import LLMClient
from app.core.config import Settings
from app.models.domain import (
    NOT_FOUND_ANSWER,
    Citation,
    QAResult,
    RetrievedChunk,
)
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    document_id: str
    query: str
    search_attempts: int
    chunks: list[RetrievedChunk]
    evidence_sufficient: bool
    answer: str
    supported: bool
    citations: list[Citation]
    retrieval_latency_ms: int
    llm_latency_ms: int


class DocumentQAAgent:
    """Run the retrieve → evaluate → (rewrite) → generate loop for one question."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        retrieval: RetrievalService,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._retrieval = retrieval
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {"rewrite": "rewrite", "generate": "generate"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        return graph.compile()

    async def answer(self, question: str, document_id: str) -> QAResult:
        """Answer one question using only chunks from `document_id`."""
        started = time.perf_counter()
        state = await self._graph.ainvoke(
            {
                "question": question,
                "document_id": document_id,
                "query": question,
                "search_attempts": 0,
                "chunks": [],
                "evidence_sufficient": False,
                "answer": NOT_FOUND_ANSWER,
                "supported": False,
                "citations": [],
                "retrieval_latency_ms": 0,
                "llm_latency_ms": 0,
            }
        )
        total_ms = int((time.perf_counter() - started) * 1000)
        return QAResult(
            question=question,
            answer=state.get("answer") or NOT_FOUND_ANSWER,
            supported=bool(state.get("supported")),
            citations=state.get("citations") or [],
            search_attempts=int(state.get("search_attempts") or 0),
            retrieval_latency_ms=int(state.get("retrieval_latency_ms") or 0),
            llm_latency_ms=int(state.get("llm_latency_ms") or 0),
            total_latency_ms=total_ms,
        )

    async def _retrieve(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        chunks = self._retrieval.search(
            state["document_id"],
            state["query"],
            k=self._settings.retrieval_top_k,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        attempt = int(state.get("search_attempts") or 0) + 1
        logger.info(
            "agent.retrieve query=%r hits=%s attempt=%s",
            state["query"][:120],
            len(chunks),
            attempt,
        )
        return {
            "chunks": chunks,
            "search_attempts": attempt,
            "retrieval_latency_ms": int(state.get("retrieval_latency_ms") or 0) + elapsed,
        }

    def _evaluate(self, state: AgentState) -> dict[str, Any]:
        """Decide whether retrieved text is related enough to call the LLM."""
        sufficient = evidence_is_sufficient(
            state["question"], state.get("chunks") or []
        )
        logger.info(
            "agent.evaluate sufficient=%s attempt=%s",
            sufficient,
            state.get("search_attempts"),
        )
        return {"evidence_sufficient": sufficient}

    def _route_after_evaluate(self, state: AgentState) -> Literal["rewrite", "generate"]:
        if state.get("evidence_sufficient"):
            return "generate"
        if int(state.get("search_attempts") or 0) >= self._settings.max_retrieval_attempts:
            return "generate"
        return "rewrite"

    async def _rewrite(self, state: AgentState) -> dict[str, Any]:
        """Ask the model for a better search query. Cheaper than a full answer call."""
        started = time.perf_counter()
        query = await self._llm.rewrite_query(state["question"], state.get("chunks") or [])
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("agent.rewrite query=%r", query[:120])
        return {
            "query": query,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed,
        }

    async def _generate(self, state: AgentState) -> dict[str, Any]:
        chunks = state.get("chunks") or []
        if not chunks:
            return {
                "answer": NOT_FOUND_ANSWER,
                "supported": False,
                "citations": [],
            }
        started = time.perf_counter()
        result = await self._llm.generate_answer(state["question"], chunks)
        elapsed = int((time.perf_counter() - started) * 1000)
        supported = bool(result.supported)
        answer = result.answer.strip() if supported else NOT_FOUND_ANSWER
        citations = _citations_from(result.used_chunk_ids, chunks, supported)
        return {
            "answer": answer,
            "supported": supported,
            "citations": citations,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed,
        }


def _citations_from(
    used_ids: list[str],
    chunks: list[RetrievedChunk],
    supported: bool,
) -> list[Citation]:
    """Map model-selected chunk ids back to page / JSON-path citations."""
    if not supported:
        return []
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    selected = [by_id[chunk_id] for chunk_id in used_ids if chunk_id in by_id]
    if not selected:
        selected = chunks[:3]
    citations: list[Citation] = []
    seen: set[tuple[str | None, int | None, str | None]] = set()
    for chunk in selected:
        key = (chunk.source, chunk.page, chunk.json_path)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source=chunk.source,
                page=chunk.page,
                json_path=chunk.json_path,
                chunk_id=chunk.chunk_id,
            )
        )
    return citations
