"""Agent loop: generate when evidence looks related; rewrite once if it does not."""

from __future__ import annotations

import pytest

from app.agents.graph import DocumentQAAgent
from app.core.config import Settings
from app.models.domain import NOT_FOUND_ANSWER, GroundedAnswer, RetrievedChunk
from app.services.retrieval import RetrievalService


class FakeCatalog:
    def get(self, document_id: str):
        return document_id


class FakeIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, query: str, document_id: str, k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks[:k]


class SequenceLLM:
    def __init__(self, answers: list[GroundedAnswer]) -> None:
        self.answers = answers
        self.generate_calls = 0
        self.rewrite_calls = 0

    async def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
        self.generate_calls += 1
        return self.answers[min(self.generate_calls - 1, len(self.answers) - 1)]

    async def rewrite_query(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.rewrite_calls += 1
        return question + " retry-token"


@pytest.mark.asyncio
async def test_agent_retrieves_and_answers(tmp_path) -> None:
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc_1",
        source="policy.pdf",
        text="The company uses AWS as its primary cloud provider.",
        page=3,
    )
    llm = SequenceLLM(
        [GroundedAnswer(answer="The company uses AWS.", supported=True, used_chunk_ids=["c1"])]
    )
    settings = Settings(data_dir=tmp_path, max_retrieval_attempts=2, retrieval_top_k=3)
    retrieval = RetrievalService(FakeCatalog(), FakeIndex([chunk]))  # type: ignore[arg-type]
    agent = DocumentQAAgent(settings, llm, retrieval)
    result = await agent.answer("Which cloud providers do you rely on?", "doc_1")
    assert result.supported is True
    assert "AWS" in result.answer
    assert result.citations[0].page == 3
    assert llm.rewrite_calls == 0


@pytest.mark.asyncio
async def test_agent_retries_then_refuses(tmp_path) -> None:
    weak = RetrievedChunk(
        chunk_id="c2",
        document_id="doc_1",
        source="policy.pdf",
        text="Office hours are Monday through Friday, 9am to 5pm.",
        page=1,
    )
    llm = SequenceLLM(
        [
            GroundedAnswer(answer=NOT_FOUND_ANSWER, supported=False),
            GroundedAnswer(answer=NOT_FOUND_ANSWER, supported=False),
        ]
    )
    settings = Settings(data_dir=tmp_path, max_retrieval_attempts=2, retrieval_top_k=3)
    retrieval = RetrievalService(FakeCatalog(), FakeIndex([weak]))  # type: ignore[arg-type]
    agent = DocumentQAAgent(settings, llm, retrieval)
    result = await agent.answer("Who is the CEO?", "doc_1")
    assert result.supported is False
    assert result.answer == NOT_FOUND_ANSWER
    assert llm.rewrite_calls == 1
    assert llm.generate_calls == 1
    assert result.search_attempts == 2
