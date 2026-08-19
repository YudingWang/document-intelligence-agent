"""Agent loop: generate when evidence looks related; rewrite once if it does not."""

from __future__ import annotations

import pytest

from app.agents.graph import DocumentQAAgent
from app.core.config import Settings
from app.models.domain import NOT_FOUND_ANSWER, ChatTurn, GroundedAnswer, RetrievedChunk
from app.tools.documents import DocumentTools, merge_chunks


class FakeCatalog:
    def get(self, document_id: str):
        return document_id


class FakeIndex:
    def __init__(self, chunks: list[RetrievedChunk] | list[list[RetrievedChunk]]) -> None:
        self.queries: list[str] = []
        if chunks and isinstance(chunks[0], list):
            self._waves: list[list[RetrievedChunk]] = chunks  # type: ignore[assignment]
            self.chunks = [item for wave in self._waves for item in wave]
        else:
            self._waves = [chunks]  # type: ignore[list-item]
            self.chunks = chunks  # type: ignore[assignment]

    def search(self, query: str, document_id: str, k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        wave = self._waves[min(len(self.queries) - 1, len(self._waves) - 1)]
        return wave[:k]

    def get_by_ids(self, ids: list[str]) -> list[RetrievedChunk]:
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


class SequenceLLM:
    def __init__(self, answers: list[GroundedAnswer]) -> None:
        self.answers = answers
        self.generate_calls = 0
        self.rewrite_calls = 0
        self.standalone_calls = 0
        self.seen_chunks: list[list[str]] = []

    async def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
        self.generate_calls += 1
        self.seen_chunks.append([chunk.chunk_id for chunk in chunks])
        return self.answers[min(self.generate_calls - 1, len(self.answers) - 1)]

    async def rewrite_query(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.rewrite_calls += 1
        return question + " retry-token"

    async def standalone_query(self, question: str, history: list[ChatTurn]) -> str:
        self.standalone_calls += 1
        if any("AWS" in turn.text for turn in history) and "region" in question.lower():
            return "What AWS hosting region does the document specify?"
        return question


def _agent(tmp_path, llm, index) -> DocumentQAAgent:
    settings = Settings(data_dir=tmp_path, max_retrieval_attempts=2, retrieval_top_k=3)
    tools = DocumentTools(FakeCatalog(), index)  # type: ignore[arg-type]
    return DocumentQAAgent(settings, llm, tools)


def _chunk(chunk_id: str, text: str, page: int = 1, document_id: str = "doc_1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source="policy.pdf",
        text=text,
        page=page,
    )


@pytest.mark.asyncio
async def test_agent_retrieves_and_answers(tmp_path) -> None:
    chunk = _chunk("c1", "The company uses AWS as its primary cloud provider.", page=3)
    llm = SequenceLLM(
        [GroundedAnswer(answer="The company uses AWS.", supported=True, used_chunk_ids=["c1"])]
    )
    result = await _agent(tmp_path, llm, FakeIndex([chunk])).answer(
        "Which cloud providers do you rely on?", "doc_1"
    )
    assert result.supported is True
    assert "AWS" in result.answer
    assert result.citations[0].page == 3
    assert llm.rewrite_calls == 0
    assert llm.standalone_calls == 0


@pytest.mark.asyncio
async def test_agent_retries_then_refuses(tmp_path) -> None:
    weak = _chunk("c2", "Office hours are Monday through Friday, 9am to 5pm.")
    llm = SequenceLLM([GroundedAnswer(answer=NOT_FOUND_ANSWER, supported=False)])
    result = await _agent(tmp_path, llm, FakeIndex([weak])).answer("Who is the CEO?", "doc_1")
    assert result.supported is False
    assert result.answer == NOT_FOUND_ANSWER
    assert llm.rewrite_calls == 1
    assert llm.generate_calls == 1
    assert result.search_attempts == 2


@pytest.mark.asyncio
async def test_retry_merges_first_and_second_retrieval(tmp_path) -> None:
    first = _chunk("c1", "Office hours are Monday through Friday.")
    second = _chunk("c2", "The company uses AWS as its primary cloud provider.")
    llm = SequenceLLM(
        [GroundedAnswer(answer="The company uses AWS.", supported=True, used_chunk_ids=["c2"])]
    )
    result = await _agent(tmp_path, llm, FakeIndex([[first], [second]])).answer(
        "Which cloud providers do you rely on?", "doc_1"
    )
    assert result.supported is True
    assert llm.rewrite_calls == 1
    assert set(llm.seen_chunks[0]) == {"c1", "c2"}


@pytest.mark.asyncio
async def test_chat_history_rewrites_follow_up(tmp_path) -> None:
    chunk = _chunk("c1", "Primary region is us-east-1 in Virginia.")
    llm = SequenceLLM(
        [GroundedAnswer(answer="us-east-1 in Virginia.", supported=True, used_chunk_ids=["c1"])]
    )
    index = FakeIndex([chunk])
    result = await _agent(tmp_path, llm, index).answer(
        "What region?",
        "doc_1",
        history=[
            ChatTurn(role="user", text="Which cloud providers do you rely on?"),
            ChatTurn(role="agent", text="AWS is the primary provider."),
        ],
    )
    assert llm.standalone_calls == 1
    assert index.queries[0] == "What AWS hosting region does the document specify?"
    assert result.question == "What region?"
    assert result.supported is True


@pytest.mark.asyncio
async def test_agent_retrieves_from_multiple_documents(tmp_path) -> None:
    aws = _chunk("a1", "The company uses AWS as its primary cloud provider.", document_id="doc_a")
    region = _chunk("b1", "Primary region is us-east-1 in Virginia.", document_id="doc_b")

    class PerDocIndex:
        def search(self, query: str, document_id: str, k: int) -> list[RetrievedChunk]:
            return [aws] if document_id == "doc_a" else [region]

        def get_by_ids(self, ids: list[str]) -> list[RetrievedChunk]:
            by_id = {aws.chunk_id: aws, region.chunk_id: region}
            return [by_id[item] for item in ids if item in by_id]

    llm = SequenceLLM(
        [GroundedAnswer(answer="AWS in us-east-1.", supported=True, used_chunk_ids=["a1", "b1"])]
    )
    result = await _agent(tmp_path, llm, PerDocIndex()).answer(
        "Which cloud provider and region?", ["doc_a", "doc_b"]
    )
    assert result.supported is True
    assert set(llm.seen_chunks[0]) == {"a1", "b1"}


def test_merge_chunks_dedupes_and_keeps_higher_score() -> None:
    low = _chunk("c1", "AWS", page=1)
    low.score = 0.1
    high = _chunk("c1", "AWS", page=1)
    high.score = 0.9
    extra = _chunk("c2", "GCP", page=2)
    extra.score = 0.5
    merged = merge_chunks([low], [high, extra], limit=8)
    assert [item.chunk_id for item in merged] == ["c1", "c2"]
    assert merged[0].score == 0.9
