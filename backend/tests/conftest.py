"""Shared fixtures: in-memory app, stub LLM, tiny PDF builder. No live OpenAI calls."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.domain import GroundedAnswer, NOT_FOUND_ANSWER, RetrievedChunk
from app.repositories.embeddings import TokenHashEmbeddings

# Terms that appear in sample_data/vendor_security.json
_SAMPLE_TERMS = (
    "aws",
    "cloud",
    "notification",
    "sla",
    "third",
    "personal",
    "datadog",
    "apm",
    "eum",
    "dem",
    "virginia",
    "oregon",
    "us-east-1",
)


class StubLLM:
    """Return a snippet from retrieved text, or a refusal. Tracks call counts."""

    def __init__(self) -> None:
        self.generate_calls = 0
        self.rewrite_calls = 0

    async def generate_answer(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        self.generate_calls += 1
        blob = " ".join(chunk.text for chunk in chunks).lower()
        question_l = question.lower()
        if "ceo" in question_l or "not in the document" in question_l:
            return GroundedAnswer(answer=NOT_FOUND_ANSWER, supported=False)
        if any(term in blob for term in _SAMPLE_TERMS):
            return GroundedAnswer(
                answer=chunks[0].text[:400],
                supported=True,
                used_chunk_ids=[chunks[0].chunk_id],
            )
        return GroundedAnswer(answer=NOT_FOUND_ANSWER, supported=False)

    async def rewrite_query(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.rewrite_calls += 1
        return f"{question} SLA AWS Datadog monitoring data center"


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        embedding_backend="fake",
        vector_backend="memory",
        data_dir=tmp_path / "data",
        max_retrieval_attempts=2,
        batch_concurrency=2,
        retrieval_top_k=4,
    )


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def client(tmp_settings: Settings, stub_llm: StubLLM) -> TestClient:
    app = create_app(
        settings=tmp_settings,
        llm=stub_llm,
        embeddings=TokenHashEmbeddings(),
    )
    with TestClient(app) as test_client:
        yield test_client


def make_pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data
