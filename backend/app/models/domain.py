"""Internal models shared by loaders, the vector index, and the agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


NOT_FOUND_ANSWER = (
    "I could not find this information in the provided document."
)


class Chunk(BaseModel):
    """One indexed text unit, with page number (PDF) or JSON path."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    page: int | None = None
    json_path: str | None = None


class RetrievedChunk(Chunk):
    score: float | None = None


class Citation(BaseModel):
    source: str
    page: int | None = None
    json_path: str | None = None
    chunk_id: str | None = None


class ChatTurn(BaseModel):
    """One prior chat message. Used only to resolve follow-up questions."""

    role: Literal["user", "agent"]
    text: str = Field(min_length=1, max_length=8000)


class GroundedAnswer(BaseModel):
    """Structured output we ask gpt-4o-mini to return."""

    answer: str
    supported: bool
    used_chunk_ids: list[str] = Field(default_factory=list)


class QAResult(BaseModel):
    question: str
    answer: str
    supported: bool
    citations: list[Citation] = Field(default_factory=list)
    search_attempts: int = 1
    retrieval_latency_ms: int = 0
    llm_latency_ms: int = 0
    total_latency_ms: int = 0


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_type: Literal["pdf", "json"]
    status: Literal["ready", "failed"] = "ready"
    pages: int | None = None
    chunks: int = 0
    is_sample: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
