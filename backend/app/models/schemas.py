"""HTTP request/response schemas (OpenAPI)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.domain import Citation, DocumentRecord, QAResult


class ErrorResponse(BaseModel):
    error: str
    code: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    configured: bool


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class ChatRequest(BaseModel):
    document_id: str
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    supported: bool
    citations: list[Citation] = Field(default_factory=list)


class BatchQAResponse(BaseModel):
    """Canonical assignment output: a list of question/answer objects."""

    document_id: str
    filename: str | None = None
    results: list[QAResult]
