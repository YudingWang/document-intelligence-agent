"""HTTP request/response schemas (OpenAPI)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.domain import ChatTurn, Citation, DocumentRecord, QAResult


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
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def collect_document_ids(self) -> "ChatRequest":
        ids: list[str] = []
        for value in [self.document_id, *self.document_ids]:
            text = (value or "").strip()
            if text and text not in ids:
                ids.append(text)
        self.document_ids = ids
        self.document_id = ids[0] if ids else None
        return self


class ChatResponse(BaseModel):
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    question: str
    answer: str
    supported: bool
    citations: list[Citation] = Field(default_factory=list)


class BatchQAResponse(BaseModel):
    """Canonical assignment output: a list of question/answer objects."""

    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    filename: str | None = None
    filenames: list[str] = Field(default_factory=list)
    results: list[QAResult]
