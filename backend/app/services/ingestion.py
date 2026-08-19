"""Deterministic ingest: validate → parse → chunk → embed → index → catalog."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import LLMServiceError, UnreadableDocumentError, UnsupportedMediaError
from app.core.secrets import require_openai_key
from app.loaders.chunking import chunk_json_leaves, chunk_pdf_pages
from app.loaders.json_loader import flatten_json, parse_json_bytes
from app.loaders.pdf_loader import load_pdf
from app.models.domain import DocumentRecord
from app.repositories.document_catalog import DocumentCatalog
from app.repositories.vector_index import VectorIndex

logger = logging.getLogger(__name__)

PDF_TYPES = {"application/pdf"}
JSON_TYPES = {"application/json", "text/json"}


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        catalog: DocumentCatalog,
        vector_index: VectorIndex,
        uploads_dir: Path,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._vector_index = vector_index
        self._uploads_dir = uploads_dir
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, *, filename: str, content_type: str | None, data: bytes) -> DocumentRecord:
        file_type = detect_file_type(filename, content_type, data)
        if self._settings.embedding_backend.lower() != "fake":
            require_openai_key(self._settings)
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        dest = self._uploads_dir / document_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

        if file_type == "pdf":
            pages = load_pdf(data, filename)
            chunks = chunk_pdf_pages(
                pages,
                document_id=document_id,
                source=filename,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )
            page_count = len(pages)
        else:
            payload = parse_json_bytes(data, filename)
            leaves = flatten_json(payload)
            if not leaves:
                raise UnreadableDocumentError(f"JSON document '{filename}' is empty.")
            chunks = chunk_json_leaves(
                leaves,
                document_id=document_id,
                source=filename,
                chunk_size=self._settings.chunk_size,
            )
            page_count = None

        if not chunks:
            raise UnreadableDocumentError(f"No indexable text was extracted from '{filename}'.")

        try:
            self._vector_index.add_chunks(chunks)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError("Failed to embed or index the document.") from exc
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            status="ready",
            pages=page_count,
            chunks=len(chunks),
        )
        self._catalog.add(record)
        logger.info(
            "ingestion.complete filename=%s type=%s chunks=%s",
            filename,
            file_type,
            len(chunks),
        )
        return record


def detect_file_type(filename: str, content_type: str | None, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    mime = (content_type or "").split(";")[0].strip().lower()
    head = data[:8]

    if suffix == ".pdf" or mime in PDF_TYPES or head.startswith(b"%PDF"):
        if suffix and suffix != ".pdf" and not head.startswith(b"%PDF"):
            raise UnsupportedMediaError("Only PDF and JSON source documents are supported.")
        return "pdf"
    if suffix == ".json" or mime in JSON_TYPES or _looks_like_json(data):
        return "json"
    raise UnsupportedMediaError("Only PDF and JSON source documents are supported.")


def _looks_like_json(data: bytes) -> bool:
    stripped = data.lstrip()
    return stripped.startswith(b"{") or stripped.startswith(b"[")
