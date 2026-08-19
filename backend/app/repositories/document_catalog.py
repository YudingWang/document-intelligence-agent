"""JSON file of DocumentRecord metadata (filename, type, chunk count)."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from app.core.exceptions import DocumentNotFoundError
from app.models.domain import DocumentRecord


class DocumentCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def add(self, record: DocumentRecord) -> DocumentRecord:
        with self._lock:
            self._records[record.document_id] = record
            self._dump()
        return record

    def get(self, document_id: str) -> DocumentRecord:
        with self._lock:
            record = self._records.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        return record

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )

    def remove(self, document_id: str) -> None:
        with self._lock:
            if document_id not in self._records:
                raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
            del self._records[document_id]
            self._dump()

    def _load(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            item["document_id"]: DocumentRecord.model_validate(item)
            for item in raw
        }

    def _dump(self) -> None:
        payload = [record.model_dump(mode="json") for record in self._records.values()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
