"""Document tools: search, neighboring section, and catalog metadata."""

from __future__ import annotations

import re

from app.models.domain import DocumentRecord, RetrievedChunk
from app.repositories.document_catalog import DocumentCatalog
from app.repositories.vector_index import VectorIndex

MAX_EVIDENCE_CHUNKS = 8
_CHUNK_INDEX = re.compile(r"^(.*)_c(\d{4})$")


class DocumentTools:
    """Thin tool boundary so the graph does not call the vector store directly."""

    def __init__(self, catalog: DocumentCatalog, vector_index: VectorIndex) -> None:
        self._catalog = catalog
        self._index = vector_index

    def get_document_info(self, document_id: str) -> DocumentRecord:
        return self._catalog.get(document_id)

    def search_document(
        self, document_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:
        self._catalog.get(document_id)
        return self._index.search(query=query, document_id=document_id, k=k)

    def read_document_section(
        self, document_id: str, chunk_id: str, window: int = 1
    ) -> list[RetrievedChunk]:
        """Return the hit plus adjacent chunks (same document, sequential ids)."""
        self._catalog.get(document_id)
        match = _CHUNK_INDEX.match(chunk_id)
        if match is None:
            return self._index.get_by_ids([chunk_id])
        prefix, index = match.group(1), int(match.group(2))
        start = max(0, index - window)
        ids = [f"{prefix}_c{i:04d}" for i in range(start, index + window + 1)]
        return [
            chunk
            for chunk in self._index.get_by_ids(ids)
            if chunk.document_id == document_id
        ]


def merge_chunks(
    existing: list[RetrievedChunk],
    incoming: list[RetrievedChunk],
    limit: int = MAX_EVIDENCE_CHUNKS,
) -> list[RetrievedChunk]:
    """Dedupe by chunk_id, keep the higher score, cap the evidence set."""
    by_id: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in existing}
    for chunk in incoming:
        previous = by_id.get(chunk.chunk_id)
        if previous is None or (chunk.score or 0) > (previous.score or 0):
            by_id[chunk.chunk_id] = chunk
    merged = list(by_id.values())
    merged.sort(key=lambda item: item.score if item.score is not None else -1.0, reverse=True)
    return merged[:limit]
