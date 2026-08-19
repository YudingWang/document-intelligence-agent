"""Search the vector index for one document. Always filters by document_id."""

from __future__ import annotations

from app.models.domain import RetrievedChunk
from app.repositories.document_catalog import DocumentCatalog
from app.repositories.vector_index import VectorIndex


class RetrievalService:
    def __init__(self, catalog: DocumentCatalog, vector_index: VectorIndex) -> None:
        self._catalog = catalog
        self._vector_index = vector_index

    def search(self, document_id: str, query: str, k: int) -> list[RetrievedChunk]:
        self._catalog.get(document_id)
        return self._vector_index.search(query=query, document_id=document_id, k=k)
