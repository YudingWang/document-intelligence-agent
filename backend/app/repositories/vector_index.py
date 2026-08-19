"""LangChain vector store wrapper. Production uses Chroma; tests use memory."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from app.core.config import Settings
from app.models.domain import Chunk, RetrievedChunk


class VectorIndex(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...

    def search(
        self,
        query: str,
        document_id: str,
        k: int,
    ) -> list[RetrievedChunk]: ...

    def get_by_ids(self, ids: list[str]) -> list[RetrievedChunk]: ...


class LangChainVectorIndex:
    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._by_id: dict[str, RetrievedChunk] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        documents = [
            Document(page_content=chunk.text, metadata=_metadata(chunk))
            for chunk in chunks
        ]
        self._store.add_documents(
            documents=documents,
            ids=[chunk.chunk_id for chunk in chunks],
        )
        for chunk in chunks:
            self._by_id[chunk.chunk_id] = RetrievedChunk(**chunk.model_dump(), score=None)

    def search(self, query: str, document_id: str, k: int) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        for document, score in self._search_with_filter(query, document_id, k):
            chunk = _document_to_chunk(document, score)
            if chunk is None or chunk.document_id != document_id:
                continue
            self._by_id[chunk.chunk_id] = chunk
            results.append(chunk)
            if len(results) >= k:
                break
        return results

    def get_by_ids(self, ids: list[str]) -> list[RetrievedChunk]:
        found: dict[str, RetrievedChunk] = {}
        missing: list[str] = []
        for chunk_id in ids:
            cached = self._by_id.get(chunk_id)
            if cached is None:
                missing.append(chunk_id)
            else:
                found[chunk_id] = cached
        if missing:
            for chunk in self._fetch_ids(missing):
                self._by_id[chunk.chunk_id] = chunk
                found[chunk.chunk_id] = chunk
        return [found[chunk_id] for chunk_id in ids if chunk_id in found]

    def _search_with_filter(
        self,
        query: str,
        document_id: str,
        k: int,
    ) -> list[tuple[Document, float | None]]:
        # Native metadata filters differ by backend; fall back to over-fetch + Python filter.
        where = {"document_id": document_id}
        attempts = (
            {"k": k, "filter": where},
            {"k": max(k * 8, k), "filter": where},
            {"k": max(k * 8, k)},
        )
        for kwargs in attempts:
            try:
                scored = self._store.similarity_search_with_score(query, **kwargs)
                return [(doc, score) for doc, score in scored]
            except Exception:  # noqa: BLE001
                try:
                    docs = self._store.similarity_search(query, **kwargs)
                    return [(doc, None) for doc in docs]
                except Exception:  # noqa: BLE001
                    continue
        return []

    def _fetch_ids(self, ids: list[str]) -> list[RetrievedChunk]:
        getter = getattr(self._store, "get_by_ids", None)
        if getter is None:
            return []
        try:
            documents = getter(ids)
        except Exception:  # noqa: BLE001
            return []
        chunks: list[RetrievedChunk] = []
        for document in documents or []:
            if document is None:
                continue
            chunk = _document_to_chunk(document, None)
            if chunk is not None:
                chunks.append(chunk)
        return chunks


def create_vector_index(settings: Settings, embeddings: Embeddings) -> LangChainVectorIndex:
    backend = settings.vector_backend.lower()
    if backend == "memory":
        return LangChainVectorIndex(InMemoryVectorStore(embedding=embeddings))
    if backend != "chroma":
        raise ValueError(f"Unsupported vector backend '{settings.vector_backend}'.")

    from langchain_chroma import Chroma

    store = Chroma(
        collection_name="document_qa",
        embedding_function=embeddings,
        persist_directory=str(settings.resolved_data_dir() / "chroma"),
    )
    return LangChainVectorIndex(store)


def _document_to_chunk(document: Document, score: float | None) -> RetrievedChunk | None:
    metadata = document.metadata or {}
    document_id = str(metadata.get("document_id") or "")
    chunk_id = str(metadata.get("chunk_id") or "")
    if not document_id or not chunk_id:
        return None
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source=str(metadata.get("source") or ""),
        text=document.page_content,
        page=_optional_int(metadata.get("page")),
        json_path=_optional_str(metadata.get("json_path")),
        score=float(score) if score is not None else None,
    )


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source": chunk.source,
    }
    if chunk.page is not None:
        payload["page"] = chunk.page
    if chunk.json_path:
        payload["json_path"] = chunk.json_path
    return payload


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
