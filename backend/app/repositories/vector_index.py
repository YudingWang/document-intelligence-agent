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


class LangChainVectorIndex:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def add_chunks(self, chunks: list[Chunk]) -> None:
        documents = [
            Document(page_content=chunk.text, metadata=_metadata(chunk))
            for chunk in chunks
        ]
        self._store.add_documents(
            documents=documents,
            ids=[chunk.chunk_id for chunk in chunks],
        )

    def search(self, query: str, document_id: str, k: int) -> list[RetrievedChunk]:
        pairs = self._search_with_filter(query, document_id, k)
        results: list[RetrievedChunk] = []
        for document, score in pairs:
            metadata = document.metadata or {}
            if metadata.get("document_id") != document_id:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id") or ""),
                    document_id=document_id,
                    source=str(metadata.get("source") or ""),
                    text=document.page_content,
                    page=_optional_int(metadata.get("page")),
                    json_path=_optional_str(metadata.get("json_path")),
                    score=float(score) if score is not None else None,
                )
            )
            if len(results) >= k:
                break
        return results

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
