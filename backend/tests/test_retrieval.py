"""Vector search must not leak chunks from another document_id."""

from __future__ import annotations

from app.models.domain import Chunk
from app.repositories.embeddings import TokenHashEmbeddings
from app.repositories.vector_index import LangChainVectorIndex
from langchain_core.vectorstores import InMemoryVectorStore


def test_search_filters_by_document_id() -> None:
    store = LangChainVectorIndex(InMemoryVectorStore(embedding=TokenHashEmbeddings()))
    store.add_chunks(
        [
            Chunk(
                chunk_id="a1",
                document_id="doc_a",
                source="a.json",
                text="Document A uses Azure exclusively.",
            ),
            Chunk(
                chunk_id="b1",
                document_id="doc_b",
                source="b.json",
                text="Document B uses Amazon Web Services AWS.",
            ),
        ]
    )
    hits = store.search("Which cloud provider is used?", "doc_b", k=3)
    assert hits
    assert all(hit.document_id == "doc_b" for hit in hits)
    assert any("AWS" in hit.text for hit in hits)
