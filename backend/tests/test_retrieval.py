"""Vector search must not leak chunks from another document_id."""

from __future__ import annotations

from app.models.domain import Chunk, DocumentRecord
from app.repositories.embeddings import TokenHashEmbeddings
from app.repositories.vector_index import LangChainVectorIndex
from app.services.ingestion import safe_filename
from app.tools.documents import DocumentTools
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

    loaded = store.get_by_ids(["b1", "a1"])
    assert [item.chunk_id for item in loaded] == ["b1", "a1"]


def test_read_section_includes_neighbors() -> None:
    store = LangChainVectorIndex(InMemoryVectorStore(embedding=TokenHashEmbeddings()))
    store.add_chunks(
        [
            Chunk(chunk_id="doc_1_c0000", document_id="doc_1", source="a.pdf", text="before", page=1),
            Chunk(chunk_id="doc_1_c0001", document_id="doc_1", source="a.pdf", text="hit", page=1),
            Chunk(chunk_id="doc_1_c0002", document_id="doc_1", source="a.pdf", text="after", page=2),
        ]
    )

    class Catalog:
        def get(self, document_id: str) -> DocumentRecord:
            return DocumentRecord(
                document_id=document_id,
                filename="a.pdf",
                file_type="pdf",
                chunks=3,
            )

    tools = DocumentTools(Catalog(), store)  # type: ignore[arg-type]
    section = tools.read_document_section("doc_1", "doc_1_c0001")
    assert [chunk.text for chunk in section] == ["before", "hit", "after"]


def test_safe_filename_keeps_basename() -> None:
    assert safe_filename("../etc/passwd.json") == "passwd.json"
    assert safe_filename("/tmp/nested/policy.pdf") == "policy.pdf"
    assert safe_filename("my file.json") == "my_file.json"
