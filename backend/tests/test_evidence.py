from app.agents.evidence import content_terms, evidence_is_sufficient
from app.models.domain import RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="doc_1",
        source="policy.pdf",
        text=text,
        page=1,
    )


def test_empty_chunks_are_insufficient() -> None:
    assert evidence_is_sufficient("Which cloud providers do you rely on?", []) is False


def test_related_chunk_is_sufficient() -> None:
    chunks = [_chunk("Acme relies on Amazon Web Services (AWS) as the primary cloud provider.")]
    assert evidence_is_sufficient("Which cloud providers do you rely on?", chunks) is True


def test_unrelated_chunk_is_insufficient() -> None:
    chunks = [_chunk("Office hours are Monday through Friday, 9am to 5pm.")]
    assert evidence_is_sufficient("Who is the CEO?", chunks) is False


def test_acronyms_count_as_content_terms() -> None:
    terms = content_terms("Do you perform APM, EUM, or DEM monitoring?")
    assert {"apm", "eum", "dem", "monitoring"} <= terms
