"""Loader and chunking tests for PDF pages, JSON paths, and question shapes."""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import InvalidInputError, UnreadableDocumentError
from app.loaders.chunking import chunk_json_leaves, chunk_pdf_pages
from app.loaders.json_loader import flatten_json, parse_json_bytes
from app.loaders.pdf_loader import load_pdf
from app.loaders.questions import parse_questions, parse_questions_file
from tests.conftest import make_pdf_bytes


def test_flatten_json_preserves_paths() -> None:
    payload = {"security": {"cloud_provider": "AWS", "regions": ["us-east-1", "us-west-2"]}}
    leaves = flatten_json(payload)
    paths = {leaf.path: leaf.value for leaf in leaves}
    assert paths["$.security.cloud_provider"] == "AWS"
    assert paths["$.security.regions[0]"] == "us-east-1"
    assert paths["$.security.regions[1]"] == "us-west-2"


def test_parse_json_rejects_malformed() -> None:
    with pytest.raises(InvalidInputError):
        parse_json_bytes(b"{not json", "broken.json")


def test_question_shapes() -> None:
    assert parse_questions({"questions": ["A", "B"]}) == ["A", "B"]
    assert parse_questions(["A", {"question": "B"}]) == ["A", "B"]
    with pytest.raises(InvalidInputError):
        parse_questions({"nope": []})
    with pytest.raises(InvalidInputError):
        parse_questions_file(b"[]", "empty.json")


def test_pdf_loader_keeps_page_numbers() -> None:
    data = make_pdf_bytes("Acme uses AWS in us-east-1.")
    pages = load_pdf(data, "policy.pdf")
    assert pages[0].page_number == 1
    assert "AWS" in pages[0].text


def test_pdf_loader_rejects_empty() -> None:
    data = make_pdf_bytes("   ")
    with pytest.raises(UnreadableDocumentError):
        load_pdf(data, "empty.pdf")


def test_chunking_assigns_ids() -> None:
    pages = load_pdf(make_pdf_bytes("Cloud provider AWS. Backup Oregon."), "p.pdf")
    chunks = chunk_pdf_pages(
        pages, document_id="doc_1", source="p.pdf", chunk_size=80, chunk_overlap=10
    )
    assert chunks
    assert chunks[0].page == 1
    assert chunks[0].chunk_id.startswith("doc_1_")

    leaves = flatten_json({"a": "one", "b": "two"})
    json_chunks = chunk_json_leaves(
        leaves, document_id="doc_2", source="a.json", chunk_size=1000
    )
    assert json_chunks[0].json_path == "$.a"
    assert "$.a" in json_chunks[0].text


def test_json_chunks_follow_object_sections() -> None:
    payload = {
        "company": "Acme",
        "security": {
            "cloud_providers": ["AWS", "GCP"],
            "monitoring": {"apm": True, "dem": False},
        },
    }
    chunks = chunk_json_leaves(
        flatten_json(payload), document_id="doc_3", source="policy.json", chunk_size=1000
    )
    by_path = {chunk.json_path: chunk.text for chunk in chunks}
    assert "$.company" in by_path
    assert "$.security.cloud_providers" in by_path
    assert "$.security.monitoring" in by_path
    assert ".." not in "".join(by_path)
    assert "AWS" in by_path["$.security.cloud_providers"]
    assert "apm" in by_path["$.security.monitoring"]
