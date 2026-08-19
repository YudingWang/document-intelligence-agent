"""HTTP tests: upload, one-shot QA, batch, chat, and error codes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_pdf_bytes

SAMPLE_JSON = Path(__file__).resolve().parents[2] / "sample_data" / "vendor_security.json"
QUESTIONS = Path(__file__).resolve().parents[2] / "sample_data" / "questions.json"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "gpt-4o-mini"


def test_upload_json_and_list(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_type"] == "json"
    assert body["status"] == "ready"
    assert body["chunks"] > 0

    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["document_id"] == body["document_id"]


def test_upload_pdf(client: TestClient) -> None:
    pdf = make_pdf_bytes("Acme relies on Amazon Web Services (AWS) in us-east-1.")
    response = client.post(
        "/api/v1/documents",
        files={"document": ("policy.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["file_type"] == "pdf"
    assert response.json()["pages"] == 1


def test_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"document": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_unknown_document(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"document_id": "missing", "message": "Which cloud provider?"},
    )
    assert response.status_code == 404


def test_one_shot_qa(client: TestClient) -> None:
    response = client.post(
        "/api/v1/qa",
        files={
            "document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json"),
            "questions": ("questions.json", QUESTIONS.read_bytes(), "application/json"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 5
    assert {item["question"] for item in body["results"]}
    for item in body["results"]:
        assert "question" in item and "answer" in item


def test_batch_and_chat(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    document_id = upload.json()["document_id"]

    batch = client.post(
        "/api/v1/qa/batch",
        data={"document_id": document_id},
        files={"questions": ("questions.json", QUESTIONS.read_bytes(), "application/json")},
    )
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 5

    chat = client.post(
        "/api/v1/chat",
        json={"document_id": document_id, "message": "Which cloud providers do you rely on?"},
    )
    assert chat.status_code == 200
    assert "answer" in chat.json()


def test_malformed_questions(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    document_id = upload.json()["document_id"]
    response = client.post(
        "/api/v1/qa/batch",
        data={"document_id": document_id},
        files={"questions": ("questions.json", b"{bad", "application/json")},
    )
    assert response.status_code == 400
