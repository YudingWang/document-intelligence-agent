"""HTTP tests: upload, one-shot QA, batch, chat, and error codes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.repositories.embeddings import TokenHashEmbeddings
from tests.conftest import StubLLM, make_pdf_bytes

SAMPLE_JSON = Path(__file__).resolve().parents[2] / "sample_data" / "vendor_security.json"
QUESTIONS = Path(__file__).resolve().parents[2] / "sample_data" / "questions.json"


def test_sample_document_is_seeded(tmp_path, stub_llm: StubLLM) -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        embedding_backend="fake",
        vector_backend="memory",
        data_dir=tmp_path / "data",
        seed_sample_document=True,
    )
    app = create_app(settings=settings, llm=stub_llm, embeddings=TokenHashEmbeddings())
    with TestClient(app) as seeded:
        listed = seeded.get("/api/v1/documents").json()["documents"]
    assert any(item["is_sample"] and item["filename"] == "sample_eval.json" for item in listed)


def test_sample_questions_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/sample/questions")
    assert response.status_code == 200
    questions = response.json()["questions"]
    assert len(questions) == 19
    assert "Where are your data centres located?" in questions


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


def test_delete_document(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    document_id = upload.json()["document_id"]
    deleted = client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["documents"] == []
    listed = client.get("/api/v1/documents")
    assert listed.json()["documents"] == []
    missing = client.post(
        "/api/v1/chat",
        json={"document_id": document_id, "message": "Which cloud provider?"},
    )
    assert missing.status_code == 404


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


def test_batch_and_chat_multiple_documents(client: TestClient) -> None:
    first = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    second = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    ids = [first.json()["document_id"], second.json()["document_id"]]
    batch = client.post(
        "/api/v1/qa/batch",
        data={"document_ids": ",".join(ids)},
        files={"questions": ("questions.json", QUESTIONS.read_bytes(), "application/json")},
    )
    assert batch.status_code == 200
    assert batch.json()["document_ids"] == ids
    chat = client.post(
        "/api/v1/chat",
        json={"document_ids": ids, "message": "Which cloud providers do you rely on?"},
    )
    assert chat.status_code == 200
    assert chat.json()["document_ids"] == ids


def test_upload_sanitizes_filename(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"document": ("../etc/my file.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "my_file.json"
    assert ".." not in response.json()["filename"]
    assert "/" not in response.json()["filename"]


def test_chat_history_is_accepted(client: TestClient, stub_llm) -> None:
    upload = client.post(
        "/api/v1/documents",
        files={"document": ("vendor_security.json", SAMPLE_JSON.read_bytes(), "application/json")},
    )
    document_id = upload.json()["document_id"]
    response = client.post(
        "/api/v1/chat",
        json={
            "document_id": document_id,
            "message": "What region?",
            "history": [
                {"role": "user", "text": "Which cloud providers do you rely on?"},
                {"role": "agent", "text": "AWS is the primary provider."},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["question"] == "What region?"
    assert stub_llm.standalone_calls == 1


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
