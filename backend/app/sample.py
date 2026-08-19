"""Locate bundled sample files used for the demo document."""

from __future__ import annotations

import json
from pathlib import Path

SAMPLE_DOCUMENT_ID = "doc_sample"
SAMPLE_FILENAME = "sample_eval.json"


def sample_data_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "sample_data"
        if candidate.is_dir() and (parent / "backend").is_dir():
            return candidate
    return None


def load_sample_questions() -> list[str]:
    root = sample_data_dir()
    if root is None:
        return []
    path = root / "sample_eval_questions.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        return []
    return [str(item).strip() for item in questions if str(item).strip()]
