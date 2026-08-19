"""Accept `{ "questions": [...] }`, a string list, or `{ "question": "..." }` items."""

from __future__ import annotations

from typing import Any

from app.core.exceptions import InvalidInputError
from app.loaders.json_loader import parse_json_bytes


def parse_questions_file(data: bytes, filename: str) -> list[str]:
    return parse_questions(parse_json_bytes(data, filename))


def parse_questions(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        if "questions" in payload:
            items = payload["questions"]
        elif "question" in payload:
            items = payload["question"]
        else:
            raise InvalidInputError(
                "Questions JSON must be a list or an object with a 'questions' array."
            )
        if isinstance(items, str):
            items = [items]
    elif isinstance(payload, list):
        items = payload
    else:
        raise InvalidInputError("Questions JSON must be a list or an object.")

    questions: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            raw = item.get("question") or item.get("q") or item.get("text")
            text = str(raw).strip() if raw else ""
        else:
            text = ""
        if text:
            questions.append(text)

    if not questions:
        raise InvalidInputError("No questions were found in the questions file.")
    return questions
