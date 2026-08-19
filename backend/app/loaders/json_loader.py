"""Parse a JSON document and flatten it into path/value leaves for indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import InvalidInputError, UnreadableDocumentError


@dataclass(frozen=True)
class JsonLeaf:
    path: str
    value: str


def parse_json_bytes(data: bytes, filename: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnreadableDocumentError(f"JSON file '{filename}' is not valid UTF-8.") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"JSON file '{filename}' is malformed: {exc.msg}.") from exc


def flatten_json(value: Any, path: str = "$") -> list[JsonLeaf]:
    """Walk objects/arrays so citations can point at `$.security.region`."""
    leaves: list[JsonLeaf] = []
    _walk(value, path, leaves)
    return leaves


def _walk(value: Any, path: str, leaves: list[JsonLeaf]) -> None:
    if isinstance(value, dict):
        if not value:
            leaves.append(JsonLeaf(path=path, value="{}"))
            return
        for key, child in value.items():
            _walk(child, f"{path}.{key}", leaves)
        return
    if isinstance(value, list):
        if not value:
            leaves.append(JsonLeaf(path=path, value="[]"))
            return
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", leaves)
        return
    leaves.append(JsonLeaf(path=path, value=_stringify(value)))


def _stringify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
