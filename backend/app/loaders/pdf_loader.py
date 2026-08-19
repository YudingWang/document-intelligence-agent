"""Extract page-level text from a PDF. Encrypted or empty files are rejected."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from app.core.exceptions import UnreadableDocumentError


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str


def load_pdf(data: bytes, filename: str) -> list[PdfPage]:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - pymupdf raises mixed errors
        raise UnreadableDocumentError(f"Unable to open PDF '{filename}'.") from exc

    if document.is_encrypted:
        raise UnreadableDocumentError(f"PDF '{filename}' is encrypted and cannot be read.")

    pages: list[PdfPage] = []
    try:
        for index, page in enumerate(document, start=1):
            text = _normalize_text(page.get_text("text") or "")
            pages.append(PdfPage(page_number=index, text=text))
    finally:
        document.close()

    if not any(page.text for page in pages):
        raise UnreadableDocumentError(
            f"PDF '{filename}' did not contain extractable text."
        )
    return pages


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return compact.strip()
