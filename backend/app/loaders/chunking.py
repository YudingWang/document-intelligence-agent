"""Turn parsed PDF pages / JSON leaves into retrieval chunks with stable ids."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.loaders.json_loader import JsonLeaf
from app.loaders.pdf_loader import PdfPage
from app.models.domain import Chunk


def chunk_pdf_pages(
    pages: list[PdfPage],
    *,
    document_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        if not page.text:
            continue
        for piece in splitter.split_text(page.text):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_c{index:04d}",
                    document_id=document_id,
                    source=source,
                    text=text,
                    page=page.page_number,
                )
            )
            index += 1
    return chunks


def chunk_json_leaves(
    leaves: list[JsonLeaf],
    *,
    document_id: str,
    source: str,
    chunk_size: int,
) -> list[Chunk]:
    """Group leaves by parent object so citations stay on one JSON section."""
    chunks: list[Chunk] = []
    index = 0
    for parent, group in _group_by_parent(leaves):
        bucket: list[str] = []
        size = 0
        for leaf in group:
            line = f"{leaf.path}: {leaf.value}"
            if bucket and size + len(line) + 1 > chunk_size:
                chunks.append(_json_chunk(document_id, source, index, bucket, parent))
                index += 1
                bucket, size = [], 0
            bucket.append(line)
            size += len(line) + 1
            if len(line) >= chunk_size:
                chunks.append(_json_chunk(document_id, source, index, bucket, parent))
                index += 1
                bucket, size = [], 0
        if bucket:
            chunks.append(_json_chunk(document_id, source, index, bucket, parent))
            index += 1
    return chunks


def json_section_path(path: str) -> str:
    """Citation path: parent object, or the leaf itself for top-level scalars."""
    if path.endswith("]"):
        bracket = path.rfind("[")
        return path[:bracket] if bracket > 0 else path
    if path.count(".") <= 1:
        return path
    return path.rsplit(".", 1)[0]


def _group_by_parent(leaves: list[JsonLeaf]) -> list[tuple[str, list[JsonLeaf]]]:
    groups: list[tuple[str, list[JsonLeaf]]] = []
    for leaf in leaves:
        parent = json_section_path(leaf.path)
        if groups and groups[-1][0] == parent:
            groups[-1][1].append(leaf)
        else:
            groups.append((parent, [leaf]))
    return groups


def _json_chunk(
    document_id: str,
    source: str,
    index: int,
    lines: list[str],
    json_path: str,
) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}_c{index:04d}",
        document_id=document_id,
        source=source,
        text="\n".join(lines),
        json_path=json_path,
    )
