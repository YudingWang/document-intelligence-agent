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
    """Pack leaves into chunks without dropping JSON paths."""
    chunks: list[Chunk] = []
    bucket: list[str] = []
    paths: list[str] = []
    size = 0
    index = 0

    def flush() -> None:
        nonlocal bucket, paths, size, index
        if not bucket:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}_c{index:04d}",
                document_id=document_id,
                source=source,
                text="\n".join(bucket),
                json_path=paths[0] if len(paths) == 1 else f"{paths[0]}..{paths[-1]}",
            )
        )
        index += 1
        bucket, paths, size = [], [], 0

    for leaf in leaves:
        line = f"{leaf.path}: {leaf.value}"
        if bucket and size + len(line) + 1 > chunk_size:
            flush()
        bucket.append(line)
        paths.append(leaf.path)
        size += len(line) + 1
        if len(line) >= chunk_size:
            flush()
    flush()
    return chunks
