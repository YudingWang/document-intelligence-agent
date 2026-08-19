"""One-shot CLI: index a document and answer a questions JSON file."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.loaders.questions import parse_questions_file
from app.runtime import build_container


async def run(document: Path, questions: Path, output: Path | None) -> None:
    configure_logging()
    container = build_container(get_settings())
    record = container.ingestion.ingest(
        filename=document.name,
        content_type=None,
        data=document.read_bytes(),
    )
    question_list = parse_questions_file(questions.read_bytes(), questions.name)
    results = await container.qa.answer_batch(record.document_id, question_list)
    payload = {
        "document_id": record.document_id,
        "filename": record.filename,
        "results": [item.model_dump() for item in results],
    }
    text = json.dumps(payload, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run document QA from the command line.")
    parser.add_argument("--document", required=True, type=Path, help="PDF or JSON source document")
    parser.add_argument("--questions", required=True, type=Path, help="Questions JSON file")
    parser.add_argument("--out", type=Path, help="Optional output JSON path")
    args = parser.parse_args()
    asyncio.run(run(args.document, args.questions, args.out))


if __name__ == "__main__":
    main()
