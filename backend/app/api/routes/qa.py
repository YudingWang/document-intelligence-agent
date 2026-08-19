"""Question-answering routes: one-shot, batch, and chat."""

from fastapi import APIRouter, File, Form, UploadFile

from app.api.deps import ContainerDep
from app.api.uploads import read_upload
from app.core.exceptions import InvalidInputError
from app.core.logging import document_id_var
from app.loaders.questions import parse_questions_file
from app.models.schemas import BatchQAResponse, ChatRequest, ChatResponse

router = APIRouter(tags=["qa"])


@router.post("/api/v1/qa", response_model=BatchQAResponse)
async def one_shot_qa(
    container: ContainerDep,
    document: UploadFile = File(..., description="Source PDF or JSON document"),
    questions: UploadFile = File(..., description="JSON list of questions"),
) -> BatchQAResponse:
    """Assignment contract: two files in, structured question/answer list out."""
    document_bytes = await read_upload(container, document)
    questions_bytes = await read_upload(container, questions)
    question_list = parse_questions_file(questions_bytes, questions.filename or "questions.json")
    record = container.ingestion.ingest(
        filename=document.filename or "document",
        content_type=document.content_type,
        data=document_bytes,
    )
    token = document_id_var.set(record.document_id)
    try:
        results = await container.qa.answer_batch(record.document_id, question_list)
    finally:
        document_id_var.reset(token)
    return _batch_payload(container, [record.document_id], results)


@router.post("/api/v1/qa/batch", response_model=BatchQAResponse)
async def batch_qa(
    container: ContainerDep,
    questions: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    document_ids: str | None = Form(default=None),
) -> BatchQAResponse:
    """Run a questions file against one or more already indexed documents."""
    ids = container.qa.resolve_ids(document_id, document_ids)
    questions_bytes = await read_upload(container, questions)
    question_list = parse_questions_file(questions_bytes, questions.filename or "questions.json")
    results = await container.qa.answer_batch(ids, question_list)
    return _batch_payload(container, ids, results)


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(container: ContainerDep, payload: ChatRequest) -> ChatResponse:
    """Answer a question, using history only to resolve follow-up references."""
    if not payload.document_ids:
        raise InvalidInputError("Select at least one document.")
    result = await container.qa.answer_one(
        payload.document_ids,
        payload.message,
        history=payload.history,
    )
    return ChatResponse(
        document_id=payload.document_ids[0],
        document_ids=payload.document_ids,
        question=result.question,
        answer=result.answer,
        supported=result.supported,
        citations=result.citations,
    )


def _batch_payload(container: ContainerDep, ids: list[str], results) -> BatchQAResponse:
    records = [container.catalog.get(item) for item in ids]
    return BatchQAResponse(
        document_id=ids[0],
        document_ids=ids,
        filename=records[0].filename,
        filenames=[item.filename for item in records],
        results=results,
    )
