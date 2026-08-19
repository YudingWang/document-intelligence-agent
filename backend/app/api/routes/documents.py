"""Upload, list, and fetch indexed source documents."""

from fastapi import APIRouter, UploadFile

from app.api.deps import ContainerDep
from app.api.uploads import read_upload
from app.models.domain import DocumentRecord
from app.models.schemas import DocumentListResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentRecord)
async def upload_document(container: ContainerDep, document: UploadFile) -> DocumentRecord:
    data = await read_upload(container, document)
    return container.ingestion.ingest(
        filename=document.filename or "document",
        content_type=document.content_type,
        data=data,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(container: ContainerDep) -> DocumentListResponse:
    return DocumentListResponse(documents=container.catalog.list())


@router.get("/{document_id}", response_model=DocumentRecord)
def get_document(container: ContainerDep, document_id: str) -> DocumentRecord:
    return container.catalog.get(document_id)
