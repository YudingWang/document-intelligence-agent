"""Liveness probe. `configured` is true once a real API key (or fake embeddings) is set."""

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.core.secrets import has_openai_key
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(container: ContainerDep) -> HealthResponse:
    settings = container.settings
    configured = has_openai_key(settings) or settings.embedding_backend == "fake"
    return HealthResponse(
        status="ok",
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        configured=configured,
    )
