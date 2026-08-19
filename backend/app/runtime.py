"""Wire settings, stores, and services into one process-wide container.

Used by the FastAPI lifespan and the CLI so both entrypoints share the same graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agents.graph import DocumentQAAgent
from app.agents.llm import LLMClient, OpenAILLMClient
from app.core.config import Settings
from app.repositories.document_catalog import DocumentCatalog
from app.repositories.embeddings import create_embeddings
from app.repositories.vector_index import create_vector_index
from app.services.ingestion import IngestionService
from app.services.qa import QAService
from app.tools.documents import DocumentTools

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    """Application services shared across HTTP handlers."""

    settings: Settings
    catalog: DocumentCatalog
    ingestion: IngestionService
    qa: QAService
    agent: DocumentQAAgent


def build_container(
    settings: Settings,
    llm: LLMClient | None = None,
    embeddings=None,
) -> AppContainer:
    """Create catalog, vector index, ingestion, and the QA agent."""
    data_dir = settings.resolved_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    embedder = embeddings or create_embeddings(settings)
    vector_index = create_vector_index(settings, embedder)
    catalog = DocumentCatalog(data_dir / "catalog.json")
    tools = DocumentTools(catalog, vector_index)
    llm_client = llm or OpenAILLMClient(settings)
    agent = DocumentQAAgent(settings, llm_client, tools)
    ingestion = IngestionService(
        settings,
        catalog,
        vector_index,
        uploads_dir=data_dir / "uploads",
    )
    qa = QAService(settings, catalog, agent)
    try:
        ingestion.ensure_sample()
    except Exception as exc:  # noqa: BLE001
        logger.warning("sample document was not seeded: %s", exc)
    return AppContainer(
        settings=settings,
        catalog=catalog,
        ingestion=ingestion,
        qa=qa,
        agent=agent,
    )
