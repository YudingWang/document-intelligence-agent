"""Fan out independent questions with a concurrency cap."""

from __future__ import annotations

import asyncio
import logging

from app.agents.graph import DocumentQAAgent
from app.core.config import Settings
from app.core.logging import document_id_var
from app.models.domain import QAResult
from app.repositories.document_catalog import DocumentCatalog

logger = logging.getLogger(__name__)


class QAService:
    def __init__(
        self,
        settings: Settings,
        catalog: DocumentCatalog,
        agent: DocumentQAAgent,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._agent = agent

    async def answer_one(self, document_id: str, question: str) -> QAResult:
        self._catalog.get(document_id)
        token = document_id_var.set(document_id)
        try:
            result = await self._agent.answer(question, document_id)
        finally:
            document_id_var.reset(token)
        logger.info(
            "qa.complete supported=%s attempts=%s total_ms=%s",
            result.supported,
            result.search_attempts,
            result.total_latency_ms,
        )
        return result

    async def answer_batch(self, document_id: str, questions: list[str]) -> list[QAResult]:
        self._catalog.get(document_id)
        semaphore = asyncio.Semaphore(self._settings.batch_concurrency)

        async def run(question: str) -> QAResult:
            async with semaphore:
                return await self.answer_one(document_id, question)

        return list(await asyncio.gather(*[run(question) for question in questions]))
