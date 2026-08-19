"""Fan out independent questions with a concurrency cap."""

from __future__ import annotations

import asyncio
import logging

from app.agents.graph import DocumentQAAgent
from app.core.config import Settings
from app.core.exceptions import InvalidInputError
from app.core.logging import document_id_var
from app.models.domain import ChatTurn, QAResult
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

    def resolve_ids(self, *groups: str | list[str] | None) -> list[str]:
        found: list[str] = []
        for group in groups:
            if group is None or group == "":
                continue
            values = group if isinstance(group, list) else str(group).split(",")
            for item in values:
                text = str(item).strip()
                if text and text not in found:
                    found.append(text)
        if not found:
            raise InvalidInputError("Select at least one document.")
        for document_id in found:
            self._catalog.get(document_id)
        return found

    async def answer_one(
        self,
        document_ids: str | list[str],
        question: str,
        history: list[ChatTurn] | None = None,
    ) -> QAResult:
        ids = self.resolve_ids(document_ids)
        token = document_id_var.set(",".join(ids))
        try:
            result = await self._agent.answer(question, ids, history=history)
        finally:
            document_id_var.reset(token)
        logger.info(
            "qa.complete supported=%s attempts=%s total_ms=%s docs=%s",
            result.supported,
            result.search_attempts,
            result.total_latency_ms,
            len(ids),
        )
        return result

    async def answer_batch(
        self, document_ids: str | list[str], questions: list[str]
    ) -> list[QAResult]:
        ids = self.resolve_ids(document_ids)
        semaphore = asyncio.Semaphore(self._settings.batch_concurrency)

        async def run(question: str) -> QAResult:
            async with semaphore:
                return await self.answer_one(ids, question)

        return list(await asyncio.gather(*[run(question) for question in questions]))
