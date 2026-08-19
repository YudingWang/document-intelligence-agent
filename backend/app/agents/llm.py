"""OpenAI chat wrapper: grounded answer (structured) and query rewrite."""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.prompts import CONTEXTUALIZE_PROMPT, REWRITE_PROMPT, SYSTEM_PROMPT
from app.core.config import Settings
from app.core.exceptions import LLMServiceError
from app.core.secrets import require_openai_key
from app.models.domain import ChatTurn, GroundedAnswer, RetrievedChunk


class LLMClient(Protocol):
    async def generate_answer(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer: ...

    async def rewrite_query(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> str: ...

    async def standalone_query(
        self, question: str, history: list[ChatTurn]
    ) -> str: ...


class OpenAILLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key or None,
        )

    def _ensure_ready(self) -> None:
        require_openai_key(self._settings)

    async def generate_answer(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        self._ensure_ready()
        llm = self._chat.with_structured_output(GroundedAnswer)
        try:
            result = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_format_generate_prompt(question, chunks)),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError("The language model failed while generating an answer.") from exc
        if isinstance(result, GroundedAnswer):
            return result
        return GroundedAnswer.model_validate(result)

    async def rewrite_query(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Short completion: better search terms, not a full answer."""
        evidence = "\n".join(chunk.text[:400] for chunk in chunks[:4]) or "(no chunks)"
        return await self._complete(
            "You rewrite retrieval queries.",
            f"{REWRITE_PROMPT}\n\nQuestion:\n{question}\n\nWeak evidence:\n{evidence}",
            fallback=question,
            error="The language model failed while rewriting the query.",
        )

    async def standalone_query(self, question: str, history: list[ChatTurn]) -> str:
        """Resolve follow-ups like 'what region?' without using history as evidence."""
        if not history:
            return question
        transcript = "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text[:500]}"
            for turn in history[-6:]
        )
        return await self._complete(
            "You rewrite follow-up questions.",
            (
                f"{CONTEXTUALIZE_PROMPT}\n\nChat history:\n{transcript}\n\n"
                f"Latest question:\n{question}"
            ),
            fallback=question,
            error="The language model failed while resolving the follow-up question.",
        )

    async def _complete(
        self, system: str, prompt: str, *, fallback: str, error: str
    ) -> str:
        self._ensure_ready()
        try:
            response = await self._chat.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(error) from exc
        text = response.content if isinstance(response.content, str) else str(response.content)
        return text.strip() or fallback


def _format_generate_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        location = []
        if chunk.page is not None:
            location.append(f"page {chunk.page}")
        if chunk.json_path:
            location.append(chunk.json_path)
        loc = f" ({', '.join(location)})" if location else ""
        blocks.append(f"[{chunk.chunk_id}] source={chunk.source}{loc}\n{chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no retrieved evidence)"
    return f"Question:\n{question}\n\nEvidence:\n{context}"
