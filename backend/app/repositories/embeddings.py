"""Embedding factory: OpenAI in production, token-hash vectors in tests."""

from __future__ import annotations

import math
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.core.exceptions import ConfigurationError

_TOKEN = re.compile(r"[a-z0-9]+")


class TokenHashEmbeddings(Embeddings):
    """Deterministic embeddings so tests can retrieve without an API key."""

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in _TOKEN.findall(text.lower()):
            vector[hash(token) % self.size] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def create_embeddings(settings: Settings) -> Embeddings:
    backend = settings.embedding_backend.lower()
    if backend == "fake":
        return TokenHashEmbeddings()
    if backend != "openai":
        raise ConfigurationError(f"Unsupported embedding backend '{settings.embedding_backend}'.")
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
    )
