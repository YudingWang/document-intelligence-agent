"""Helpers for checking that a real OpenAI key is present (never log the value)."""

from app.core.config import Settings
from app.core.exceptions import ConfigurationError


def has_openai_key(settings: Settings) -> bool:
    key = settings.openai_api_key.strip()
    return bool(key) and not key.startswith("sk-your-")


def require_openai_key(settings: Settings) -> None:
    if not has_openai_key(settings):
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. Put your key in .env before ingesting or answering."
        )
