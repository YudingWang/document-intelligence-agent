"""Environment-backed settings. `.env` is loaded from cwd and parent folders."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_files() -> tuple[Path, ...]:
    files: list[Path] = [Path.cwd() / ".env"]
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate not in files:
            files.append(candidate)
    return tuple(files)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.0

    max_file_size_mb: int = 25
    max_retrieval_attempts: int = 2
    batch_concurrency: int = 3
    retrieval_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 150

    vector_backend: str = "chroma"
    embedding_backend: str = "openai"
    data_dir: Path = Field(default=Path("./data"))
    seed_sample_document: bool = True

    @field_validator("openai_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        forbidden = {"gpt-4", "gpt-4-32k", "gpt-4-turbo", "gpt-4.1"}
        normalized = value.strip().lower()
        if normalized in forbidden or "16k" in normalized:
            raise ValueError(
                "This challenge requires gpt-4o-mini. Do not use GPT-4 or 16k models."
            )
        return value

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def resolved_data_dir(self) -> Path:
        path = Path(self.data_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
