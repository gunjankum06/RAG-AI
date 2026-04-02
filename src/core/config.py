from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "nomic-embed-text"
    ollama_timeout_seconds: int = Field(default=300, ge=10)
    ollama_max_retries: int = Field(default=3, ge=0, le=10)

    # ── Vector store ──────────────────────────────────────────────────
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    chroma_persist_dir: Path = Path("./storage/chroma")
    faiss_index_dir: Path = Path("./storage/faiss")

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=100, le=10_000)
    chunk_overlap: int = Field(default=200, ge=0)

    # ── Retrieval ─────────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=100)
    rerank_enabled: bool = True
    rerank_top_n: int = Field(default=3, ge=1, le=50)

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_batch_size: int = Field(default=64, ge=1, le=512)

    # ── API ───────────────────────────────────────────────────────────
    api_key: str = "change-me-to-a-secure-random-string"
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # ── Environment ───────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            msg = f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        return v

    @field_validator("rerank_top_n")
    @classmethod
    def rerank_top_n_must_be_lte_top_k(cls, v: int, info) -> int:
        top_k = info.data.get("top_k", 5)
        if v > top_k:
            msg = f"rerank_top_n ({v}) must be ≤ top_k ({top_k})"
            raise ValueError(msg)
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()
