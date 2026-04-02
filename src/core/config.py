from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "nomic-embed-text"

    # Vector store
    vector_store_type: str = "chroma"  # "chroma" | "faiss"
    chroma_persist_dir: Path = Path("./storage/chroma")
    faiss_index_dir: Path = Path("./storage/faiss")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    rerank_enabled: bool = True
    rerank_top_n: int = 3

    # Embeddings
    embedding_batch_size: int = 64

    # API
    api_key: str = "change-me-to-a-secure-random-string"
    rate_limit_per_minute: int = 60

    # Logging
    log_level: str = "INFO"


settings = Settings()
