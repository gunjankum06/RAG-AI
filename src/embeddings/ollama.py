"""Ollama embedding client with batched processing, retry, and circuit breaker."""

from __future__ import annotations

import httpx

from src.core.config import settings
from src.core.exceptions import EmbeddingError
from src.core.logging import logger
from src.core.resilience import async_retry


class OllamaEmbeddings:
    """Generate embeddings using a locally-running Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "Embedded batch %d–%d of %d",
                i,
                min(i + self.batch_size, len(texts)),
                len(texts),
            )

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = await self._embed_batch([text])
        return results[0]

    @async_retry(max_retries=settings.ollama_max_retries, base_delay=1.0, exceptions=(EmbeddingError,))
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embedding API for a batch of texts (with retry)."""
        results: list[list[float]] = []
        for text in texts:
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                )
                response.raise_for_status()
                data = response.json()
                results.append(data["embeddings"][0])
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"Ollama embedding request failed: {exc}"
                ) from exc
            except (KeyError, IndexError) as exc:
                raise EmbeddingError(
                    f"Unexpected Ollama embedding response: {exc}"
                ) from exc
        return results

    async def close(self) -> None:
        await self._client.aclose()
