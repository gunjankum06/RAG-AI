"""LRU cache for embedding vectors to avoid redundant Ollama calls."""

from __future__ import annotations

from collections import OrderedDict

from src.core.logging import logger


class EmbeddingCache:
    """Simple in-memory LRU cache for embedding vectors."""

    def __init__(self, max_size: int = 10_000):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> list[float] | None:
        if text in self._cache:
            self._cache.move_to_end(text)
            self._hits += 1
            return self._cache[text]
        self._misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        if text in self._cache:
            self._cache.move_to_end(text)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[text] = embedding

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Embedding cache cleared")
