"""Tests for retrieval components."""

import pytest

from src.embeddings.cache import EmbeddingCache
from src.retrieval.context import build_context
from src.vectorstore.base import SearchResult


# ── Context Builder ───────────────────────────────────────────────────


class TestBuildContext:
    def test_empty_results(self):
        assert build_context([]) == ""

    def test_formats_sources(self):
        results = [
            SearchResult(content="Chunk one content", metadata={"filename": "doc1.pdf"}, score=0.95),
            SearchResult(content="Chunk two content", metadata={"filename": "doc2.txt"}, score=0.80),
        ]
        context = build_context(results)
        assert "doc1.pdf" in context
        assert "doc2.txt" in context
        assert "0.950" in context
        assert "Chunk one content" in context

    def test_truncates_large_context(self):
        big_chunk = "x" * 20_000
        results = [SearchResult(content=big_chunk, metadata={}, score=0.9)]
        context = build_context(results, max_tokens=100)
        assert len(context) <= 500  # 100 * 4 + some header overhead

    def test_multiple_sources_separator(self):
        results = [
            SearchResult(content="A", metadata={"filename": "a.txt"}, score=0.9),
            SearchResult(content="B", metadata={"filename": "b.txt"}, score=0.8),
        ]
        context = build_context(results)
        assert "---" in context

    def test_missing_filename_fallback(self):
        results = [SearchResult(content="X", metadata={"source": "/path/file.pdf"}, score=0.9)]
        context = build_context(results)
        assert "/path/file.pdf" in context


# ── Embedding Cache ───────────────────────────────────────────────────


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("hello", [0.1, 0.2])
        assert cache.get("hello") == [0.1, 0.2]

    def test_cache_miss(self):
        cache = EmbeddingCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = EmbeddingCache(max_size=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])  # Evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]

    def test_hit_rate_stats(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("x", [1.0])
        cache.get("x")  # hit
        cache.get("y")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("a", [1.0])
        cache.clear()
        assert cache.get("a") is None
        assert cache.stats()["size"] == 0
