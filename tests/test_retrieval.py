"""Tests for retrieval components."""

from src.retrieval.context import build_context
from src.vectorstore.base import SearchResult


def test_build_context_empty():
    assert build_context([]) == ""


def test_build_context_formats_sources():
    results = [
        SearchResult(
            content="Chunk one content",
            metadata={"filename": "doc1.pdf"},
            score=0.95,
        ),
        SearchResult(
            content="Chunk two content",
            metadata={"filename": "doc2.txt"},
            score=0.80,
        ),
    ]
    context = build_context(results)
    assert "doc1.pdf" in context
    assert "doc2.txt" in context
    assert "0.950" in context
    assert "Chunk one content" in context


def test_build_context_truncates():
    # Create results that exceed max_tokens
    big_chunk = "x" * 20_000
    results = [SearchResult(content=big_chunk, metadata={}, score=0.9)]
    context = build_context(results, max_tokens=100)
    assert len(context) <= 500  # 100 * 4 + some header overhead
