"""Tests for the ingestion pipeline components."""

from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.dedup import compute_hash, deduplicate


class FakeDocument:
    """Minimal stand-in for langchain Document."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def test_compute_hash_deterministic():
    assert compute_hash("hello") == compute_hash("hello")
    assert compute_hash("hello") != compute_hash("world")


def test_deduplicate_removes_duplicates():
    docs = [
        FakeDocument("same content"),
        FakeDocument("same content"),
        FakeDocument("different content"),
    ]
    unique = deduplicate(docs)
    assert len(unique) == 2


def test_deduplicate_preserves_order():
    docs = [
        FakeDocument("first"),
        FakeDocument("second"),
        FakeDocument("first"),
    ]
    unique = deduplicate(docs)
    assert unique[0].page_content == "first"
    assert unique[1].page_content == "second"


def test_chunk_documents():
    # Create a long document that should be split
    long_text = "This is a sentence. " * 200  # ~4000 chars
    docs = [FakeDocument(long_text, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
