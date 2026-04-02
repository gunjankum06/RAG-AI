"""Tests for the ingestion pipeline components."""

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.dedup import compute_hash, deduplicate


class FakeDocument:
    """Minimal stand-in for langchain Document."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ── Hashing ───────────────────────────────────────────────────────────


class TestComputeHash:
    def test_deterministic(self):
        assert compute_hash("hello") == compute_hash("hello")

    def test_different_inputs(self):
        assert compute_hash("hello") != compute_hash("world")

    def test_empty_string(self):
        h = compute_hash("")
        assert isinstance(h, str) and len(h) == 64

    def test_unicode(self):
        h = compute_hash("日本語テスト")
        assert isinstance(h, str) and len(h) == 64


# ── Deduplication ─────────────────────────────────────────────────────


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        docs = [
            FakeDocument("same content"),
            FakeDocument("same content"),
            FakeDocument("different content"),
        ]
        unique = deduplicate(docs)
        assert len(unique) == 2

    def test_preserves_order(self):
        docs = [
            FakeDocument("first"),
            FakeDocument("second"),
            FakeDocument("first"),
        ]
        unique = deduplicate(docs)
        assert unique[0].page_content == "first"
        assert unique[1].page_content == "second"

    def test_adds_content_hash_metadata(self):
        docs = [FakeDocument("test")]
        unique = deduplicate(docs)
        assert "content_hash" in unique[0].metadata

    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_all_unique(self):
        docs = [FakeDocument(f"unique content {i}") for i in range(10)]
        assert len(deduplicate(docs)) == 10


# ── Chunking ──────────────────────────────────────────────────────────


class TestChunkDocuments:
    def test_splits_long_document(self):
        long_text = "This is a sentence. " * 200  # ~4000 chars
        docs = [FakeDocument(long_text, metadata={"source": "test.txt"})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_adds_chunk_index(self):
        long_text = "This is a sentence. " * 200
        docs = [FakeDocument(long_text)]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_preserves_metadata(self):
        long_text = "This is a sentence. " * 200
        docs = [FakeDocument(long_text, metadata={"source": "test.txt", "custom": "value"})]
        chunks = chunk_documents(docs)
        assert chunks[0].metadata.get("custom") == "value"

    def test_short_document_single_chunk(self):
        docs = [FakeDocument("Short text")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1

    def test_empty_input(self):
        assert chunk_documents([]) == []
