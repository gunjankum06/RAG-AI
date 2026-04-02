"""Shared test fixtures and configuration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.vectorstore.base import SearchResult


@pytest.fixture
def mock_embeddings():
    """Mock OllamaEmbeddings that returns deterministic vectors."""
    emb = AsyncMock()
    emb.embed_query = AsyncMock(return_value=[0.1] * 768)
    emb.embed_texts = AsyncMock(return_value=[[0.1] * 768])
    emb.close = AsyncMock()
    return emb


@pytest.fixture
def mock_vector_store():
    """Mock vector store implementing BaseVectorStore interface."""
    store = AsyncMock()
    store.add_documents = AsyncMock(return_value=5)
    store.search = AsyncMock(
        return_value=[
            SearchResult(
                content="Test chunk content",
                metadata={"filename": "test.pdf", "chunk_index": 0},
                score=0.95,
            ),
        ]
    )
    store.delete_collection = AsyncMock()
    store.list_collections = AsyncMock(return_value=[{"name": "default", "count": 10}])
    store.collection_count = AsyncMock(return_value=10)
    store.health_check = AsyncMock(return_value=True)
    return store


@pytest.fixture
def mock_llm():
    """Mock OllamaLLM that returns a canned response."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="This is a test answer based on the documents.")
    llm.health_check = AsyncMock(return_value=True)
    llm.close = AsyncMock()
    return llm


@pytest.fixture
def sample_documents():
    """Create sample LangChain-like document objects."""

    class FakeDoc:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    return [
        FakeDoc("First document content about machine learning.", {"source": "doc1.txt", "filename": "doc1.txt"}),
        FakeDoc("Second document about natural language processing.", {"source": "doc2.txt", "filename": "doc2.txt"}),
        FakeDoc("Third document covering retrieval augmented generation.", {"source": "doc3.txt", "filename": "doc3.txt"}),
    ]


@pytest.fixture
def sample_search_results():
    """Pre-built search results for retrieval tests."""
    return [
        SearchResult(content="Machine learning is a subset of AI.", metadata={"filename": "ml.pdf", "chunk_index": 0}, score=0.95),
        SearchResult(content="NLP deals with text understanding.", metadata={"filename": "nlp.pdf", "chunk_index": 1}, score=0.87),
        SearchResult(content="RAG combines retrieval with generation.", metadata={"filename": "rag.pdf", "chunk_index": 2}, score=0.82),
    ]
