"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class BaseVectorStore(ABC):
    """Interface that all vector store backends must implement."""

    @abstractmethod
    async def add_documents(
        self, documents: list[Document], collection: str = "default"
    ) -> int:
        """Add documents to the store. Returns number of documents added."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        collection: str = "default",
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar documents by embedding vector."""

    @abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Delete an entire collection."""

    @abstractmethod
    async def list_collections(self) -> list[dict]:
        """List all collections with their document counts."""

    @abstractmethod
    async def collection_count(self, collection: str) -> int:
        """Return the number of documents in a collection."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the vector store is reachable and healthy."""
