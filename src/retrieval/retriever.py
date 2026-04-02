"""Vector search retriever with optional metadata filtering."""

from __future__ import annotations

from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.logging import logger
from src.embeddings.cache import EmbeddingCache
from src.embeddings.ollama import OllamaEmbeddings
from src.vectorstore.base import BaseVectorStore, SearchResult


class Retriever:
    """Retrieves relevant chunks from the vector store for a given query."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: OllamaEmbeddings | None = None,
        cache: EmbeddingCache | None = None,
    ):
        self._store = vector_store
        self._embeddings = embeddings or OllamaEmbeddings()
        self._cache = cache or EmbeddingCache()

    async def retrieve(
        self,
        query: str,
        collection: str = "default",
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Embed the query and return the top-K most similar chunks."""
        top_k = top_k or settings.top_k

        # Check cache
        cached = self._cache.get(query)
        if cached is not None:
            query_embedding = cached
        else:
            try:
                query_embedding = await self._embeddings.embed_query(query)
                self._cache.put(query, query_embedding)
            except Exception as exc:
                raise RetrievalError(f"Failed to embed query: {exc}") from exc

        results = await self._store.search(
            query_embedding=query_embedding,
            collection=collection,
            top_k=top_k,
        )

        logger.info(
            "Retrieved %d chunks for query (collection=%s, top_k=%d)",
            len(results),
            collection,
            top_k,
        )
        return results
