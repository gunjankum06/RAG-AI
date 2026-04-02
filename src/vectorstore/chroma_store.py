"""ChromaDB vector store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.core.logging import logger
from src.embeddings.ollama import OllamaEmbeddings
from src.vectorstore.base import BaseVectorStore, SearchResult

if TYPE_CHECKING:
    from langchain_core.documents import Document


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB-backed vector store."""

    def __init__(self, embeddings: OllamaEmbeddings | None = None):
        persist_dir = str(settings.chroma_persist_dir)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embeddings = embeddings or OllamaEmbeddings()
        logger.info("ChromaDB initialized at %s", persist_dir)

    def _get_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_documents(
        self, documents: list[Document], collection: str = "default"
    ) -> int:
        if not documents:
            return 0

        col = self._get_collection(collection)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Use content hash as stable ID (enables idempotent ingestion)
        ids = [
            doc.metadata.get("content_hash", f"doc_{i}")
            for i, doc in enumerate(documents)
        ]

        # Embed in batches
        try:
            embeddings = await self._embeddings.embed_texts(texts)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed documents: {exc}") from exc

        # Upsert into Chroma
        try:
            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

        logger.info("Upserted %d chunks into collection '%s'", len(documents), collection)
        return len(documents)

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "default",
        top_k: int = 5,
    ) -> list[SearchResult]:
        col = self._get_collection(collection)
        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB returns distances; convert to similarity score
                score = 1.0 - dist
                search_results.append(
                    SearchResult(content=doc, metadata=meta, score=score)
                )

        return search_results

    async def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(name=collection)
            logger.info("Deleted collection '%s'", collection)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete collection '{collection}': {exc}"
            ) from exc

    async def list_collections(self) -> list[dict]:
        collections = self._client.list_collections()
        result = []
        for col in collections:
            result.append({"name": col.name, "count": col.count()})
        return result

    async def collection_count(self, collection: str) -> int:
        col = self._get_collection(collection)
        return col.count()

    async def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False
