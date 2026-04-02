"""FAISS vector store implementation with persistence."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.core.logging import logger
from src.embeddings.ollama import OllamaEmbeddings
from src.vectorstore.base import BaseVectorStore, SearchResult

if TYPE_CHECKING:
    from langchain_core.documents import Document

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store with file-based persistence."""

    def __init__(self, embeddings: OllamaEmbeddings | None = None):
        if faiss is None:
            raise VectorStoreError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            )
        self._base_dir = Path(settings.faiss_index_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings or OllamaEmbeddings()
        self._indices: dict[str, faiss.IndexFlatIP] = {}
        self._metadata: dict[str, list[dict]] = {}
        self._documents: dict[str, list[str]] = {}
        logger.info("FAISS store initialized at %s", self._base_dir)

    def _collection_dir(self, collection: str) -> Path:
        return self._base_dir / collection

    def _load_collection(self, collection: str) -> None:
        """Load a collection from disk if it exists and isn't already loaded."""
        if collection in self._indices:
            return

        col_dir = self._collection_dir(collection)
        index_path = col_dir / "index.faiss"
        meta_path = col_dir / "metadata.pkl"

        if index_path.exists() and meta_path.exists():
            self._indices[collection] = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self._metadata[collection] = data["metadata"]
            self._documents[collection] = data["documents"]
            logger.info("Loaded FAISS collection '%s' from disk", collection)

    def _save_collection(self, collection: str) -> None:
        """Persist a collection to disk."""
        col_dir = self._collection_dir(collection)
        col_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._indices[collection], str(col_dir / "index.faiss"))
        with open(col_dir / "metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "metadata": self._metadata[collection],
                    "documents": self._documents[collection],
                },
                f,
            )

    def _ensure_index(self, collection: str, dim: int) -> None:
        """Create an index for the collection if one doesn't exist."""
        self._load_collection(collection)
        if collection not in self._indices:
            # Inner-product index (use normalized vectors for cosine similarity)
            self._indices[collection] = faiss.IndexFlatIP(dim)
            self._metadata[collection] = []
            self._documents[collection] = []

    async def add_documents(
        self, documents: list[Document], collection: str = "default"
    ) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            embeddings = await self._embeddings.embed_texts(texts)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed documents: {exc}") from exc

        vectors = np.array(embeddings, dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        self._ensure_index(collection, dim)
        self._indices[collection].add(vectors)
        self._metadata[collection].extend(metadatas)
        self._documents[collection].extend(texts)

        self._save_collection(collection)
        logger.info("Added %d vectors to FAISS collection '%s'", len(documents), collection)
        return len(documents)

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "default",
        top_k: int = 5,
    ) -> list[SearchResult]:
        self._load_collection(collection)
        if collection not in self._indices:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        n_stored = self._indices[collection].ntotal
        k = min(top_k, n_stored)
        if k == 0:
            return []

        scores, indices = self._indices[collection].search(query_vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    content=self._documents[collection][idx],
                    metadata=self._metadata[collection][idx],
                    score=float(score),
                )
            )
        return results

    async def delete_collection(self, collection: str) -> None:
        col_dir = self._collection_dir(collection)
        if col_dir.exists():
            import shutil
            shutil.rmtree(col_dir)
        self._indices.pop(collection, None)
        self._metadata.pop(collection, None)
        self._documents.pop(collection, None)
        logger.info("Deleted FAISS collection '%s'", collection)

    async def list_collections(self) -> list[dict]:
        results: list[dict] = []
        if self._base_dir.exists():
            for child in self._base_dir.iterdir():
                if child.is_dir() and (child / "index.faiss").exists():
                    self._load_collection(child.name)
                    count = self._indices.get(child.name, None)
                    results.append({
                        "name": child.name,
                        "count": count.ntotal if count else 0,
                    })
        return results

    async def collection_count(self, collection: str) -> int:
        self._load_collection(collection)
        idx = self._indices.get(collection)
        return idx.ntotal if idx else 0

    async def health_check(self) -> bool:
        return faiss is not None
