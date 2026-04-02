"""Shared FastAPI dependencies: auth, rate limiting, service factories."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.core.config import settings
from src.core.logging import logger
from src.embeddings.ollama import OllamaEmbeddings
from src.llm.chain import RAGChain
from src.llm.ollama import OllamaLLM
from src.retrieval.retriever import Retriever
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.chroma_store import ChromaVectorStore

# ── API Key Auth ──────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Rate Limiting (in-memory, per-key) ────────────────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


async def rate_limiter(request: Request, api_key: str = Depends(verify_api_key)) -> str:
    now = time.time()
    window = 60.0
    max_requests = settings.rate_limit_per_minute

    # Prune old entries
    _rate_limit_store[api_key] = [
        ts for ts in _rate_limit_store[api_key] if now - ts < window
    ]

    if len(_rate_limit_store[api_key]) >= max_requests:
        logger.warning("Rate limit exceeded for key ending ...%s", api_key[-6:])
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    _rate_limit_store[api_key].append(now)
    return api_key


# ── Service Singletons ───────────────────────────────────────────────

_embeddings: OllamaEmbeddings | None = None
_vector_store: BaseVectorStore | None = None
_retriever: Retriever | None = None
_llm: OllamaLLM | None = None
_rag_chain: RAGChain | None = None


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings  # noqa: PLW0603
    if _embeddings is None:
        _embeddings = OllamaEmbeddings()
    return _embeddings


def get_vector_store() -> BaseVectorStore:
    global _vector_store  # noqa: PLW0603
    if _vector_store is None:
        embeddings = get_embeddings()
        if settings.vector_store_type == "faiss":
            from src.vectorstore.faiss_store import FAISSVectorStore
            _vector_store = FAISSVectorStore(embeddings=embeddings)
        else:
            _vector_store = ChromaVectorStore(embeddings=embeddings)
    return _vector_store


def get_retriever() -> Retriever:
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        _retriever = Retriever(
            vector_store=get_vector_store(),
            embeddings=get_embeddings(),
        )
    return _retriever


def get_llm() -> OllamaLLM:
    global _llm  # noqa: PLW0603
    if _llm is None:
        _llm = OllamaLLM()
    return _llm


def get_rag_chain() -> RAGChain:
    global _rag_chain  # noqa: PLW0603
    if _rag_chain is None:
        _rag_chain = RAGChain(retriever=get_retriever(), llm=get_llm())
    return _rag_chain
