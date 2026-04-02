"""Shared FastAPI dependencies: auth, rate limiting, service lifecycle."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.core.config import settings
from src.core.logging import logger
from src.core.resilience import CircuitBreaker
from src.embeddings.ollama import OllamaEmbeddings
from src.guardrails.engine import GuardrailsEngine
from src.llm.chain import RAGChain
from src.llm.ollama import OllamaLLM
from src.retrieval.retriever import Retriever
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.chroma_store import ChromaVectorStore

# ── API Key Auth (timing-safe comparison) ─────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if not api_key or not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Rate Limiting (sliding window, per-key) ───────────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


async def rate_limiter(request: Request, api_key: str = Depends(verify_api_key)) -> str:
    now = time.time()
    window = 60.0
    max_requests = settings.rate_limit_per_minute

    timestamps = _rate_limit_store[api_key]
    # Prune expired entries
    _rate_limit_store[api_key] = timestamps = [
        ts for ts in timestamps if now - ts < window
    ]

    if len(timestamps) >= max_requests:
        logger.warning("Rate limit exceeded for key ending ...%s", api_key[-6:])
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(window - (now - timestamps[0])))},
        )

    timestamps.append(now)
    return api_key


# ── Circuit Breakers ──────────────────────────────────────────────────

ollama_circuit = CircuitBreaker("ollama", failure_threshold=5, recovery_timeout=30.0)


# ── Service Registry (lazy singletons with cleanup) ──────────────────


class _ServiceRegistry:
    """Manages singleton lifecycle for all long-lived services."""

    def __init__(self) -> None:
        self._embeddings: OllamaEmbeddings | None = None
        self._vector_store: BaseVectorStore | None = None
        self._retriever: Retriever | None = None
        self._llm: OllamaLLM | None = None
        self._rag_chain: RAGChain | None = None
        self._guardrails: GuardrailsEngine | None = None

    @property
    def embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings()
        return self._embeddings

    @property
    def vector_store(self) -> BaseVectorStore:
        if self._vector_store is None:
            if settings.vector_store_type == "faiss":
                from src.vectorstore.faiss_store import FAISSVectorStore
                self._vector_store = FAISSVectorStore(embeddings=self.embeddings)
            else:
                self._vector_store = ChromaVectorStore(embeddings=self.embeddings)
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store=self.vector_store,
                embeddings=self.embeddings,
            )
        return self._retriever

    @property
    def llm(self) -> OllamaLLM:
        if self._llm is None:
            self._llm = OllamaLLM()
        return self._llm

    @property
    def guardrails(self) -> GuardrailsEngine:
        if self._guardrails is None:
            self._guardrails = GuardrailsEngine()
        return self._guardrails

    @property
    def rag_chain(self) -> RAGChain:
        if self._rag_chain is None:
            self._rag_chain = RAGChain(
                retriever=self.retriever,
                llm=self.llm,
                guardrails=self.guardrails,
            )
        return self._rag_chain

    async def shutdown(self) -> None:
        """Release all held resources."""
        if self._embeddings:
            await self._embeddings.close()
        if self._llm:
            await self._llm.close()
        logger.info("Service registry shut down")


registry = _ServiceRegistry()


# ── FastAPI-compatible getters (for Depends()) ────────────────────────


def get_embeddings() -> OllamaEmbeddings:
    return registry.embeddings


def get_vector_store() -> BaseVectorStore:
    return registry.vector_store


def get_retriever() -> Retriever:
    return registry.retriever


def get_llm() -> OllamaLLM:
    return registry.llm


def get_rag_chain() -> RAGChain:
    return registry.rag_chain
