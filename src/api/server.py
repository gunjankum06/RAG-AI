"""FastAPI application — entry point with middleware, lifespan, and routes."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import get_llm, get_vector_store, registry
from src.api.middleware import RequestContextMiddleware
from src.api.routes import collections, ingest, query
from src.api.schemas import HealthResponse
from src.core.config import settings
from src.core.exceptions import CircuitOpenError, RAGError
from src.core.logging import logger
from src.core.observability import setup_observability, shutdown_observability


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup validation and graceful shutdown."""
    setup_observability()

    logger.info(
        "RAG AI starting (env=%s, vector_store=%s, llm=%s)",
        settings.environment,
        settings.vector_store_type,
        settings.llm_model,
    )
    _startup_time = time.monotonic()

    # Validate critical configuration
    if settings.api_key == "change-me-to-a-secure-random-string":
        logger.warning("⚠ Using default API key — set API_KEY in .env for production")

    # Ensure storage directories exist
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)

    # Pre-warm services and validate connectivity
    try:
        llm = get_llm()
        if await llm.health_check():
            logger.info("Ollama connectivity verified")
        else:
            logger.warning("Ollama is not reachable — queries will fail until it's available")
    except Exception:
        logger.warning("Could not reach Ollama during startup — continuing anyway")

    logger.info("RAG AI ready (startup took %.1fs)", time.monotonic() - _startup_time)

    yield

    # Graceful shutdown: release all resources
    logger.info("RAG AI shutting down — releasing resources")
    await registry.shutdown()
    shutdown_observability()
    logger.info("RAG AI shutdown complete")


app = FastAPI(
    title="RAG AI",
    description="Production-grade Retrieval-Augmented Generation API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# ── Middleware Stack (order matters — outermost first) ────────────────

app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)


# ── Exception Handlers ───────────────────────────────────────────────


@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError):
    logger.error("RAGError: %s (retriable=%s)", exc, exc.retriable)
    status = 503 if isinstance(exc, CircuitOpenError) else 500
    return JSONResponse(
        status_code=status,
        content={"detail": str(exc), "retriable": exc.retriable},
    )


# ── Routes ────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(collections.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Check the health of Ollama and the vector store."""
    llm = get_llm()
    vs = get_vector_store()

    ollama_ok = await llm.health_check()
    vs_ok = await vs.health_check()

    status = "healthy" if (ollama_ok and vs_ok) else "degraded"
    return HealthResponse(status=status, ollama=ollama_ok, vector_store=vs_ok)


@app.get("/", tags=["system"])
async def root():
    return {
        "service": "RAG AI",
        "version": "2.0.0",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health",
    }
