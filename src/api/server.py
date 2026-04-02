"""FastAPI application — entry point with middleware, lifespan, and routes."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import get_llm, get_vector_store
from src.api.routes import collections, ingest, query
from src.api.schemas import HealthResponse
from src.core.exceptions import RAGError
from src.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("RAG AI server starting up")
    yield
    logger.info("RAG AI server shutting down")


app = FastAPI(
    title="RAG AI",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ───────────────────────────────────────────────

@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError):
    logger.error("RAGError: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Routes ────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(collections.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of Ollama and the vector store."""
    llm = get_llm()
    vs = get_vector_store()

    ollama_ok = await llm.health_check()
    vs_ok = await vs.health_check()

    status = "healthy" if (ollama_ok and vs_ok) else "degraded"
    return HealthResponse(status=status, ollama=ollama_ok, vector_store=vs_ok)


@app.get("/")
async def root():
    return {
        "service": "RAG AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
