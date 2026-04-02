"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Query ─────────────────────────────────────────────────────────────

class ChatMessageSchema(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=10_000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5_000)
    collection: str = Field(default="default", max_length=100)
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    stream: bool = False
    chat_history: list[ChatMessageSchema] = Field(default_factory=list)


class SourceSchema(BaseModel):
    content: str
    filename: str | None = None
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSchema] = Field(default_factory=list)


# ── Ingestion ─────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    task_id: str
    status: str  # "processing" | "completed" | "failed"
    documents_loaded: int = 0
    chunks_created: int = 0
    chunks_stored: int = 0
    message: str = ""


# ── Collections ───────────────────────────────────────────────────────

class CollectionSchema(BaseModel):
    name: str
    count: int


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama: bool
    vector_store: bool
