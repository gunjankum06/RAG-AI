"""Pydantic schemas for API request/response models."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


# ── Query ─────────────────────────────────────────────────────────────


class ChatMessageSchema(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=10_000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5_000)
    collection: str = Field(default="default", min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    stream: bool = False
    chat_history: list[ChatMessageSchema] = Field(default_factory=list, max_length=50)

    @field_validator("collection")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            msg = "Collection name must be alphanumeric with hyphens/underscores only"
            raise ValueError(msg)
        return v


class SourceSchema(BaseModel):
    content: str
    filename: str | None = None
    score: float = 0.0
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSchema] = Field(default_factory=list)
    model: str | None = None


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
    version: str = "2.0.0"


# ── Error ─────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    detail: str
    retriable: bool = False
