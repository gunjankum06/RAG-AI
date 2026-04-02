# High-Level Design (HLD) - RAG AI

## 1. Overview

RAG AI is a production-oriented Retrieval-Augmented Generation platform built around FastAPI, Ollama, and pluggable vector stores.

Primary goals:

- private/local-first LLM and embeddings
- reliable ingestion and retrieval APIs
- security guardrails and DLP controls
- observable request lifecycle for operations and debugging

## 2. System Context

The platform receives documents and user questions through HTTP APIs.

- Ingestion API stores chunked vectorized content in ChromaDB or FAISS.
- Query API retrieves relevant chunks, optionally reranks, applies guardrails, and generates answers via Ollama.
- Streamlit UI is an optional client.

## 3. Core Components

### 3.1 API Layer

- FastAPI app lifecycle, middleware, routing
- API key authentication and rate limiting
- health endpoints and OpenAPI docs

### 3.2 Ingestion Pipeline

- load supported files (`.pdf`, `.txt`, `.md`, `.docx`)
- deduplicate by content hash
- split documents into chunks
- create embeddings and store vectors
- enforce DLP policy during ingestion

### 3.3 Retrieval + Generation

- embed query
- vector search in selected collection
- optional reranking
- context assembly
- prompt construction with chat history
- answer generation (sync or stream)

### 3.4 Safety and Governance

- guardrails for injection/topic/grounding/quality/PII
- OWASP-focused threat checks
- DLP policy for block/redact/audit workflows

### 3.5 Storage

- Chroma persistent store (`storage/chroma`)
- FAISS index store (`storage/faiss`)

## 4. Request Flow

### 4.1 Ingestion Flow

1. Client uploads files via `POST /api/v1/ingest`.
2. API validates API key, rate limits, and file constraints.
3. Pipeline loads, deduplicates, chunks, and embeds content.
4. Vectors and metadata are persisted.
5. API returns ingestion summary.

### 4.2 Query Flow

1. Client sends question via `POST /api/v1/query`.
2. API validates auth and request schema.
3. Guardrails input checks run.
4. Retriever fetches candidate chunks.
5. Optional reranker reorders candidates.
6. Context is built and prompt is rendered.
7. LLM generates response.
8. Guardrails output checks and redaction run.
9. API returns answer + sources (+ guardrails report).

## 5. Observability Design (Arize/Phoenix)

The system supports optional OpenTelemetry tracing export over OTLP.

### 5.1 Design Goals

- trace each RAG stage with low operational overhead
- support both local Phoenix and Arize Cloud
- keep observability optional and disabled by default

### 5.2 Configuration

Observability is controlled via environment variables:

- `ARIZE_ENABLED`
- `ARIZE_SERVICE_NAME`
- `ARIZE_PROJECT_NAME`
- `ARIZE_OTLP_ENDPOINT`
- `ARIZE_API_KEY`
- `ARIZE_SPACE_KEY`

When disabled, tracing paths are no-op.

### 5.3 Instrumented Spans

The following spans are emitted during query execution:

- `rag.query` / `rag.query_stream`
- `rag.guardrails.input`
- `rag.retrieve`
- `rag.rerank`
- `rag.build_context`
- `rag.generate` / `rag.generate_stream`
- `rag.guardrails.output`

### 5.4 Span Attributes

Representative attributes:

- `rag.collection`
- `rag.top_k`
- `rag.fetch_k`
- `rag.rerank_enabled`
- `rag.candidate_count`
- `rag.source_count`
- `rag.question_length`
- `llm.model`

This enables latency decomposition and rapid troubleshooting in production.

## 6. Security Model

- API key auth using timing-safe comparison
- per-key sliding window rate limiting
- file upload size limits and filename sanitization
- configurable CORS policy
- guardrails + DLP enforcement paths

## 7. Reliability and Operations

- startup checks and graceful shutdown
- retry with backoff and circuit breaker patterns for model calls
- structured logs with request correlation IDs
- health endpoint for liveness/dependency checks

## 8. Deployment Options

- local Python process (`uvicorn`)
- Docker Compose (API + UI)
- configurable vector backend for workload profile

## 9. Trade-offs

- Local Ollama improves privacy but requires host capacity and model lifecycle management.
- Reranking improves quality but increases latency.
- Strict guardrails improve safety but can reduce answer recall.
- Full tracing improves diagnosability at the cost of small runtime overhead.
