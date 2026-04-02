# High-Level Design (HLD) — RAG AI System

> **Document Version:** 2.2.0  
> **Author:** Engineering Team  
> **Last Updated:** 2026-04-02  
> **Status:** Approved  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Goals & Non-Goals](#2-system-goals--non-goals)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Component Design](#4-component-design)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Data Model & Storage](#6-data-model--storage)
7. [API Design](#7-api-design)
8. [Security Architecture](#8-security-architecture)
9. [Scalability & Performance](#9-scalability--performance)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Failure Handling & Resilience](#11-failure-handling--resilience)
12. [Technology Decisions & Rationale](#12-technology-decisions--rationale)
13. [Constraints & Assumptions](#13-constraints--assumptions)
14. [Guardrails AI — Input/Output Safety](#14-guardrails-ai--inputoutput-safety)
15. [Future Roadmap](#15-future-roadmap)
16. [Changelog](#16-changelog)

---

## 1. Introduction

### 1.1 Purpose

This document describes the high-level design of **RAG AI** — a production-grade Retrieval-Augmented Generation system that enables users to chat with their documents using locally-hosted LLMs. The system is fully private (no data leaves the network), free to operate (no API key costs), and designed to scale from a single developer workstation to a multi-user enterprise deployment.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|-------------|
| Document ingestion (PDF, TXT, MD, DOCX) | Real-time document sync from cloud drives |
| Vector embedding & storage (ChromaDB / FAISS) | Multi-tenant SaaS architecture |
| Retrieval with cross-encoder reranking | Custom model fine-tuning |
| LLM generation via Ollama (streaming) | Cloud-hosted LLM providers (OpenAI, Anthropic) |
| REST API with auth, rate limiting | GraphQL API |
| Streamlit chat UI | Production-grade React frontend |
| Docker Compose deployment | Kubernetes Helm charts (future) |

### 1.3 Target Audience

- Backend engineers implementing or extending the system
- DevOps engineers deploying and scaling it
- Technical leads evaluating the architecture
- QA engineers understanding component boundaries

### 1.4 Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — enhancing LLM answers with retrieved context from a knowledge base |
| **Embedding** | A dense vector representation of text, used for semantic similarity search |
| **Chunk** | A sized segment of a document, typically 500–1500 characters, the unit of storage and retrieval |
| **Vector Store** | A database optimized for storing and searching high-dimensional vectors |
| **Reranking** | A second-pass scoring of retrieved results using a cross-encoder model for higher precision |
| **Cross-encoder** | A model that jointly encodes (query, document) pairs for precise relevance scoring |
| **Collection** | A logical namespace within the vector store to isolate document sets |
| **Top-K** | The number of most-similar chunks retrieved from the vector store |
| **SSE** | Server-Sent Events — a protocol for streaming token-by-token responses to the client |

---

## 2. System Goals & Non-Goals

### 2.1 Functional Goals

| ID | Goal | Priority |
|----|------|----------|
| F1 | Ingest documents in PDF, TXT, MD, DOCX formats via API | P0 |
| F2 | Split documents into overlapping chunks with metadata | P0 |
| F3 | Generate embeddings locally using Ollama | P0 |
| F4 | Store and search vectors using ChromaDB or FAISS | P0 |
| F5 | Answer natural language questions with source citations | P0 |
| F6 | Stream LLM responses token-by-token via SSE | P0 |
| F7 | Support conversational context (multi-turn chat) | P1 |
| F8 | Rerank retrieved chunks using a cross-encoder | P1 |
| F9 | Deduplicate documents on re-ingestion | P1 |
| F10 | Manage multiple collections (create, list, delete) | P1 |

### 2.2 Non-Functional Goals

| ID | Goal | Target |
|----|------|--------|
| NF1 | **Latency** — Query response (non-streaming, excl. LLM time) | < 500ms p95 |
| NF2 | **Throughput** — Concurrent API requests | 50 req/s (single node) |
| NF3 | **Ingestion speed** — Embedding throughput | 100 chunks/min (CPU) |
| NF4 | **Availability** — Health check endpoint | 99.5% uptime |
| NF5 | **Privacy** — No data leaves the network | 100% local |
| NF6 | **Security** — API key authentication on all mutating endpoints | All /api/v1/* |

### 2.3 Non-Goals

- **Multi-tenancy** — The system uses a single API key; per-user isolation is not implemented.
- **GPU requirement** — The system must run on CPU-only machines (GPU optional for performance).
- **Real-time sync** — Documents are ingested on-demand, not watched/synced automatically.
- **Custom training** — No fine-tuning pipelines; uses off-the-shelf Ollama models.

---

## 3. System Architecture Overview

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Clients                                      │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐        │
│    │ Streamlit UI │    │   curl/SDK   │    │  3rd-party Apps  │        │
│    └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘        │
└───────────┼───────────────────┼─────────────────────┼──────────────────┘
            │                   │                     │
            ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       API Gateway (FastAPI)                              │
│  ┌───────────┐  ┌──────────────┐  ┌─────────┐  ┌────────────────┐     │
│  │   CORS    │  │  API Key Auth │  │  Rate   │  │  Exception     │     │
│  │ Middleware │  │  Middleware   │  │ Limiter │  │  Handler       │     │
│  └───────────┘  └──────────────┘  └─────────┘  └────────────────┘     │
│                                                                         │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────────────┐    │
│  │ POST /api/v1/    │  │ POST /api/v1/  │  │ GET/DELETE /api/v1/ │    │
│  │   ingest         │  │   query        │  │   collections       │    │
│  └────────┬─────────┘  └───────┬────────┘  └─────────────────────┘    │
└───────────┼─────────────────────┼─────────────────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────────┐  ┌─────────────────────────────────────────────┐
│  Ingestion Pipeline  │  │          Retrieval Pipeline                 │
│  ┌────────────────┐  │  │  ┌──────────────┐  ┌────────────────────┐ │
│  │ Document       │  │  │  │ Query        │  │ Cross-Encoder      │ │
│  │ Loaders (4     │  │  │  │ Embedding    │  │ Reranker           │ │
│  │ formats)       │  │  │  │ (cached)     │  │ (ms-marco-MiniLM)  │ │
│  ├────────────────┤  │  │  ├──────────────┤  ├────────────────────┤ │
│  │ Recursive      │  │  │  │ Vector       │  │ Context Window     │ │
│  │ Chunker        │  │  │  │ Similarity   │  │ Assembly           │ │
│  ├────────────────┤  │  │  │ Search       │  │ (truncation-safe)  │ │
│  │ SHA-256        │  │  │  ├──────────────┤  ├────────────────────┤ │
│  │ Deduplicator   │  │  │  │ RAG Chain    │  │ Streaming LLM      │ │
│  └────────┬───────┘  │  │  │ (prompt +    │  │ Response           │ │
│           │          │  │  │  history)    │  │ (SSE)              │ │
└───────────┼──────────┘  │  └──────┬───────┘  └────────────────────┘ │
            │             └─────────┼─────────────────────────────────┘
            ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Data Layer                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────┐          │
│  │       ChromaDB            │  │         FAISS              │          │
│  │  (Persistent Client)      │  │  (File-persisted Index)    │          │
│  │  - HNSW cosine index      │  │  - IndexFlatIP (L2-norm)   │          │
│  │  - Metadata filtering     │  │  - Pickle metadata store   │          │
│  │  - Collection management  │  │  - Shutil-based deletion   │          │
│  └───────────────────────────┘  └───────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
            │                       │
            ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Ollama (Local LLM Runtime)                            │
│  ┌───────────────────┐  ┌──────────────────────┐                       │
│  │ llama3 / mistral  │  │ nomic-embed-text     │                       │
│  │ (Generation)      │  │ (Embeddings)         │                       │
│  │ /api/generate     │  │ /api/embed           │                       │
│  └───────────────────┘  └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Layered Architecture

The system follows a **clean layered architecture** with clear dependency directions:

```
┌──────────────────────────────────────────┐
│  Presentation Layer (API + UI)           │  ← Entry points
├──────────────────────────────────────────┤
│  Application Layer (Pipelines + Chains)  │  ← Business logic orchestration
├──────────────────────────────────────────┤
│  Domain Layer (Retriever, Reranker, …)   │  ← Core domain logic
├──────────────────────────────────────────┤
│  Infrastructure Layer (Stores, Clients)  │  ← External system adapters
└──────────────────────────────────────────┘
```

| Layer | Modules | Responsibilities |
|-------|---------|-----------------|
| **Presentation** | `src/api/`, `src/ui/` | HTTP routing, request validation, serialization, SSE streaming |
| **Application** | `src/ingestion/pipeline.py`, `src/llm/chain.py` | Orchestrate multi-step workflows (ingest, query) |
| **Domain** | `src/retrieval/`, `src/ingestion/chunker.py`, `src/ingestion/dedup.py` | Chunking logic, deduplication, reranking, context assembly |
| **Infrastructure** | `src/vectorstore/`, `src/embeddings/`, `src/llm/ollama.py` | Ollama HTTP clients, ChromaDB/FAISS adapters, embedding cache |
| **Cross-cutting** | `src/core/` | Configuration, logging, exceptions |

---

## 4. Component Design

### 4.1 Component Catalog

| Component | Module | Type | Description |
|-----------|--------|------|-------------|
| **FastAPI Server** | `src/api/server.py` | Entry point | Application bootstrap, middleware stack, route registration |
| **Auth Middleware** | `src/api/dependencies.py` | Security | API key validation via `X-API-Key` header |
| **Rate Limiter** | `src/api/dependencies.py` | Security | In-memory sliding-window rate limiter (per key) |
| **Pydantic Schemas** | `src/api/schemas.py` | Validation | Request/response models with field-level constraints |
| **Ingest Route** | `src/api/routes/ingest.py` | API | File upload → temp storage → pipeline invocation |
| **Query Route** | `src/api/routes/query.py` | API | Question → RAG chain → response (or SSE stream) |
| **Collections Route** | `src/api/routes/collections.py` | API | CRUD for vector store collections |
| **Document Loader** | `src/ingestion/loader.py` | Ingestion | Format-specific loaders (PDF, TXT, MD, DOCX) |
| **Chunker** | `src/ingestion/chunker.py` | Ingestion | Recursive character splitting with overlap |
| **Deduplicator** | `src/ingestion/dedup.py` | Ingestion | SHA-256 content hashing to eliminate duplicates |
| **Ingestion Pipeline** | `src/ingestion/pipeline.py` | Orchestration | Chains: load → chunk → dedup → embed → store |
| **Ollama Embeddings** | `src/embeddings/ollama.py` | Infrastructure | Batched async HTTP client for `/api/embed` |
| **Embedding Cache** | `src/embeddings/cache.py` | Performance | LRU cache (10K entries) for embedding vectors |
| **Vector Store Base** | `src/vectorstore/base.py` | Abstraction | ABC defining the store interface |
| **ChromaDB Store** | `src/vectorstore/chroma_store.py` | Infrastructure | Persistent ChromaDB with HNSW cosine index |
| **FAISS Store** | `src/vectorstore/faiss_store.py` | Infrastructure | File-persisted FAISS with L2-normalized IP search |
| **Retriever** | `src/retrieval/retriever.py` | Domain | Embeds query → vector search → returns results |
| **Reranker** | `src/retrieval/reranker.py` | Domain | Cross-encoder precision scoring of candidates |
| **Context Builder** | `src/retrieval/context.py` | Domain | Assembles retrieved chunks into a prompt-safe context |
| **Ollama LLM** | `src/llm/ollama.py` | Infrastructure | Async HTTP client for `/api/generate` (streaming) |
| **RAG Chain** | `src/llm/chain.py` | Orchestration | Prompt formatting, history management, generation |
| **Streamlit UI** | `src/ui/app.py` | Presentation | Chat interface with file upload and source display |
| **Settings** | `src/core/config.py` | Configuration | `pydantic-settings` with `.env` loading |
| **Logger** | `src/core/logging.py` | Observability | Structured logging to stdout |
| **Exceptions** | `src/core/exceptions.py` | Error handling | Typed exception hierarchy |

### 4.2 Component Interaction Matrix

Shows which components call which (→ = calls):

| Caller → Callee | Loader | Chunker | Dedup | Embeddings | VectorStore | Retriever | Reranker | Context | LLM | Chain |
|-----------------|:------:|:-------:|:-----:|:----------:|:-----------:|:---------:|:--------:|:-------:|:---:|:-----:|
| **Ingest Route** | | | | | | | | | | |
| → Pipeline | ✓ | ✓ | ✓ | | ✓ | | | | | |
| **Query Route** | | | | | | | | | | |
| → Chain | | | | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Pipeline** | ✓ | ✓ | ✓ | | ✓ | | | | | |
| **VectorStore** | | | | ✓ | | | | | | |
| **Retriever** | | | | ✓ | ✓ | | | | | |
| **Chain** | | | | | | ✓ | ✓ | ✓ | ✓ | |

### 4.3 Detailed Component Descriptions

#### 4.3.1 Document Loader (`src/ingestion/loader.py`)

**Responsibility:** Load raw files into LangChain `Document` objects with source metadata.

```
Input:  File path (PDF, TXT, MD, DOCX)
Output: List[Document] with page_content + metadata
```

**Design Decisions:**
- Uses a **registry pattern** (`LOADER_MAP`) mapping file extensions to loader classes.
- Injects `source` and `filename` into metadata for every loaded document.
- Non-recursive directory loading (no hidden file scanning).

**Supported Loaders:**

| Extension | Loader Class | Library |
|-----------|-------------|---------|
| `.pdf` | `PyPDFLoader` | pypdf |
| `.txt` | `TextLoader` | langchain |
| `.md` | `UnstructuredMarkdownLoader` | unstructured |
| `.docx` | `Docx2txtLoader` | docx2txt |

#### 4.3.2 Chunker (`src/ingestion/chunker.py`)

**Responsibility:** Split documents into overlapping chunks optimized for embedding and retrieval.

**Algorithm:** Recursive character text splitting with a prioritized separator hierarchy:
1. `\n\n` (paragraph boundary)
2. `\n` (line boundary)
3. `. ` (sentence boundary)
4. ` ` (word boundary)
5. `""` (character boundary — last resort)

**Parameters:**

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `chunk_size` | 1000 chars | Balances context richness vs. embedding quality |
| `chunk_overlap` | 200 chars | Prevents loss of context at chunk boundaries |

**Output metadata:** Each chunk receives a `chunk_index` for ordering.

#### 4.3.3 Deduplicator (`src/ingestion/dedup.py`)

**Responsibility:** Prevent storing duplicate content across re-ingestion runs.

**Algorithm:**
1. Compute `SHA-256(page_content)` for each chunk.
2. Track seen hashes in an in-memory set.
3. Drop chunks with previously-seen hashes.
4. Attach `content_hash` to surviving chunk metadata.

**Key property:** The `content_hash` is also used as the ChromaDB document ID, making upserts idempotent — re-ingesting the same file overwrites rather than duplicates.

#### 4.3.4 Embedding Client (`src/embeddings/ollama.py`)

**Responsibility:** Generate dense vector representations of text using Ollama's local embedding model.

**Design:**
- Async HTTP client (`httpx.AsyncClient`) with 120s timeout.
- Batched processing: sends up to `EMBEDDING_BATCH_SIZE` (default 64) texts per batch.
- Calls Ollama's `/api/embed` endpoint.
- Returns `List[List[float]]` — one vector per input text.

**Error handling:** Raises `EmbeddingError` on HTTP failures or malformed responses.

#### 4.3.5 Embedding Cache (`src/embeddings/cache.py`)

**Responsibility:** Avoid redundant Ollama API calls for previously-seen text.

**Implementation:** `OrderedDict`-based LRU cache with configurable max size (default 10,000 entries).

**Metrics exposed:** `hits`, `misses`, `hit_rate`, `size`.

**When cache helps most:**
- Repeated queries (e.g., retry or refresh).
- Re-ingestion of unchanged documents.

#### 4.3.6 Vector Store — Abstract Base (`src/vectorstore/base.py`)

**Responsibility:** Define a contract that all vector store backends must implement.

**Interface:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_documents` | `(docs, collection) → int` | Store embedded documents |
| `search` | `(embedding, collection, top_k) → List[SearchResult]` | Similarity search |
| `delete_collection` | `(collection) → None` | Drop a collection |
| `list_collections` | `() → List[dict]` | Enumerate collections |
| `collection_count` | `(collection) → int` | Count docs in collection |
| `health_check` | `() → bool` | Connectivity check |

**`SearchResult` dataclass:** `content: str`, `metadata: dict`, `score: float`

#### 4.3.7 Vector Store — ChromaDB (`src/vectorstore/chroma_store.py`)

**Configuration:**
- `PersistentClient` with configurable path (`CHROMA_PERSIST_DIR`).
- HNSW index with cosine distance metric.

**Idempotent upsert:** Uses `content_hash` as document ID. Re-ingesting the same content updates rather than duplicates.

**Score conversion:** ChromaDB returns distance; converted to similarity via `score = 1.0 - distance`.

#### 4.3.8 Vector Store — FAISS (`src/vectorstore/faiss_store.py`)

**Configuration:**
- `IndexFlatIP` (inner product) with L2-normalized vectors for cosine similarity.
- File-persisted: `index.faiss` + `metadata.pkl` per collection directory.
- Lazy-loaded: collections are read from disk on first access.

**Trade-offs vs. ChromaDB:**

| Aspect | ChromaDB | FAISS |
|--------|----------|-------|
| Metadata filtering | Native | Manual (post-filter) |
| Persistence | Built-in | Custom (pickle) |
| Search speed (>100K) | Good | Excellent |
| Memory usage | Moderate | Low |
| Production maturity | High | High (Meta-backed) |

#### 4.3.9 Retriever (`src/retrieval/retriever.py`)

**Responsibility:** Orchestrate query embedding → cache check → vector search.

**Flow:**
```
query string
    │
    ├──▶ Check Embedding Cache
    │         ├─ HIT  → use cached vector
    │         └─ MISS → call OllamaEmbeddings.embed_query()
    │                       └─ cache result
    │
    ▼
vector store.search(embedding, top_k)
    │
    ▼
List[SearchResult]
```

#### 4.3.10 Reranker (`src/retrieval/reranker.py`)

**Responsibility:** Improve retrieval precision by scoring (query, chunk) pairs with a cross-encoder.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — a lightweight 22M-parameter model trained on MS MARCO passage ranking.

**Strategy:**
1. Retrieve `3 × top_k` candidates from the vector store (cast a wider net).
2. Score each candidate with the cross-encoder.
3. Return top `RERANK_TOP_N` results.

**Why rerank?** Bi-encoder embeddings (used for vector search) are fast but approximate. Cross-encoders jointly attend to query + document and produce more accurate relevance scores, especially for nuanced queries.

**Lazy loading:** The model is loaded on first rerank call and kept in memory.

#### 4.3.11 Context Builder (`src/retrieval/context.py`)

**Responsibility:** Assemble retrieved chunks into a single context string safe for the LLM prompt.

**Formatting:**
```
[Source 1: report.pdf | Score: 0.950]
<chunk content>

---

[Source 2: notes.md | Score: 0.820]
<chunk content>
```

**Truncation:** Enforces a `max_tokens` limit (~3000 tokens ≈ 12K chars) to prevent context overflow.

#### 4.3.12 RAG Chain (`src/llm/chain.py`)

**Responsibility:** Combine retrieved context + chat history + question into a prompt, generate an answer.

**Prompt Template:**
```
System: You are a helpful AI assistant. Answer based ONLY on the provided context...
Context: {assembled context}
Chat History: {last 10 turns}
Question: {user question}
Answer:
```

**Chat history management:** Retains the last 10 message exchanges to maintain conversational coherence without overflowing the context window.

**Two modes:**
- `query()` → returns `RAGResponse(answer, sources)` — full response.
- `query_stream()` → yields tokens via `AsyncIterator[str]` — for SSE.

---

## 5. Data Flow Diagrams

### 5.1 Document Ingestion Flow

```
User uploads files via POST /api/v1/ingest
    │
    ▼
┌─────────────────────────────────────┐
│ 1. API Layer                         │
│    - Validate API key               │
│    - Check rate limit               │
│    - Save uploads to temp dir       │
│    - Generate task_id (UUID)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. Document Loading                  │
│    - Detect file extension          │
│    - Select appropriate loader      │
│    - Load → List[Document]          │
│    - Attach source metadata         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Chunking                          │
│    - RecursiveCharacterTextSplitter │
│    - chunk_size=1000, overlap=200   │
│    - Attach chunk_index metadata    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. Deduplication                     │
│    - SHA-256 hash each chunk        │
│    - Remove duplicates (in-batch)   │
│    - Attach content_hash metadata   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. Embedding                         │
│    - Batch texts (64 per batch)     │
│    - POST /api/embed to Ollama      │
│    - Returns List[List[float]]      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 6. Vector Storage                    │
│    - Upsert (id=content_hash)       │
│    - Store: vectors + docs + meta   │
│    - Persist to disk                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 7. Response                          │
│    { task_id, status: "completed",  │
│      documents_loaded: N,           │
│      chunks_stored: M }             │
└─────────────────────────────────────┘
```

### 5.2 Query & Answer Flow

```
User sends POST /api/v1/query { question, collection, top_k, rerank, stream }
    │
    ▼
┌─────────────────────────────────────┐
│ 1. API Layer                         │
│    - Validate API key & rate limit  │
│    - Parse & validate request body  │
│    - Route to RAG Chain             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. Query Embedding                   │
│    - Check LRU cache for query      │
│    - If miss: embed via Ollama      │
│    - Cache the embedding            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Vector Search                     │
│    - Search with top_k × 3         │
│      (over-fetch for reranking)     │
│    - Returns candidates with scores │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. Reranking (if enabled)            │
│    - Cross-encoder scores each      │
│      (query, chunk) pair            │
│    - Sort by score descending       │
│    - Truncate to RERANK_TOP_N       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. Context Assembly                  │
│    - Format chunks with sources     │
│    - Truncate to max_tokens         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 6. Prompt Construction               │
│    - System prompt + context        │
│    - Chat history (last 10 turns)   │
│    - Current question               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 7. LLM Generation                    │
│    - POST /api/generate to Ollama   │
│    - stream=true → SSE to client    │
│    - stream=false → full response   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 8. Response                          │
│    { answer: "...",                  │
│      sources: [{content, filename,  │
│                  score}] }           │
└─────────────────────────────────────┘
```

---

## 6. Data Model & Storage

### 6.1 Document Metadata Schema

Every chunk stored in the vector store carries this metadata:

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `source` | string | Loader | Full file path of origin document |
| `filename` | string | Loader | File name (e.g., `report.pdf`) |
| `page` | int | Loader (PDF) | Page number (PDF only) |
| `chunk_index` | int | Chunker | Sequential index within the chunked document |
| `content_hash` | string | Dedup | SHA-256 hash of chunk content (also used as vector ID) |

### 6.2 Storage Layout

```
storage/
├── chroma/                    # ChromaDB persistent data
│   ├── chroma.sqlite3         # Metadata database
│   └── <collection-uuid>/    # HNSW index files per collection
│       ├── data_level0.bin
│       ├── header.bin
│       ├── index_metadata.json
│       └── length.bin
│
└── faiss/                     # FAISS persistent data
    └── <collection-name>/
        ├── index.faiss        # FAISS index binary
        └── metadata.pkl       # Pickled metadata + document texts
```

### 6.3 Embedding Dimensions

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `nomic-embed-text` | 768 | Default embedding model |
| `mxbai-embed-large` | 1024 | Alternative (higher quality, slower) |
| `all-minilm` | 384 | Lighter alternative |

### 6.4 API Data Contracts

#### IngestResponse
```json
{
  "task_id": "uuid-string",
  "status": "completed | failed | processing",
  "documents_loaded": 5,
  "chunks_created": 142,
  "chunks_stored": 138,
  "message": "Successfully ingested 5 file(s)"
}
```

#### QueryResponse
```json
{
  "answer": "Based on the documents, ...",
  "sources": [
    {
      "content": "Relevant chunk text...",
      "filename": "report.pdf",
      "score": 0.953
    }
  ]
}
```

---

## 7. API Design

### 7.1 Endpoint Summary

| Method | Path | Auth | Rate Limited | Description |
|--------|------|:----:|:------------:|-------------|
| `GET` | `/` | No | No | Service info |
| `GET` | `/health` | No | No | Health check (Ollama + vector store) |
| `POST` | `/api/v1/ingest` | Yes | Yes | Upload & ingest documents |
| `POST` | `/api/v1/query` | Yes | Yes | RAG query (streaming or non-streaming) |
| `GET` | `/api/v1/collections` | Yes | Yes | List all collections |
| `DELETE` | `/api/v1/collections/{name}` | Yes | Yes | Delete a collection |

### 7.2 Authentication

- **Mechanism:** Static API key via `X-API-Key` HTTP header.
- **Validation:** Constant-time string comparison against `API_KEY` env var.
- **Failure:** HTTP 401 `{"detail": "Invalid or missing API key"}`.

### 7.3 Rate Limiting

- **Algorithm:** Sliding window (in-memory, per-key).
- **Default:** 60 requests per minute per key.
- **Failure:** HTTP 429 `{"detail": "Rate limit exceeded"}`.
- **Production upgrade:** Replace with Redis-backed limiter for multi-node.

### 7.4 Error Responses

| HTTP Code | Condition | Body |
|-----------|-----------|------|
| 400 | Invalid request body (Pydantic validation) | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| 401 | Missing or invalid API key | `{"detail": "Invalid or missing API key"}` |
| 429 | Rate limit exceeded | `{"detail": "Rate limit exceeded"}` |
| 500 | Internal error (RAGError subclass) | `{"detail": "Error description"}` |

---

## 8. Security Architecture

### 8.1 Threat Model

| Threat | Mitigation | Component |
|--------|-----------|-----------|
| **Unauthorized access** | API key authentication on all `/api/v1/*` | `dependencies.py` |
| **Brute force** | Rate limiting (60/min per key) | `dependencies.py` |
| **Injection via filenames** | Temp dir isolation, no shell execution | `routes/ingest.py` |
| **SSRF via Ollama URL** | Config-only URL, no user-supplied URLs forwarded | `config.py` |
| **XSS in responses** | JSON serialization (no raw HTML) | FastAPI default |
| **Data exfiltration** | All processing is local, no external API calls | Architecture |
| **DoS via large files** | FastAPI upload limits, temp file cleanup | `routes/ingest.py` |
| **Path traversal** | Sanitized via `Path` objects, no user paths in storage | `loader.py` |

### 8.2 Data Privacy

- **Zero cloud exposure:** Both LLM and embeddings run locally via Ollama.
- **No telemetry:** No analytics or external reporting.
- **Temp file cleanup:** Uploaded files are deleted from temp directories after ingestion.
- **Storage isolation:** Vector data stored in configurable local directories only.

### 8.3 Security Checklist

- [x] API key authentication
- [x] Rate limiting per key
- [x] CORS middleware (configurable origins)
- [x] Input validation via Pydantic schemas
- [x] No dynamic SQL / command injection surfaces
- [x] Temp file cleanup in `finally` blocks
- [x] Typed exception handling (no stack traces in responses)
- [ ] HTTPS termination (delegate to reverse proxy / nginx)
- [ ] API key rotation mechanism (future)
- [ ] Audit logging (future)

---

## 9. Scalability & Performance

### 9.1 Performance Characteristics

| Operation | Bottleneck | Typical Latency | Optimization |
|-----------|-----------|----------------|-------------|
| **Embedding (single)** | Ollama API call | 50–200ms | Batching, caching |
| **Embedding (batch × 64)** | Ollama compute | 3–10s | GPU acceleration |
| **Vector search (Chroma)** | HNSW traversal | 5–20ms | Index tuning |
| **Vector search (FAISS)** | IndexFlatIP | 1–5ms | IVF quantization |
| **Reranking (5 candidates)** | Cross-encoder inference | 100–500ms | GPU, reduce candidates |
| **LLM generation** | Token generation | 2–30s | Smaller model, GPU |
| **Full query (no stream)** | LLM generation | 3–35s | Streaming to hide latency |

### 9.2 Scaling Strategies

```
                    Single Node                      Multi-Node
               ┌─────────────────┐          ┌──────────────────────┐
 < 10K docs    │ ChromaDB local  │          │                      │
               │ Ollama CPU      │          │                      │
               │ 1 API process   │          │                      │
               └─────────────────┘          │                      │
                                            │                      │
               ┌─────────────────┐          │                      │
 10K–100K      │ FAISS IVF index │   ───▶   │  nginx load balancer │
 docs          │ Ollama GPU      │          │  2–4 API replicas    │
               │ 2–4 uvicorn     │          │  Redis cache/limiter │
               │   workers       │          │  Shared storage NFS  │
               └─────────────────┘          └──────────────────────┘
                                            
               ┌─────────────────┐          ┌──────────────────────┐
 100K+ docs    │ Chroma client-  │   ───▶   │  K8s cluster         │
               │   server mode   │          │  N API pods (HPA)    │
               │ Dedicated GPU   │          │  Chroma cluster      │
               │   Ollama node   │          │  GPU node pool       │
               └─────────────────┘          └──────────────────────┘
```

### 9.3 Caching Strategy

| Cache Layer | What | TTL | Max Size | Implementation |
|------------|------|-----|----------|----------------|
| **Embedding cache** | query text → vector | Session | 10K entries | In-memory LRU (`OrderedDict`) |
| **Future: Response cache** | (query + collection) → answer | 5 min | 1K entries | Redis |
| **Future: Model cache** | Reranker model weights | Process lifetime | 1 model | Module-level global |

### 9.4 Bottleneck Analysis

```
Ingestion Pipeline Bottleneck Analysis:

  Load (fast) ──▶ Chunk (fast) ──▶ Dedup (fast) ──▶ Embed (SLOW) ──▶ Store (moderate)
                                                        ▲
                                                        │
                                                  Bottleneck: Ollama
                                                  embed throughput
                                                  
  Mitigation: Batching (64/batch), GPU acceleration, async I/O

Query Pipeline Bottleneck Analysis:

  Embed (fast*) ──▶ Search (fast) ──▶ Rerank (moderate) ──▶ LLM Gen (SLOW)
       ▲                                                        ▲
       │                                                        │
  *Cached in LRU                                         Bottleneck: Token
                                                         generation speed
  Mitigation: Streaming (hide latency), smaller models, GPU
```

---

## 10. Deployment Architecture

### 10.1 Docker Compose (Default)

```
┌────────────────────────────────────────────────────────┐
│                    Docker Host                          │
│                                                        │
│  ┌────────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │  rag-ollama    │ │  rag-api     │ │  rag-ui      │ │
│  │  :11434        │ │  :8000       │ │  :8501       │ │
│  │                │◀│              │ │              │◀│
│  │  ollama/ollama │ │  FastAPI +   │ │  Streamlit   │ │
│  │  (GPU pass-    │ │  uvicorn     │ │              │ │
│  │   through opt) │ │              │ │              │ │
│  └───────┬────────┘ └──────┬───────┘ └──────────────┘ │
│          │                 │                           │
│          │    ┌────────────┘                           │
│          │    │                                        │
│          ▼    ▼                                        │
│  ┌──────────────────┐  ┌─────────────────────┐        │
│  │ ollama_data vol  │  │ ./storage bind mount │        │
│  │ (model weights)  │  │ (vectors + indexes)  │        │
│  └──────────────────┘  └─────────────────────┘        │
└────────────────────────────────────────────────────────┘
```

### 10.2 Service Configuration

| Service | Image | Ports | Volumes | Depends On |
|---------|-------|-------|---------|-----------|
| `ollama` | `ollama/ollama:latest` | 11434 | `ollama_data` | — |
| `api` | Custom (Dockerfile) | 8000 | `./storage`, `./data` | ollama |
| `ui` | Custom (Dockerfile) | 8501 | — | api |

### 10.3 Production Deployment Checklist

- [ ] Set a strong, random `API_KEY` in `.env`
- [ ] Restrict `CORS` origins to your domain(s)
- [ ] Place behind reverse proxy (nginx/Caddy) with HTTPS
- [ ] Configure `LOG_LEVEL=WARNING` for production noise reduction
- [ ] Set up persistent volume for `storage/` directory
- [ ] Pull required Ollama models before first request
- [ ] Monitor `/health` endpoint
- [ ] Set appropriate `RATE_LIMIT_PER_MINUTE` for your use case
- [ ] Back up `storage/` directory regularly

---

## 11. Failure Handling & Resilience

### 11.1 Error Classification

| Error Type | Exception Class | HTTP Code | Retry? | User Message |
|-----------|-----------------|:---------:|:------:|-------------|
| Invalid input | Pydantic `ValidationError` | 400 | No | Field-specific validation errors |
| Auth failure | `HTTPException` | 401 | No | "Invalid or missing API key" |
| Rate exceeded | `HTTPException` | 429 | Yes (after cooldown) | "Rate limit exceeded" |
| Unsupported file | `IngestionError` | 500 | No | "Unsupported file type '.xyz'" |
| Ollama unreachable | `EmbeddingError` / `LLMError` | 500 | Yes | "Ollama connection failed" |
| Vector store error | `VectorStoreError` | 500 | Yes | "Vector store operation failed" |
| Unknown | `Exception` | 500 | Depends | Generic error message |

### 11.2 Failure Scenarios & Recovery

| Scenario | Impact | Detection | Recovery |
|----------|--------|-----------|----------|
| **Ollama down** | No embeddings, no generation | `/health` returns `ollama: false` | Restart Ollama; API returns 500 with clear message |
| **Vector store corrupt** | No search results | `health_check()` fails | Re-ingest from `data/` directory |
| **Disk full** | Persistence fails | OS error on write | Alert on disk usage; clean old collections |
| **OOM on reranker** | Rerank fails | Process crash | Disable reranking (`RERANK_ENABLED=false`) |
| **Temp dir not cleaned** | Disk leak | Monitoring | `finally` block ensures cleanup; add cron if needed |

### 11.3 Graceful Degradation

```
Full Capability (all systems healthy)
    │
    ├── Ollama down → Embedding + Generation fail
    │   └── API returns 500 with "Ollama unavailable"
    │       Health endpoint shows degraded status
    │
    ├── Reranker OOM → Reranking disabled
    │   └── Falls back to raw vector search scores
    │       Quality degrades slightly, system stays up
    │
    └── Vector store corrupt → Search returns empty
        └── LLM responds "I don't have enough info..."
            User can re-ingest documents
```

---

## 12. Technology Decisions & Rationale

### 12.1 ADR (Architecture Decision Records)

#### ADR-001: Local-First with Ollama

**Context:** Need an LLM inference engine that is free, private, and runs on commodity hardware.

**Decision:** Use Ollama for both generation and embeddings.

**Rationale:**
- Zero cost — no per-token API fees.
- Full privacy — no data leaves the machine.
- Model flexibility — switch models via config.
- CPU-compatible — GPU optional for speed.

**Trade-offs:**
- Slower than cloud APIs on CPU.
- Limited to models Ollama supports.

#### ADR-002: ChromaDB as Default Vector Store

**Context:** Need a persistent, production-ready vector store with metadata filtering.

**Decision:** ChromaDB as default, FAISS as alternative.

**Rationale:**
- ChromaDB: built-in persistence, metadata filtering, Python-native.
- FAISS as fallback: lower memory, faster raw search, Meta-backed.
- Abstraction layer (`BaseVectorStore`) allows swapping without code changes.

#### ADR-003: Cross-Encoder Reranking

**Context:** Bi-encoder search (embedding similarity) has recall limitations for nuanced queries.

**Decision:** Two-stage retrieval — vector search (recall) → cross-encoder rerank (precision).

**Rationale:**
- Over-fetch (3× top_k) candidates from vector search.
- Cross-encoder (`ms-marco-MiniLM-L-6-v2`) jointly scores query+document pairs.
- Dramatically improves precision on ambiguous queries.
- Minimal latency cost (~100ms for 15 candidates).

#### ADR-004: FastAPI over Flask/Django

**Context:** Need an async-capable HTTP framework with auto-generated docs.

**Decision:** FastAPI.

**Rationale:**
- Native async/await support (critical for Ollama HTTP calls).
- Auto-generated OpenAPI docs at `/docs`.
- Pydantic integration for request/response validation.
- Dependency injection system for auth/rate limiting.

#### ADR-005: Content-Hash Deduplication

**Context:** Users may re-upload the same files or overlapping document sets.

**Decision:** SHA-256 hash of chunk content as the deduplication key and vector store ID.

**Rationale:**
- Deterministic: same content always produces same hash.
- Enables idempotent upserts: re-ingestion overwrites, not duplicates.
- Fast: SHA-256 is negligible compared to embedding time.

---

## 13. Constraints & Assumptions

### 13.1 Constraints

| ID | Constraint | Impact |
|----|-----------|--------|
| C1 | Must run fully offline (no external API calls) | Limits to local models via Ollama |
| C2 | Must run on CPU-only machines | LLM generation is slower than GPU setups |
| C3 | Single API key authentication | No per-user isolation or audit trails |
| C4 | In-memory rate limiting | Resets on server restart; not suitable for multi-node |
| C5 | Python 3.11+ requirement | Excludes older runtime environments |

### 13.2 Assumptions

| ID | Assumption | Risk if Invalid |
|----|-----------|----------------|
| A1 | Ollama is running and accessible at `OLLAMA_BASE_URL` | All embedding/generation calls fail |
| A2 | Required models are pre-pulled (`llama3`, `nomic-embed-text`) | First request will timeout |
| A3 | Sufficient disk space for vector store persistence | Writes will fail silently or crash |
| A4 | Documents are well-formatted and text-extractable | Poor chunking, garbage embeddings |
| A5 | Single-node deployment (MVP) | Rate limiting and caching are in-memory |

---

## 14. Guardrails AI — Input/Output Safety

### 14.1 Overview

The system integrates a **Guardrails AI** module that validates both user input (pre-retrieval) and LLM output (post-generation). Each validator can be independently enabled/disabled via environment variables.

```
User Query ──► [Input Guardrails] ──► Retrieve ──► Rerank ──► LLM ──► [Output Guardrails] ──► Response
                    │                                                        │
              Prompt Injection                                       Context Grounding
              Topic Relevance                                        PII Detection
                    │                                                Response Quality
                    ▼                                                        │
              Block (if fail)                                         Warn / Redact
```

### 14.2 Input Validators

| Validator | Purpose | Failure Action |
|-----------|---------|----------------|
| **Prompt Injection Detector** | Detects instruction-override, jailbreak, role-play patterns via 20+ regex rules | Block query |
| **Topic Relevance** | Rejects queries containing code-execution patterns (`os.system`, `exec()`, SQL injection) | Block query |

### 14.3 Output Validators

| Validator | Purpose | Failure Action |
|-----------|---------|----------------|
| **Context Grounding** | Computes tri-gram overlap between response and retrieved context; flags hallucination below threshold | Warn in report |
| **PII Detector** | Regex-based scan for email, phone, SSN, credit card, IP address | Warn / Redact |
| **Response Quality** | Checks for empty, too-short, too-long, or refusal responses | Warn in report |

### 14.4 Configuration

All guardrails settings are toggled via environment variables:

```bash
GUARDRAILS_ENABLED=true
GUARDRAILS_BLOCK_ON_FAILURE=true     # Block on input failures
GUARDRAILS_CHECK_INJECTION=true
GUARDRAILS_CHECK_TOPIC=true
GUARDRAILS_CHECK_GROUNDING=true
GUARDRAILS_CHECK_PII=true
GUARDRAILS_CHECK_QUALITY=true
GUARDRAILS_PII_REDACT=false          # Replace PII with [REDACTED]
GUARDRAILS_GROUNDING_THRESHOLD=0.3   # 0.0–1.0
```

### 14.5 API Response

The `/query` response includes a `guardrails` field with validation results:

```json
{
  "answer": "...",
  "sources": [...],
  "guardrails": {
    "passed": true,
    "blocked": false,
    "block_reason": "",
    "input_checks": [
      {"status": "pass", "validator": "prompt_injection", "message": "No prompt injection detected"}
    ],
    "output_checks": [
      {"status": "pass", "validator": "context_grounding", "message": "Response is grounded (score: 0.65)", "grounding_score": 0.65},
      {"status": "pass", "validator": "pii_detection", "message": "No PII detected"},
      {"status": "pass", "validator": "response_quality", "message": "Response quality checks passed"}
    ]
  }
}
```

---

## 14.6 OWASP Agent Threat Protection (v2.2.0)

### 14.6.1 Sensitive Data Classifier

Multi-category scanner (`src/guardrails/data_classifier.py`) with 30+ regex patterns across 5 data categories:

| Category | Examples | Severity |
|----------|----------|----------|
| **PII** | Email, phone, SSN, driver's license, passport, DOB, physical address, IP | Medium–Critical |
| **PHI** (HIPAA) | Medical record number, health plan ID, ICD-10 codes, medications, lab results | Medium–Critical |
| **PCI-DSS** | Credit card, CVV, bank account, routing number, IBAN | High–Critical |
| **Credentials** | Passwords, Bearer tokens, Basic auth, JWT tokens | Critical |
| **Secrets** | AWS keys, GitHub PATs, private keys, connection strings, API keys, Slack webhooks | High–Critical |

### 14.6.2 DLP Engine

The Data Loss Prevention engine (`src/guardrails/dlp.py`) applies threshold-based policies:

```
Sensitive Data → Classify → [severity ≥ BLOCK_THRESHOLD] → BLOCK (reject entirely)
                          → [severity ≥ REDACT_THRESHOLD] → REDACT (replace with placeholders)
                          → [below thresholds]            → AUDIT (log and allow)
```

DLP runs at three points:
1. **Query input** — blocks queries containing leaked credentials/secrets
2. **LLM response** — redacts sensitive data before returning to user
3. **Document ingestion** — scans chunks before they enter the vector store

### 14.6.3 OWASP Agent Threat Validators

| Validator | OWASP Threat | What It Detects |
|-----------|-------------|-----------------|
| `DataExfiltrationDetector` | LLM06 — Data Exfiltration | URL-based exfil, encoding tricks, markdown/HTML tracking pixels, webhook callbacks |
| `IndirectInjectionDetector` | Indirect Prompt Injection | Hidden instructions in documents, system tags, zero-width chars, role redefinition |
| `ExcessiveAgencyDetector` | Excessive Agency | Demands to execute commands, access files, connect to DBs, make autonomous decisions |
| `SystemPromptLeakDetector` | Information Disclosure | Attempts to extract system prompts, reveal instructions, dump configuration |

### 14.6.4 Security Audit Logger

Dedicated audit sink (`src/guardrails/audit.py`) writes structured JSON to stderr — separate from application logs:

```json
{
  "timestamp": "2026-04-02T12:00:00Z",
  "audit_event": "sensitive_data_detected",
  "correlation_id": "abc-123",
  "severity": "critical",
  "category": "pii,secrets",
  "action": "block",
  "detail": "Sensitive data detected in query: 3 finding(s)",
  "metadata": {"finding_count": 3, "categories": ["pii", "secrets"]}
}
```

Audit events: `sensitive_data_detected`, `query_blocked`, `response_redacted`, `ingestion_scan`

---

## 15. Future Roadmap

| Phase | Feature | Effort | Priority |
|-------|---------|--------|----------|
| **v1.1** | Redis-backed rate limiting & caching | Medium | P1 |
| **v1.1** | Webhook/callback for async ingestion | Medium | P1 |
| **v1.2** | Hybrid search (vector + BM25 keyword) | Medium | P1 |
| **v1.2** | Metadata filtering in queries (date, source, tag) | Low | P2 |
| **v1.3** | Multi-user auth (JWT + per-user collections) | High | P2 |
| **v1.3** | Admin dashboard (ingestion stats, cache metrics) | Medium | P2 |
| **v2.0** | Kubernetes Helm chart | High | P2 |
| **v2.0** | GPU-optimized Ollama deployment | Medium | P2 |
| **v2.1** | Agent-style RAG (tool use, multi-step reasoning) | High | P3 |
| **v2.1** | Support for images/tables (multimodal RAG) | High | P3 |

---

## 16. Changelog

| Version | Date | Author | Change Description |
|---------|------|--------|--------------------|
| 2.2.0 | 2026-04-02 | Engineering Team | OWASP Agent Threat Protection — sensitive data classifier (PII/PHI/PCI/credentials/secrets, 30+ patterns), DLP engine with block/redact/audit policies, security audit logger, OWASP validators (data exfiltration, indirect injection, excessive agency, system prompt leak), ingestion-time DLP scanning, 93 total guardrails+OWASP tests |
| 2.1.0 | 2026-04-02 | Engineering Team | Guardrails AI integration — prompt injection detection, topic relevance filtering, context-grounding hallucination check, PII detection & optional redaction, response quality validation, configurable per-validator toggles, guardrails report in query API response, comprehensive test suite (30+ tests) |
| 1.0.0 | 2026-04-02 | Engineering Team | Initial HLD — full system design covering architecture, components, data flows, security, scalability, deployment, and technology decisions |
| 2.0.0 | 2026-04-02 | Engineering Team | Principal Engineer upgrade — added resilience primitives (retry with exponential backoff + jitter, circuit breaker), request ID middleware with correlation ID propagation, structured JSON logging, HMAC timing-safe auth, config validation with Pydantic field validators, service registry pattern with graceful shutdown, multi-stage Dockerfile with non-root user, Docker healthchecks, GitHub Actions CI/CD pipeline, Makefile automation, comprehensive test fixtures, Bandit security linting |

---

*End of High-Level Design Document*
