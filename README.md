# RAG AI — Production-Grade Retrieval-Augmented Generation

A **scalable, production-ready** RAG system powered by **Ollama** (local LLM) and **ChromaDB** / **FAISS** for vector storage. Designed for real-world deployment with async processing, API-first architecture, document deduplication, reranking, caching, and observability.

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              FastAPI Gateway                │
                          │  (Auth · Rate Limit · CORS · Health)       │
                          └────────┬──────────────────┬────────────────┘
                                   │                  │
                    ┌──────────────▼──┐       ┌───────▼────────┐
                    │  Ingestion API  │       │   Query API    │
                    │  POST /ingest   │       │  POST /query   │
                    └──────┬──────────┘       └───────┬────────┘
                           │                          │
              ┌────────────▼────────────┐   ┌────────▼─────────────────┐
              │   Document Pipeline     │   │   Retrieval Pipeline     │
              │  ┌───────────────────┐  │   │  ┌────────────────────┐  │
              │  │ Load & Detect     │  │   │  │ Embed Query        │  │
              │  │ Deduplicate       │  │   │  │ Vector Search      │  │
              │  │ Chunk (Recursive) │  │   │  │ Rerank (Cross-Enc) │  │
              │  │ Embed (Batched)   │  │   │  │ Context Assembly   │  │
              │  │ Store Vectors     │  │   │  │ LLM Generation     │  │
              │  └───────────────────┘  │   │  │ Stream Response    │  │
              └────────────┬────────────┘   │  └────────────────────┘  │
                           │                └──────────┬───────────────┘
                           ▼                           ▼
              ┌──────────────────────────────────────────────────┐
              │          Vector Store (ChromaDB / FAISS)          │
              │          + Metadata & Document Hashes             │
              └──────────────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │           Ollama (Local LLM Runtime)             │
              │     llama3 · mistral · nomic-embed-text          │
              └──────────────────────────────────────────────────┘
```

## Key Production Features

| Category | Feature |
|----------|---------|
| **Scale** | Async ingestion, batched embeddings, connection pooling |
| **Reliability** | Health checks, graceful shutdown, retry with backoff |
| **Security** | API key auth, rate limiting, CORS, input sanitization |
| **Quality** | Cross-encoder reranking, hybrid search, deduplication |
| **Safety** | Guardrails AI — prompt injection, hallucination, PII detection |
| **Data Protection** | DLP engine — PII/PHI/PCI/secrets/credentials scanning & redaction |
| **OWASP** | Agent threat protection — exfiltration, indirect injection, excessive agency |
| **Ops** | Structured JSON logging, Docker-ready |
| **Caching** | LRU cache for embeddings, query result caching |
| **Flexibility** | Pluggable vector store (Chroma / FAISS), swappable models |

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | [Ollama](https://ollama.com/) | Local inference, no API costs, data stays private |
| Embeddings | `nomic-embed-text` via Ollama | High-quality local embeddings |
| Vector DB | [ChromaDB](https://www.trychroma.com/) | Persistent, metadata filtering, production-tested |
| Alt Vector DB | [FAISS](https://github.com/facebookresearch/faiss) | Blazing-fast similarity search at scale |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranking of retrieved chunks |
| Guardrails | [Guardrails AI](https://www.guardrailsai.com/) | Input/output safety — injection, hallucination, PII |
| DLP | Custom DLP Engine | OWASP-aligned PII/PHI/PCI/secrets/credentials protection |
| API | [FastAPI](https://fastapi.tiangolo.com/) | Async, OpenAPI docs, dependency injection |
| Orchestration | [LangChain](https://python.langchain.com/) | Loaders, splitters, chain abstractions |
| UI | [Streamlit](https://streamlit.io/) | Rapid chat interface prototyping |
| Containerization | Docker + Compose | Reproducible deployments |
| Language | Python 3.11+ | Async-native, rich ecosystem |

## Prerequisites

Install the following before you start:

1. **Python 3.11+**
2. **Ollama** (running locally) — [https://ollama.com/download](https://ollama.com/download)
3. **Git**
4. Optional: **Docker Desktop** (if using Docker Compose)

Pull required models once:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Verify Ollama:

```bash
curl http://localhost:11434/api/tags
```

## Quick Start (First 15 Minutes)

This path is optimized for first-time setup and validation.

### 1. Clone and enter the repo

```bash
git clone <repo-url> rag-ai
cd rag-ai
```

### 2. Create and activate a virtual environment

Create venv:

```bash
python -m venv .venv
```

Activate:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```cmd
:: Windows CMD
.venv\Scripts\activate.bat
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install python-multipart
```

### 4. Create `.env`

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Edit `.env` and set at least:

```env
API_KEY=replace-with-a-strong-random-secret
```

### 5. Start API server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup lines:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6. Verify health

```bash
curl http://127.0.0.1:8000/health
```

Expected response shape:

```json
{"status":"healthy","ollama":true,"vector_store":true,"version":"2.2.0"}
```

### 7. Ingest your first document

Supported file types: `.pdf`, `.txt`, `.md`, `.docx`

PowerShell example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/ingest" `
       -H "X-API-Key: replace-with-a-strong-random-secret" `
       -F "collection=default" `
       -F "files=@data\sample.txt"
```

Bash example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ingest" \
       -H "X-API-Key: replace-with-a-strong-random-secret" \
       -F "collection=default" \
       -F "files=@data/sample.txt"
```

Expected successful fields:

- `status: completed`
- `documents_loaded > 0`
- `chunks_stored > 0`

### 8. Query your collection

PowerShell example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/query" `
       -H "X-API-Key: replace-with-a-strong-random-secret" `
       -H "Content-Type: application/json" `
       -d "{\"question\":\"Summarize the uploaded document\",\"collection\":\"default\",\"top_k\":5,\"rerank\":true,\"stream\":false,\"chat_history\":[]}"
```

Bash example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
       -H "X-API-Key: replace-with-a-strong-random-secret" \
       -H "Content-Type: application/json" \
       -d '{"question":"Summarize the uploaded document","collection":"default","top_k":5,"rerank":true,"stream":false,"chat_history":[]}'
```

### 9. Open API docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Streamlit UI (Optional)

Start in another terminal (venv activated):

```bash
streamlit run src/ui/app.py
```

Open `http://127.0.0.1:8501` and set:

- API URL: `http://127.0.0.1:8000`
- API Key: your `.env` API key
- Collection: `default` (or your custom collection)

## Docker Compose

### 1. Create `.env`

```powershell
Copy-Item .env.example .env
```

Set `API_KEY` in `.env`.

### 2. Start all services

```bash
docker compose up --build
```

Services:

- API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:8501`

### 3. Stop services

```bash
docker compose down
```

## How To Use The API

All protected endpoints require header:

```text
X-API-Key: <your-api-key>
```

### Endpoint summary

- `GET /health` - health status (no API key required)
- `POST /api/v1/ingest` - upload and index files
- `POST /api/v1/query` - ask questions over indexed docs
- `GET /api/v1/collections` - list collections
- `DELETE /api/v1/collections/{name}` - delete a collection

### `POST /api/v1/ingest`

Form fields:

- `files` (required, multi-file)
- `collection` (optional, default: `default`, allowed: letters/numbers/`_`/`-`)

### `POST /api/v1/query`

JSON body example:

```json
{
       "question": "What does the policy say about data retention?",
       "collection": "default",
       "top_k": 5,
       "rerank": true,
       "stream": false,
       "chat_history": [
              {"role": "user", "content": "We uploaded the policy doc earlier."}
       ]
}
```

## Development Commands

If `make` is available:

```bash
make dev         # install runtime + dev deps
make run         # start FastAPI
make run-ui      # start Streamlit
make test        # run tests
make lint        # lint and format checks
make format      # auto-fix lint + format
```

Windows note: GNU `make` may not be installed by default. If unavailable, run Python/pip/uvicorn commands directly.

## Quick Demo Script

Use this when you want to verify end-to-end behavior quickly.

### Run the prepared scripts (recommended)

PowerShell:

```powershell
$env:API_KEY="replace-with-a-strong-random-secret"
.\scripts\demo.ps1
```

PowerShell with custom URL/collection:

```powershell
.\scripts\demo.ps1 -ApiUrl "http://127.0.0.1:8000" -Collection "demo" -ApiKey "replace-with-a-strong-random-secret"
```

Bash:

```bash
API_KEY="replace-with-a-strong-random-secret" bash scripts/demo.sh
```

The scripts will:

- create `data/demo.txt`
- call `/health`
- ingest the file into the selected collection
- run a sample query and print the response

### PowerShell demo

```powershell
# 1) Set your API key from .env
$API_KEY = "replace-with-a-strong-random-secret"

# 2) Create a small sample file
"RAG AI demo document. This project supports ingestion and question answering." | Out-File -FilePath data\demo.txt -Encoding utf8

# 3) Ingest
curl.exe -X POST "http://127.0.0.1:8000/api/v1/ingest" `
         -H "X-API-Key: $API_KEY" `
         -F "collection=demo" `
         -F "files=@data\demo.txt"

# 4) Query
curl.exe -X POST "http://127.0.0.1:8000/api/v1/query" `
         -H "X-API-Key: $API_KEY" `
         -H "Content-Type: application/json" `
         -d "{\"question\":\"What is this demo document about?\",\"collection\":\"demo\",\"top_k\":5,\"rerank\":true,\"stream\":false,\"chat_history\":[]}"
```

### Bash demo

```bash
# 1) Set your API key from .env
API_KEY="replace-with-a-strong-random-secret"

# 2) Create a small sample file
echo "RAG AI demo document. This project supports ingestion and question answering." > data/demo.txt

# 3) Ingest
curl -X POST "http://127.0.0.1:8000/api/v1/ingest" \
       -H "X-API-Key: ${API_KEY}" \
       -F "collection=demo" \
       -F "files=@data/demo.txt"

# 4) Query
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
       -H "X-API-Key: ${API_KEY}" \
       -H "Content-Type: application/json" \
       -d '{"question":"What is this demo document about?","collection":"demo","top_k":5,"rerank":true,"stream":false,"chat_history":[]}'
```

Expected result:

- Ingest response should show `status: completed`.
- Query response should include `answer` and at least one `sources` entry.

## Project Structure

```
rag-ai/
├── src/
│   ├── api/
│   │   ├── server.py            # FastAPI app, middleware, lifespan
│   │   ├── routes/
│   │   │   ├── ingest.py        # POST /ingest — file upload & processing
│   │   │   ├── query.py         # POST /query — retrieval & generation
│   │   │   └── collections.py   # Collection management endpoints
│   │   ├── dependencies.py      # Auth, rate limiting, shared deps
│   │   └── schemas.py           # Pydantic request/response models
│   ├── core/
│   │   ├── config.py            # Settings via pydantic-settings (.env)
│   │   ├── logging.py           # Structured JSON logging
│   │   └── exceptions.py        # Custom exception hierarchy
│   ├── ingestion/
│   │   ├── loader.py            # Multi-format document loaders
│   │   ├── chunker.py           # Recursive text splitting
│   │   ├── dedup.py             # Content-hash deduplication
│   │   └── pipeline.py          # Orchestrates load → chunk → embed → store
│   ├── embeddings/
│   │   ├── ollama.py            # Ollama embedding client (batched)
│   │   └── cache.py             # LRU embedding cache
│   ├── vectorstore/
│   │   ├── base.py              # Abstract vector store interface
│   │   ├── chroma_store.py      # ChromaDB implementation
│   │   └── faiss_store.py       # FAISS implementation
│   ├── retrieval/
│   │   ├── retriever.py         # Vector search + metadata filtering
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── context.py           # Context window assembly
│   ├── llm/
│   │   ├── ollama.py            # Ollama LLM client (streaming)
│   │   └── chain.py             # RAG prompt + chain + guardrails
│   ├── guardrails/
│   │   ├── validators.py        # Injection, grounding, PII, quality validators
│   │   ├── engine.py            # GuardrailsEngine orchestration
│   │   ├── owasp_validators.py  # OWASP agent threat validators
│   │   ├── data_classifier.py   # PII/PHI/PCI/secrets classifier (30+ patterns)
│   │   ├── dlp.py               # Data Loss Prevention engine
│   │   └── audit.py             # Security audit logger
│   └── ui/
│       └── app.py               # Streamlit chat interface
├── data/                        # Drop documents here
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

All settings via environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3` | Model for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` or `faiss` |
| `CHROMA_PERSIST_DIR` | `./storage/chroma` | ChromaDB data directory |
| `FAISS_INDEX_DIR` | `./storage/faiss` | FAISS index directory |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Chunks to retrieve |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANK_TOP_N` | `3` | Chunks after reranking |
| `API_KEY` | *(required)* | API authentication key |
| `RATE_LIMIT_PER_MINUTE` | `60` | Requests per minute per key |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `EMBEDDING_BATCH_SIZE` | `64` | Batch size for embedding calls |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Request timeout for Ollama calls |
| `OLLAMA_MAX_RETRIES` | `3` | Retry attempts for failed Ollama calls |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum file upload size |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `ENVIRONMENT` | `development` | `development`, `staging`, or `production` |
| `GUARDRAILS_ENABLED` | `true` | Enable/disable guardrails validation |
| `GUARDRAILS_BLOCK_ON_FAILURE` | `true` | Block queries that fail input guardrails |
| `GUARDRAILS_CHECK_INJECTION` | `true` | Enable prompt injection detection |
| `GUARDRAILS_CHECK_TOPIC` | `true` | Enable code-execution / off-topic filtering |
| `GUARDRAILS_CHECK_GROUNDING` | `true` | Enable context-grounding validation |
| `GUARDRAILS_CHECK_PII` | `true` | Enable PII detection in responses |
| `GUARDRAILS_CHECK_QUALITY` | `true` | Enable response quality checks |
| `GUARDRAILS_PII_REDACT` | `false` | Redact detected PII in responses |
| `GUARDRAILS_GROUNDING_THRESHOLD` | `0.3` | Min grounding score (0.0–1.0) |
| `OWASP_CHECK_EXFILTRATION` | `true` | Detect data exfiltration attempts |
| `OWASP_CHECK_EXCESSIVE_AGENCY` | `true` | Block excessive agent actions |
| `OWASP_CHECK_INDIRECT_INJECTION` | `true` | Detect indirect prompt injection in context |
| `OWASP_CHECK_SYSTEM_PROMPT_LEAK` | `true` | Block system prompt extraction |
| `DLP_ENABLED` | `true` | Enable Data Loss Prevention engine |
| `DLP_SCAN_INGESTION` | `true` | Scan documents before storing |
| `DLP_BLOCK_SEVERITY` | `critical` | Block if severity ≥ threshold |
| `DLP_REDACT_SEVERITY` | `high` | Redact if severity ≥ threshold |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `401 Invalid or missing API key` | Wrong or missing `X-API-Key` header | Send the exact API key from `.env` |
| `429 Rate limit exceeded` | Too many requests/min | Slow requests or increase `RATE_LIMIT_PER_MINUTE` |
| Health shows `"ollama": false` | Ollama not running / wrong URL | Start Ollama and verify `OLLAMA_BASE_URL` |
| Ingest returns "No supported files provided" | Unsupported file extension | Use `.pdf`, `.txt`, `.md`, or `.docx` |
| `413` on upload | File exceeds max limit | Increase `MAX_UPLOAD_SIZE_MB` or upload smaller files |
| Query quality is poor | Few chunks retrieved or no rerank | Increase `TOP_K`, keep `RERANK_ENABLED=true`, ingest better docs |
| CORS issues in browser app | Origin not allowed | Set `CORS_ORIGINS` appropriately in `.env` |

## Security Notes

- Always change `API_KEY` before exposing the service outside localhost.
- In production, set strict `CORS_ORIGINS` (avoid `[*]`).
- Keep guardrails and DLP enabled unless you have a specific reason to disable them.
- Do not upload secrets or regulated data unless your policy allows it.

## Validation Checklist

Use this checklist after setup:

1. `GET /health` returns `status: healthy`.
2. `POST /api/v1/ingest` stores chunks successfully.
3. `POST /api/v1/query` returns an `answer` and non-empty `sources`.
4. Streamlit UI can ingest and answer from the same collection.
5. `GET /api/v1/collections` shows your collection.

## Scaling Guide

| Scale | Recommendation |
|-------|---------------|
| **< 10K docs** | Single-node ChromaDB, default settings |
| **10K–100K docs** | FAISS with IVF index, increase batch size |
| **100K+ docs** | ChromaDB client-server mode, dedicated Ollama GPU node |
| **Multi-user** | Deploy behind nginx, add Redis for rate limiting & caching |
| **Enterprise** | Kubernetes Helm chart, horizontal API pods, Chroma cluster |

## Documentation

- [High-Level Design (HLD)](docs/HLD.md) — Detailed system architecture, component design, data flows, security model, scalability strategies, and technology decisions.

## Changelog

| Version | Date | Author | Change Description |
|---------|------|--------|--------------------|
| 1.0.0 | 2026-04-02 | Engineering Team | Initial release — full project scaffold with FastAPI server, ingestion pipeline (PDF/TXT/MD/DOCX), ChromaDB & FAISS vector stores, cross-encoder reranking, streaming LLM via Ollama, Streamlit chat UI, Docker Compose deployment, API key auth, rate limiting, embedding cache, and test suite |
| 2.0.0 | 2026-04-02 | Engineering Team | Principal Engineer upgrade — request ID tracing & timing middleware, structured JSON logging with correlation IDs, retry with exponential backoff + jitter, circuit breaker pattern, HMAC timing-safe auth, file size limits, input sanitization, config validators, multi-stage Docker build with non-root user, Docker healthchecks, GitHub Actions CI/CD, Makefile, service registry with graceful shutdown, comprehensive test suite with conftest fixtures, Bandit/Bugbear linting |
| 2.1.0 | 2026-04-02 | Engineering Team | Guardrails AI integration — prompt injection detection, topic relevance filtering, context-grounding hallucination check, PII detection & optional redaction, response quality validation, configurable per-validator toggles, guardrails report in API response, comprehensive test suite |
| 2.2.0 | 2026-04-02 | Engineering Team | OWASP Agent Threat Protection — sensitive data classifier (PII/PHI/PCI/credentials/secrets, 30+ patterns), DLP engine with block/redact/audit policies, security audit logger, OWASP validators (data exfiltration, indirect injection, excessive agency, system prompt leak), ingestion-time DLP scanning, 93 total guardrails+OWASP tests |

## License

MIT
