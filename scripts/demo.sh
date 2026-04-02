#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"
COLLECTION="${COLLECTION:-demo}"
API_KEY="${API_KEY:-}"

if [[ -z "${API_KEY}" ]]; then
  echo "API key missing. Set API_KEY env var." >&2
  exit 1
fi

mkdir -p data
echo "RAG AI demo document. This project supports ingestion and question answering." > data/demo.txt

echo "Checking health..."
curl -q -sS "${API_URL}/health" | sed 's/.*/&/'
echo

echo "Ingesting demo document..."
INGEST_RESPONSE="$(curl -q -sS -X POST "${API_URL}/api/v1/ingest" \
  -H "X-API-Key: ${API_KEY}" \
  -F "collection=${COLLECTION}" \
  -F "files=@data/demo.txt")"
echo "${INGEST_RESPONSE}"

if [[ "${INGEST_RESPONSE}" != *'"status":"completed"'* ]]; then
  echo "Ingest failed." >&2
  if [[ "${INGEST_RESPONSE}" == *'/api/embed'* ]] || [[ "${INGEST_RESPONSE}" == *'404'* ]]; then
    echo "Hint: check Ollama version/model compatibility and embedding endpoint support." >&2
  fi
  exit 1
fi

echo

echo "Querying demo collection..."
curl -q -sS -X POST "${API_URL}/api/v1/query" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this demo document about?\",\"collection\":\"${COLLECTION}\",\"top_k\":5,\"rerank\":true,\"stream\":false,\"chat_history\":[]}"
echo

echo "Demo complete."
