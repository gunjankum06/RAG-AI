"""Tests for the FastAPI API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.core.config import settings

client = TestClient(app)


# ── System Endpoints ──────────────────────────────────────────────────


class TestSystemEndpoints:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "RAG AI"
        assert data["version"] == "2.0.0"
        assert "docs" in data
        assert "environment" in data

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "vector_store" in data
        assert "version" in data

    def test_request_id_header(self):
        response = client.get("/")
        assert "x-request-id" in response.headers
        assert "x-response-time-ms" in response.headers

    def test_custom_request_id_propagated(self):
        custom_id = "test-req-12345"
        response = client.get("/", headers={"X-Request-ID": custom_id})
        assert response.headers.get("x-request-id") == custom_id


# ── Auth Guard ────────────────────────────────────────────────────────


class TestAuthGuard:
    def test_query_without_api_key(self):
        response = client.post("/api/v1/query", json={"question": "test"})
        assert response.status_code == 401

    def test_query_with_wrong_key(self):
        response = client.post(
            "/api/v1/query",
            json={"question": "test"},
            headers={"X-API-Key": "wrong-key-here"},
        )
        assert response.status_code == 401

    def test_ingest_without_api_key(self):
        response = client.post("/api/v1/ingest")
        assert response.status_code == 401

    def test_list_collections_without_api_key(self):
        response = client.get("/api/v1/collections")
        assert response.status_code == 401

    def test_delete_collection_without_api_key(self):
        response = client.delete("/api/v1/collections/default")
        assert response.status_code == 401


# ── Input Validation ──────────────────────────────────────────────────


class TestInputValidation:
    def test_query_empty_question(self):
        response = client.post(
            "/api/v1/query",
            json={"question": ""},
            headers={"X-API-Key": settings.api_key},
        )
        assert response.status_code == 422

    def test_query_invalid_collection_name(self):
        response = client.post(
            "/api/v1/query",
            json={"question": "test", "collection": "invalid name!"},
            headers={"X-API-Key": settings.api_key},
        )
        assert response.status_code == 422

    def test_query_top_k_out_of_range(self):
        response = client.post(
            "/api/v1/query",
            json={"question": "test", "top_k": 999},
            headers={"X-API-Key": settings.api_key},
        )
        assert response.status_code == 422


# ── Resilience ────────────────────────────────────────────────────────


class TestResilience:
    def test_circuit_breaker_state(self):
        from src.core.resilience import CircuitBreaker

        cb = CircuitBreaker("test-service", failure_threshold=2, recovery_timeout=1.0)
        assert cb.state == CircuitBreaker.CLOSED

    def test_config_validators(self):
        """Ensure config validators catch bad combinations."""
        from pydantic import ValidationError
        from src.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(chunk_size=100, chunk_overlap=200)
