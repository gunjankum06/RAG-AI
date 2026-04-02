"""Tests for the FastAPI API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.core.config import settings

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "RAG AI"
    assert "docs" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "vector_store" in data


def test_query_without_api_key():
    response = client.post(
        "/api/v1/query",
        json={"question": "test"},
    )
    assert response.status_code == 401


def test_ingest_without_api_key():
    response = client.post("/api/v1/ingest")
    assert response.status_code == 401


def test_list_collections_without_api_key():
    response = client.get("/api/v1/collections")
    assert response.status_code == 401
