"""Custom exception hierarchy for the RAG AI application.

All domain exceptions inherit from RAGError, which carries an optional
`retriable` flag so callers can decide whether to retry.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for all RAG errors."""

    def __init__(self, message: str = "", *, retriable: bool = False) -> None:
        super().__init__(message)
        self.retriable = retriable


class IngestionError(RAGError):
    """Error during document ingestion."""


class EmbeddingError(RAGError):
    """Error generating embeddings."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message, retriable=True)


class VectorStoreError(RAGError):
    """Error interacting with the vector store."""


class RetrievalError(RAGError):
    """Error during retrieval pipeline."""


class LLMError(RAGError):
    """Error communicating with the LLM."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message, retriable=True)


class ConfigurationError(RAGError):
    """Invalid or missing configuration."""


class CircuitOpenError(RAGError):
    """Raised when a circuit breaker is open and rejecting calls."""

    def __init__(self, service: str = "") -> None:
        super().__init__(f"Circuit breaker open for {service}", retriable=False)
