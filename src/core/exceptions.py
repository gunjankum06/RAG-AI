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


class GuardrailsError(RAGError):
    """Raised when a guardrails check blocks the request."""


class SensitiveDataError(RAGError):
    """Raised when sensitive data (PII/PHI/PCI/secrets) is detected."""

    def __init__(self, message: str = "Sensitive data policy violation") -> None:
        super().__init__(message, retriable=False)
