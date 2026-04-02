"""Custom exception hierarchy for the RAG AI application."""


class RAGError(Exception):
    """Base exception for all RAG errors."""


class IngestionError(RAGError):
    """Error during document ingestion."""


class EmbeddingError(RAGError):
    """Error generating embeddings."""


class VectorStoreError(RAGError):
    """Error interacting with the vector store."""


class RetrievalError(RAGError):
    """Error during retrieval pipeline."""


class LLMError(RAGError):
    """Error communicating with the LLM."""


class ConfigurationError(RAGError):
    """Invalid or missing configuration."""
