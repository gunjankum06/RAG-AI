"""Multi-format document loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from src.core.exceptions import IngestionError
from src.core.logging import logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}

SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())


def load_document(file_path: Path) -> list[Document]:
    """Load a single document, returning LangChain Document objects."""
    suffix = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        raise IngestionError(
            f"Unsupported file type '{suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Loading document: %s", file_path.name)
    try:
        loader = loader_cls(str(file_path))
        docs = loader.load()
    except Exception as exc:
        raise IngestionError(f"Failed to load {file_path.name}: {exc}") from exc

    # Attach source metadata
    for doc in docs:
        doc.metadata.setdefault("source", str(file_path))
        doc.metadata.setdefault("filename", file_path.name)

    logger.info("Loaded %d page(s) from %s", len(docs), file_path.name)
    return docs


def load_directory(directory: Path) -> list[Document]:
    """Load all supported documents from a directory (non-recursive)."""
    if not directory.is_dir():
        raise IngestionError(f"Directory not found: {directory}")

    all_docs: list[Document] = []
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_docs.extend(load_document(file_path))

    logger.info("Total documents loaded from %s: %d pages", directory, len(all_docs))
    return all_docs
