"""Recursive text splitting for chunking documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.logging import logger

if TYPE_CHECKING:
    from langchain_core.documents import Document


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info(
        "Chunked %d document(s) into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
