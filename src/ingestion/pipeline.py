"""Ingestion pipeline: load → chunk → deduplicate → embed → store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.core.logging import logger
from src.ingestion.chunker import chunk_documents
from src.ingestion.dedup import deduplicate
from src.ingestion.loader import load_directory, load_document

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from src.vectorstore.base import BaseVectorStore


async def ingest_files(
    file_paths: list[Path],
    vector_store: BaseVectorStore,
    collection: str = "default",
) -> dict:
    """Ingest a list of files through the full pipeline.

    Returns a summary dict with counts.
    """
    # 1. Load
    all_docs: list[Document] = []
    for path in file_paths:
        if path.is_dir():
            all_docs.extend(load_directory(path))
        else:
            all_docs.extend(load_document(path))

    if not all_docs:
        logger.warning("No documents loaded from provided paths")
        return {"documents_loaded": 0, "chunks_created": 0, "chunks_stored": 0}

    # 2. Chunk
    chunks = chunk_documents(all_docs)

    # 3. Deduplicate
    unique_chunks = deduplicate(chunks)

    # 4. Store (embedding happens inside the vector store)
    await vector_store.add_documents(unique_chunks, collection=collection)

    summary = {
        "documents_loaded": len(all_docs),
        "chunks_created": len(chunks),
        "chunks_stored": len(unique_chunks),
    }
    logger.info("Ingestion complete: %s", summary)
    return summary
