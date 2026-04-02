"""Ingestion pipeline: load → chunk → deduplicate → DLP scan → embed → store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.core.config import settings
from src.core.logging import logger
from src.guardrails.audit import audit_ingestion_scan
from src.guardrails.dlp import DLPAction, DLPEngine
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

    # 4. DLP scan — check chunks for sensitive data before storage
    dlp_blocked = 0
    dlp_redacted = 0
    safe_chunks: list[Document] = []

    if settings.dlp_enabled and settings.dlp_scan_ingestion:
        dlp = DLPEngine()
        for chunk in unique_chunks:
            decision = dlp.scan_document(chunk.page_content)
            filename = chunk.metadata.get("filename", chunk.metadata.get("source", "unknown"))

            if decision.action == DLPAction.BLOCK:
                dlp_blocked += 1
                audit_ingestion_scan(
                    filename=filename,
                    action="block",
                    finding_count=decision.classification.total_count if decision.classification else 0,
                    categories=sorted(
                        c.value for c in decision.classification.categories_found
                    ) if decision.classification else [],
                )
                continue  # skip this chunk

            if decision.action == DLPAction.REDACT:
                dlp_redacted += 1
                chunk.page_content = decision.sanitized_text
                audit_ingestion_scan(
                    filename=filename,
                    action="redact",
                    finding_count=decision.classification.total_count if decision.classification else 0,
                    categories=sorted(
                        c.value for c in decision.classification.categories_found
                    ) if decision.classification else [],
                )

            safe_chunks.append(chunk)
    else:
        safe_chunks = unique_chunks

    # 5. Store (embedding happens inside the vector store)
    await vector_store.add_documents(safe_chunks, collection=collection)

    summary = {
        "documents_loaded": len(all_docs),
        "chunks_created": len(chunks),
        "chunks_stored": len(safe_chunks),
        "dlp_blocked": dlp_blocked,
        "dlp_redacted": dlp_redacted,
    }
    logger.info("Ingestion complete: %s", summary)
    return summary
