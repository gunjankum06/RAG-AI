"""Content-hash deduplication for document chunks."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from src.core.logging import logger

if TYPE_CHECKING:
    from langchain_core.documents import Document


def compute_hash(text: str) -> str:
    """Compute a SHA-256 hash of the text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate(chunks: list[Document]) -> list[Document]:
    """Remove duplicate chunks based on content hash.

    Adds a `content_hash` metadata field to each surviving chunk.
    """
    seen: set[str] = set()
    unique: list[Document] = []

    for chunk in chunks:
        content_hash = compute_hash(chunk.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            chunk.metadata["content_hash"] = content_hash
            unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed:
        logger.info("Deduplication removed %d duplicate chunk(s)", removed)
    logger.info("Unique chunks after dedup: %d", len(unique))
    return unique
