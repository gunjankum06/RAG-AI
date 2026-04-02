"""Context window assembly from retrieved chunks."""

from __future__ import annotations

from src.vectorstore.base import SearchResult


def build_context(results: list[SearchResult], max_tokens: int = 3000) -> str:
    """Assemble a context string from search results.

    Each chunk is formatted with its source and relevance score.
    Truncates if the assembled context exceeds max_tokens (approximated as chars / 4).
    """
    if not results:
        return ""

    max_chars = max_tokens * 4  # rough approximation
    parts: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        source = result.metadata.get("filename", result.metadata.get("source", "unknown"))
        header = f"[Source {i}: {source} | Score: {result.score:.3f}]"
        block = f"{header}\n{result.content}"

        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                parts.append(block[:remaining] + "\n... (truncated)")
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(parts)
