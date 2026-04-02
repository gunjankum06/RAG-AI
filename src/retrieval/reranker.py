"""Cross-encoder reranker for improving retrieval precision."""

from __future__ import annotations

from src.core.config import settings
from src.core.logging import logger
from src.vectorstore.base import SearchResult

_reranker_model = None
_reranker_unavailable = False


def _get_reranker():
    """Lazy-load the cross-encoder model."""
    global _reranker_model  # noqa: PLW0603
    global _reranker_unavailable  # noqa: PLW0603

    if _reranker_unavailable:
        return None

    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder reranker model loaded")
        except Exception as exc:
            _reranker_unavailable = True
            logger.warning("Reranker unavailable; falling back to vector search ordering: %s", exc)
            return None
    return _reranker_model


def rerank(
    query: str,
    results: list[SearchResult],
    top_n: int | None = None,
) -> list[SearchResult]:
    """Rerank search results using a cross-encoder model.

    Args:
        query: The original user query.
        results: Candidate chunks from vector search.
        top_n: Number of results to return after reranking.

    Returns:
        Reranked and truncated list of SearchResult.
    """
    if not results:
        return results

    top_n = top_n or settings.rerank_top_n
    model = _get_reranker()
    if model is None:
        return results[:top_n]

    pairs = [(query, r.content) for r in results]
    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result.score = float(score)

    reranked = sorted(results, key=lambda r: r.score, reverse=True)[:top_n]

    logger.info(
        "Reranked %d → %d results (top score=%.4f)",
        len(results),
        len(reranked),
        reranked[0].score if reranked else 0.0,
    )
    return reranked
