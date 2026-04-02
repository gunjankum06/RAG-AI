"""Query routes — retrieval-augmented generation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_rag_chain, rate_limiter
from src.api.schemas import QueryRequest, QueryResponse, SourceSchema
from src.core.logging import logger
from src.llm.chain import ChatMessage

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    _api_key: str = Depends(rate_limiter),
):
    """Query documents with RAG — supports both streaming and non-streaming."""
    rag_chain = get_rag_chain()

    chat_history = [
        ChatMessage(role=msg.role, content=msg.content)
        for msg in body.chat_history
    ]

    if body.stream:
        return StreamingResponse(
            _stream_response(rag_chain, body, chat_history),
            media_type="text/event-stream",
        )

    # Non-streaming
    result = await rag_chain.query(
        question=body.question,
        collection=body.collection,
        chat_history=chat_history,
        top_k=body.top_k,
        use_rerank=body.rerank,
    )

    sources = [
        SourceSchema(
            content=s.content[:500],
            filename=s.metadata.get("filename"),
            score=s.score,
        )
        for s in result.sources
    ]

    logger.info(
        "Query answered (collection=%s, sources=%d)",
        body.collection,
        len(sources),
    )
    return QueryResponse(answer=result.answer, sources=sources)


async def _stream_response(rag_chain, body: QueryRequest, chat_history: list[ChatMessage]):
    """Yield server-sent events for streaming responses."""
    async for token in rag_chain.query_stream(
        question=body.question,
        collection=body.collection,
        chat_history=chat_history,
        top_k=body.top_k,
        use_rerank=body.rerank,
    ):
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"
