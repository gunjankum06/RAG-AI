"""RAG chain — combines retrieval context with LLM generation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.core.config import settings
from src.core.logging import logger
from src.core.observability import start_span
from src.guardrails.engine import GuardrailsEngine, GuardrailsReport
from src.llm.ollama import OllamaLLM
from src.retrieval.context import build_context
from src.retrieval.reranker import rerank
from src.retrieval.retriever import Retriever
from src.vectorstore.base import SearchResult

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information, say so — do not make up answers.

Rules:
- Be concise and accurate.
- Cite which source(s) you used when possible.
- If the context is insufficient, say "I don't have enough information to answer that based on the available documents."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    guardrails: GuardrailsReport | None = None


class RAGChain:
    """Orchestrates retrieval and generation for RAG queries."""

    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM | None = None,
        guardrails: GuardrailsEngine | None = None,
    ):
        self._retriever = retriever
        self._llm = llm or OllamaLLM()
        self._guardrails = guardrails or GuardrailsEngine()

    async def query(
        self,
        question: str,
        collection: str = "default",
        chat_history: list[ChatMessage] | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> RAGResponse:
        """Run a full RAG query and return the answer with sources."""
        use_rerank = use_rerank if use_rerank is not None else settings.rerank_enabled
        effective_top_k = top_k or settings.top_k

        with start_span(
            "rag.query",
            {
                "rag.collection": collection,
                "rag.top_k": effective_top_k,
                "rag.rerank_enabled": use_rerank,
                "rag.question_length": len(question),
            },
        ):

            # ── Guardrails: input validation ──────────────────────────────
            if self._guardrails.enabled:
                with start_span("rag.guardrails.input"):
                    input_report = self._guardrails.check(query=question)
                if input_report.blocked:
                    logger.warning("Query blocked by guardrails: %s", input_report.block_reason)
                    return RAGResponse(
                        answer="I'm unable to process this query due to safety guidelines.",
                        guardrails=input_report,
                    )

            # 1. Retrieve
            # Fetch more candidates if we'll rerank
            fetch_k = effective_top_k * 3 if use_rerank else effective_top_k
            with start_span("rag.retrieve", {"rag.fetch_k": fetch_k}):
                results = await self._retriever.retrieve(
                    query=question,
                    collection=collection,
                    top_k=fetch_k,
                )

            # 2. Rerank
            if use_rerank and results:
                with start_span("rag.rerank", {"rag.candidate_count": len(results)}):
                    results = rerank(question, results)

            # 3. Build context
            with start_span("rag.build_context", {"rag.source_count": len(results)}):
                context = build_context(results)

            # 4. Format prompt
            history_str = self._format_history(chat_history)
            prompt = RAG_SYSTEM_PROMPT.format(
                context=context,
                chat_history=history_str,
                question=question,
            )

            # 5. Generate
            logger.info("Generating answer for: %s", question[:80])
            with start_span("rag.generate", {"llm.model": settings.llm_model}):
                answer = await self._llm.generate(prompt)

            # ── Guardrails: output validation ─────────────────────────────
            guardrails_report = None
            if self._guardrails.enabled:
                with start_span("rag.guardrails.output"):
                    guardrails_report = self._guardrails.check(
                        query=question,
                        response=answer,
                        context=context,
                    )
                    answer = self._guardrails.redact_pii(answer)

            return RAGResponse(answer=answer, sources=results, guardrails=guardrails_report)

    async def query_stream(
        self,
        question: str,
        collection: str = "default",
        chat_history: list[ChatMessage] | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream a RAG answer token by token."""
        use_rerank = use_rerank if use_rerank is not None else settings.rerank_enabled
        effective_top_k = top_k or settings.top_k

        with start_span(
            "rag.query_stream",
            {
                "rag.collection": collection,
                "rag.top_k": effective_top_k,
                "rag.rerank_enabled": use_rerank,
                "rag.question_length": len(question),
            },
        ):
            # ── Guardrails: input validation (blocks before streaming) ────
            if self._guardrails.enabled:
                with start_span("rag.guardrails.input"):
                    input_report = self._guardrails.check(query=question)
                if input_report.blocked:
                    logger.warning("Streaming query blocked by guardrails: %s", input_report.block_reason)
                    yield "I'm unable to process this query due to safety guidelines."
                    return

            fetch_k = effective_top_k * 3 if use_rerank else effective_top_k
            with start_span("rag.retrieve", {"rag.fetch_k": fetch_k}):
                results = await self._retriever.retrieve(
                    query=question,
                    collection=collection,
                    top_k=fetch_k,
                )

            if use_rerank and results:
                with start_span("rag.rerank", {"rag.candidate_count": len(results)}):
                    results = rerank(question, results)

            with start_span("rag.build_context", {"rag.source_count": len(results)}):
                context = build_context(results)
            history_str = self._format_history(chat_history)
            prompt = RAG_SYSTEM_PROMPT.format(
                context=context,
                chat_history=history_str,
                question=question,
            )

            logger.info("Streaming answer for: %s", question[:80])
            with start_span("rag.generate_stream", {"llm.model": settings.llm_model}):
                async for token in self._llm.generate_stream(prompt):
                    yield token

    @staticmethod
    def _format_history(chat_history: list[ChatMessage] | None) -> str:
        if not chat_history:
            return "(No prior conversation)"
        lines: list[str] = []
        for msg in chat_history[-10:]:  # Keep last 10 exchanges
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
