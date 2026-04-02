"""Ollama LLM client with streaming, retry, and connection pooling."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from src.core.config import settings
from src.core.exceptions import LLMError
from src.core.logging import logger
from src.core.resilience import async_retry


class OllamaLLM:
    """Async client for Ollama text generation."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    @async_retry(max_retries=settings.ollama_max_retries, base_delay=2.0, exceptions=(LLMError,))
    async def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming, with retry)."""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning(
                    "Ollama /api/generate not found at %s; using local fallback response",
                    self.base_url,
                )
                return self._fallback_response(prompt)
            raise LLMError(f"Ollama generation failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama generation failed: {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise LLMError(f"Unexpected Ollama response: {exc}") from exc

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens from Ollama one at a time."""
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
                timeout=settings.ollama_timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning(
                    "Ollama /api/generate not found at %s; streaming local fallback response",
                    self.base_url,
                )
                for token in self._fallback_response(prompt).split():
                    yield f"{token} "
                return
            raise LLMError(f"Ollama streaming failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self._client.get(
                f"{self.base_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _fallback_response(prompt: str) -> str:
        """Fallback response when Ollama generation endpoint is unavailable."""
        question = "your question"
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                question = line.replace("Question:", "", 1).strip() or question
                break

        return (
            "Model generation endpoint is unavailable in the current local runtime. "
            f"I can still process retrieval and safety checks, but I cannot generate a full LLM answer for: '{question}'."
        )
