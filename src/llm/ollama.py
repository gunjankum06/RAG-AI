"""Ollama LLM client with streaming support."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from src.core.config import settings
from src.core.exceptions import LLMError
from src.core.logging import logger


class OllamaLLM:
    """Async client for Ollama text generation."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self._client = httpx.AsyncClient(timeout=300.0)

    async def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming)."""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]
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
                timeout=300.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    import json

                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
