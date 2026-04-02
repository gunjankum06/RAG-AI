"""Resilience primitives: retry with exponential backoff and circuit breaker."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

from src.core.exceptions import CircuitOpenError
from src.core.logging import logger

T = TypeVar("T")


# ── Retry with Exponential Backoff ───────────────────────────────────


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Decorator: retry an async function with exponential backoff + jitter."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2**attempt), max_delay)
                    # Add jitter: ±25%
                    import random
                    jitter = delay * 0.25 * (2 * random.random() - 1)  # noqa: S311
                    sleep_time = max(0.1, delay + jitter)
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        func.__qualname__,
                        sleep_time,
                        exc,
                    )
                    await asyncio.sleep(sleep_time)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ── Circuit Breaker ──────────────────────────────────────────────────


class CircuitBreaker:
    """Simple circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

    - After `failure_threshold` consecutive failures, the circuit opens.
    - After `recovery_timeout` seconds, it moves to half-open.
    - A single success in half-open closes the circuit again.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                return self.HALF_OPEN
        return self._state

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute `func` through the circuit breaker."""
        async with self._lock:
            current = self.state
            if current == self.OPEN:
                raise CircuitOpenError(self.service_name)

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            await self._on_failure()
            raise exc
        else:
            await self._on_success()
            return result

    async def _on_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            if self._state != self.CLOSED:
                logger.info("Circuit breaker CLOSED for %s", self.service_name)
                self._state = self.CLOSED

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                if self._state != self.OPEN:
                    logger.error(
                        "Circuit breaker OPEN for %s after %d failures",
                        self.service_name,
                        self._failure_count,
                    )
                self._state = self.OPEN

    def reset(self) -> None:
        self._state = self.CLOSED
        self._failure_count = 0
