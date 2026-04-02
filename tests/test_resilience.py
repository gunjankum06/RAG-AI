"""Tests for resilience primitives: retry and circuit breaker."""

import asyncio

import pytest

from src.core.resilience import CircuitBreaker, async_retry


# ── Retry ─────────────────────────────────────────────────────────────


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_without_retry(self):
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        @async_retry(max_retries=2, base_delay=0.01, exceptions=(ValueError,))
        async def always_fail():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await always_fail()

    @pytest.mark.asyncio
    async def test_does_not_retry_unexpected_exception(self):
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        async def wrong_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong")

        with pytest.raises(TypeError):
            await wrong_error()
        assert call_count == 1


# ── Circuit Breaker ───────────────────────────────────────────────────


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60.0)

        async def fail():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitBreaker.OPEN

    @pytest.mark.asyncio
    async def test_allows_success(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        async def succeed():
            return 42

        result = await cb.call(succeed)
        assert result == 42
        assert cb.state == CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_resets_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        async def fail():
            raise RuntimeError("boom")

        async def succeed():
            return "ok"

        # One failure
        with pytest.raises(RuntimeError):
            await cb.call(fail)

        # Success resets count
        await cb.call(succeed)
        assert cb._failure_count == 0

    def test_reset_method(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb._state = CircuitBreaker.OPEN
        cb._failure_count = 10
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb._failure_count == 0
