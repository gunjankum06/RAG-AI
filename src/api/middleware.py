"""FastAPI middleware: request ID injection, request timing, and logging."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.logging import correlation_id_var, logger


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID, measure duration, and log every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or accept an existing request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        correlation_id_var.set(request_id)

        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            status = response.status_code if response else 500
            logger.info(
                "%s %s → %d (%.1fms)",
                request.method,
                request.url.path,
                status,
                duration_ms,
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
