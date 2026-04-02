"""Structured logging with JSON output and correlation ID propagation."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

from src.core.config import settings

# Correlation ID propagated across async contexts
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


class JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get("-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable log format for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | [%(correlation_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        record.correlation_id = correlation_id_var.get("-")  # type: ignore[attr-defined]
        return super().format(record)


def setup_logging() -> logging.Logger:
    """Configure application logging based on settings."""
    root_logger = logging.getLogger("rag_ai")
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if settings.log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(TextFormatter())
        root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


logger = setup_logging()
