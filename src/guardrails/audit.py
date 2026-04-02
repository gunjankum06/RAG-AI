"""Security audit logger — tamper-evident logging for sensitive data events.

Aligned with OWASP LLM Top 10:
  - LLM06: Sensitive Information Disclosure (audit trail)
  - OWASP Agent Threats: Logging & Monitoring Failures

Emits structured audit events for every DLP decision, guardrail
violation, and sensitive-data access attempt.  Events are written to
a dedicated audit logger (separate from application logs) and include
correlation IDs for traceability.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from src.core.logging import correlation_id_var


class _AuditFormatter(logging.Formatter):
    """JSON formatter for audit log lines — immutable structure."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_event": getattr(record, "audit_event", "unknown"),
            "correlation_id": correlation_id_var.get("-"),
            "severity": getattr(record, "severity", "info"),
            "category": getattr(record, "category", ""),
            "action": getattr(record, "action", ""),
            "detail": record.getMessage(),
            "metadata": getattr(record, "audit_metadata", {}),
        }
        return json.dumps(entry, default=str)


def _setup_audit_logger() -> logging.Logger:
    """Create a dedicated audit logger (separate sink from app logs)."""
    audit = logging.getLogger("rag_ai.audit")
    audit.setLevel(logging.INFO)
    audit.propagate = False  # Don't bubble up to root logger

    if not audit.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_AuditFormatter())
        audit.addHandler(handler)

    return audit


_audit = _setup_audit_logger()


# ── Public API ────────────────────────────────────────────────────────


def audit_dlp_event(
    *,
    event: str,
    action: str,
    context: str,
    severity: str = "info",
    category: str = "",
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a DLP audit event."""
    record = _audit.makeRecord(
        name="rag_ai.audit",
        level=logging.INFO,
        fn="",
        lno=0,
        msg=detail,
        args=(),
        exc_info=None,
    )
    record.audit_event = event  # type: ignore[attr-defined]
    record.severity = severity  # type: ignore[attr-defined]
    record.category = category  # type: ignore[attr-defined]
    record.action = action  # type: ignore[attr-defined]
    record.audit_metadata = metadata or {}  # type: ignore[attr-defined]
    _audit.handle(record)


def audit_sensitive_data_detected(
    *,
    context: str,
    action: str,
    categories: list[str],
    finding_count: int,
    highest_severity: str,
) -> None:
    """Convenience: log a sensitive-data detection event."""
    audit_dlp_event(
        event="sensitive_data_detected",
        action=action,
        context=context,
        severity=highest_severity,
        category=",".join(categories),
        detail=f"Sensitive data detected in {context}: {finding_count} finding(s)",
        metadata={
            "finding_count": finding_count,
            "categories": categories,
            "highest_severity": highest_severity,
        },
    )


def audit_query_blocked(*, reason: str, query_preview: str = "") -> None:
    """Convenience: log a blocked query event."""
    # Never log full query — only first 40 chars
    safe_preview = query_preview[:40] + "..." if len(query_preview) > 40 else query_preview
    audit_dlp_event(
        event="query_blocked",
        action="block",
        context="query",
        severity="high",
        detail=f"Query blocked: {reason}",
        metadata={"query_preview": safe_preview},
    )


def audit_response_redacted(*, finding_count: int, categories: list[str]) -> None:
    """Convenience: log a response redaction event."""
    audit_dlp_event(
        event="response_redacted",
        action="redact",
        context="response",
        severity="medium",
        detail=f"Response redacted: {finding_count} sensitive item(s)",
        metadata={"finding_count": finding_count, "categories": categories},
    )


def audit_ingestion_scan(
    *,
    filename: str,
    action: str,
    finding_count: int,
    categories: list[str],
) -> None:
    """Convenience: log an ingestion-time DLP scan event."""
    # Never log filename with path — only the basename
    safe_name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    audit_dlp_event(
        event="ingestion_scan",
        action=action,
        context="ingestion",
        severity="medium" if finding_count > 0 else "info",
        category=",".join(categories),
        detail=f"Ingestion scan for '{safe_name}': {action} ({finding_count} finding(s))",
        metadata={
            "filename": safe_name,
            "finding_count": finding_count,
            "categories": categories,
        },
    )
