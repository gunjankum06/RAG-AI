"""Data Loss Prevention (DLP) — policy enforcement for the RAG pipeline.

Aligned with OWASP LLM Top 10 (LLM06 — Sensitive Information Disclosure)
and OWASP Agent Security threats:
  - Prevents sensitive data from leaking through LLM responses
  - Scans ingested documents and quarantines high-risk content
  - Enforces redaction-before-storage or redaction-before-response policies
  - Produces audit-grade events for every policy decision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.config import settings
from src.core.logging import logger
from src.guardrails.data_classifier import (
    ClassificationResult,
    DataCategory,
    SensitiveDataClassifier,
    Severity,
)


class DLPAction(str, Enum):
    """What the DLP engine decided to do."""

    ALLOW = "allow"          # No sensitive data found
    REDACT = "redact"        # Sensitive tokens replaced with placeholders
    BLOCK = "block"          # Request/response completely rejected
    AUDIT_ONLY = "audit"     # Logged but allowed through (dev/staging)


@dataclass
class DLPDecision:
    """Result of a DLP policy evaluation."""

    action: DLPAction
    original_text: str | None = None      # Only stored when action != ALLOW
    sanitized_text: str = ""
    classification: ClassificationResult | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "action": self.action.value,
            "reason": self.reason,
        }
        if self.classification:
            d["classification"] = self.classification.to_dict()
        return d


class DLPEngine:
    """Data Loss Prevention engine — enforces sensitive-data policies.

    Policies (configured via settings):
    - **ingestion**: scan uploaded documents before they are embedded
    - **response**: scan LLM responses before returning to user
    - **query**: scan user queries for leaked credentials / PII

    Severity thresholds control when to block vs. redact vs. audit-only.
    """

    def __init__(self) -> None:
        self._classifier = SensitiveDataClassifier(
            min_severity=Severity.LOW,
        )
        self._block_threshold = Severity[settings.dlp_block_severity.upper()]
        self._redact_threshold = Severity[settings.dlp_redact_severity.upper()]

        self._severity_order = {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }

        logger.info(
            "DLP engine initialised (block≥%s, redact≥%s, ingestion_scan=%s)",
            settings.dlp_block_severity,
            settings.dlp_redact_severity,
            settings.dlp_scan_ingestion,
        )

    # ── Public API ────────────────────────────────────────────────────

    def scan_query(self, text: str) -> DLPDecision:
        """Scan a user query for leaked credentials or sensitive data."""
        if not settings.dlp_enabled:
            return DLPDecision(action=DLPAction.ALLOW, sanitized_text=text)
        return self._evaluate(text, context="query")

    def scan_response(self, text: str) -> DLPDecision:
        """Scan an LLM response before it reaches the user."""
        if not settings.dlp_enabled:
            return DLPDecision(action=DLPAction.ALLOW, sanitized_text=text)
        return self._evaluate(text, context="response")

    def scan_document(self, text: str) -> DLPDecision:
        """Scan a document chunk before ingestion into the vector store."""
        if not settings.dlp_enabled or not settings.dlp_scan_ingestion:
            return DLPDecision(action=DLPAction.ALLOW, sanitized_text=text)
        return self._evaluate(text, context="ingestion")

    # ── Internal ──────────────────────────────────────────────────────

    def _evaluate(self, text: str, *, context: str) -> DLPDecision:
        """Run the classifier and apply the policy."""
        result = self._classifier.classify(text)

        if not result.has_sensitive_data:
            return DLPDecision(
                action=DLPAction.ALLOW,
                sanitized_text=text,
                classification=result,
            )

        highest_ord = self._severity_order[result.highest_severity]
        block_ord = self._severity_order[self._block_threshold]
        redact_ord = self._severity_order[self._redact_threshold]

        # ── Decide action ─────────────────────────────────────────────
        if highest_ord >= block_ord:
            action = DLPAction.BLOCK
            reason = (
                f"Blocked: {result.highest_severity.value}-severity "
                f"sensitive data detected in {context} "
                f"({', '.join(f.data_type for f in result.findings)})"
            )
            logger.warning("DLP BLOCK [%s]: %s", context, reason)
            return DLPDecision(
                action=action,
                sanitized_text="",
                classification=result,
                reason=reason,
            )

        if highest_ord >= redact_ord:
            sanitized = self._classifier.redact(text)
            reason = (
                f"Redacted: {result.highest_severity.value}-severity "
                f"sensitive data in {context} "
                f"({result.total_count} finding(s))"
            )
            logger.info("DLP REDACT [%s]: %s", context, reason)
            return DLPDecision(
                action=DLPAction.REDACT,
                sanitized_text=sanitized,
                classification=result,
                reason=reason,
            )

        # Below redact threshold → audit only
        reason = (
            f"Audit: {result.highest_severity.value}-severity "
            f"sensitive data noted in {context}"
        )
        logger.info("DLP AUDIT [%s]: %s", context, reason)
        return DLPDecision(
            action=DLPAction.AUDIT_ONLY,
            sanitized_text=text,
            classification=result,
            reason=reason,
        )
