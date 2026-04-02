"""Sensitive data classification — PII, PHI, PCI-DSS, secrets, credentials.

Aligned with OWASP Top 10 for LLM Applications (2025):
  - LLM06: Sensitive Information Disclosure
  - OWASP Agent Threat: Data exfiltration prevention

Each category has its own set of regex patterns and a severity level.
The classifier returns structured findings so callers can decide to
block, redact, or simply audit-log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataCategory(str, Enum):
    """Top-level data sensitivity categories."""

    PII = "pii"            # Personally Identifiable Information
    PHI = "phi"            # Protected Health Information (HIPAA)
    PCI = "pci"            # Payment Card Industry (PCI-DSS)
    CREDENTIALS = "creds"  # Passwords, API keys, tokens
    SECRETS = "secrets"    # Private keys, connection strings


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SensitiveFinding:
    """A single sensitive-data match."""

    category: DataCategory
    data_type: str          # e.g. "ssn", "credit_card", "aws_key"
    severity: Severity
    count: int = 1
    sample: str = ""        # First N chars of match (for audit, never the full value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "data_type": self.data_type,
            "severity": self.severity.value,
            "count": self.count,
        }


@dataclass
class ClassificationResult:
    """Aggregated result of a sensitive-data scan."""

    has_sensitive_data: bool = False
    findings: list[SensitiveFinding] = field(default_factory=list)
    highest_severity: Severity = Severity.LOW

    @property
    def categories_found(self) -> set[DataCategory]:
        return {f.category for f in self.findings}

    @property
    def total_count(self) -> int:
        return sum(f.count for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_sensitive_data": self.has_sensitive_data,
            "highest_severity": self.highest_severity.value,
            "total_findings": self.total_count,
            "categories": sorted(c.value for c in self.categories_found),
            "findings": [f.to_dict() for f in self.findings],
        }


# ── Pattern definitions ──────────────────────────────────────────────

# (display_name, category, severity, regex_pattern)
_PATTERN_DEFS: list[tuple[str, DataCategory, Severity, str]] = [
    # ── PII ───────────────────────────────────────────────────────────
    (
        "email",
        DataCategory.PII,
        Severity.MEDIUM,
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    ),
    (
        "phone_us",
        DataCategory.PII,
        Severity.MEDIUM,
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    ),
    (
        "ssn",
        DataCategory.PII,
        Severity.CRITICAL,
        r"\b\d{3}-\d{2}-\d{4}\b",
    ),
    (
        "drivers_license",
        DataCategory.PII,
        Severity.HIGH,
        r"\b[A-Z]\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d\b",
    ),
    (
        "passport_us",
        DataCategory.PII,
        Severity.HIGH,
        r"\b[A-Z]\d{8}\b",
    ),
    (
        "date_of_birth",
        DataCategory.PII,
        Severity.MEDIUM,
        r"\b(?:DOB|Date\s+of\s+Birth|Born)\s*[:=]?\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
    ),
    (
        "ip_address",
        DataCategory.PII,
        Severity.LOW,
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    ),
    (
        "physical_address",
        DataCategory.PII,
        Severity.MEDIUM,
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Ln|Lane|Way|Ct|Court|Pl|Place)\b",
    ),

    # ── PHI (HIPAA) ──────────────────────────────────────────────────
    (
        "medical_record_number",
        DataCategory.PHI,
        Severity.CRITICAL,
        r"\b(?:MRN|Medical\s+Record)\s*[:=#]?\s*[A-Z0-9]{6,12}\b",
    ),
    (
        "health_plan_id",
        DataCategory.PHI,
        Severity.HIGH,
        r"\b(?:Health\s+Plan|Insurance)\s*(?:ID|Number|#)\s*[:=]?\s*[A-Z0-9]{6,15}\b",
    ),
    (
        "diagnosis_code",
        DataCategory.PHI,
        Severity.HIGH,
        r"\b(?:ICD[-\s]?10|Diagnosis)\s*[:=]?\s*[A-Z]\d{2}(?:\.\d{1,4})?\b",
    ),
    (
        "medication",
        DataCategory.PHI,
        Severity.MEDIUM,
        r"\b(?:prescribed|Rx|medication|drug)\s*[:=]?\s*\w+\s+\d+\s*(?:mg|mcg|ml|g)\b",
    ),
    (
        "lab_result",
        DataCategory.PHI,
        Severity.MEDIUM,
        r"\b(?:lab\s+result|blood\s+test|A1C|CBC|BMP|TSH|INR)\s*[:=]?\s*\d+\.?\d*\s*(?:mg/dL|mmol/L|%|IU/L)?\b",
    ),

    # ── PCI-DSS ───────────────────────────────────────────────────────
    (
        "credit_card",
        DataCategory.PCI,
        Severity.CRITICAL,
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    ),
    (
        "cvv",
        DataCategory.PCI,
        Severity.CRITICAL,
        r"\b(?:CVV|CVC|CVV2|CID)\s*[:=]?\s*\d{3,4}\b",
    ),
    (
        "bank_account",
        DataCategory.PCI,
        Severity.HIGH,
        r"\b(?:Account|Acct)\s*(?:Number|No|#)\s*[:=]?\s*\d{8,17}\b",
    ),
    (
        "routing_number",
        DataCategory.PCI,
        Severity.HIGH,
        r"\b(?:Routing|ABA)\s*(?:Number|No|#)\s*[:=]?\s*\d{9}\b",
    ),
    (
        "iban",
        DataCategory.PCI,
        Severity.HIGH,
        r"\b[A-Z]{2}\d{2}\s?(?:\d{4}\s?){3,7}\d{1,4}\b",
    ),

    # ── Credentials ───────────────────────────────────────────────────
    (
        "password_in_text",
        DataCategory.CREDENTIALS,
        Severity.CRITICAL,
        r"(?:password|passwd|pwd)\s*[:=]\s*\S{6,}",
    ),
    (
        "bearer_token",
        DataCategory.CREDENTIALS,
        Severity.CRITICAL,
        r"\b(?:Bearer|Token)\s+[A-Za-z0-9\-._~+/]+=*\b",
    ),
    (
        "basic_auth",
        DataCategory.CREDENTIALS,
        Severity.CRITICAL,
        r"\bBasic\s+[A-Za-z0-9+/]{20,}={0,2}\b",
    ),
    (
        "jwt_token",
        DataCategory.CREDENTIALS,
        Severity.CRITICAL,
        r"\beyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b",
    ),

    # ── Secrets ───────────────────────────────────────────────────────
    (
        "aws_access_key",
        DataCategory.SECRETS,
        Severity.CRITICAL,
        r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b",
    ),
    (
        "aws_secret_key",
        DataCategory.SECRETS,
        Severity.CRITICAL,
        r"\b[A-Za-z0-9/+=]{40}\b",
    ),
    (
        "github_pat",
        DataCategory.SECRETS,
        Severity.CRITICAL,
        r"\bgh[ps]_[A-Za-z0-9_]{36,}\b",
    ),
    (
        "private_key_header",
        DataCategory.SECRETS,
        Severity.CRITICAL,
        r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
    ),
    (
        "connection_string",
        DataCategory.SECRETS,
        Severity.CRITICAL,
        r"(?:mongodb|postgres|mysql|redis|amqp)(?:\+\w+)?://\S+:\S+@\S+",
    ),
    (
        "generic_api_key",
        DataCategory.SECRETS,
        Severity.HIGH,
        r"(?:api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9\-._]{20,}['\"]?",
    ),
    (
        "slack_webhook",
        DataCategory.SECRETS,
        Severity.HIGH,
        r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
    ),
]


class SensitiveDataClassifier:
    """Multi-category sensitive data scanner.

    Scans text for PII, PHI, PCI, credentials, and secrets using compiled
    regex patterns.  Returns a structured :class:`ClassificationResult`.
    """

    _SEVERITY_ORDER = {
        Severity.LOW: 0,
        Severity.MEDIUM: 1,
        Severity.HIGH: 2,
        Severity.CRITICAL: 3,
    }

    def __init__(
        self,
        *,
        categories: set[DataCategory] | None = None,
        min_severity: Severity = Severity.LOW,
    ) -> None:
        min_ord = self._SEVERITY_ORDER[min_severity]
        self._patterns: list[tuple[str, DataCategory, Severity, re.Pattern[str]]] = []
        for name, cat, sev, raw_pattern in _PATTERN_DEFS:
            if categories and cat not in categories:
                continue
            if self._SEVERITY_ORDER[sev] < min_ord:
                continue
            self._patterns.append((name, cat, sev, re.compile(raw_pattern, re.IGNORECASE)))

    def classify(self, text: str) -> ClassificationResult:
        """Scan *text* and return all sensitive data findings."""
        findings: list[SensitiveFinding] = []
        highest = Severity.LOW

        for name, cat, sev, pattern in self._patterns:
            matches = pattern.findall(text)
            if not matches:
                continue
            first_match = matches[0] if isinstance(matches[0], str) else str(matches[0])
            # Never store full match — only first 4 chars + mask
            sample = first_match[:4] + "***" if len(first_match) > 4 else "***"
            findings.append(
                SensitiveFinding(
                    category=cat,
                    data_type=name,
                    severity=sev,
                    count=len(matches),
                    sample=sample,
                )
            )
            if self._SEVERITY_ORDER[sev] > self._SEVERITY_ORDER[highest]:
                highest = sev

        return ClassificationResult(
            has_sensitive_data=len(findings) > 0,
            findings=findings,
            highest_severity=highest,
        )

    def redact(self, text: str) -> str:
        """Replace all sensitive data matches with category-tagged placeholders."""
        result = text
        for name, cat, _sev, pattern in self._patterns:
            result = pattern.sub(f"[{cat.value.upper()}_{name.upper()}_REDACTED]", result)
        return result
