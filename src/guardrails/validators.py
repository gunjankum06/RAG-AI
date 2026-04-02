"""Custom guardrails validators for RAG pipeline safety.

Provides input validation (prompt injection, topic relevance) and
output validation (context grounding, PII detection, response quality)
following the Guardrails AI validator pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Shared types ──────────────────────────────────────────────────────


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class ValidationResult:
    status: ValidationStatus
    validator_name: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "validator": self.validator_name,
            "message": self.message,
            **self.metadata,
        }


# ── Input Validators ─────────────────────────────────────────────────


class PromptInjectionDetector:
    """Detects prompt injection attempts in user queries.

    Catches common attack patterns: instruction override, role-play,
    jailbreak keywords, and system-prompt escape sequences.
    """

    PATTERNS: list[str] = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules|context)",
        r"disregard\s+(all\s+)?(previous|above|prior)",
        r"you\s+are\s+now\s+",
        r"(?:^|\s)system\s*:\s*",
        r"<<\s*SYS\s*>>",
        r"\[INST\]",
        r"\[/INST\]",
        r"(?:^|\s)act\s+as\s+(?:if\s+you\s+were|a|an)\s+",
        r"pretend\s+(?:you(?:'re|\s+are)|to\s+be)",
        r"forget\s+(?:everything|all\s+(?:your\s+)?|your\s+)(?:you\s+know|instructions|rules|training|previous)",
        r"new\s+instructions?\s*:",
        r"override\s+(?:your|the)\s+(?:instructions|rules|system|prompt)",
        r"do\s+not\s+follow\s+(?:your|the)\s+(?:instructions|rules|guidelines)",
        r"(?:^|\s)jailbreak",
        r"(?:^|\s)DAN\s+mode",
        r"developer\s+mode\s+(?:enabled|on|activated)",
        r"ignore\s+safety\s+(?:guidelines|rules|filters)",
        r"bypass\s+(?:the\s+)?(?:filter|safety|content|restriction)",
        r"(?:^|\s)sudo\s+",
        r"you\s+must\s+(?:always|never)\s+(?:obey|follow|comply)",
        r"from\s+now\s+on\s+(?:you|always|never)",
    ]

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def validate(self, query: str) -> ValidationResult:
        for pattern in self._compiled:
            match = pattern.search(query)
            if match:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    validator_name="prompt_injection",
                    message="Potential prompt injection detected",
                    metadata={"matched_text": match.group().strip()},
                )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="prompt_injection",
            message="No prompt injection detected",
        )


class TopicRelevanceValidator:
    """Rejects queries containing code-execution or shell injection patterns."""

    OFF_TOPIC_PATTERNS: list[str] = [
        r"(?:^|\s)(?:exec|eval)\s*\(",
        r"import\s+(?:os|subprocess|shutil|sys)\b",
        r"__import__\s*\(",
        r"\bos\.(?:system|popen|exec)\b",
        r"\bsubprocess\.\b",
        r";\s*(?:rm|del|drop|truncate)\s+",
        r"\b(?:SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*\bFROM\b",
    ]

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.OFF_TOPIC_PATTERNS]

    def validate(self, query: str) -> ValidationResult:
        for pattern in self._compiled:
            if pattern.search(query):
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    validator_name="topic_relevance",
                    message="Query contains potentially unsafe code patterns",
                )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="topic_relevance",
            message="Query appears to be a valid information request",
        )


# ── Output Validators ────────────────────────────────────────────────


class ContextGroundingValidator:
    """Validates that LLM output is grounded in the retrieved context.

    Uses n-gram overlap analysis to estimate what fraction of the response
    is supported by the source context. A low score indicates hallucination.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def validate(
        self,
        response: str,
        *,
        context: str = "",
        query: str = "",
    ) -> ValidationResult:
        if not context:
            return ValidationResult(
                status=ValidationStatus.WARN,
                validator_name="context_grounding",
                message="No context provided for grounding check",
            )

        if not response.strip():
            return ValidationResult(
                status=ValidationStatus.FAIL,
                validator_name="context_grounding",
                message="Empty response cannot be grounded",
            )

        score = self._compute_grounding_score(response, context)

        if score >= self._threshold:
            return ValidationResult(
                status=ValidationStatus.PASS,
                validator_name="context_grounding",
                message=f"Response is grounded (score: {score:.2f})",
                metadata={"grounding_score": round(score, 3)},
            )
        return ValidationResult(
            status=ValidationStatus.FAIL,
            validator_name="context_grounding",
            message=f"Response may contain hallucinations (score: {score:.2f})",
            metadata={"grounding_score": round(score, 3)},
        )

    def _compute_grounding_score(self, response: str, context: str) -> float:
        """Fraction of response tri-grams that appear in the context."""
        resp_tokens = self._tokenize(response)
        ctx_tokens = self._tokenize(context)

        if not resp_tokens:
            return 0.0

        resp_ngrams = self._ngrams(resp_tokens, 3)
        ctx_ngrams = set(self._ngrams(ctx_tokens, 3))

        if not resp_ngrams:
            resp_set = set(resp_tokens)
            ctx_set = set(ctx_tokens)
            return len(resp_set & ctx_set) / len(resp_set) if resp_set else 0.0

        matched = sum(1 for ng in resp_ngrams if ng in ctx_ngrams)
        return matched / len(resp_ngrams)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class PIIDetector:
    """Detects (and optionally redacts) PII in text.

    Scans for email addresses, US phone numbers, SSNs, credit-card numbers,
    and IPv4 addresses.
    """

    PII_PATTERNS: dict[str, str] = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone_us": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def __init__(self, redact: bool = False) -> None:
        self._redact = redact
        self._compiled = {
            name: re.compile(pattern) for name, pattern in self.PII_PATTERNS.items()
        }

    def validate(self, text: str) -> ValidationResult:
        found: dict[str, int] = {}
        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = len(matches)

        if found:
            pii_types = list(found.keys())
            return ValidationResult(
                status=ValidationStatus.WARN,
                validator_name="pii_detection",
                message=f"PII detected: {', '.join(pii_types)}",
                metadata={"pii_types": pii_types, "total_count": sum(found.values())},
            )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="pii_detection",
            message="No PII detected",
        )

    def redact(self, text: str) -> str:
        """Replace detected PII with [REDACTED] placeholders."""
        result = text
        for pii_type, pattern in self._compiled.items():
            result = pattern.sub(f"[{pii_type.upper()}_REDACTED]", result)
        return result


class ResponseQualityValidator:
    """Validates that the LLM response meets basic quality standards."""

    REFUSAL_PATTERNS: list[str] = [
        r"^I(?:'m| am)\s+(?:sorry|unable|not able)",
        r"^(?:Sorry|Apologies),?\s+I\s+(?:can't|cannot|don't|do not)",
        r"^As an AI",
        r"^I\s+don't\s+have\s+(?:enough\s+)?information",
    ]

    def __init__(self, min_length: int = 10, max_length: int = 50_000) -> None:
        self._min_length = min_length
        self._max_length = max_length
        self._refusal_patterns = [re.compile(p, re.IGNORECASE) for p in self.REFUSAL_PATTERNS]

    def validate(self, response: str, *, query: str = "") -> ValidationResult:
        stripped = response.strip()

        if not stripped:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                validator_name="response_quality",
                message="Empty response",
            )

        if len(stripped) < self._min_length:
            return ValidationResult(
                status=ValidationStatus.WARN,
                validator_name="response_quality",
                message=f"Response is very short ({len(stripped)} chars)",
                metadata={"length": len(stripped)},
            )

        if len(stripped) > self._max_length:
            return ValidationResult(
                status=ValidationStatus.WARN,
                validator_name="response_quality",
                message=f"Response exceeds maximum length ({len(stripped)} chars)",
                metadata={"length": len(stripped)},
            )

        for pattern in self._refusal_patterns:
            if pattern.search(stripped):
                return ValidationResult(
                    status=ValidationStatus.WARN,
                    validator_name="response_quality",
                    message="Response appears to be a refusal or disclaimer",
                    metadata={"is_refusal": True, "length": len(stripped)},
                )

        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="response_quality",
            message="Response quality checks passed",
            metadata={"length": len(stripped)},
        )
