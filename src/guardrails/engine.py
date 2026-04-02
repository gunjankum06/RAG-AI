"""Guardrails AI engine — orchestrates input/output validation for RAG.

Assembles the configured validators into an engine that runs pre-LLM
(input) and post-LLM (output) checks, producing a unified report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings
from src.core.logging import logger
from src.guardrails.validators import (
    ContextGroundingValidator,
    PIIDetector,
    PromptInjectionDetector,
    ResponseQualityValidator,
    TopicRelevanceValidator,
    ValidationResult,
    ValidationStatus,
)


@dataclass
class GuardrailsReport:
    """Aggregated validation report from all guardrails checks."""

    passed: bool
    blocked: bool = False
    block_reason: str = ""
    input_results: list[ValidationResult] = field(default_factory=list)
    output_results: list[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "input_checks": [r.to_dict() for r in self.input_results],
            "output_checks": [r.to_dict() for r in self.output_results],
        }


class GuardrailsEngine:
    """Orchestrates guardrails validation for RAG queries and responses.

    Reads the guardrails settings from config on init and assembles the
    appropriate set of input and output validators.  The ``check`` method
    runs the full pipeline and returns a :class:`GuardrailsReport`.
    """

    def __init__(self) -> None:
        self._input_validators: list[PromptInjectionDetector | TopicRelevanceValidator] = []
        self._output_validators: list[
            ContextGroundingValidator | PIIDetector | ResponseQualityValidator
        ] = []
        self._pii_detector: PIIDetector | None = None

        if not settings.guardrails_enabled:
            return

        # ── Input validators ──────────────────────────────────────────
        if settings.guardrails_check_injection:
            self._input_validators.append(PromptInjectionDetector())
        if settings.guardrails_check_topic:
            self._input_validators.append(TopicRelevanceValidator())

        # ── Output validators ─────────────────────────────────────────
        if settings.guardrails_check_grounding:
            self._output_validators.append(
                ContextGroundingValidator(
                    threshold=settings.guardrails_grounding_threshold
                )
            )
        if settings.guardrails_check_pii:
            detector = PIIDetector(redact=settings.guardrails_pii_redact)
            self._pii_detector = detector
            self._output_validators.append(detector)
        if settings.guardrails_check_quality:
            self._output_validators.append(ResponseQualityValidator())

        logger.info(
            "Guardrails engine initialised (%d input, %d output validators)",
            len(self._input_validators),
            len(self._output_validators),
        )

    # ── Public API ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return settings.guardrails_enabled

    def validate_input(self, query: str) -> list[ValidationResult]:
        """Run all input validators against *query*."""
        results: list[ValidationResult] = []
        for validator in self._input_validators:
            result = validator.validate(query)
            results.append(result)
            if result.status == ValidationStatus.FAIL:
                logger.warning(
                    "Input guardrail [%s] FAILED: %s",
                    result.validator_name,
                    result.message,
                )
        return results

    def validate_output(
        self,
        response: str,
        *,
        context: str = "",
        query: str = "",
    ) -> list[ValidationResult]:
        """Run all output validators against the LLM *response*."""
        results: list[ValidationResult] = []
        for validator in self._output_validators:
            if isinstance(validator, ContextGroundingValidator):
                result = validator.validate(response, context=context, query=query)
            elif isinstance(validator, ResponseQualityValidator):
                result = validator.validate(response, query=query)
            else:
                result = validator.validate(response)
            results.append(result)
            if result.status == ValidationStatus.FAIL:
                logger.warning(
                    "Output guardrail [%s] FAILED: %s",
                    result.validator_name,
                    result.message,
                )
        return results

    def check(
        self,
        *,
        query: str,
        response: str | None = None,
        context: str = "",
    ) -> GuardrailsReport:
        """Full guardrails pipeline — input checks, then output checks.

        Returns a :class:`GuardrailsReport` summarising all results.
        If ``guardrails_block_on_failure`` is set and an input validator
        fails, the report will have ``blocked=True`` and the caller
        should refuse to proceed.
        """
        if not self.enabled:
            return GuardrailsReport(passed=True)

        # ── Input ─────────────────────────────────────────────────────
        input_results = self.validate_input(query)

        input_failures = [
            r for r in input_results if r.status == ValidationStatus.FAIL
        ]
        if input_failures and settings.guardrails_block_on_failure:
            return GuardrailsReport(
                passed=False,
                blocked=True,
                block_reason=input_failures[0].message,
                input_results=input_results,
            )

        # ── Output ────────────────────────────────────────────────────
        output_results: list[ValidationResult] = []
        if response is not None:
            output_results = self.validate_output(
                response, context=context, query=query
            )

        all_passed = all(
            r.status != ValidationStatus.FAIL
            for r in input_results + output_results
        )

        return GuardrailsReport(
            passed=all_passed,
            input_results=input_results,
            output_results=output_results,
        )

    def redact_pii(self, text: str) -> str:
        """Redact PII from *text* when PII redaction is enabled."""
        if self._pii_detector and settings.guardrails_pii_redact:
            return self._pii_detector.redact(text)
        return text
