"""Guardrails AI engine — orchestrates input/output validation for RAG.

Assembles the configured validators into an engine that runs pre-LLM
(input) and post-LLM (output) checks, producing a unified report.
Integrates DLP scanning and OWASP agent threat validators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings
from src.core.logging import logger
from src.guardrails.audit import (
    audit_query_blocked,
    audit_response_redacted,
    audit_sensitive_data_detected,
)
from src.guardrails.dlp import DLPAction, DLPEngine
from src.guardrails.owasp_validators import (
    DataExfiltrationDetector,
    ExcessiveAgencyDetector,
    IndirectInjectionDetector,
    SystemPromptLeakDetector,
)
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
    dlp_action: str = ""
    dlp_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "input_checks": [r.to_dict() for r in self.input_results],
            "output_checks": [r.to_dict() for r in self.output_results],
            "dlp_action": self.dlp_action,
            "dlp_detail": self.dlp_detail,
        }


class GuardrailsEngine:
    """Orchestrates guardrails validation for RAG queries and responses.

    Reads the guardrails settings from config on init and assembles the
    appropriate set of input and output validators.  The ``check`` method
    runs the full pipeline and returns a :class:`GuardrailsReport`.
    """

    def __init__(self) -> None:
        self._input_validators: list[Any] = []
        self._output_validators: list[Any] = []
        self._pii_detector: PIIDetector | None = None
        self._dlp: DLPEngine | None = None

        if not settings.guardrails_enabled:
            return

        # ── Input validators ──────────────────────────────────────────
        if settings.guardrails_check_injection:
            self._input_validators.append(PromptInjectionDetector())
        if settings.guardrails_check_topic:
            self._input_validators.append(TopicRelevanceValidator())

        # OWASP agent threat validators (input)
        if settings.owasp_check_exfiltration:
            self._input_validators.append(DataExfiltrationDetector())
        if settings.owasp_check_excessive_agency:
            self._input_validators.append(ExcessiveAgencyDetector())
        if settings.owasp_check_system_prompt_leak:
            self._input_validators.append(SystemPromptLeakDetector())

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

        # OWASP: indirect injection in retrieved context
        if settings.owasp_check_indirect_injection:
            self._output_validators.append(IndirectInjectionDetector())

        # ── DLP engine ────────────────────────────────────────────────
        if settings.dlp_enabled:
            self._dlp = DLPEngine()

        logger.info(
            "Guardrails engine initialised (%d input, %d output validators, DLP=%s)",
            len(self._input_validators),
            len(self._output_validators),
            "on" if self._dlp else "off",
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
            elif isinstance(validator, IndirectInjectionDetector):
                # Scan the context for indirect injections
                result = validator.validate(context) if context else ValidationResult(
                    status=ValidationStatus.PASS,
                    validator_name="indirect_injection",
                    message="No context to scan",
                )
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
        """Full guardrails pipeline — input checks, DLP, then output checks.

        Returns a :class:`GuardrailsReport` summarising all results.
        If ``guardrails_block_on_failure`` is set and an input validator
        fails, the report will have ``blocked=True`` and the caller
        should refuse to proceed.
        """
        if not self.enabled:
            return GuardrailsReport(passed=True)

        dlp_action = ""
        dlp_detail = ""

        # ── DLP: scan query for leaked credentials ────────────────────
        if self._dlp:
            query_dlp = self._dlp.scan_query(query)
            if query_dlp.action == DLPAction.BLOCK:
                audit_query_blocked(reason=query_dlp.reason, query_preview=query)
                return GuardrailsReport(
                    passed=False,
                    blocked=True,
                    block_reason="Query contains sensitive data (credentials/secrets)",
                    dlp_action=query_dlp.action.value,
                    dlp_detail=query_dlp.reason,
                )
            dlp_action = query_dlp.action.value
            dlp_detail = query_dlp.reason

        # ── Input validators ──────────────────────────────────────────
        input_results = self.validate_input(query)

        input_failures = [
            r for r in input_results if r.status == ValidationStatus.FAIL
        ]
        if input_failures and settings.guardrails_block_on_failure:
            audit_query_blocked(
                reason=input_failures[0].message,
                query_preview=query,
            )
            return GuardrailsReport(
                passed=False,
                blocked=True,
                block_reason=input_failures[0].message,
                input_results=input_results,
                dlp_action=dlp_action,
                dlp_detail=dlp_detail,
            )

        # ── Output checks + DLP response scan ─────────────────────────
        output_results: list[ValidationResult] = []
        if response is not None:
            output_results = self.validate_output(
                response, context=context, query=query
            )

            # DLP scan on the LLM response
            if self._dlp:
                resp_dlp = self._dlp.scan_response(response)
                dlp_action = resp_dlp.action.value
                dlp_detail = resp_dlp.reason

                if resp_dlp.classification and resp_dlp.classification.has_sensitive_data:
                    audit_sensitive_data_detected(
                        context="response",
                        action=resp_dlp.action.value,
                        categories=sorted(
                            c.value for c in resp_dlp.classification.categories_found
                        ),
                        finding_count=resp_dlp.classification.total_count,
                        highest_severity=resp_dlp.classification.highest_severity.value,
                    )

        all_passed = all(
            r.status != ValidationStatus.FAIL
            for r in input_results + output_results
        )

        return GuardrailsReport(
            passed=all_passed,
            input_results=input_results,
            output_results=output_results,
            dlp_action=dlp_action,
            dlp_detail=dlp_detail,
        )

    def redact_pii(self, text: str) -> str:
        """Redact PII from *text* when PII redaction is enabled."""
        if self._pii_detector and settings.guardrails_pii_redact:
            text = self._pii_detector.redact(text)
        # Also use DLP engine for comprehensive redaction
        if self._dlp:
            resp_dlp = self._dlp.scan_response(text)
            if resp_dlp.action == DLPAction.REDACT:
                text = resp_dlp.sanitized_text
                if resp_dlp.classification:
                    audit_response_redacted(
                        finding_count=resp_dlp.classification.total_count,
                        categories=sorted(
                            c.value for c in resp_dlp.classification.categories_found
                        ),
                    )
        return text
