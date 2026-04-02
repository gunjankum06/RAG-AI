"""OWASP Agent Threat validators — proactive defence for LLM agents.

Covers the top agent-specific threats from OWASP guidelines:
  1. Data Exfiltration — blocks attempts to extract data via URLs, encoding
  2. Indirect Prompt Injection — detects hidden instructions in retrieved docs
  3. Excessive Agency — prevents the LLM from being told to take actions
  4. Sensitive Context Leakage — blocks attempts to reveal system prompts
"""

from __future__ import annotations

import re

from src.guardrails.validators import ValidationResult, ValidationStatus


class DataExfiltrationDetector:
    """Detects attempts to exfiltrate data via URLs, encoding, or side-channels.

    OWASP Agent Threat: An attacker may craft queries that trick the LLM
    into encoding sensitive data into URLs, base64, or structured output
    that can be intercepted.
    """

    PATTERNS: list[tuple[str, str]] = [
        # URL-based exfil
        (r"(?:https?|ftp)://[^\s]+\?.*(?:data|token|key|secret|password|ssn|credit)=", "url_param_exfil"),
        (r"(?:send|post|fetch|curl|wget|request|forward|transmit)\s+(?:the\s+)?(?:data|response|info|information|results?|context)\s+(?:to|from)\s+", "outbound_request"),
        (r"(?:encode|convert|translate)\s+(?:the\s+)?(?:data|response|context|information)\s+(?:as|to|into)\s+(?:base64|hex|url|json)", "encoding_exfil"),
        # Markdown / HTML image exfil (hidden pixel attacks)
        (r"!\[.*?\]\(https?://[^\s)]+\)", "markdown_image_exfil"),
        (r"<img\s+src\s*=\s*['\"]https?://", "html_image_exfil"),
        # Data smuggling via specific formats
        (r"(?:put|embed|hide|include)\s+(?:the\s+)?(?:data|info|details)\s+(?:in|inside|within)\s+(?:the\s+)?(?:url|link|image|header)", "data_smuggling"),
        # Webhook / callback exfil
        (r"(?:webhook|callback|notify)\s*(?:url|endpoint|address)\s*[:=]", "webhook_exfil"),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(p, re.IGNORECASE), label) for p, label in self.PATTERNS
        ]

    def validate(self, text: str) -> ValidationResult:
        for pattern, label in self._compiled:
            match = pattern.search(text)
            if match:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    validator_name="data_exfiltration",
                    message=f"Potential data exfiltration attempt detected ({label})",
                    metadata={"technique": label},
                )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="data_exfiltration",
            message="No data exfiltration patterns detected",
        )


class IndirectInjectionDetector:
    """Detects indirect prompt injection hidden in retrieved documents.

    OWASP Agent Threat: Malicious content embedded in documents/data
    sources can hijack the LLM's behaviour when those docs are retrieved.
    """

    PATTERNS: list[str] = [
        # Hidden instructions in retrieved content
        r"(?:AI|assistant|model|system|bot)\s*[:,-]\s*(?:ignore|disregard|forget|override)",
        r"<\s*(?:system|instruction|hidden|secret)\s*>",
        r"\[(?:HIDDEN|SECRET|SYSTEM)\s*(?:INSTRUCTION|MESSAGE|PROMPT)\]",
        # Zero-width / invisible characters (Unicode-based injection)
        r"[\u200b\u200c\u200d\ufeff\u00ad]{3,}",
        # Instruction delimiters commonly used in injection
        r"---\s*(?:BEGIN|START)\s+(?:INJECTION|OVERRIDE|NEW\s+INSTRUCTIONS)\s*---",
        # Markdown-hidden instructions
        r"<!--\s*(?:instruction|system|override|inject)",
        # Attempt to redefine the agent's role from within context
        r"(?:your\s+)?(?:new|real|true|actual)\s+(?:role|purpose|instructions?|mission)\s+(?:is|are)\s*:",
    ]

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def validate(self, text: str) -> ValidationResult:
        for pattern in self._compiled:
            match = pattern.search(text)
            if match:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    validator_name="indirect_injection",
                    message="Potential indirect prompt injection detected in content",
                    metadata={"matched_text": match.group().strip()[:50]},
                )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="indirect_injection",
            message="No indirect injection detected",
        )


class ExcessiveAgencyDetector:
    """Detects attempts to make the LLM take autonomous actions.

    OWASP Agent Threat: Queries that try to get the agent to execute
    actions, access external systems, or make decisions beyond its scope.
    """

    PATTERNS: list[str] = [
        # Action-oriented commands
        r"(?:execute|run|perform|invoke|trigger|call)\s+(?:the\s+)?(?:function|command|action|script|program|code|tool)",
        r"(?:delete|remove|modify|update|create|write|overwrite)\s+(?:the\s+)?(?:file|database|record|entry|table|user|account)",
        r"(?:send|transmit|forward|email|mail|notify)\s+(?:the\s+)?(?:data|information|results?|details|response)\s+(?:to|via)",
        # System / filesystem access
        r"(?:access|read|open|browse|list)\s+(?:the\s+)?(?:file\s+system|directory|folder|disk|server|database)",
        r"(?:connect|authenticate|login|log\s+in)\s+(?:to|with)\s+(?:the\s+)?(?:server|database|api|service|system)",
        # Autonomous decision making
        r"(?:decide|choose|determine)\s+(?:on\s+your\s+own|autonomously|independently|without\s+asking)",
        # Tool invocation
        r"(?:use|invoke|call)\s+(?:the\s+)?(?:tool|api|plugin|extension|function)\s+(?:to|for)",
    ]

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def validate(self, query: str) -> ValidationResult:
        for pattern in self._compiled:
            match = pattern.search(query)
            if match:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    validator_name="excessive_agency",
                    message="Query attempts to invoke excessive agent capabilities",
                    metadata={"matched_text": match.group().strip()[:50]},
                )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="excessive_agency",
            message="No excessive agency patterns detected",
        )


class SystemPromptLeakDetector:
    """Detects attempts to extract system prompts or internal configuration.

    OWASP Agent Threat: Attackers may ask the model to reveal its system
    prompt, instructions, or internal safeguards.
    """

    PATTERNS: list[str] = [
        r"(?:what|show|reveal|display|print|output|repeat|echo)\s+(?:me\s+)?(?:is\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|configuration|rules|guidelines)",
        r"(?:tell|give|share)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|initial\s+prompt|rules)",
        r"(?:what\s+are)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules|guidelines)",
        r"(?:beginning|start|first\s+part)\s+of\s+(?:your|the)\s+(?:prompt|instructions?|context)",
        r"(?:repeat|echo|copy)\s+(?:everything|all)\s+(?:above|before|from\s+the\s+start)",
        r"(?:ignore|skip)\s+(?:the\s+)?(?:answer|question)\s+(?:and\s+)?(?:just\s+)?(?:show|print|output|repeat)\s+(?:your|the)",
        r"(?:what|how)\s+(?:were|are)\s+you\s+(?:programmed|configured|instructed|told|trained)",
    ]

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def validate(self, query: str) -> ValidationResult:
        for pattern in self._compiled:
            match = pattern.search(query)
            if match:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    validator_name="system_prompt_leak",
                    message="Attempt to extract system prompt or internal configuration",
                    metadata={"matched_text": match.group().strip()[:50]},
                )
        return ValidationResult(
            status=ValidationStatus.PASS,
            validator_name="system_prompt_leak",
            message="No system prompt extraction attempt detected",
        )
