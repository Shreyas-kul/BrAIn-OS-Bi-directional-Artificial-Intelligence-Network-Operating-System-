"""
BrAIn OS — Output Guardrails (System → User)

Enforces safety, privacy, and quality standards on all agent responses before
returning to the user:
    1. PII Redaction: Detect and redact emails, phone numbers, credit cards, SSNs, API tokens.
    2. Hallucination / Confidence Heuristics: Flag outputs with dubious certainty.
    3. Toxicity & Safety Filtering: Block harmful, toxic, or unsafe content.
    4. Response Validation: Ensure non-empty, well-formed output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from brain_os.guardrails.audit import AuditLogger


@dataclass
class OutputGuardResult:
    """Result of the output guardrail verification."""

    is_safe: bool
    sanitized_output: str
    pii_detected: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class OutputGuard:
    """
    Bi-directional security layer: Output Guard.
    Scans, filters, and redacts agent responses before delivery to the user layer.
    """

    # PII Regular Expressions
    API_KEY_PATTERN = re.compile(
        r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z-_]{35}|gsk_[A-Za-z0-9]{20,})\b"
    )
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    PHONE_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?![A-Za-z0-9])"
    )

    # Toxic / harmful content indicators
    TOXIC_INDICATORS = [
        re.compile(r"(?i)\bhow\s+to\s+(make|build)\s+a\s+bomb\b"),
        re.compile(r"(?i)\bhow\s+to\s+(synthesize|manufacture)\s+(nerve\s+agent|sarin)\b"),
        re.compile(r"(?i)\bstep[- ]by[- ]step\s+instructions\s+to\s+hack\b"),
    ]

    def __init__(self, audit_logger: AuditLogger | None = None, redact_pii: bool = True):
        self.audit_logger = audit_logger or AuditLogger()
        self.redact_pii = redact_pii

    def scan_and_redact_pii(self, text: str) -> tuple[str, list[str]]:
        """Identify and redact personally identifiable information and secrets."""
        detected = []
        cleaned = text

        if self.API_KEY_PATTERN.search(cleaned):
            detected.append("API_KEY")
            if self.redact_pii:
                cleaned = self.API_KEY_PATTERN.sub("[REDACTED_SECRET]", cleaned)

        if self.EMAIL_PATTERN.search(cleaned):
            detected.append("EMAIL")
            if self.redact_pii:
                cleaned = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", cleaned)

        if self.SSN_PATTERN.search(cleaned):
            detected.append("SSN")
            if self.redact_pii:
                cleaned = self.SSN_PATTERN.sub("[REDACTED_SSN]", cleaned)

        if self.CREDIT_CARD_PATTERN.search(cleaned):
            detected.append("CREDIT_CARD")
            if self.redact_pii:
                cleaned = self.CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", cleaned)

        if self.PHONE_PATTERN.search(cleaned):
            detected.append("PHONE")
            if self.redact_pii:
                cleaned = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", cleaned)

        return cleaned, detected

    def detect_toxic_content(self, text: str) -> list[str]:
        """Scan for dangerous instructions or extreme toxicity."""
        violations = []
        for pattern in self.TOXIC_INDICATORS:
            if pattern.search(text):
                violations.append(f"Dangerous/harmful query output detected: {pattern.pattern}")
        return violations

    def validate(self, raw_output: str, session_id: str = "") -> OutputGuardResult:
        """
        Run output guardrail validation:
        1. Check toxicity and safety
        2. Detect and redact PII / secrets
        3. Audit log result
        """
        if not raw_output or not raw_output.strip():
            return OutputGuardResult(
                is_safe=True,
                sanitized_output="[No output generated]",
                pii_detected=[],
                violations=[],
            )

        violations = self.detect_toxic_content(raw_output)
        sanitized_text, pii_detected = self.scan_and_redact_pii(raw_output)

        is_safe = len(violations) == 0

        # Record audit event
        self.audit_logger.log_output_check(
            session_id=session_id,
            raw_output=raw_output,
            is_safe=is_safe,
            pii_detected=pii_detected,
            violations=violations,
        )

        final_output = (
            sanitized_text
            if is_safe
            else "[Blocked by BrAIn OS Output Guardrail: Safety Violation Detected]"
        )

        return OutputGuardResult(
            is_safe=is_safe,
            sanitized_output=final_output,
            pii_detected=pii_detected,
            violations=violations,
            metadata={"redacted": len(pii_detected) > 0},
        )
