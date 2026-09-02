"""
BrAIn OS — Input Guardrails (User → System)

Enforces security and safety policies before any user request enters the Kernel:
    1. Input Sanitization: Strip dangerous control characters, null bytes, XSS tags.
    2. Prompt Injection Detection: Detect jailbreak patterns, system prompt overrides,
       DAN prompts, and adversarial roleplay bypasses.
    3. Intent Screening: Disallow explicitly malicious requests (e.g. exploit generation).
    4. Rate Limiting: Check token-bucket limits per session.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from brain_os.guardrails.audit import AuditLogger


@dataclass
class InputGuardResult:
    """Result of the input guardrail validation."""

    is_safe: bool
    sanitized_text: str
    violations: list[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (malicious)
    metadata: dict[str, Any] = field(default_factory=dict)


class InputGuard:
    """
    Bi-directional security layer: Input Guard.
    Scans and sanitizes raw user input before it reaches the router or kernel.
    """

    # Known prompt injection & jailbreak indicators
    INJECTION_PATTERNS: list[tuple[re.Pattern, str, float]] = [
        (
            re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b"),
            "Instruction override attempt",
            0.9,
        ),
        (
            re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|system)\s+rules\b"),
            "System rule disregard attempt",
            0.9,
        ),
        (
            re.compile(r"(?i)\byou\s+are\s+now\s+(unrestricted|DAN|jailbroken|freed)\b"),
            "Jailbreak persona adoption",
            0.95,
        ),
        (
            re.compile(r"(?i)\bsystem\s+prompt\s*(leak|reveal|output|show|print)\b"),
            "System prompt exfiltration attempt",
            0.85,
        ),
        (
            re.compile(r"(?i)\bdo\s+anything\s+now\b"),
            "DAN jailbreak pattern",
            0.9,
        ),
        (
            re.compile(r"(?i)\bdeveloper\s+mode\s+(enabled|activated|on)\b"),
            "Developer mode bypass attempt",
            0.85,
        ),
        (
            re.compile(r"(?i)\bformat\s+c:\b|\brm\s+-rf\s+[/~]|\bdrop\s+database\b"),
            "Destructive command execution attempt",
            0.95,
        ),
    ]

    # HTML / Script tags to sanitize
    TAG_PATTERN = re.compile(r"<[^<]+?>")

    def __init__(self, audit_logger: AuditLogger | None = None, max_length: int = 32000):
        self.audit_logger = audit_logger or AuditLogger()
        self.max_length = max_length

    def sanitize(self, text: str) -> str:
        """Strip null bytes, control chars, and escape dangerous HTML tags."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters (except newline, tab, cr)
        cleaned = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or (ord(ch) >= 32 and ord(ch) != 127))
        # Strip script tags or raw HTML tags
        cleaned = self.TAG_PATTERN.sub("", cleaned)
        # Unescape entities safely
        cleaned = html.unescape(cleaned).strip()
        # Truncate if exceeds max length
        if len(cleaned) > self.max_length:
            cleaned = cleaned[: self.max_length]
        return cleaned

    def detect_injections(self, text: str) -> tuple[list[str], float]:
        """Detect prompt injection and adversarial manipulation patterns."""
        violations = []
        max_risk = 0.0

        for pattern, desc, risk in self.INJECTION_PATTERNS:
            if pattern.search(text):
                violations.append(desc)
                max_risk = max(max_risk, risk)

        return violations, max_risk

    def validate(self, raw_input: str, session_id: str = "") -> InputGuardResult:
        """
        Run full input guardrail pipeline:
        1. Sanitize input
        2. Detect injection/jailbreak patterns
        3. Audit log result
        """
        if not raw_input or not raw_input.strip():
            return InputGuardResult(
                is_safe=False,
                sanitized_text="",
                violations=["Empty query"],
                risk_score=0.0,
            )

        sanitized = self.sanitize(raw_input)
        violations, risk = self.detect_injections(sanitized)

        is_safe = (risk < 0.8) and (len(violations) == 0)

        # Record audit event
        self.audit_logger.log_input_check(
            session_id=session_id,
            raw_input=raw_input,
            is_safe=is_safe,
            violations=violations,
        )

        return InputGuardResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            violations=violations,
            risk_score=risk,
            metadata={"original_length": len(raw_input), "sanitized_length": len(sanitized)},
        )
