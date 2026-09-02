"""
BrAIn OS — Structured Audit Logger

Every action in BrAIn OS is audited to structured JSON logs:
    {agent, action, tool, input_hash, output_hash, timestamp, status}

Provides an immutable audit trail for security, debugging, compliance,
and forensic analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for audit integrity tracking."""
    if data is None:
        return "none"
    if isinstance(data, (dict, list)):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class AuditEvent:
    """A single structured audit log record."""

    event_id: str
    timestamp: str
    event_type: str  # e.g., "input_guard", "output_guard", "tool_call", "agent_exec"
    session_id: str = ""
    agent_name: str = ""
    agent_pid: str = ""
    action: str = ""
    tool: str | None = None
    input_hash: str = ""
    output_hash: str = ""
    status: str = "SUCCESS"  # SUCCESS | BLOCKED | FAILED
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Structured JSON audit logger.
    Writes audit records to disk and structlog/standard logging.
    """

    def __init__(self, log_dir: str = "./logs", log_file: str = "audit.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure the audit log directory exists."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create audit log dir %s: %s", self.log_dir, e)

    def log(self, event: AuditEvent) -> None:
        """Record an audit event."""
        json_line = event.to_json()
        logger.info("AUDIT: %s", json_line)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
        except Exception as e:
            logger.error("Failed to write audit log to file: %s", e)

    def log_input_check(
        self,
        session_id: str,
        raw_input: str,
        is_safe: bool,
        violations: list[str],
    ) -> AuditEvent:
        """Log input guard check."""
        event = AuditEvent(
            event_id=compute_hash(f"{session_id}:{raw_input}:{datetime.now(timezone.utc)}"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="input_guard",
            session_id=session_id,
            action="input_validation",
            input_hash=compute_hash(raw_input),
            status="SUCCESS" if is_safe else "BLOCKED",
            details={"violations": violations, "safe": is_safe},
        )
        self.log(event)
        return event

    def log_output_check(
        self,
        session_id: str,
        raw_output: str,
        is_safe: bool,
        pii_detected: list[str],
        violations: list[str],
    ) -> AuditEvent:
        """Log output guard check."""
        event = AuditEvent(
            event_id=compute_hash(f"{session_id}:{raw_output}:{datetime.now(timezone.utc)}"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="output_guard",
            session_id=session_id,
            action="output_validation",
            output_hash=compute_hash(raw_output),
            status="SUCCESS" if is_safe else "BLOCKED",
            details={
                "pii_detected": pii_detected,
                "violations": violations,
                "safe": is_safe,
            },
        )
        self.log(event)
        return event

    def log_tool_access(
        self,
        agent_name: str,
        agent_pid: str,
        tool: str,
        allowed: bool,
        reason: str = "",
    ) -> AuditEvent:
        """Log tool access check."""
        event = AuditEvent(
            event_id=compute_hash(f"{agent_pid}:{tool}:{datetime.now(timezone.utc)}"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="access_control",
            agent_name=agent_name,
            agent_pid=agent_pid,
            action="tool_permission_check",
            tool=tool,
            status="SUCCESS" if allowed else "BLOCKED",
            details={"reason": reason, "allowed": allowed},
        )
        self.log(event)
        return event
