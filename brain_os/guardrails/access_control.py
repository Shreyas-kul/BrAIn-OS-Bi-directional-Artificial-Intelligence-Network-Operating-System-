"""
BrAIn OS — Access Control Layer (Sandboxing & Capability Whitelist)

Enforces OS-level principle of least privilege:
    - Each agent is a sandboxed process with an explicit whitelist of allowed tools.
    - Research Agent can web_search, but NOT file_write or code_exec.
    - Code Agent can file_write and code_exec, but NOT web_search.
    - Any unauthorized tool invocation is blocked and audited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from brain_os.guardrails.audit import AuditLogger

logger = logging.getLogger(__name__)


class PermissionDeniedError(Exception):
    """Raised when an agent attempts to invoke a disallowed tool."""

    def __init__(self, agent_name: str, tool_name: str, allowed_tools: set[str]):
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.allowed_tools = allowed_tools
        super().__init__(
            f"Access Denied: Agent '{agent_name}' is not authorized to invoke tool '{tool_name}'. "
            f"Allowed tools: {sorted(list(allowed_tools))}"
        )


@dataclass
class AccessDecision:
    """Result of a permission check."""

    allowed: bool
    agent_name: str
    tool_name: str
    reason: str


class AccessController:
    """
    RBAC and capability whitelist manager.
    Enforces the tool access matrix defined in agent_capabilities.yaml.
    """

    DEFAULT_CAPABILITIES: dict[str, set[str]] = {
        "research": {"web_search", "file_read", "llm_call"},
        "code": {"file_read", "file_write", "code_exec", "llm_call"},
        "data": {"file_read", "db_query", "llm_call"},
        "reasoning": {"file_read", "llm_call"},
        "conversation": {"llm_call"},
    }

    def __init__(
        self,
        capabilities_file: str | Path | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.audit_logger = audit_logger or AuditLogger()
        self.capabilities: dict[str, set[str]] = dict(self.DEFAULT_CAPABILITIES)

        if capabilities_file:
            self._load_from_yaml(Path(capabilities_file))
        else:
            default_path = (
                Path(__file__).resolve().parent.parent / "config" / "agent_capabilities.yaml"
            )
            if default_path.exists():
                self._load_from_yaml(default_path)

    def _load_from_yaml(self, path: Path) -> None:
        """Load agent capabilities from YAML config."""
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "agents" in data:
                    for agent_key, cfg in data["agents"].items():
                        caps = set(cfg.get("capabilities", []))
                        self.capabilities[agent_key.lower()] = caps
        except Exception as e:
            logger.warning("Could not load capabilities from %s: %s; using defaults", path, e)

    def is_tool_allowed(self, agent_name: str, tool_name: str) -> bool:
        """Check if an agent is authorized to call a specific tool."""
        agent_key = agent_name.lower().replace("_agent", "").replace(" ", "_")
        allowed = self.capabilities.get(agent_key, set())
        return tool_name in allowed

    def check_and_audit(
        self,
        agent_name: str,
        agent_pid: str,
        tool_name: str,
    ) -> AccessDecision:
        """Evaluate permission and record in audit log."""
        allowed = self.is_tool_allowed(agent_name, tool_name)
        agent_key = agent_name.lower().replace("_agent", "").replace(" ", "_")
        allowed_tools = self.capabilities.get(agent_key, set())

        reason = (
            f"Authorized tool '{tool_name}' for agent '{agent_name}'"
            if allowed
            else f"Unauthorized tool '{tool_name}' for agent '{agent_name}'. Allowed: {allowed_tools}"
        )

        self.audit_logger.log_tool_access(
            agent_name=agent_name,
            agent_pid=agent_pid,
            tool=tool_name,
            allowed=allowed,
            reason=reason,
        )

        return AccessDecision(
            allowed=allowed,
            agent_name=agent_name,
            tool_name=tool_name,
            reason=reason,
        )

    def enforce(self, agent_name: str, agent_pid: str, tool_name: str) -> None:
        """Enforce permission check, raising PermissionDeniedError if rejected."""
        decision = self.check_and_audit(agent_name, agent_pid, tool_name)
        if not decision.allowed:
            agent_key = agent_name.lower().replace("_agent", "").replace(" ", "_")
            raise PermissionDeniedError(
                agent_name=agent_name,
                tool_name=tool_name,
                allowed_tools=self.capabilities.get(agent_key, set()),
            )
