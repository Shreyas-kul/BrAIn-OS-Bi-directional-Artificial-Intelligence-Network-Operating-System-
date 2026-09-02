"""
BrAIn OS — Agent Process Model

Each sub-agent is modeled as an OS process with:
    - Unique PID (UUID)
    - Status (READY → RUNNING → BLOCKED → TERMINATED)
    - Priority level (P0-Critical to P3-Low)
    - Capability whitelist (which tools it can use)
    - Isolated memory space
    - Checkpoint ID for resume-on-failure

The ProcessTable acts as /proc — a registry of all active agent processes
that the scheduler and orchestrator query to manage execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from brain_os.memory.working_context import WorkingContext


class ProcessStatus(str, Enum):
    """Agent process lifecycle states — mirrors OS process states."""

    READY = "READY"  # Waiting in queue, ready to execute
    RUNNING = "RUNNING"  # Currently executing
    BLOCKED = "BLOCKED"  # Waiting for I/O (LLM call, tool result)
    SUSPENDED = "SUSPENDED"  # Paused by scheduler (preempted)
    TERMINATED = "TERMINATED"  # Execution complete (success or failure)
    FAILED = "FAILED"  # Execution failed after retries


@dataclass
class AgentTask:
    """A unit of work submitted to the scheduler."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_task_id: str | None = None  # For sub-tasks in DAG execution


@dataclass
class AgentResult:
    """Result of an agent's execution."""

    task_id: str
    agent_pid: str
    agent_name: str
    status: ProcessStatus
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0  # Seconds
    tokens_used: int = 0
    checkpoint_id: str | None = None  # For resumable tasks


@dataclass
class AgentProcess:
    """
    An agent process — the fundamental unit of execution in BrAIn OS.

    Each agent is a process with its own isolated memory space,
    constrained capabilities, and lifecycle state.

    OS Analogy:
        - pid → Unix PID
        - status → RUNNING/BLOCKED/ZOMBIE
        - priority → nice value
        - capabilities → file descriptor access permissions
        - memory_space → virtual memory (mmap)
        - checkpoint → process snapshot for resume
    """

    pid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # e.g., "research", "code", "data"
    display_name: str = ""  # e.g., "Research Agent"
    status: ProcessStatus = ProcessStatus.READY
    priority: int = 2  # 0 (highest/critical) → 3 (lowest)
    capabilities: set[str] = field(default_factory=set)  # Allowed tools
    memory_space: WorkingContext = field(default_factory=WorkingContext)
    system_prompt: str = ""
    description: str = ""

    # Execution tracking
    current_task: AgentTask | None = None
    checkpoint_id: str | None = None  # LangGraph checkpoint ID for resume
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cpu_time: float = 0.0  # Total execution time in seconds
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Routing metadata (set by MoE router)
    gate_score: float = 0.0  # How confident the router is in this selection

    def __lt__(self, other: AgentProcess) -> bool:
        """Comparison for PriorityQueue ordering (lower priority = higher precedence)."""
        return self.priority < other.priority

    @property
    def is_active(self) -> bool:
        """Check if the process is in an active state."""
        return self.status in {ProcessStatus.READY, ProcessStatus.RUNNING, ProcessStatus.BLOCKED}

    @property
    def uptime(self) -> float:
        """Seconds since process creation."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if this agent has permission to use a tool (least-privilege enforcement)."""
        return tool_name in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Serialize process info for monitoring/logging."""
        return {
            "pid": self.pid,
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status.value,
            "priority": self.priority,
            "capabilities": sorted(self.capabilities),
            "cpu_time": self.cpu_time,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "gate_score": self.gate_score,
            "created_at": self.created_at.isoformat(),
            "uptime": self.uptime,
        }


class ProcessTable:
    """
    Agent process registry — the /proc of BrAIn OS.

    Tracks all active and recently terminated agent processes.
    Queried by the scheduler, orchestrator, and monitoring endpoints.
    """

    def __init__(self):
        self._processes: dict[str, AgentProcess] = {}

    def register(self, process: AgentProcess) -> None:
        """Register a new agent process in the table."""
        self._processes[process.pid] = process

    def unregister(self, pid: str) -> AgentProcess | None:
        """Remove a process from the table."""
        return self._processes.pop(pid, None)

    def get(self, pid: str) -> AgentProcess | None:
        """Get a process by PID."""
        return self._processes.get(pid)

    def get_by_name(self, name: str) -> list[AgentProcess]:
        """Get all processes with a given agent name."""
        return [p for p in self._processes.values() if p.name == name]

    def get_active(self) -> list[AgentProcess]:
        """Get all processes in active states."""
        return [p for p in self._processes.values() if p.is_active]

    def get_by_status(self, status: ProcessStatus) -> list[AgentProcess]:
        """Get all processes with a specific status."""
        return [p for p in self._processes.values() if p.status == status]

    def count(self) -> int:
        """Total registered processes."""
        return len(self._processes)

    def count_active(self) -> int:
        """Count of processes in active states."""
        return len(self.get_active())

    def all(self) -> list[AgentProcess]:
        """Get all registered processes."""
        return list(self._processes.values())

    def list_active(self) -> list[AgentProcess]:
        """Alias for get_active."""
        return self.get_active()

    def terminate(self, pid: str) -> None:
        """Mark a process as terminated."""
        proc = self.get(pid)
        if proc:
            proc.status = ProcessStatus.TERMINATED

    def to_dict(self) -> dict[str, Any]:
        """Serialize the process table for monitoring."""
        return {
            "total": self.count(),
            "active": self.count_active(),
            "by_status": {
                status.value: len(self.get_by_status(status))
                for status in ProcessStatus
            },
            "processes": [p.to_dict() for p in self._processes.values()],
        }


# Alias for memory space
MemorySpace = WorkingContext


@dataclass
class IPCMessage:
    """Inter-process communication message between agents via Redis message bus."""

    sender_pid: str
    recipient_pid: str
    content: Any
    message_type: str = "text"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
