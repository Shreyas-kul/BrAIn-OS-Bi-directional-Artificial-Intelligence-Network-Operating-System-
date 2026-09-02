"""
BrAIn OS — Performance & System Metrics

Tracks real-time runtime statistics for the OS:
    - Total tasks processed / failed
    - Average execution latency
    - Active agent processes
    - Guardrail blocks (input & output)
    - Memory hits / misses
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class KernelMetrics:
    """System-wide runtime telemetry metrics."""

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    input_guard_violations: int = 0
    output_guard_redactions: int = 0
    total_execution_time: float = 0.0
    agent_executions: dict[str, int] = field(default_factory=dict)
    memory_reads: int = 0
    memory_writes: int = 0

    def record_task_start(self) -> None:
        self.total_tasks += 1

    def record_task_success(self, duration: float, agent_name: str = "") -> None:
        self.successful_tasks += 1
        self.total_execution_time += duration
        if agent_name:
            self.agent_executions[agent_name] = (
                self.agent_executions.get(agent_name, 0) + 1
            )

    def record_task_failure(self) -> None:
        self.failed_tasks += 1

    def record_input_violation(self) -> None:
        self.input_guard_violations += 1

    def record_output_redaction(self) -> None:
        self.output_guard_redactions += 1

    def record_memory_read(self) -> None:
        self.memory_reads += 1

    def record_memory_write(self) -> None:
        self.memory_writes += 1

    @property
    def avg_latency(self) -> float:
        if self.successful_tasks == 0:
            return 0.0
        return round(self.total_execution_time / self.successful_tasks, 4)

    def snapshot(self) -> dict[str, Any]:
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return {
            "uptime_seconds": round(uptime, 2),
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "avg_latency_seconds": self.avg_latency,
            "input_guard_violations": self.input_guard_violations,
            "output_guard_redactions": self.output_guard_redactions,
            "agent_executions": dict(self.agent_executions),
            "memory_reads": self.memory_reads,
            "memory_writes": self.memory_writes,
        }


# Global metrics instance
metrics = KernelMetrics()
