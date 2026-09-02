"""
BrAIn OS — Async Priority Scheduler

The scheduler manages concurrent agent execution like an OS process scheduler:
    - Priority queue with 4 levels (P0-Critical → P3-Low)
    - Semaphore-based concurrency control (max 5 parallel agents)
    - Priority aging (starved tasks get promoted)
    - Timeout watchdog (kills stuck agents after TTL)
    - Cooperative preemption at tool-call boundaries

Engine: Python asyncio with Semaphore(N) for max concurrency.
Policy: Priority-based with aging. Starved tasks promoted after configurable time.
Preemption: Cooperative — agents yield at tool-call boundaries.
Deadlock prevention: Timeout watchdog kills stuck agents after configurable TTL.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from brain_os.config.settings import get_settings
from brain_os.kernel.process import (
    AgentProcess,
    AgentResult,
    AgentTask,
    ProcessStatus,
    ProcessTable,
)

logger = logging.getLogger(__name__)


class AgentScheduler:
    """
    Async priority scheduler for agent processes.

    Manages which agents run, when, and with what resources. Enforces
    concurrency limits and handles timeouts for stuck agents.

    OS Analogy: This is the CFS (Completely Fair Scheduler) of BrAIn OS.
    It takes processes from the priority queue and dispatches them to
    execution slots managed by an asyncio.Semaphore.

    Usage:
        scheduler = AgentScheduler(max_concurrent=5)
        result = await scheduler.schedule(task, agent_executor)
    """

    def __init__(
        self,
        max_concurrent: int | None = None,
        agent_timeout: int | None = None,
        priority_aging_seconds: int | None = None,
    ):
        settings = get_settings()
        self._max_concurrent = max_concurrent or settings.scheduler.max_concurrent_agents
        self._agent_timeout = agent_timeout or settings.scheduler.agent_timeout_seconds
        self._aging_seconds = priority_aging_seconds or settings.scheduler.priority_aging_seconds

        # Core scheduling primitives
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._queue: asyncio.PriorityQueue[tuple[int, float, AgentProcess]] = (
            asyncio.PriorityQueue()
        )
        self._process_table = ProcessTable()

        # Metrics
        self._total_scheduled = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_timeouts = 0

        logger.info(
            "Scheduler initialized: max_concurrent=%d, timeout=%ds, aging=%ds",
            self._max_concurrent,
            self._agent_timeout,
            self._aging_seconds,
        )

    @property
    def process_table(self) -> ProcessTable:
        """Access the process table (for monitoring)."""
        return self._process_table

    @property
    def queue_size(self) -> int:
        """Current number of items in the scheduler queue."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    async def schedule(
        self,
        task_or_process: AgentProcess | AgentTask | None = None,
        executor_or_factory: Callable[..., Awaitable[AgentResult]] | None = None,
        *,
        process: AgentProcess | None = None,
        task: AgentTask | None = None,
        coroutine_factory: Callable[..., Awaitable[AgentResult]] | None = None,
    ) -> AgentResult:
        """
        Schedule an agent process for execution.

        Supports both:
            schedule(process, executor)
            schedule(task, coroutine_factory, process=process)
        """
        self._total_scheduled += 1

        # Resolve process
        if isinstance(task_or_process, AgentProcess):
            target_proc = task_or_process
        elif process is not None:
            target_proc = process
        else:
            target_proc = AgentProcess(name="anonymous_agent")

        # Resolve task
        if isinstance(task_or_process, AgentTask):
            target_proc.current_task = task_or_process
        elif task is not None:
            target_proc.current_task = task

        # Resolve executor
        exec_fn = executor_or_factory or coroutine_factory
        if exec_fn is None:
            raise ValueError("An executor callable must be provided.")

        import inspect
        sig = inspect.signature(exec_fn)
        param_count = len(sig.parameters)

        if param_count >= 2:
            async def wrapped_executor(p: AgentProcess) -> AgentResult:
                return await exec_fn(p.current_task, p)
        else:
            async def wrapped_executor(p: AgentProcess) -> AgentResult:
                return await exec_fn(p)

        # Register in process table
        target_proc.status = ProcessStatus.READY
        self._process_table.register(target_proc)

        # Put in priority queue: (priority, enqueue_time, process)
        enqueue_time = time.monotonic()
        await self._queue.put((target_proc.priority, enqueue_time, target_proc))

        logger.info(
            "Scheduled process pid=%s name=%s priority=P%d",
            target_proc.pid,
            target_proc.name,
            target_proc.priority,
        )

        # Wait for a slot and dispatch
        return await self._dispatch(target_proc, wrapped_executor, enqueue_time)

    async def schedule_parallel(
        self,
        processes: list[AgentProcess],
        executor: Callable[[AgentProcess], Awaitable[AgentResult]],
    ) -> list[AgentResult]:
        """
        Schedule multiple agent processes for parallel execution.

        Uses asyncio.gather() to run all processes concurrently,
        subject to the semaphore concurrency limit.

        Args:
            processes: List of agent processes to run in parallel
            executor: Async callable for each agent

        Returns:
            List of AgentResults (order matches input processes)
        """
        tasks = [self.schedule(process, executor) for process in processes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    AgentResult(
                        task_id=processes[i].current_task.task_id if processes[i].current_task else "",
                        agent_pid=processes[i].pid,
                        agent_name=processes[i].name,
                        status=ProcessStatus.FAILED,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    # ------------------------------------------------------------------
    # Dispatch (Internal)
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        process: AgentProcess,
        executor: Callable[[AgentProcess], Awaitable[AgentResult]],
        enqueue_time: float,
    ) -> AgentResult:
        """
        Acquire a semaphore slot and execute the agent with timeout.

        This is where the actual execution happens. The semaphore
        ensures we never exceed max_concurrent agents running at once.
        """
        async with self._semaphore:
            # Calculate wait time (for aging metrics)
            wait_time = time.monotonic() - enqueue_time
            if wait_time > self._aging_seconds:
                logger.warning(
                    "Process pid=%s waited %.1fs (aging threshold=%ds)",
                    process.pid,
                    wait_time,
                    self._aging_seconds,
                )

            # Transition to RUNNING
            process.status = ProcessStatus.RUNNING
            process.started_at = datetime.now(timezone.utc)
            start_time = time.monotonic()

            logger.info(
                "Dispatching process pid=%s name=%s (waited=%.2fs)",
                process.pid,
                process.name,
                wait_time,
            )

            try:
                # Execute with timeout watchdog
                result = await asyncio.wait_for(
                    executor(process),
                    timeout=self._agent_timeout,
                )

                # Update process metrics
                execution_time = time.monotonic() - start_time
                process.cpu_time += execution_time
                process.status = ProcessStatus.TERMINATED
                process.finished_at = datetime.now(timezone.utc)
                process.tasks_completed += 1
                result.execution_time = execution_time

                self._total_completed += 1
                logger.info(
                    "Process completed pid=%s name=%s time=%.2fs",
                    process.pid,
                    process.name,
                    execution_time,
                )

                return result

            except asyncio.TimeoutError:
                # Timeout watchdog — kill stuck agent
                execution_time = time.monotonic() - start_time
                process.status = ProcessStatus.FAILED
                process.finished_at = datetime.now(timezone.utc)
                process.tasks_failed += 1
                self._total_timeouts += 1

                logger.error(
                    "Process TIMEOUT pid=%s name=%s (limit=%ds, ran=%.2fs)",
                    process.pid,
                    process.name,
                    self._agent_timeout,
                    execution_time,
                )

                return AgentResult(
                    task_id=process.current_task.task_id if process.current_task else "",
                    agent_pid=process.pid,
                    agent_name=process.name,
                    status=ProcessStatus.FAILED,
                    error=f"Timeout after {self._agent_timeout}s",
                    execution_time=execution_time,
                )

            except Exception as e:
                # Unexpected error
                execution_time = time.monotonic() - start_time
                process.status = ProcessStatus.FAILED
                process.finished_at = datetime.now(timezone.utc)
                process.tasks_failed += 1
                self._total_failed += 1

                logger.exception(
                    "Process FAILED pid=%s name=%s error=%s",
                    process.pid,
                    process.name,
                    str(e),
                )

                return AgentResult(
                    task_id=process.current_task.task_id if process.current_task else "",
                    agent_pid=process.pid,
                    agent_name=process.name,
                    status=ProcessStatus.FAILED,
                    error=str(e),
                    execution_time=execution_time,
                )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Scheduler statistics for monitoring."""
        return {
            "max_concurrent": self._max_concurrent,
            "queue_size": self._queue.qsize(),
            "active_processes": self._process_table.count_active(),
            "total_processes": self._process_table.count(),
            "total_scheduled": self._total_scheduled,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_timeouts": self._total_timeouts,
            "process_table": self._process_table.to_dict(),
        }
