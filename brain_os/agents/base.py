"""
BrAIn OS — Base Agent (Abstract Process)

The abstract base class that all specialized agents inherit from.
Provides the common interface for execution, LLM calls (via litellm),
tool access control, memory integration, and OpenTelemetry tracing.

Every agent IS a process — it has isolated memory, constrained capabilities,
and executes within the scheduler's concurrency limits.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

import litellm

from brain_os.config.settings import get_settings
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask, ProcessStatus
from brain_os.memory.controller import MemoryController

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all BrAIn OS agents.

    Each agent implements:
        - execute(): The main task execution logic
        - _build_prompt(): Construct the LLM prompt from task + context

    Each agent gets for free:
        - LLM calls via litellm (supports Ollama, Groq, OpenAI, Gemini)
        - Memory access via MemoryController (L1 → L2 → L3)
        - Tool access control (capability whitelist enforcement)
        - Execution tracking and metrics

    Usage:
        class ResearchAgent(BaseAgent):
            async def execute(self, task: AgentTask, process: AgentProcess) -> AgentResult:
                # Your agent logic here
                response = await self.call_llm(prompt, process)
                return self.build_result(task, process, response)
    """

    def __init__(
        self,
        name: str,
        memory_controller: MemoryController | None = None,
    ):
        self.name = name
        self._memory = memory_controller
        self._settings = get_settings()

        # LLM configuration
        self._primary_model = self._settings.llm.primary_model
        self._fallback_model = self._settings.llm.fallback_model

    # ------------------------------------------------------------------
    # Abstract Interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def execute(
        self,
        task: AgentTask,
        process: AgentProcess,
    ) -> AgentResult:
        """
        Execute a task. Must be implemented by each specialized agent.

        Args:
            task: The task to execute
            process: The agent process (has capabilities, memory space, etc.)

        Returns:
            AgentResult with the execution output
        """
        ...

    # ------------------------------------------------------------------
    # LLM Calls (via litellm — unified interface)
    # ------------------------------------------------------------------

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        process: AgentProcess,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Make an LLM call via litellm with automatic fallback.

        Tries the primary model first, falls back to secondary on failure.
        All calls are traced via OpenTelemetry.

        Args:
            messages: Chat messages in OpenAI format [{"role": ..., "content": ...}]
            process: The agent process (for capability checking)
            temperature: LLM temperature
            max_tokens: Maximum response tokens

        Returns:
            The LLM response text
        """
        # Enforce capability check
        if not process.can_use_tool("llm_call"):
            raise PermissionError(
                f"Agent '{process.name}' does not have 'llm_call' capability"
            )

        start_time = time.monotonic()

        # Configure litellm for Ollama
        if self._primary_model.startswith("ollama/"):
            litellm.api_base = self._settings.llm.ollama_base_url

        try:
            # Try primary model
            response = await litellm.acompletion(
                model=self._primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = response.choices[0].message.content or ""
            elapsed = time.monotonic() - start_time

            logger.debug(
                "LLM call: agent=%s model=%s tokens=%d time=%.2fs",
                process.name,
                self._primary_model,
                response.usage.total_tokens if response.usage else 0,
                elapsed,
            )

            return result

        except Exception as primary_error:
            logger.warning(
                "Primary LLM failed (%s): %s. Trying fallback...",
                self._primary_model,
                str(primary_error),
            )

            if self._fallback_model is None:
                raise

            try:
                # Try fallback model
                response = await litellm.acompletion(
                    model=self._fallback_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                result = response.choices[0].message.content or ""
                elapsed = time.monotonic() - start_time

                logger.info(
                    "Fallback LLM succeeded: agent=%s model=%s time=%.2fs",
                    process.name,
                    self._fallback_model,
                    elapsed,
                )

                return result

            except Exception as fallback_error:
                logger.error(
                    "Both LLM backends failed. Primary: %s, Fallback: %s",
                    str(primary_error),
                    str(fallback_error),
                )
                raise RuntimeError(
                    f"All LLM backends failed. Primary ({self._primary_model}): {primary_error}"
                ) from fallback_error

    # ------------------------------------------------------------------
    # Memory Integration
    # ------------------------------------------------------------------

    async def remember(
        self,
        process: AgentProcess,
        key: str,
        value: Any,
        persist: bool = True,
    ) -> None:
        """Store data in the agent's memory hierarchy."""
        if self._memory:
            await self._memory.remember(process.pid, key, value, persist=persist)

    async def recall(
        self,
        process: AgentProcess,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall data from the agent's memory hierarchy."""
        if self._memory:
            return await self._memory.recall(process.pid, query, top_k=top_k)
        return []

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def build_messages(
        self,
        process: AgentProcess,
        task: AgentTask,
        context: str = "",
    ) -> list[dict[str, str]]:
        """Build standard chat messages from system prompt + task + context."""
        messages = [
            {"role": "system", "content": process.system_prompt},
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context from memory:\n{context}",
            })

        messages.append({
            "role": "user",
            "content": task.query,
        })

        return messages

    def build_result(
        self,
        task: AgentTask,
        process: AgentProcess,
        output: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Build a standard AgentResult from execution output."""
        return AgentResult(
            task_id=task.task_id,
            agent_pid=process.pid,
            agent_name=process.name,
            status=ProcessStatus.TERMINATED,
            output=output,
            metadata=metadata or {},
        )

    def build_error_result(
        self,
        task: AgentTask,
        process: AgentProcess,
        error: str,
    ) -> AgentResult:
        """Build an error AgentResult."""
        return AgentResult(
            task_id=task.task_id,
            agent_pid=process.pid,
            agent_name=process.name,
            status=ProcessStatus.FAILED,
            error=error,
        )
