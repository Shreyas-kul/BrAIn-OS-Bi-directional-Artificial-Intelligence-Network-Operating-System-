"""
BrAIn OS — Code Agent

💻 Specialist in code generation, debugging, code review, and software engineering.

Capabilities: file_read, file_write, code_exec, llm_call
Priority: P1 (High)

Routes queries about writing code, fixing bugs, and technical implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.agents.base import BaseAgent
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask

logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """
    Code sub-agent — writes, debugs, and reviews code.

    Tools available: file_read, file_write, code_exec, llm_call
    Restricted from: web_search, db_query
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="code", **kwargs)

    async def execute(
        self,
        task: AgentTask,
        process: AgentProcess,
    ) -> AgentResult:
        """
        Execute a coding task.

        Workflow:
            1. Recall relevant code context from memory
            2. Generate/debug/review code using LLM
            3. Store code artifacts in memory
            4. Return structured code output
        """
        try:
            # Step 1: Recall relevant context
            memories = await self.recall(process, task.query, top_k=5)
            context = ""
            if memories:
                context_items = [
                    f"- {m.get('value', '')}" for m in memories if m.get("value")
                ]
                context = "\n".join(context_items)

            # Step 2: Build messages with code-specific instructions
            messages = self.build_messages(process, task, context=context)
            response = await self.call_llm(messages, process)

            # Step 3: Store code output in memory
            await self.remember(
                process,
                key=f"code_{task.task_id[:8]}",
                value=response,
                persist=True,
            )

            return self.build_result(
                task,
                process,
                output=response,
                metadata={"memories_used": len(memories)},
            )

        except Exception as e:
            logger.exception("Code agent failed: %s", e)
            return self.build_error_result(task, process, str(e))
