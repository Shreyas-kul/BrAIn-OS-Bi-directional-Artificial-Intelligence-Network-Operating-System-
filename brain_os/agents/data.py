"""
BrAIn OS — Data Agent

📊 Specialist in data analysis, visualization, SQL queries, and statistical reasoning.

Capabilities: file_read, db_query, llm_call
Priority: P1 (High)

Routes queries about analyzing datasets, creating charts, and database operations.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.agents.base import BaseAgent
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """
    Data sub-agent — analyzes data and produces insights.

    Tools available: file_read, db_query, llm_call
    Restricted from: web_search, file_write, code_exec
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="data", **kwargs)

    async def execute(
        self,
        task: AgentTask,
        process: AgentProcess,
    ) -> AgentResult:
        """
        Execute a data analysis task.

        Workflow:
            1. Recall relevant data context from memory
            2. Analyze data and generate insights using LLM
            3. Store analysis results in memory
            4. Return structured analysis output
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

            # Step 2: Build messages and call LLM
            messages = self.build_messages(process, task, context=context)
            response = await self.call_llm(messages, process)

            # Step 3: Store analysis in memory
            await self.remember(
                process,
                key=f"data_{task.task_id[:8]}",
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
            logger.exception("Data agent failed: %s", e)
            return self.build_error_result(task, process, str(e))
