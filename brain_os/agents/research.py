"""
BrAIn OS — Research Agent

🔍 Specialist in web search, document analysis, and information synthesis.

Capabilities: web_search, file_read, llm_call
Priority: P2 (Normal)

Routes queries about finding information, summarizing articles,
and answering factual questions.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.agents.base import BaseAgent
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Research sub-agent — finds, analyzes, and synthesizes information.

    Tools available: web_search, file_read, llm_call
    Restricted from: file_write, code_exec, db_query
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="research", **kwargs)

    async def execute(
        self,
        task: AgentTask,
        process: AgentProcess,
    ) -> AgentResult:
        """
        Execute a research task.

        Workflow:
            1. Recall relevant context from memory
            2. Synthesize information using LLM
            3. Store findings in memory for future recall
            4. Return structured research output
        """
        try:
            # Step 1: Recall relevant context from memory
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

            # Step 3: Store findings in memory
            await self.remember(
                process,
                key=f"research_{task.task_id[:8]}",
                value=response,
                persist=True,
            )

            # Step 4: Return result
            return self.build_result(
                task,
                process,
                output=response,
                metadata={"memories_used": len(memories)},
            )

        except Exception as e:
            logger.exception("Research agent failed: %s", e)
            return self.build_error_result(task, process, str(e))
