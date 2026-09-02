"""
BrAIn OS — Conversation Agent

💬 Specialist in natural dialogue, summarization, and general Q&A.

Capabilities: llm_call
Priority: P3 (Low)

Routes queries that are conversational, require summarization,
or don't fit other specialists. The most restricted agent — can only
talk to the LLM, no tools access.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.agents.base import BaseAgent
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask

logger = logging.getLogger(__name__)


class ConversationAgent(BaseAgent):
    """
    Conversation sub-agent — engages in helpful dialogue.

    Tools available: llm_call (ONLY)
    Restricted from: web_search, file_read, file_write, code_exec, db_query

    This is the most restricted agent, following the principle of least privilege.
    It can only make LLM calls — no file system, no web, no database access.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="conversation", **kwargs)

    async def execute(
        self,
        task: AgentTask,
        process: AgentProcess,
    ) -> AgentResult:
        """
        Execute a conversational task.

        Workflow:
            1. Recall conversation history from memory
            2. Generate response using LLM
            3. Store conversation turn in memory
            4. Return response
        """
        try:
            # Step 1: Recall relevant conversation history
            memories = await self.recall(process, task.query, top_k=3)
            context = ""
            if memories:
                context_items = [
                    f"- {m.get('value', '')}" for m in memories if m.get("value")
                ]
                context = "\n".join(context_items)

            # Step 2: Build messages and call LLM
            messages = self.build_messages(process, task, context=context)
            response = await self.call_llm(messages, process)

            # Step 3: Store conversation turn
            await self.remember(
                process,
                key=f"conv_{task.task_id[:8]}",
                value=f"User: {task.query}\nAssistant: {response}",
                persist=False,  # Conversations stay in L1+L2, not persisted to L3
            )

            return self.build_result(
                task,
                process,
                output=response,
                metadata={"memories_used": len(memories)},
            )

        except Exception as e:
            logger.exception("Conversation agent failed: %s", e)
            return self.build_error_result(task, process, str(e))
