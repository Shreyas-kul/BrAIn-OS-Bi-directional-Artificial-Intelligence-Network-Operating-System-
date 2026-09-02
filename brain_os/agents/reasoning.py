"""
BrAIn OS — Reasoning Agent

🤖 Specialist in chain-of-thought reasoning, planning, verification, and
complex problem decomposition.

Capabilities: file_read, llm_call
Priority: P0 (Critical — used for planning and verification)

Routes queries requiring deep logical analysis, multi-step planning,
and task verification. This agent is the "brain" that plans DAG execution
and verifies other agents' outputs.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.agents.base import BaseAgent
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """
    Reasoning sub-agent — thinks deeply, plans, and verifies.

    Tools available: file_read, llm_call
    Restricted from: web_search, file_write, code_exec, db_query

    This is the highest-priority agent (P0) because it's used for:
        - Planning multi-step task DAGs
        - Verifying other agents' outputs
        - Complex logical reasoning
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="reasoning", **kwargs)

    async def execute(
        self,
        task: AgentTask,
        process: AgentProcess,
    ) -> AgentResult:
        """
        Execute a reasoning task.

        Workflow:
            1. Recall relevant context from memory
            2. Apply chain-of-thought reasoning via LLM
            3. Store reasoning chain in memory
            4. Return structured reasoning output
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

            # Step 2: Build CoT-enhanced messages
            cot_instruction = (
                "\n\nIMPORTANT: Think step-by-step. Show your reasoning process explicitly. "
                "Number each step. Identify assumptions and potential issues. "
                "If this is a verification task, be critical and thorough."
            )

            messages = self.build_messages(process, task, context=context)
            # Append CoT instruction to the user message
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += cot_instruction

            response = await self.call_llm(messages, process, temperature=0.3)

            # Step 3: Store reasoning chain
            await self.remember(
                process,
                key=f"reasoning_{task.task_id[:8]}",
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
            logger.exception("Reasoning agent failed: %s", e)
            return self.build_error_result(task, process, str(e))

    async def plan_task(
        self,
        task: AgentTask,
        process: AgentProcess,
        available_agents: list[str],
    ) -> list[dict[str, Any]]:
        """
        Plan a multi-step task by decomposing it into a DAG of steps.

        Used by the orchestrator to build execution plans for complex tasks.

        Args:
            task: The task to plan
            process: The reasoning agent's process
            available_agents: Names of available specialist agents

        Returns:
            List of plan steps, each with:
                {"name": str, "agent": str, "description": str, "depends_on": list[str]}
        """
        plan_prompt = (
            f"You are a task planner. Break down this task into steps.\n\n"
            f"Available specialist agents: {', '.join(available_agents)}\n\n"
            f"Task: {task.query}\n\n"
            f"Respond with a JSON array of steps. Each step has:\n"
            f'- "name": short step name\n'
            f'- "agent": which specialist agent should handle it\n'
            f'- "description": what this step should accomplish\n'
            f'- "depends_on": list of step names this depends on (empty for first steps)\n\n'
            f"Respond ONLY with the JSON array, no other text."
        )

        messages = [
            {"role": "system", "content": process.system_prompt},
            {"role": "user", "content": plan_prompt},
        ]

        response = await self.call_llm(messages, process, temperature=0.2)

        # Parse the plan (with fallback to single-step)
        import json

        try:
            # Try to extract JSON from the response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            plan = json.loads(response)
            if isinstance(plan, list):
                return plan
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse plan, falling back to single-step")

        # Fallback: single-step plan using the conversation agent
        return [
            {
                "name": "execute",
                "agent": "conversation",
                "description": task.query,
                "depends_on": [],
            }
        ]
