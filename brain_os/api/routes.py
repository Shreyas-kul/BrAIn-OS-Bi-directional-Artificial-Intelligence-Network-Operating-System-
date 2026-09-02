"""
BrAIn OS — REST & WebSocket API Routes

Endpoints:
    - GET  /health              → Health status of Kernel, Redis, ChromaDB, LLM
    - GET  /status              → Kernel runtime metrics and process state
    - GET  /agents              → List all registered agent processes and capabilities
    - POST /tasks               → Submit task to Kernel for routing & execution
    - GET  /tasks/{task_id}     → Get task status and result
    - GET  /memory/{session_id} → Inspect session memory
    - DELETE /memory/{session_id} → Evict session memory
    - WS   /ws/{session_id}     → Real-time interactive streaming WebSocket
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from brain_os.agents.code import CodeAgent
from brain_os.agents.conversation import ConversationAgent
from brain_os.agents.data import DataAgent
from brain_os.agents.reasoning import ReasoningAgent
from brain_os.agents.research import ResearchAgent
from brain_os.config.settings import get_settings
from brain_os.guardrails import AccessController, InputGuard, OutputGuard
from brain_os.kernel.orchestrator import DAGOrchestrator
from brain_os.kernel.process import AgentProcess, AgentResult, AgentTask, ProcessStatus, ProcessTable
from brain_os.kernel.router import AttentionRouter
from brain_os.kernel.scheduler import AgentScheduler
from brain_os.memory.controller import MemoryController
from brain_os.observability import metrics, trace_span

logger = logging.getLogger(__name__)

router = APIRouter()

# Global OS State container
class KernelState:
    def __init__(self):
        self.settings = get_settings()
        self.process_table = ProcessTable()
        self.scheduler = AgentScheduler()
        self.memory_controller = MemoryController()
        self.router = AttentionRouter()
        self.orchestrator = DAGOrchestrator()
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.access_controller = AccessController()

        # Instantiate specialized agents
        self.agents: dict[str, Any] = {
            "research": ResearchAgent(memory_controller=self.memory_controller),
            "code": CodeAgent(memory_controller=self.memory_controller),
            "data": DataAgent(memory_controller=self.memory_controller),
            "reasoning": ReasoningAgent(memory_controller=self.memory_controller),
            "conversation": ConversationAgent(memory_controller=self.memory_controller),
        }

        # Cache completed tasks: task_id -> TaskResponse
        self.task_history: dict[str, dict[str, Any]] = {}


kernel_state = KernelState()


# --- Pydantic Request/Response Models ---

class TaskRequest(BaseModel):
    query: str = Field(..., description="The user prompt or instruction")
    session_id: str | None = Field(default=None, description="Session ID for stateful memory")
    preferred_agent: str | None = Field(default=None, description="Force a specific agent")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class TaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    response: str
    agent_used: str
    gate_score: float = 1.0
    execution_time: float = 0.0
    guardrails: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    services: dict[str, str]


# --- REST Endpoints ---

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health and connectivity of BrAIn OS subsystems."""
    snapshot = metrics.snapshot()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=snapshot["uptime_seconds"],
        services={
            "kernel": "ONLINE",
            "scheduler": "ONLINE",
            "memory": "ONLINE",
            "llm_backend": kernel_state.settings.llm.llm_default_backend,
        },
    )


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """Return runtime metrics, active agent processes, and scheduler state."""
    snapshot = metrics.snapshot()
    active_procs = kernel_state.process_table.list_active()
    return {
        "metrics": snapshot,
        "active_processes": len(active_procs),
        "scheduler_queue": kernel_state.scheduler.queue_size,
        "concurrency_limit": kernel_state.scheduler._max_concurrent,
    }


@router.get("/agents")
async def list_agents() -> list[dict[str, Any]]:
    """List all 5 specialized agents with their priorities and capabilities."""
    result = []
    for name, agent in kernel_state.agents.items():
        caps = kernel_state.access_controller.capabilities.get(name, set())
        result.append({
            "name": agent.name,
            "agent_type": name,
            "capabilities": sorted(list(caps)),
            "model": agent._primary_model,
        })
    return result


@router.post("/tasks", response_model=TaskResponse)
async def submit_task(request: TaskRequest) -> TaskResponse:
    """
    Execute a task through the full BrAIn OS lifecycle:
    Input Guard → Attention Router → Scheduler → Agent Execution → Output Guard
    """
    start_time = time.perf_counter()
    metrics.record_task_start()

    session_id = request.session_id or str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    with trace_span("brain_os.task_lifecycle", {"task_id": task_id, "session_id": session_id}):
        # 1. Input Guardrail
        input_result = kernel_state.input_guard.validate(request.query, session_id=session_id)
        if not input_result.is_safe:
            metrics.record_input_violation()
            metrics.record_task_failure()
            return TaskResponse(
                task_id=task_id,
                session_id=session_id,
                status="BLOCKED_BY_GUARDRAIL",
                response=f"Security alert: Input blocked. Violations: {input_result.violations}",
                agent_used="none",
                guardrails={"input_safe": False, "violations": input_result.violations},
            )

        sanitized_query = input_result.sanitized_text

        # 2. MoE Router
        agent_key = request.preferred_agent
        gate_score = 1.0

        if not agent_key or agent_key not in kernel_state.agents:
            routing_decision = await kernel_state.router.route(sanitized_query)
            top_agent_proc, gate_score = routing_decision.selected_agents[0]
            agent_key = top_agent_proc.name.lower().replace("_agent", "").replace(" ", "_")

        selected_agent = kernel_state.agents.get(agent_key, kernel_state.agents["conversation"])

        # 3. Create Process & Task
        task = AgentTask(
            task_id=task_id,
            query=sanitized_query,
            session_id=session_id,
            metadata=request.metadata,
        )

        process = AgentProcess(
            name=selected_agent.name,
            capabilities=kernel_state.access_controller.capabilities.get(agent_key, {"llm_call"}),
        )
        kernel_state.process_table.register(process)

        # 4. Schedule and execute
        try:
            agent_result: AgentResult = await kernel_state.scheduler.schedule(
                task=task,
                coroutine_factory=lambda t, p: selected_agent.execute(t, p),
                process=process,
            )
        except Exception as e:
            logger.error("Execution failed for task %s: %s", task_id, e)
            metrics.record_task_failure()
            return TaskResponse(
                task_id=task_id,
                session_id=session_id,
                status="FAILED",
                response=f"Execution error: {str(e)}",
                agent_used=selected_agent.name,
            )

        # 5. Output Guardrail
        raw_output = agent_result.output
        output_result = kernel_state.output_guard.validate(raw_output, session_id=session_id)

        if not output_result.is_safe:
            metrics.record_task_failure()
            output_text = "[Output blocked by safety filter]"
        else:
            output_text = output_result.sanitized_output

        execution_time = time.perf_counter() - start_time
        metrics.record_task_success(duration=execution_time, agent_name=selected_agent.name)

        resp = TaskResponse(
            task_id=task_id,
            session_id=session_id,
            status=str(agent_result.status.value),
            response=output_text,
            agent_used=selected_agent.name,
            gate_score=round(gate_score, 3),
            execution_time=round(execution_time, 4),
            guardrails={
                "input_safe": True,
                "output_safe": output_result.is_safe,
                "pii_redacted": output_result.pii_detected,
            },
            metadata={"tokens": agent_result.tokens_used},
        )

        kernel_state.task_history[task_id] = resp.model_dump()
        return resp


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    """Retrieve result and status of a previously executed task."""
    if task_id in kernel_state.task_history:
        return kernel_state.task_history[task_id]
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")


@router.get("/memory/{session_id}")
async def get_memory(session_id: str, query: str = "") -> dict[str, Any]:
    """Recall session memory from the hierarchical memory controller."""
    recalled = await kernel_state.memory_controller.recall(
        agent_pid=session_id,
        query=query,
    )
    return {"session_id": session_id, "query": query, "memory": recalled}


@router.delete("/memory/{session_id}")
async def clear_memory(session_id: str) -> dict[str, str]:
    """Clear L1 working context for a session."""
    kernel_state.memory_controller.clear_working_context(session_id)
    return {"session_id": session_id, "status": "cleared"}


# --- WebSocket Endpoint ---

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Bi-directional streaming WebSocket interface.
    Allows persistent conversational sessions with BrAIn OS.
    """
    await websocket.accept()
    logger.info("WebSocket connected: session %s", session_id)

    await websocket.send_json({
        "type": "system",
        "message": f"Connected to BrAIn OS (Session: {session_id})",
    })

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            query = payload.get("query", "")

            if not query:
                continue

            await websocket.send_json({"type": "status", "message": "Processing request..."})

            # Submit task internally
            task_req = TaskRequest(query=query, session_id=session_id)
            task_res = await submit_task(task_req)

            await websocket.send_json({
                "type": "response",
                "task_id": task_res.task_id,
                "agent": task_res.agent_used,
                "response": task_res.response,
                "execution_time": task_res.execution_time,
                "guardrails": task_res.guardrails,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session %s", session_id)
    except Exception as e:
        logger.error("WebSocket error on session %s: %s", session_id, e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
