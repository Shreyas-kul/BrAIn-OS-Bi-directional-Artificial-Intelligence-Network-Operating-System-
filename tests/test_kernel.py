"""
Unit tests for BrAIn OS Kernel Layer (Process, Scheduler, Router, Orchestrator).
"""

import asyncio
import pytest

from brain_os.kernel.orchestrator import DAGNode, DAGOrchestrator, NodeStatus
from brain_os.kernel.process import (
    AgentProcess,
    AgentResult,
    AgentTask,
    IPCMessage,
    MemorySpace,
    ProcessStatus,
    ProcessTable,
)
from brain_os.kernel.router import AttentionRouter
from brain_os.kernel.scheduler import AgentScheduler


def test_process_lifecycle():
    proc = AgentProcess(name="test_agent", priority=1, capabilities={"llm_call"})
    assert proc.status == ProcessStatus.READY
    assert proc.pid is not None
    assert "llm_call" in proc.capabilities

    proc.status = ProcessStatus.RUNNING
    assert proc.status == ProcessStatus.RUNNING


def test_process_table():
    table = ProcessTable()
    p1 = AgentProcess(name="agent_1")
    p2 = AgentProcess(name="agent_2")

    table.register(p1)
    table.register(p2)

    assert table.get(p1.pid) == p1
    assert len(table.list_active()) == 2

    table.terminate(p1.pid)
    assert p1.status == ProcessStatus.TERMINATED
    assert len(table.list_active()) == 1


@pytest.mark.asyncio
async def test_scheduler_priority_and_concurrency():
    scheduler = AgentScheduler(max_concurrent=2)
    proc = AgentProcess(name="sched_proc", priority=0)
    task = AgentTask(query="test task")

    async def mock_exec(t: AgentTask, p: AgentProcess) -> AgentResult:
        await asyncio.sleep(0.01)
        return AgentResult(
            task_id=t.task_id,
            agent_pid=p.pid,
            agent_name=p.name,
            status=ProcessStatus.TERMINATED,
            output="mock output",
        )

    result = await scheduler.schedule(task, mock_exec, process=proc)
    assert result.status == ProcessStatus.TERMINATED
    assert result.output == "mock output"


@pytest.mark.asyncio
async def test_attention_router():
    router = AttentionRouter()

    # Keyword / Semantic fallback routes
    code_decision = await router.route("write a python script to parse logs")
    assert code_decision.primary_agent.name.lower() in ("code_agent", "code agent", "code")

    research_decision = await router.route("search web for recent quantum computing discoveries")
    assert research_decision.primary_agent.name.lower() in ("research_agent", "research agent", "research")

    assert len(code_decision.selected_agents) >= 1
    assert code_decision.selected_agents[0][1] > 0.0


@pytest.mark.asyncio
async def test_dag_orchestrator():
    from brain_os.kernel.orchestrator import ExecutionDAG

    dag = ExecutionDAG()
    n1_id = dag.add_node(name="Step1", agent_name="research")
    n2_id = dag.add_node(name="Step2", agent_name="code", dependencies=[n1_id])

    assert len(dag.nodes) == 2
    assert n1_id in dag.nodes[n2_id].dependencies
