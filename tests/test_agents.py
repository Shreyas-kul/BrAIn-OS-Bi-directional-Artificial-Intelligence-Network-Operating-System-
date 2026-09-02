"""
Unit tests for BrAIn OS Specialized Agents.
"""

import pytest

from brain_os.agents.code import CodeAgent
from brain_os.agents.conversation import ConversationAgent
from brain_os.agents.data import DataAgent
from brain_os.agents.reasoning import ReasoningAgent
from brain_os.agents.research import ResearchAgent
from brain_os.kernel.process import AgentProcess, AgentTask


def test_agent_initialization():
    research = ResearchAgent()
    code = CodeAgent()
    data = DataAgent()
    reasoning = ReasoningAgent()
    conversation = ConversationAgent()

    assert research.name == "research"
    assert code.name == "code"
    assert data.name == "data"
    assert reasoning.name == "reasoning"
    assert conversation.name == "conversation"


def test_agent_message_construction():
    code_agent = CodeAgent()
    task = AgentTask(query="Write a quicksort implementation in Python")
    process = AgentProcess(name="code")

    messages = code_agent.build_messages(process, task)
    assert len(messages) >= 2
    assert any("quicksort" in m["content"] for m in messages)
