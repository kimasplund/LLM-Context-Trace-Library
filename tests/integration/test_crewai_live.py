"""Live CrewAI tests: a scripted stub LLM, no network or API key."""

import os
import time

import pytest

# Telemetry must be off before crewai is imported, so nothing tries the network.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-fake")

pytest.importorskip("crewai")

from crewai import BaseLLM
from crewai.tools import tool

from lctl.core.events import EventType
from lctl.integrations.crewai import LCTLAgent, LCTLCrew, LCTLTask

pytestmark = [
    pytest.mark.integration,
    # Upstream: building any crewai Agent trips two deprecated fields of crewai's own
    # (function_calling_llm, and allow_code_execution, whose warning text is just
    # "deprecated"); LCTL passes neither.
    pytest.mark.filterwarnings("ignore:function_calling_llm is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:deprecated$:DeprecationWarning"),
]


class ScriptedLLM(BaseLLM):
    """Replays canned ReAct text, one reply per call, with a small delay."""

    def __init__(self, replies):
        super().__init__(model="stub")
        self.replies = list(replies)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        time.sleep(0.01)
        return self.replies.pop(0)

    def supports_function_calling(self):
        return False


_FINAL = "Thought: I now know the final answer\nFinal Answer: {}"


@tool("Search Database")
def search_database(query: str) -> str:
    """Search the internal database for information."""
    return f"Found 3 results for: {query}"


def _crew():
    researcher_llm = ScriptedLLM(
        [
            'Thought: use the tool\nAction: Search Database\nAction Input: {"query": "orders"}',
            _FINAL.format("three results"),
        ]
    )
    writer_llm = ScriptedLLM([_FINAL.format("a short report")])
    researcher = LCTLAgent(
        role="researcher", goal="find", backstory="b", tools=[search_database], llm=researcher_llm
    )
    writer = LCTLAgent(role="writer", goal="write", backstory="b", llm=writer_llm)
    tasks = [
        LCTLTask(description="Research {topic}", expected_output="notes", agent=researcher),
        LCTLTask(description="Write it up", expected_output="report", agent=writer),
    ]
    crew = LCTLCrew(agents=[researcher, writer], tasks=tasks, chain_id="crew-live")
    crew.kickoff(inputs={"topic": "orders"})
    return crew


@pytest.fixture(scope="module")
def events():
    return _crew().session.chain.events


def _steps(events):
    return [e for e in events if e.agent != "crew-manager" and e.type == EventType.STEP_END]


def test_steps_attributed_to_agent_roles(events):
    agents = {e.agent for e in events if e.type in (EventType.STEP_START, EventType.STEP_END)}
    assert "unknown" not in agents
    assert {"researcher", "writer"} <= agents


def test_step_durations_are_measured(events):
    ends = _steps(events)
    assert ends
    assert any(e.data["duration_ms"] > 0 for e in ends)


def test_no_spurious_tool_result_step(events):
    for e in events:
        assert "ToolResult" not in (e.agent or "")
        assert "ToolResult" not in str(e.data.get("intent", ""))
    # A ToolResult has no thought, so recording it as a step yields the generic summary.
    starts = [e for e in events if e.type == EventType.STEP_START and e.agent != "crew-manager"]
    assert starts
    assert all(e.data["input_summary"] != "Agent step" for e in starts)


def test_tool_result_attached_as_tool_call(events):
    calls = [e for e in events if e.type == EventType.TOOL_CALL]
    assert [c.data["tool"] for c in calls] == ["Search Database"]
    assert calls[0].agent == "researcher"
    assert "Found 3 results" in str(calls[0].data["output"])
