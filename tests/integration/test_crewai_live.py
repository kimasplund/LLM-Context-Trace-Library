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
    pytest.mark.filterwarnings(
        "ignore:function_calling_llm is deprecated:DeprecationWarning:crewai.*"
    ),
    pytest.mark.filterwarnings("ignore:deprecated$:DeprecationWarning:crewai.*"),
]


class ScriptedLLM(BaseLLM):
    """Replays canned ReAct text, one reply per call, with a small delay."""

    def __init__(self, replies, delay=0.01):
        super().__init__(model="stub")
        self.replies = list(replies)
        self.delay = delay

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        time.sleep(self.delay)
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


_TOOL_THEN_FINAL = [
    'Thought: use the tool\nAction: Search Database\nAction Input: {"query": "orders"}',
    _FINAL.format("three results"),
]


def _agents(runs=1, researcher_delay=0.01, **researcher_kwargs):
    researcher = LCTLAgent(
        role="researcher",
        goal="find",
        backstory="b",
        tools=[search_database],
        llm=ScriptedLLM(_TOOL_THEN_FINAL * runs, delay=researcher_delay),
        **researcher_kwargs,
    )
    writer = LCTLAgent(
        role="writer",
        goal="write",
        backstory="b",
        llm=ScriptedLLM([_FINAL.format("a short report")] * runs),
    )
    return researcher, writer


def _tasks(researcher, writer):
    return [
        LCTLTask(description="Research {topic}", expected_output="notes", agent=researcher),
        LCTLTask(description="Write it up", expected_output="report", agent=writer),
    ]


def test_user_agent_step_callback_is_chained_and_restored():
    from crewai import Crew

    plain_calls = []
    r, w = _agents(step_callback=plain_calls.append)
    Crew(agents=[r.agent, w.agent], tasks=[t.task for t in _tasks(r, w)]).kickoff(
        inputs={"topic": "x"}
    )

    calls = []
    original = calls.append
    r, w = _agents(step_callback=original)
    crew = LCTLCrew(agents=[r, w], tasks=_tasks(r, w))
    crew.kickoff(inputs={"topic": "x"})

    assert plain_calls
    assert len(calls) == len(plain_calls)
    assert r.agent.step_callback is original
    assert w.agent.step_callback is None
    assert "researcher" in {e.agent for e in crew.session.chain.events}


def test_same_agents_traced_by_two_crews_in_sequence():
    r, w = _agents(runs=2)
    first = LCTLCrew(agents=[r, w], tasks=_tasks(r, w), chain_id="one")
    second = LCTLCrew(agents=[r, w], tasks=_tasks(r, w), chain_id="two")
    first.kickoff(inputs={"topic": "x"})
    second.kickoff(inputs={"topic": "x"})
    for crew in (first, second):
        agents = {e.agent for e in crew.session.chain.events}
        assert {"researcher", "writer"} <= agents
    assert r.agent.step_callback is None


def test_durations_start_at_the_agents_task_not_the_run():
    r, w = _agents(researcher_delay=0.15)
    crew = LCTLCrew(agents=[r, w], tasks=_tasks(r, w))
    crew.kickoff(inputs={"topic": "x"})
    writer_ends = [
        e for e in crew.session.chain.events if e.type == EventType.STEP_END and e.agent == "writer"
    ]
    assert writer_ends
    assert writer_ends[0].data["duration_ms"] < 100


def test_second_kickoff_excludes_idle_time():
    r, w = _agents(runs=2)
    crew = LCTLCrew(agents=[r, w], tasks=_tasks(r, w))
    crew.kickoff(inputs={"topic": "x"})
    time.sleep(0.5)
    before = len(crew.session.chain.events)
    crew.kickoff(inputs={"topic": "x"})
    first = next(
        e
        for e in crew.session.chain.events[before:]
        if e.type == EventType.STEP_END and e.agent == "researcher"
    )
    assert first.data["duration_ms"] < 400


def test_hierarchical_manager_steps_are_not_unknown():
    from crewai import Process

    manager = LCTLAgent(
        role="manager",
        goal="coordinate",
        backstory="b",
        allow_delegation=True,
        llm=ScriptedLLM(
            [
                "Thought: delegate\nAction: Delegate work to coworker\n"
                'Action Input: {"task": "Research orders", "context": "none", '
                '"coworker": "researcher"}',
                _FINAL.format("all done"),
            ]
        ),
    )
    worker = LCTLAgent(
        role="researcher",
        goal="find",
        backstory="b",
        allow_delegation=False,
        llm=ScriptedLLM([_FINAL.format("three results")]),
    )
    task = LCTLTask(description="Report on orders", expected_output="report")
    crew = LCTLCrew(
        agents=[worker], tasks=[task], process=Process.hierarchical, manager_agent=manager
    )
    crew.kickoff()
    agents = {e.agent for e in crew.session.chain.events}
    assert "unknown" not in agents
    assert "manager" in agents


def test_hierarchical_auto_manager_steps_are_not_unknown():
    from crewai import Process

    worker = LCTLAgent(
        role="researcher",
        goal="find",
        backstory="b",
        allow_delegation=False,
        llm=ScriptedLLM([_FINAL.format("three results")]),
    )
    manager_llm = ScriptedLLM(
        [
            "Thought: delegate\nAction: Delegate work to coworker\n"
            'Action Input: {"task": "Research orders", "context": "none", '
            '"coworker": "researcher"}',
            _FINAL.format("all done"),
        ]
    )
    task = LCTLTask(description="Report on orders", expected_output="report")
    crew = LCTLCrew(
        agents=[worker], tasks=[task], process=Process.hierarchical, manager_llm=manager_llm
    )
    crew.kickoff()
    events = crew.session.chain.events
    assert "unknown" not in {e.agent for e in events}


# Upstream: Crew.copy() warns about any method or partial callback; LCTL's callbacks need state,
# so they cannot be module-level functions. It only concerns CrewAI checkpointing.
@pytest.mark.filterwarnings("ignore:(method|partial) callbacks cannot be serialized:UserWarning")
def test_kickoff_for_each_is_attributed_and_restored():
    r, w = _agents(runs=2)
    original = r.agent.step_callback
    crew = LCTLCrew(agents=[r, w], tasks=_tasks(r, w))
    crew.kickoff_for_each([{"topic": "a"}, {"topic": "b"}])
    agents = {e.agent for e in crew.session.chain.events}
    assert "unknown" not in agents
    assert {"researcher", "writer"} <= agents
    assert r.agent.step_callback is original
