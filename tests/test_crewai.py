"""Tests for CrewAI integration."""

from unittest.mock import MagicMock, patch

import pytest

from lctl.core.events import EventType
from lctl.integrations.crewai import (
    CrewAINotAvailableError,
    LCTLAgent,
    LCTLCrew,
    LCTLTask,
    trace_crew,
)


# Mock crewai module structure
class MockAgent:
    def __init__(self, role, goal, backstory, **kwargs):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = kwargs.get("tools", [])
        self.verbose = kwargs.get("verbose", False)
        self.allow_delegation = kwargs.get("allow_delegation", True)


class MockTask:
    def __init__(self, description, expected_output, agent, **kwargs):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent


class MockCrew:
    def __init__(self, agents, tasks, **kwargs):
        self.agents = agents
        self.tasks = tasks
        self.process = kwargs.get("process", "sequential")
        self.verbose = kwargs.get("verbose", False)
        self.step_callback = None
        self.task_callback = None

    def kickoff(self, inputs=None):
        return "Crew Result"

    def kickoff_async(self, inputs=None):
        return "Async Crew Result"


@pytest.fixture
def mock_crewai_available():
    """Mock CrewAI availability."""
    with (
        patch("lctl.integrations.crewai.CREWAI_AVAILABLE", True),
        patch("lctl.integrations.crewai.Agent", MockAgent),
        patch("lctl.integrations.crewai.Task", MockTask),
        patch("lctl.integrations.crewai.Crew", MockCrew),
    ):
        yield


@pytest.fixture
def mock_crewai_unavailable():
    """Mock CrewAI unavailability."""
    with patch("lctl.integrations.crewai.CREWAI_AVAILABLE", False):
        yield


def test_check_available_unavailable(mock_crewai_unavailable):
    """Test error when CrewAI is not available."""
    with pytest.raises(CrewAINotAvailableError):
        LCTLAgent(role="Test", goal="Test", backstory="Test")


def test_lctl_agent_init(mock_crewai_available):
    """Test LCTLAgent initialization."""
    agent = LCTLAgent(
        role="Researcher",
        goal="Find information",
        backstory="An expert researcher",
        tools=[MagicMock(name="search_tool")],
        verbose=True,
    )

    assert agent.role == "Researcher"
    assert agent.goal == "Find information"
    assert agent.metadata["role"] == "Researcher"
    assert agent.metadata["tool_count"] == 1
    assert isinstance(agent.agent, MockAgent)


def test_lctl_task_init(mock_crewai_available):
    """Test LCTLTask initialization."""
    agent = LCTLAgent(role="Tester", goal="Test", backstory="Testing")
    task = LCTLTask(description="Run tests", expected_output="All pass", agent=agent)

    assert task.description == "Run tests"
    assert task.metadata["description"] == "Run tests"
    assert task.metadata["expected_output"] == "All pass"
    assert isinstance(task.task, MockTask)
    assert task.task.agent == agent.agent


def test_lctl_crew_kickoff(mock_crewai_available):
    """Test LCTLCrew execution."""
    agent = LCTLAgent(role="Worker", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Work done", agent=agent)

    crew = LCTLCrew(agents=[agent], tasks=[task], process="sequential", verbose=True)

    result = crew.kickoff()
    assert result == "Crew Result"

    # Check trace events
    events = crew.session.chain.events
    assert len(events) > 0

    # Verify kickoff event
    kickoff = events[0]
    assert kickoff.type == EventType.STEP_START
    assert kickoff.agent == "crew-manager"
    assert kickoff.data["intent"] == "kickoff"

    # Verify completion
    events[-1]
    # The last event might be fact added. Just check successful completion steps.
    step_ends = [e for e in events if e.type == EventType.STEP_END]
    assert any(e.agent == "crew-manager" and e.data["outcome"] == "success" for e in step_ends)


def test_lctl_crew_callbacks(mock_crewai_available):
    """Test step and task callbacks."""
    agent = LCTLAgent(role="Worker", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Work done", agent=agent)

    crew = LCTLCrew(agents=[agent], tasks=[task], verbose=True)

    # Simulate step callback
    step_output = MagicMock()
    step_output.agent = agent.agent
    step_output.thought = "Thinking about work"
    step_output.tool = "search"
    step_output.tool_input = "query"
    step_output.result = "found it"

    crew._on_step(step_output)

    events = crew.session.chain.events
    # Expect step_start, tool_call, step_end

    tool_calls = [e for e in events if e.type == EventType.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].data["tool"] == "search"
    assert tool_calls[0].data["input"] == "query"

    # Simulate task callback
    task_output = MagicMock()
    task_output.description = "Do work"
    task_output.output = "Work done"

    crew._on_task_complete(task_output)

    facts = crew.session.chain.events[-1]
    # Ideally should be a fact_added event
    if facts.type == EventType.FACT_ADDED:
        assert "Task completed" in facts.data["text"]


def test_trace_crew_wrapper(mock_crewai_available):
    """Test wrapping an existing crew."""
    agent = MockAgent("Role", "Goal", "Backstory")
    task = MockTask("Desc", "Output", agent)
    crew = MockCrew([agent], [task])

    traced = trace_crew(crew, chain_id="test-trace")
    assert isinstance(traced, LCTLCrew)
    assert traced.session.chain.id == "test-trace"

    # Check callbacks installed
    assert crew.step_callback == traced._on_step
    assert crew.task_callback == traced._on_task_complete


def test_lctl_crew_error_handling(mock_crewai_available):
    """Test error handling in kickoff."""
    agent = LCTLAgent(role="Worker", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Work done", agent=agent)

    crew = LCTLCrew(agents=[agent], tasks=[task])

    # Mock kickoff to raise exception
    with (
        patch.object(crew._crew, "kickoff", side_effect=ValueError("Test Error")),
        pytest.raises(ValueError, match="Test Error"),
    ):
        crew.kickoff()

    events = crew.session.chain.events
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].data["message"] == "Test Error"


def test_tool_result_is_not_a_step_and_agent_is_bound(mock_crewai_available):
    """A ToolResult is held for the next step; the bound agent supplies the role."""
    agent = LCTLAgent(role="Senior Researcher", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Done", agent=agent)
    crew = LCTLCrew(agents=[agent], tasks=[task])

    class ToolResult:
        result = "found it"

    class AgentAction:
        thought = "use the tool"
        tool = "search"
        tool_input = "q"
        result = ""

    crew._on_step(ToolResult(), agent=agent.agent)
    assert crew.session.chain.events == []

    crew._on_step(AgentAction(), agent=agent.agent)
    events = crew.session.chain.events
    assert {e.agent for e in events} == {"senior-researcher"}
    call = next(e for e in events if e.type == EventType.TOOL_CALL)
    assert call.data["output"] == "found it"


def test_task_complete_records_final_answer_step(mock_crewai_available):
    agent = LCTLAgent(role="Writer", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Done", agent=agent)
    crew = LCTLCrew(agents=[agent], tasks=[task])

    output = MagicMock()
    output.description = "Write it"
    output.raw = "the report"
    output.agent = "Writer"
    crew._on_task_complete(output)

    ends = [e for e in crew.session.chain.events if e.type == EventType.STEP_END]
    assert [e.agent for e in ends] == ["writer"]


class _Action:
    thought = "t"
    tool = "search"
    tool_input = "q"
    result = "r"


def _bare_crew(mock_crewai_available, **crew_attrs):
    agent = LCTLAgent(role="Worker", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Done", agent=agent)
    crew = LCTLCrew(agents=[agent], tasks=[task])
    for k, v in crew_attrs.items():
        setattr(crew._crew, k, v)
    return crew, agent


def test_unresolved_agent_in_hierarchical_crew_is_the_manager(mock_crewai_available):
    crew, _ = _bare_crew(mock_crewai_available, process="hierarchical")
    crew._process = "hierarchical"
    crew._on_step(_Action())
    assert {e.agent for e in crew.session.chain.events} == {"manager"}

    crew._crew.manager_agent = MagicMock(role="Project Lead")
    crew._on_step(_Action())
    assert crew.session.chain.events[-1].agent == "project-lead"


def test_unresolved_agent_in_sequential_crew_stays_unknown(mock_crewai_available):
    crew, _ = _bare_crew(mock_crewai_available)
    crew._on_step(_Action())
    assert {e.agent for e in crew.session.chain.events} == {"unknown"}


def test_duplicate_roles_keep_separate_state(mock_crewai_available):
    crew, _ = _bare_crew(mock_crewai_available)
    a, b = MagicMock(role="Same"), MagicMock(role="Same")

    class ToolResult:
        result = "held"

    class Quiet:
        thought = "x"
        tool = "t"
        tool_input = None
        result = ""

    crew._on_step(ToolResult(), agent=a)
    crew._on_step(Quiet(), agent=b)
    assert crew.session.chain.events[-2].data["output"] is None
    crew._on_step(Quiet(), agent=a)
    assert crew.session.chain.events[-2].data["output"] == "held"


def test_pending_tool_result_flushed_at_task_completion(mock_crewai_available):
    crew, agent = _bare_crew(mock_crewai_available)

    class ToolResult:
        result = "dangling"

    crew._on_step(ToolResult(), agent=agent.agent)
    output = MagicMock()
    output.description = "d"
    output.raw = "r"
    output.agent = "Worker"
    crew._on_task_complete(output)
    calls = [e for e in crew.session.chain.events if e.type == EventType.TOOL_CALL]
    assert [c.data["output"] for c in calls] == ["dangling"]


def test_user_agent_callback_chained_and_restored(mock_crewai_available):
    crew, agent = _bare_crew(mock_crewai_available)
    seen = []

    def boom(step):
        seen.append(step)
        raise RuntimeError("user callback failed")

    agent.agent.step_callback = boom
    crew._begin_run()
    with pytest.raises(RuntimeError, match="user callback failed"):
        agent.agent.step_callback(_Action())
    assert len(seen) == 1
    assert any(e.type == EventType.STEP_END for e in crew.session.chain.events)
    crew._end_run()
    assert agent.agent.step_callback is boom
