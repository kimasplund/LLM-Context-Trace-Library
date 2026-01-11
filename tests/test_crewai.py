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
    with patch("lctl.integrations.crewai.CREWAI_AVAILABLE", True), \
         patch("lctl.integrations.crewai.Agent", MockAgent), \
         patch("lctl.integrations.crewai.Task", MockTask), \
         patch("lctl.integrations.crewai.Crew", MockCrew):
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
        verbose=True
    )
    
    assert agent.role == "Researcher"
    assert agent.goal == "Find information"
    assert agent.metadata["role"] == "Researcher"
    assert agent.metadata["tool_count"] == 1
    assert isinstance(agent.agent, MockAgent)

def test_lctl_task_init(mock_crewai_available):
    """Test LCTLTask initialization."""
    agent = LCTLAgent(role="Tester", goal="Test", backstory="Testing")
    task = LCTLTask(
        description="Run tests",
        expected_output="All pass",
        agent=agent
    )
    
    assert task.description == "Run tests"
    assert task.metadata["description"] == "Run tests"
    assert task.metadata["expected_output"] == "All pass"
    assert isinstance(task.task, MockTask)
    assert task.task.agent == agent.agent

def test_lctl_crew_kickoff(mock_crewai_available):
    """Test LCTLCrew execution."""
    agent = LCTLAgent(role="Worker", goal="Work", backstory="Working")
    task = LCTLTask(description="Do work", expected_output="Work done", agent=agent)
    
    crew = LCTLCrew(
        agents=[agent],
        tasks=[task],
        process="sequential",
        verbose=True
    )
    
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
    completion = events[-1]
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
    with patch.object(crew._crew, 'kickoff', side_effect=ValueError("Test Error")):
        with pytest.raises(ValueError):
            crew.kickoff()
            
    events = crew.session.chain.events
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].data["message"] == "Test Error"
