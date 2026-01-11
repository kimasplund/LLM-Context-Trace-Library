"""Tests for OpenAI Agents SDK integration (lctl/integrations/openai_agents.py)."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lctl.core.events import Chain, Event, EventType, ReplayEngine
from lctl.core.session import LCTLSession

# Check if pytest-asyncio is available
try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

# Skip marker for async tests
skip_if_no_asyncio = pytest.mark.skipif(
    not PYTEST_ASYNCIO_AVAILABLE,
    reason="pytest-asyncio not installed"
)


class TestOpenAIAgentsAvailability:
    """Tests for OpenAI Agents SDK availability checking."""

    def test_is_available_returns_bool(self):
        """Test that is_available returns a boolean."""
        from lctl.integrations.openai_agents import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_openai_agents_available_constant_exists(self):
        """Test that OPENAI_AGENTS_AVAILABLE constant exists."""
        from lctl.integrations.openai_agents import OPENAI_AGENTS_AVAILABLE

        assert isinstance(OPENAI_AGENTS_AVAILABLE, bool)


class TestOpenAIAgentsNotAvailableError:
    """Tests for OpenAIAgentsNotAvailableError."""

    def test_error_is_import_error(self):
        """Test that error is an ImportError subclass."""
        from lctl.integrations.openai_agents import OpenAIAgentsNotAvailableError

        assert issubclass(OpenAIAgentsNotAvailableError, ImportError)

    def test_error_has_message(self):
        """Test that error has installation instructions."""
        from lctl.integrations.openai_agents import OpenAIAgentsNotAvailableError

        error = OpenAIAgentsNotAvailableError()
        assert "openai-agents" in str(error)
        assert "pip install" in str(error)


class TestLCTLOpenAIAgentTracerWithoutSDK:
    """Tests for LCTLOpenAIAgentTracer when SDK is not installed."""

    def test_tracer_requires_sdk(self):
        """Test that tracer raises error when SDK not available."""
        from lctl.integrations.openai_agents import (
            OPENAI_AGENTS_AVAILABLE,
            LCTLOpenAIAgentTracer,
            OpenAIAgentsNotAvailableError,
        )

        if not OPENAI_AGENTS_AVAILABLE:
            with pytest.raises(OpenAIAgentsNotAvailableError):
                LCTLOpenAIAgentTracer()


class MockAgent:
    """Mock Agent class for testing."""

    def __init__(
        self,
        name: str = "test_agent",
        instructions: str = "Test instructions",
        tools: list = None,
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []


class MockTool:
    """Mock Tool class for testing."""

    def __init__(self, name: str = "test_tool"):
        self.name = name


class MockRunContextWrapper:
    """Mock RunContextWrapper for testing."""

    def __init__(self, input_text: str = "test input"):
        self.input = input_text


class MockUsage:
    """Mock Usage class for testing."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockOutput:
    """Mock output for testing."""

    def __init__(
        self,
        final_output: str = "test output",
        usage: MockUsage = None,
    ):
        self.final_output = final_output
        self.usage = usage or MockUsage()


@pytest.fixture
def mock_openai_agents_available():
    """Fixture to mock OpenAI Agents SDK as available."""
    with patch.dict(
        "lctl.integrations.openai_agents.__dict__",
        {"OPENAI_AGENTS_AVAILABLE": True},
    ):
        yield


class TestLCTLOpenAIAgentTracerBasics:
    """Tests for basic LCTLOpenAIAgentTracer functionality."""

    @pytest.fixture
    def tracer(self, mock_openai_agents_available):
        """Create a tracer for testing."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            return LCTLOpenAIAgentTracer(chain_id="test-chain")

    def test_tracer_creation(self, tracer):
        """Test basic tracer creation."""
        assert tracer.session is not None
        assert tracer.chain.id == "test-chain"

    def test_tracer_creation_with_auto_id(self, mock_openai_agents_available):
        """Test tracer creation with auto-generated ID."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            tracer = LCTLOpenAIAgentTracer()
            assert tracer.chain.id.startswith("openai-agent-")

    def test_tracer_creation_with_existing_session(self, mock_openai_agents_available):
        """Test tracer creation with existing session."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            session = LCTLSession(chain_id="existing-session")
            tracer = LCTLOpenAIAgentTracer(session=session)
            assert tracer.session is session
            assert tracer.chain.id == "existing-session"

    def test_tracer_verbose_mode(self, mock_openai_agents_available):
        """Test tracer with verbose mode enabled."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            tracer = LCTLOpenAIAgentTracer(verbose=True)
            assert tracer._verbose is True

    def test_run_config_returns_dict(self, tracer):
        """Test that run_config returns a dictionary with hooks."""
        config = tracer.run_config
        assert isinstance(config, dict)
        assert "hooks" in config

    def test_create_hooks_returns_hooks(self, tracer):
        """Test that create_hooks returns hooks object."""
        hooks = tracer.create_hooks()
        assert hooks is not None
        assert hooks is tracer.create_hooks()

    def test_to_dict(self, tracer):
        """Test to_dict returns chain dictionary."""
        result = tracer.to_dict()
        assert "chain" in result
        assert result["chain"]["id"] == "test-chain"


class TestLCTLOpenAIAgentTracerExport:
    """Tests for tracer export functionality."""

    @pytest.fixture
    def tracer(self, mock_openai_agents_available):
        """Create a tracer for testing."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            return LCTLOpenAIAgentTracer(chain_id="export-test")

    def test_export_json(self, tracer, tmp_path: Path):
        """Test exporting tracer to JSON file."""
        export_path = tmp_path / "trace.lctl.json"
        tracer.export(str(export_path))

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "export-test"

    def test_export_yaml(self, tracer, tmp_path: Path):
        """Test exporting tracer to YAML file."""
        import yaml

        export_path = tmp_path / "trace.lctl.yaml"
        tracer.export(str(export_path))

        assert export_path.exists()
        data = yaml.safe_load(export_path.read_text())
        assert data["chain"]["id"] == "export-test"


class TestLCTLOpenAIAgentTracerManualRecording:
    """Tests for manual event recording methods."""

    @pytest.fixture
    def tracer(self, mock_openai_agents_available):
        """Create a tracer for testing."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            return LCTLOpenAIAgentTracer(chain_id="manual-test")

    def test_record_tool_call(self, tracer):
        """Test manual tool call recording."""
        tracer.record_tool_call(
            tool="search",
            input_data={"query": "test"},
            output_data={"results": []},
            duration_ms=100,
        )

        events = tracer.chain.events
        assert len(events) == 1
        assert events[0].type == EventType.TOOL_CALL
        assert events[0].data["tool"] == "search"
        assert events[0].data["duration_ms"] == 100

    def test_record_handoff(self, tracer):
        """Test manual handoff recording."""
        tracer.record_handoff("agent1", "agent2")

        events = tracer.chain.events
        assert len(events) == 1
        assert events[0].type == EventType.FACT_ADDED
        assert "Handoff from agent1 to agent2" in events[0].data["text"]

    def test_record_error(self, tracer):
        """Test manual error recording."""
        error = ValueError("Test error")
        tracer.record_error("test_agent", error, recoverable=True)

        events = tracer.chain.events
        assert len(events) == 1
        assert events[0].type == EventType.ERROR
        assert events[0].data["type"] == "ValueError"
        assert events[0].data["recoverable"] is True


class TestAgentRunContext:
    """Tests for AgentRunContext context manager."""

    @pytest.fixture
    def tracer(self, mock_openai_agents_available):
        """Create a tracer for testing."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            return LCTLOpenAIAgentTracer(chain_id="context-test")

    def test_context_manager_records_start_end(self, tracer):
        """Test that context manager records start and end events."""
        with tracer.trace_agent_run("test_agent", "test input"):
            pass

        events = tracer.chain.events
        assert len(events) == 2
        assert events[0].type == EventType.STEP_START
        assert events[1].type == EventType.STEP_END

    def test_context_manager_records_agent_name(self, tracer):
        """Test that context manager records agent name."""
        with tracer.trace_agent_run("my_agent", "input"):
            pass

        events = tracer.chain.events
        assert events[0].agent == "my_agent"
        assert events[1].agent == "my_agent"

    def test_context_manager_records_input_summary(self, tracer):
        """Test that context manager records input summary."""
        with tracer.trace_agent_run("agent", "my input summary"):
            pass

        event = tracer.chain.events[0]
        assert event.data["input_summary"] == "my input summary"

    def test_context_manager_records_success(self, tracer):
        """Test that context manager records success outcome."""
        with tracer.trace_agent_run("agent", "input"):
            pass

        event = tracer.chain.events[1]
        assert event.data["outcome"] == "success"

    def test_context_manager_records_duration(self, tracer):
        """Test that context manager records duration."""
        with tracer.trace_agent_run("agent", "input"):
            time.sleep(0.05)

        event = tracer.chain.events[1]
        assert event.data["duration_ms"] >= 40

    def test_context_manager_records_error(self, tracer):
        """Test that context manager records error on exception."""
        with pytest.raises(RuntimeError):
            with tracer.trace_agent_run("failing_agent", "input"):
                raise RuntimeError("Test failure")

        events = tracer.chain.events
        assert len(events) == 3
        assert events[1].data["outcome"] == "error"
        assert events[2].type == EventType.ERROR
        assert events[2].data["type"] == "RuntimeError"

    def test_context_manager_propagates_exception(self, tracer):
        """Test that context manager re-raises exceptions."""
        with pytest.raises(ValueError, match="test error"):
            with tracer.trace_agent_run("agent", "input"):
                raise ValueError("test error")

    def test_set_output(self, tracer):
        """Test setting output summary."""
        with tracer.trace_agent_run("agent", "input") as ctx:
            ctx.set_output("My output summary")

        event = tracer.chain.events[1]
        assert event.data["output_summary"] == "My output summary"

    def test_set_usage(self, tracer):
        """Test setting token usage."""
        with tracer.trace_agent_run("agent", "input") as ctx:
            ctx.set_usage(tokens_in=100, tokens_out=50)

        event = tracer.chain.events[1]
        assert event.data["tokens"]["input"] == 100
        assert event.data["tokens"]["output"] == 50

    def test_record_tool_call_in_context(self, tracer):
        """Test recording tool call within context."""
        with tracer.trace_agent_run("agent", "input") as ctx:
            ctx.record_tool_call("search", "query", "results", 100)

        events = tracer.chain.events
        assert len(events) == 3
        assert events[1].type == EventType.TOOL_CALL


class TestTracedAgent:
    """Tests for TracedAgent wrapper."""

    @pytest.fixture
    def traced_agent(self, mock_openai_agents_available):
        """Create a traced agent for testing."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import TracedAgent

            mock_agent = MockAgent(name="test_agent", instructions="Test instructions")
            return TracedAgent(mock_agent, chain_id="traced-agent-test")

    def test_traced_agent_creation(self, traced_agent):
        """Test TracedAgent creation."""
        assert traced_agent.agent is not None
        assert traced_agent.tracer is not None

    def test_traced_agent_records_config(self, traced_agent):
        """Test that TracedAgent records agent configuration."""
        events = traced_agent.tracer.chain.events
        assert len(events) == 1
        assert events[0].type == EventType.FACT_ADDED
        assert "test_agent" in events[0].data["text"]

    def test_traced_agent_provides_run_config(self, traced_agent):
        """Test that TracedAgent provides run_config."""
        config = traced_agent.run_config
        assert isinstance(config, dict)
        assert "hooks" in config

    def test_traced_agent_export(self, traced_agent, tmp_path: Path):
        """Test TracedAgent export."""
        export_path = tmp_path / "traced_agent.lctl.json"
        traced_agent.export(str(export_path))
        assert export_path.exists()

    def test_traced_agent_to_dict(self, traced_agent):
        """Test TracedAgent to_dict."""
        result = traced_agent.to_dict()
        assert "chain" in result
        assert result["chain"]["id"] == "traced-agent-test"

    def test_traced_agent_proxies_attributes(self, traced_agent):
        """Test that TracedAgent proxies attribute access."""
        assert traced_agent.name == "test_agent"
        assert traced_agent.instructions == "Test instructions"


class TestTraceAgentFunction:
    """Tests for trace_agent convenience function."""

    def test_trace_agent_returns_traced_agent(self, mock_openai_agents_available):
        """Test that trace_agent returns TracedAgent."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import TracedAgent, trace_agent

            mock_agent = MockAgent()
            traced = trace_agent(mock_agent, chain_id="func-test")

            assert isinstance(traced, TracedAgent)
            assert traced.tracer.chain.id == "func-test"

    def test_trace_agent_with_session(self, mock_openai_agents_available):
        """Test trace_agent with existing session."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import trace_agent

            session = LCTLSession(chain_id="existing")
            mock_agent = MockAgent()
            traced = trace_agent(mock_agent, session=session)

            assert traced.tracer.session is session


class TestLCTLRunHooks:
    """Tests for LCTLRunHooks implementation."""

    @pytest.fixture
    def hooks(self, mock_openai_agents_available):
        """Create hooks for testing."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLRunHooks

            session = LCTLSession(chain_id="hooks-test")
            return LCTLRunHooks(session, verbose=False)

    @skip_if_no_asyncio
    @pytest.mark.asyncio
    async def test_on_agent_start(self, hooks):
        """Test on_agent_start hook."""
        context = MockRunContextWrapper("test input")
        agent = MockAgent(name="test_agent")

        await hooks.on_agent_start(context, agent)

        events = hooks._session.chain.events
        assert len(events) >= 1
        assert events[0].type == EventType.STEP_START
        assert events[0].agent == "test_agent"

    @skip_if_no_asyncio
    @pytest.mark.asyncio
    async def test_on_agent_end(self, hooks):
        """Test on_agent_end hook."""
        context = MockRunContextWrapper()
        agent = MockAgent(name="test_agent")
        output = MockOutput(final_output="test result")

        await hooks.on_agent_start(context, agent)
        await hooks.on_agent_end(context, agent, output)

        events = hooks._session.chain.events
        step_end_events = [e for e in events if e.type == EventType.STEP_END]
        assert len(step_end_events) == 1
        assert step_end_events[0].data["outcome"] == "success"

    @skip_if_no_asyncio
    @pytest.mark.asyncio
    async def test_on_tool_start_end(self, hooks):
        """Test on_tool_start and on_tool_end hooks."""
        context = MockRunContextWrapper()
        agent = MockAgent()
        tool = MockTool(name="search_tool")

        await hooks.on_tool_start(context, agent, tool, {"query": "test"})
        await hooks.on_tool_end(context, agent, tool, {"results": []})

        events = hooks._session.chain.events
        tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].data["tool"] == "search_tool"

    @skip_if_no_asyncio
    @pytest.mark.asyncio
    async def test_on_handoff(self, hooks):
        """Test on_handoff hook."""
        context = MockRunContextWrapper()
        from_agent = MockAgent(name="agent1")
        to_agent = MockAgent(name="agent2")

        await hooks.on_handoff(context, from_agent, to_agent)

        events = hooks._session.chain.events
        handoff_events = [
            e for e in events
            if e.type == EventType.FACT_ADDED and "Handoff" in e.data.get("text", "")
        ]
        assert len(handoff_events) == 1
        assert "agent1" in handoff_events[0].data["text"]
        assert "agent2" in handoff_events[0].data["text"]

    @skip_if_no_asyncio
    @pytest.mark.asyncio
    async def test_on_error(self, hooks):
        """Test on_error hook."""
        context = MockRunContextWrapper()
        agent = MockAgent(name="failing_agent")
        error = RuntimeError("Test error")

        await hooks.on_agent_start(context, agent)
        await hooks.on_error(context, agent, error)

        events = hooks._session.chain.events
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].data["type"] == "RuntimeError"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        """Test _truncate with short text."""
        from lctl.integrations.openai_agents import _truncate

        text = "Short text"
        assert _truncate(text) == text

    def test_truncate_long_text(self):
        """Test _truncate with long text."""
        from lctl.integrations.openai_agents import _truncate

        text = "x" * 300
        result = _truncate(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_truncate_exact_length(self):
        """Test _truncate with exact length text."""
        from lctl.integrations.openai_agents import _truncate

        text = "x" * 200
        result = _truncate(text, max_length=200)
        assert result == text

    def test_extract_usage_with_usage_object(self):
        """Test _extract_usage with Usage object."""
        from lctl.integrations.openai_agents import _extract_usage

        usage = MockUsage(input_tokens=100, output_tokens=50)
        result = _extract_usage(usage)
        assert result["input"] == 100
        assert result["output"] == 50

    def test_extract_usage_with_none(self):
        """Test _extract_usage with None."""
        from lctl.integrations.openai_agents import _extract_usage

        result = _extract_usage(None)
        assert result["input"] == 0
        assert result["output"] == 0


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, mock_openai_agents_available, tmp_path: Path):
        """Test complete workflow: create tracer, record events, export, analyze."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            tracer = LCTLOpenAIAgentTracer(chain_id="integration-test")

            with tracer.trace_agent_run("planner", "Plan the task") as ctx:
                ctx.record_tool_call("search", "query", "results", 50)
                ctx.set_usage(tokens_in=100, tokens_out=50)
                ctx.set_output("Plan created")

            tracer.record_handoff("planner", "executor")

            with tracer.trace_agent_run("executor", "Execute the plan") as ctx:
                ctx.record_tool_call("code_gen", "template", "code", 100)
                ctx.set_usage(tokens_in=200, tokens_out=100)
                ctx.set_output("Execution complete")

            export_path = tmp_path / "integration.lctl.json"
            tracer.export(str(export_path))

            loaded_chain = Chain.load(export_path)
            assert loaded_chain.id == "integration-test"

            engine = ReplayEngine(loaded_chain)
            state = engine.replay_all()

            assert state.metrics["event_count"] > 0
            assert state.metrics["total_tokens_in"] == 300
            assert state.metrics["total_tokens_out"] == 150

    def test_error_recovery_workflow(self, mock_openai_agents_available):
        """Test workflow with error recovery."""
        with patch(
            "lctl.integrations.openai_agents._check_openai_agents_available",
            return_value=None,
        ):
            from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

            tracer = LCTLOpenAIAgentTracer(chain_id="error-recovery")

            try:
                with tracer.trace_agent_run("risky_agent", "Risky operation"):
                    raise ValueError("Simulated failure")
            except ValueError:
                pass

            with tracer.trace_agent_run("recovery_agent", "Recovery operation") as ctx:
                ctx.set_output("Recovered successfully")

            data = tracer.to_dict()
            error_events = [e for e in data["events"] if e["type"] == "error"]
            assert len(error_events) == 1

            step_ends = [e for e in data["events"] if e["type"] == "step_end"]
            outcomes = [e["data"]["outcome"] for e in step_ends]
            assert "error" in outcomes
            assert "success" in outcomes


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        """Test that all exported items exist."""
        from lctl.integrations import openai_agents

        for name in openai_agents.__all__:
            assert hasattr(openai_agents, name), f"Missing export: {name}"

    def test_expected_exports(self):
        """Test that expected items are exported."""
        from lctl.integrations.openai_agents import __all__

        expected = [
            "OPENAI_AGENTS_AVAILABLE",
            "OpenAIAgentsNotAvailableError",
            "LCTLRunHooks",
            "LCTLTracingProcessor",
            "LCTLOpenAIAgentTracer",
            "AgentRunContext",
            "TracedAgent",
            "trace_agent",
            "is_available",
        ]

        for item in expected:
            assert item in __all__, f"Missing from __all__: {item}"
