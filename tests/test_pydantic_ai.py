"""Tests for PydanticAI integration (lctl/integrations/pydantic_ai.py)."""

import sys
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from lctl.core.events import Chain, EventType, ReplayEngine
from lctl.core.session import LCTLSession

# Check if pydantic-ai is installed
try:
    import pydantic_ai
    from pydantic_ai import Agent, RunContext, AgentRunResult, RunUsage
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart, 
        ToolReturnPart,
        UserPromptPart
    )
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False


class TestPydanticAIEvents:
    """Mock events for testing."""

    @staticmethod
    def create_mock_usage(request=10, response=20):
        return RunUsage(
            input_tokens=request,
            output_tokens=response,
            input_audio_tokens=0,
            output_audio_tokens=0
        )

    @staticmethod
    def create_mock_result(data="test result", usage=None, messages=None):
        result = MagicMock(spec=AgentRunResult)
        result.data = data
        result.usage.return_value = usage or TestPydanticAIEvents.create_mock_usage()
        result.all_messages.return_value = messages or []
        return result


@pytest.fixture
def mock_pydantic_ai():
    """Fixture to check availability or skip."""
    if not PYDANTIC_AI_AVAILABLE:
        pytest.skip("PydanticAI not available")
    return True


class TestPydanticAIAvailability:
    """Tests for availability checking."""

    def test_availability_check(self):
        """Test is available check."""
        from lctl.integrations.pydantic_ai import PYDANTIC_AI_AVAILABLE
        # This test runs whether installed or not, just asserts bool type
        assert isinstance(PYDANTIC_AI_AVAILABLE, bool)

    def test_not_available_error(self):
        """Test error message."""
        from lctl.integrations.pydantic_ai import PydanticAINotAvailableError
        error = PydanticAINotAvailableError()
        assert "pip install pydantic-ai" in str(error)


@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="PydanticAI not installed")
class TestLCTLPydanticAITracer:
    """Tests for tracer functionality."""

    @pytest.fixture
    def tracer(self):
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer
        return LCTLPydanticAITracer(chain_id="test-pydantic-ai")

    def test_tracer_creation(self, tracer):
        assert tracer.session is not None
        assert tracer.chain.id == "test-pydantic-ai"

    def test_trace_run_context(self, tracer):
        with tracer.trace_run_context("agent", "input") as ctx:
            assert ctx.agent_name == "agent"
        
        events = tracer.chain.events
        assert len(events) >= 1
        assert events[0].type == EventType.STEP_START

    @pytest.mark.asyncio
    async def test_trace_agent_run_execution(self, tracer):
        """Test tracing an agent run."""
        agent = Agent(model="test")
        
        # Mock the run method
        mock_run = AsyncMock()
        mock_result = TestPydanticAIEvents.create_mock_result()
        mock_run.return_value = mock_result
        agent.run = mock_run
        
        # Use tracer to wrap
        tracer.trace_agent(agent)
        
        # Call the wrapped method
        result = await agent.run("Hello")
        
        assert result.data == "test result"
        
        events = tracer.chain.events
        
        # START exists
        assert events[0].type == EventType.STEP_START
        
        # Check success content in the step_end event
        step_end = [e for e in events if e.type == EventType.STEP_END][-1]
        assert step_end.data["outcome"] == "success"
        assert "test result" in step_end.data["output_summary"]
        assert step_end.data["tokens"]["input"] == 10
        assert step_end.data["tokens"]["output"] == 20

    def test_trace_agent_run_sync_execution(self, tracer):
        """Test tracing a sync agent run."""
        agent = Agent(model="test")
        
        # Mock run_sync
        mock_run = MagicMock()
        mock_result = TestPydanticAIEvents.create_mock_result()
        mock_run.return_value = mock_result
        agent.run_sync = mock_run
        
        tracer.trace_agent(agent)
        
        result = agent.run_sync("Hello")
        
        assert result.data == "test result"
        
        events = tracer.chain.events
        assert len(events) >= 2
        step_end = [e for e in events if e.type == EventType.STEP_END][-1]
        assert step_end.data["outcome"] == "success"
        assert step_end.data["tokens"]["input"] == 10

    @pytest.mark.asyncio
    async def test_trace_agent_error(self, tracer):
        """Test error handling in run."""
        agent = Agent(model="test")
        
        mock_run = AsyncMock()
        mock_run.side_effect = ValueError("Run failed")
        agent.run = mock_run
        
        tracer.trace_agent(agent)
        
        with pytest.raises(ValueError, match="Run failed"):
            await agent.run("Hello")
            
        events = tracer.chain.events
        error_event = [e for e in events if e.type == EventType.ERROR]
        assert len(error_event) == 1
        assert error_event[0].data["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_record_tool_calls(self, tracer):
        """Test recording of tool calls from message history."""
        agent = Agent(model="test")
        mock_run = AsyncMock()
        
        # Create messages with tool call
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content="Use tool"),
            ]),
            ModelRequest(parts=[
                ToolCallPart(tool_name="my_tool", args={"x": 1})
            ]),
            ModelResponse(parts=[
                ToolReturnPart(tool_name="my_tool", content="tool output", tool_call_id="1")
            ])
        ]
        
        mock_result = TestPydanticAIEvents.create_mock_result(messages=messages)
        mock_run.return_value = mock_result
        agent.run = mock_run
        
        tracer.trace_agent(agent)
        await agent.run("Hello")
        
        events = tracer.chain.events
        
        # Check handling of tool call
        tool_call = [e for e in events if e.type == EventType.TOOL_CALL]
        assert len(tool_call) == 1
        assert tool_call[0].data["tool"] == "my_tool"
        assert tool_call[0].data["input"]["x"] == 1

        # Check return value (recorded as fact)
        facts = [e for e in events if e.type == EventType.FACT_ADDED]
        tool_return = [f for f in facts if "my_tool returned: tool output" in f.data["text"]]
        assert len(tool_return) == 1

    def test_export_and_dict(self, tracer, tmp_path):
        """Test export functionality."""
        export_path = tmp_path / "trace.lctl.json"
        
        with tracer.trace_run_context("agent", "input"):
            pass
            
        tracer.export(str(export_path))
        assert export_path.exists()
        
        d = tracer.to_dict()
        assert d["chain"]["id"] == "test-pydantic-ai"


@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="PydanticAI not installed")
class TestHelperFunctions:
    
    def test_trace_agent_convenience(self):
        """Test trace_agent convenience function."""
        from lctl.integrations.pydantic_ai import trace_agent
        agent = Agent(model="test")
        tracer = trace_agent(agent, chain_id="conv-test")
        
        assert tracer.chain.id == "conv-test"
        # Check if wrapped (rudimentary check, since we wrap instance method)
        # We can check if `run` behaves as traced by running it?
        # But `run` is now a wrapped function.
        assert getattr(agent.run, "__wrapped__", None) is not None
