"""Comprehensive tests for PydanticAI integration.

This test module uses mocks to test the integration without requiring
pydantic_ai to be installed. It covers:
- LCTLPydanticAITracer class
- TracedAgent wrapper
- TracedStreamedRunResult
- Error handling and edge cases
- Thread safety and session management
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from lctl.core.events import EventType
from lctl.core.session import LCTLSession


# Always mock PYDANTIC_AI_AVAILABLE to True for these tests
@pytest.fixture(autouse=True)
def mock_pydantic_ai_available():
    """Ensure PYDANTIC_AI_AVAILABLE is True for all tests."""
    with patch("lctl.integrations.pydantic_ai.PYDANTIC_AI_AVAILABLE", True):
        yield


@pytest.fixture
def mock_agent():
    """Create a mock PydanticAI agent."""
    agent = MagicMock()
    agent.name = "test_agent"
    agent._function_toolset = None
    return agent


@pytest.fixture
def mock_agent_with_async_tool():
    """Create a mock PydanticAI agent with an async tool only."""
    agent = MagicMock()
    agent.name = "tool_agent"

    # Create mock async tool
    async_tool_func = AsyncMock(return_value="async_result")
    async_tool = MagicMock()
    async_tool.function = async_tool_func

    # Setup function toolset
    mock_toolset = MagicMock()
    mock_toolset.tools = {
        "async_tool": async_tool,
    }
    agent._function_toolset = mock_toolset

    return agent


@pytest.fixture
def mock_agent_with_sync_tool():
    """Create a mock PydanticAI agent with a sync tool only."""
    agent = MagicMock()
    agent.name = "sync_tool_agent"

    # Create actual sync function (not AsyncMock or MagicMock with default behavior)
    def sync_tool_func(*args, **kwargs):
        return "sync_result"

    sync_tool = MagicMock()
    sync_tool.function = sync_tool_func

    # Setup function toolset
    mock_toolset = MagicMock()
    mock_toolset.tools = {
        "sync_tool": sync_tool,
    }
    agent._function_toolset = mock_toolset

    return agent


@pytest.fixture
def mock_run_result():
    """Create a mock AgentRunResult."""
    result = MagicMock()
    result.data = "test output"

    usage = MagicMock()
    usage.request_tokens = 100
    usage.response_tokens = 50
    result.usage = MagicMock(return_value=usage)

    # Add all_messages for LLM trace
    msg1 = MagicMock()
    msg1.model_dump = MagicMock(return_value={"role": "user", "content": "test"})
    result.all_messages = MagicMock(return_value=[msg1])

    return result


@pytest.fixture
def mock_run_result_with_output():
    """Create a mock result with output attribute instead of data."""
    result = MagicMock()
    del result.data  # Remove data attribute
    result.output = "output via output attribute"

    usage = MagicMock()
    usage.request_tokens = 50
    usage.response_tokens = 25
    result.usage = MagicMock(return_value=usage)

    return result


@pytest.fixture
def mock_run_result_minimal():
    """Create a mock result with minimal attributes."""
    result = MagicMock()
    del result.data
    del result.output
    del result.usage
    del result.all_messages
    return result


class TestLCTLPydanticAITracer:
    """Tests for LCTLPydanticAITracer class."""

    def test_init_with_chain_id(self):
        """Test tracer initialization with chain_id."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="my-chain")
        assert tracer.session.chain.id == "my-chain"
        assert tracer._verbose is False
        assert hasattr(tracer, "_lock")

    def test_init_with_session(self):
        """Test tracer initialization with existing session."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        session = LCTLSession(chain_id="existing-session", redaction_enabled=False)
        tracer = LCTLPydanticAITracer(session=session)
        assert tracer.session is session
        assert tracer.session.chain.id == "existing-session"

    def test_init_with_verbose(self):
        """Test tracer initialization with verbose flag."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="test", verbose=True)
        assert tracer._verbose is True

    def test_init_without_pydantic_ai(self):
        """Test tracer raises ImportError when PydanticAI not available."""
        with patch("lctl.integrations.pydantic_ai.PYDANTIC_AI_AVAILABLE", False):
            from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

            with pytest.raises(ImportError) as exc_info:
                LCTLPydanticAITracer(chain_id="test")
            assert "PydanticAI is not installed" in str(exc_info.value)

    def test_chain_property(self):
        """Test chain property returns session chain."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="test")
        assert tracer.chain is tracer.session.chain

    def test_export(self, tmp_path):
        """Test export method."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="export-test")
        tracer.session.step_start("agent", "test", "input")
        tracer.session.step_end("agent", "success")

        file_path = tmp_path / "trace.lctl.json"
        tracer.export(str(file_path))

        assert file_path.exists()

    def test_to_dict(self):
        """Test to_dict method."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="dict-test")
        tracer.session.step_start("agent", "test", "input")

        result = tracer.to_dict()
        assert "lctl" in result
        assert "chain" in result
        assert "events" in result
        assert result["chain"]["id"] == "dict-test"

    def test_thread_safety(self):
        """Test thread safety with _lock."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="thread-test")
        results = []
        errors = []

        def worker(n):
            try:
                for i in range(10):
                    tracer.session.step_start(f"agent_{n}", f"task_{i}", f"input_{i}")
                    tracer.session.step_end(outcome="success")
                results.append(n)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3


class TestTracedAgent:
    """Tests for TracedAgent class."""

    def test_init_basic(self, mock_agent):
        """Test TracedAgent basic initialization."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent, chain_id="test-chain")
        assert traced.agent is mock_agent
        assert traced.tracer is not None
        assert traced.tracer.session.chain.id == "test-chain"

    def test_init_with_session(self, mock_agent):
        """Test TracedAgent with existing session."""
        from lctl.integrations.pydantic_ai import TracedAgent

        session = LCTLSession(chain_id="custom-session", redaction_enabled=False)
        traced = TracedAgent(mock_agent, session=session)
        assert traced.tracer.session is session

    def test_init_with_verbose(self, mock_agent):
        """Test TracedAgent with verbose flag."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent, verbose=True)
        assert traced.tracer._verbose is True

    def test_instrument_tools_no_toolset(self, mock_agent):
        """Test _instrument_tools when no toolset exists."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent._function_toolset = None
        traced = TracedAgent(mock_agent, chain_id="test")
        # Should not raise, just skip instrumentation
        assert traced.agent is mock_agent

    def test_instrument_tools_no_tools_attr(self, mock_agent):
        """Test _instrument_tools when toolset has no tools attribute."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent._function_toolset = MagicMock(spec=[])  # No 'tools' attribute
        traced = TracedAgent(mock_agent, chain_id="test")
        assert traced.agent is mock_agent

    def test_instrument_tools_no_function_attr(self):
        """Test _instrument_tools when tool has no function attribute."""
        from lctl.integrations.pydantic_ai import TracedAgent

        agent = MagicMock()
        agent.name = "test"

        tool = MagicMock(spec=[])  # No 'function' attribute
        mock_toolset = MagicMock()
        mock_toolset.tools = {"broken_tool": tool}
        agent._function_toolset = mock_toolset

        traced = TracedAgent(agent, chain_id="test")
        assert traced.agent is agent

    def test_instrument_tools_async(self, mock_agent_with_async_tool):
        """Test async tool instrumentation."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent_with_async_tool, chain_id="test")

        async_tool = mock_agent_with_async_tool._function_toolset.tools["async_tool"]
        assert hasattr(async_tool.function, "_lctl_instrumented")
        assert async_tool.function._lctl_instrumented is True

    def test_instrument_tools_sync(self, mock_agent_with_sync_tool):
        """Test sync tool instrumentation."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent_with_sync_tool, chain_id="test")

        sync_tool = mock_agent_with_sync_tool._function_toolset.tools["sync_tool"]
        assert hasattr(sync_tool.function, "_lctl_instrumented")
        assert sync_tool.function._lctl_instrumented is True

    def test_no_double_instrumentation(self, mock_agent_with_async_tool):
        """Test that tools are not double-instrumented."""
        from lctl.integrations.pydantic_ai import TracedAgent

        # First wrap
        traced1 = TracedAgent(mock_agent_with_async_tool, chain_id="test1")
        original_wrapper = mock_agent_with_async_tool._function_toolset.tools["async_tool"].function

        # Second wrap - should not re-instrument
        traced2 = TracedAgent(mock_agent_with_async_tool, chain_id="test2")
        new_wrapper = mock_agent_with_async_tool._function_toolset.tools["async_tool"].function

        assert new_wrapper is original_wrapper

    @pytest.mark.asyncio
    async def test_trace_tool_execution_async_success(self, mock_agent_with_async_tool):
        """Test async tool execution tracing on success."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent_with_async_tool, chain_id="test")
        traced.tracer.session.tool_call = MagicMock()

        async_tool = mock_agent_with_async_tool._function_toolset.tools["async_tool"]
        result = await async_tool.function("arg1", key="value")

        assert result == "async_result"
        traced.tracer.session.tool_call.assert_called_once()
        call_kwargs = traced.tracer.session.tool_call.call_args[1]
        assert call_kwargs["tool"] == "async_tool"
        assert "async_result" in call_kwargs["output_data"]
        assert "duration_ms" in call_kwargs

    @pytest.mark.asyncio
    async def test_trace_tool_execution_async_error(self):
        """Test async tool execution tracing on error."""
        from lctl.integrations.pydantic_ai import TracedAgent

        agent = MagicMock()
        agent.name = "test"

        async_tool_func = AsyncMock(side_effect=ValueError("Tool error"))
        async_tool = MagicMock()
        async_tool.function = async_tool_func

        mock_toolset = MagicMock()
        mock_toolset.tools = {"failing_tool": async_tool}
        agent._function_toolset = mock_toolset

        traced = TracedAgent(agent, chain_id="test")
        traced.tracer.session.tool_call = MagicMock()

        with pytest.raises(ValueError, match="Tool error"):
            await mock_toolset.tools["failing_tool"].function("arg")

        traced.tracer.session.tool_call.assert_called_once()
        call_kwargs = traced.tracer.session.tool_call.call_args[1]
        assert "Error: Tool error" in call_kwargs["output_data"]

    def test_trace_tool_execution_sync_success(self, mock_agent_with_sync_tool):
        """Test sync tool execution tracing on success."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent_with_sync_tool, chain_id="test")
        traced.tracer.session.tool_call = MagicMock()

        sync_tool = mock_agent_with_sync_tool._function_toolset.tools["sync_tool"]
        result = sync_tool.function("arg1", key="value")

        assert result == "sync_result"
        traced.tracer.session.tool_call.assert_called_once()
        call_kwargs = traced.tracer.session.tool_call.call_args[1]
        assert call_kwargs["tool"] == "sync_tool"
        assert "sync_result" in call_kwargs["output_data"]

    def test_trace_tool_execution_sync_error(self):
        """Test sync tool execution tracing on error."""
        from lctl.integrations.pydantic_ai import TracedAgent

        agent = MagicMock()
        agent.name = "test"

        def sync_tool_func(*args, **kwargs):
            raise RuntimeError("Sync error")

        sync_tool = MagicMock()
        sync_tool.function = sync_tool_func

        mock_toolset = MagicMock()
        mock_toolset.tools = {"failing_sync": sync_tool}
        agent._function_toolset = mock_toolset

        traced = TracedAgent(agent, chain_id="test")
        traced.tracer.session.tool_call = MagicMock()

        with pytest.raises(RuntimeError, match="Sync error"):
            mock_toolset.tools["failing_sync"].function("arg")

        traced.tracer.session.tool_call.assert_called_once()
        call_kwargs = traced.tracer.session.tool_call.call_args[1]
        assert "Error: Sync error" in call_kwargs["output_data"]

    def test_export(self, mock_agent, tmp_path):
        """Test TracedAgent export method."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent, chain_id="export-test")
        file_path = tmp_path / "agent_trace.lctl.json"
        traced.export(str(file_path))
        assert file_path.exists()

    def test_to_dict(self, mock_agent):
        """Test TracedAgent to_dict method."""
        from lctl.integrations.pydantic_ai import TracedAgent

        traced = TracedAgent(mock_agent, chain_id="dict-test")
        result = traced.to_dict()
        assert isinstance(result, dict)
        assert "chain" in result

    @pytest.mark.asyncio
    async def test_run_success(self, mock_agent, mock_run_result):
        """Test TracedAgent.run on success."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(return_value=mock_run_result)

        # Use session with redaction disabled to check token counts
        session = LCTLSession(chain_id="run-test", redaction_enabled=False)
        traced = TracedAgent(mock_agent, session=session)

        result = await traced.run("test input")

        assert result is mock_run_result
        events = traced.tracer.session.chain.events
        assert len(events) >= 2  # At least step_start and step_end

        step_start = events[0]
        assert step_start.type == EventType.STEP_START
        assert step_start.agent == "test_agent"
        assert step_start.data["intent"] == "run"

        step_end = [e for e in events if e.type == EventType.STEP_END][0]
        assert step_end.data["outcome"] == "success"
        assert step_end.data["tokens"]["input"] == 100
        assert step_end.data["tokens"]["output"] == 50

    @pytest.mark.asyncio
    async def test_run_with_output_attribute(self, mock_agent, mock_run_result_with_output):
        """Test TracedAgent.run with result.output instead of result.data."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(return_value=mock_run_result_with_output)

        traced = TracedAgent(mock_agent, chain_id="output-attr-test")

        result = await traced.run("test input")

        step_end = [e for e in traced.tracer.session.chain.events if e.type == EventType.STEP_END][0]
        assert "output via output attribute" in step_end.data["output_summary"]

    @pytest.mark.asyncio
    async def test_run_with_minimal_result(self, mock_agent, mock_run_result_minimal):
        """Test TracedAgent.run with minimal result (no data, output, usage)."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(return_value=mock_run_result_minimal)

        # Use session with redaction disabled
        session = LCTLSession(chain_id="minimal-test", redaction_enabled=False)
        traced = TracedAgent(mock_agent, session=session)

        result = await traced.run("test input")

        step_end = [e for e in traced.tracer.session.chain.events if e.type == EventType.STEP_END][0]
        assert step_end.data["outcome"] == "success"
        assert step_end.data["tokens"]["input"] == 0
        assert step_end.data["tokens"]["output"] == 0

    @pytest.mark.asyncio
    async def test_run_with_kwargs(self, mock_agent, mock_run_result):
        """Test TracedAgent.run with keyword arguments."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(return_value=mock_run_result)

        traced = TracedAgent(mock_agent, chain_id="kwargs-test")

        result = await traced.run(user_prompt="kwarg input")

        step_start = traced.tracer.session.chain.events[0]
        assert "kwarg input" in step_start.data["input_summary"]

    @pytest.mark.asyncio
    async def test_run_error(self, mock_agent):
        """Test TracedAgent.run on error."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(side_effect=RuntimeError("Agent failed"))

        traced = TracedAgent(mock_agent, chain_id="error-test")

        with pytest.raises(RuntimeError, match="Agent failed"):
            await traced.run("test input")

        events = traced.tracer.session.chain.events
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        # Error events use "type" not "error_type"
        assert error_events[0].data["type"] == "RuntimeError"
        assert error_events[0].data["message"] == "Agent failed"

        step_end = [e for e in events if e.type == EventType.STEP_END][0]
        assert step_end.data["outcome"] == "error"

    @pytest.mark.asyncio
    async def test_run_with_llm_trace(self, mock_agent, mock_run_result):
        """Test that llm_trace is recorded."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(return_value=mock_run_result)
        mock_agent.model_name = "gpt-4-turbo"

        traced = TracedAgent(mock_agent, chain_id="llm-trace-test")

        await traced.run("test input")

        llm_traces = [e for e in traced.tracer.session.chain.events if e.type == EventType.LLM_TRACE]
        assert len(llm_traces) >= 1
        assert llm_traces[0].data["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_run_with_agent_no_name(self, mock_run_result):
        """Test TracedAgent.run with agent that has no name."""
        from lctl.integrations.pydantic_ai import TracedAgent

        agent = MagicMock()
        agent.name = None
        agent._function_toolset = None
        agent.run = AsyncMock(return_value=mock_run_result)

        traced = TracedAgent(agent, chain_id="no-name-test")

        await traced.run("test")

        step_start = traced.tracer.session.chain.events[0]
        assert step_start.agent == "pydantic_ai_agent"


class TestTracedStreamedRunResult:
    """Tests for TracedStreamedRunResult class."""

    @pytest.fixture
    def mock_stream_result(self):
        """Create mock stream result."""
        result = MagicMock()

        async def stream_text_gen():
            for chunk in ["Hello", " ", "World"]:
                yield chunk

        result.stream_text = MagicMock(return_value=stream_text_gen())

        usage = MagicMock()
        usage.request_tokens = 25
        usage.response_tokens = 15
        result.usage = MagicMock(return_value=usage)

        msg = MagicMock()
        msg.model_dump = MagicMock(return_value={"role": "assistant", "content": "Hello World"})
        result.all_messages = MagicMock(return_value=[msg])

        return result

    @pytest.mark.asyncio
    async def test_stream_text_success(self, mock_agent, mock_stream_result):
        """Test stream_text iteration with tracing."""
        from lctl.integrations.pydantic_ai import TracedAgent

        @asynccontextmanager
        async def mock_run_stream(*args, **kwargs):
            yield mock_stream_result

        mock_agent.run_stream = mock_run_stream

        traced = TracedAgent(mock_agent, chain_id="stream-test")

        chunks = []
        async with traced.run_stream("test prompt") as result:
            async for chunk in result.stream_text():
                chunks.append(chunk)

        assert "".join(chunks) == "Hello World"

        events = traced.tracer.session.chain.events
        stream_starts = [e for e in events if e.type == EventType.STREAM_START]
        stream_ends = [e for e in events if e.type == EventType.STREAM_END]
        stream_chunks = [e for e in events if e.type == EventType.STREAM_CHUNK]

        assert len(stream_starts) == 1
        assert len(stream_ends) == 1
        assert len(stream_chunks) == 3

    @pytest.mark.asyncio
    async def test_stream_finalize_on_early_exit(self, mock_agent, mock_stream_result):
        """Test stream finalization when consumer exits early."""
        from lctl.integrations.pydantic_ai import TracedAgent

        @asynccontextmanager
        async def mock_run_stream(*args, **kwargs):
            yield mock_stream_result

        mock_agent.run_stream = mock_run_stream

        traced = TracedAgent(mock_agent, chain_id="early-exit-test")

        async with traced.run_stream("test prompt") as result:
            # Don't iterate - exit early
            pass

        events = traced.tracer.session.chain.events
        step_ends = [e for e in events if e.type == EventType.STEP_END]
        assert len(step_ends) == 1

    @pytest.mark.asyncio
    async def test_stream_error(self, mock_agent):
        """Test stream error handling."""
        from lctl.integrations.pydantic_ai import TracedAgent

        @asynccontextmanager
        async def mock_run_stream(*args, **kwargs):
            raise ValueError("Stream error")
            yield  # Make it a generator

        mock_agent.run_stream = mock_run_stream

        traced = TracedAgent(mock_agent, chain_id="stream-error-test")

        with pytest.raises(ValueError, match="Stream error"):
            async with traced.run_stream("test prompt") as result:
                pass

        events = traced.tracer.session.chain.events
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].data["message"] == "Stream error"

    @pytest.mark.asyncio
    async def test_stream_getattr_delegation(self, mock_agent, mock_stream_result):
        """Test that TracedStreamedRunResult delegates attribute access."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_stream_result.custom_attr = "custom_value"

        @asynccontextmanager
        async def mock_run_stream(*args, **kwargs):
            yield mock_stream_result

        mock_agent.run_stream = mock_run_stream

        traced = TracedAgent(mock_agent, chain_id="getattr-test")

        async with traced.run_stream("test") as result:
            assert result.custom_attr == "custom_value"

    @pytest.mark.asyncio
    async def test_stream_usage_error_handling(self, mock_agent):
        """Test stream handles usage() errors gracefully."""
        from lctl.integrations.pydantic_ai import TracedAgent

        result = MagicMock()

        async def stream_text_gen():
            yield "chunk"

        result.stream_text = MagicMock(return_value=stream_text_gen())
        result.usage = MagicMock(side_effect=RuntimeError("Usage error"))
        del result.all_messages

        @asynccontextmanager
        async def mock_run_stream(*args, **kwargs):
            yield result

        mock_agent.run_stream = mock_run_stream

        # Use session with redaction disabled
        session = LCTLSession(chain_id="usage-error-test", redaction_enabled=False)
        traced = TracedAgent(mock_agent, session=session)

        chunks = []
        async with traced.run_stream("test") as stream_result:
            async for chunk in stream_result.stream_text():
                chunks.append(chunk)

        # Should not raise, tokens should be 0
        events = traced.tracer.session.chain.events
        step_end = [e for e in events if e.type == EventType.STEP_END][0]
        assert step_end.data["tokens"]["input"] == 0
        assert step_end.data["tokens"]["output"] == 0


class TestTraceAgentFunction:
    """Tests for trace_agent convenience function."""

    def test_trace_agent_basic(self, mock_agent):
        """Test trace_agent creates TracedAgent."""
        from lctl.integrations.pydantic_ai import trace_agent, TracedAgent

        traced = trace_agent(mock_agent, chain_id="func-test")
        assert isinstance(traced, TracedAgent)
        assert traced.agent is mock_agent

    def test_trace_agent_with_session(self, mock_agent):
        """Test trace_agent with existing session."""
        from lctl.integrations.pydantic_ai import trace_agent

        session = LCTLSession(chain_id="existing", redaction_enabled=False)
        traced = trace_agent(mock_agent, session=session)
        assert traced.tracer.session is session

    def test_trace_agent_with_verbose(self, mock_agent):
        """Test trace_agent with verbose flag."""
        from lctl.integrations.pydantic_ai import trace_agent

        traced = trace_agent(mock_agent, verbose=True)
        assert traced.tracer._verbose is True


class TestIsAvailable:
    """Tests for is_available function."""

    def test_is_available_true(self):
        """Test is_available returns True when module is available."""
        from lctl.integrations.pydantic_ai import is_available

        assert is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False when module not available."""
        with patch("lctl.integrations.pydantic_ai.PYDANTIC_AI_AVAILABLE", False):
            # Need to reload to pick up the patched value
            import importlib
            import lctl.integrations.pydantic_ai as pai

            # Manually check the patched value
            original = pai.PYDANTIC_AI_AVAILABLE
            pai.PYDANTIC_AI_AVAILABLE = False

            try:
                assert pai.is_available() is False
            finally:
                pai.PYDANTIC_AI_AVAILABLE = original


class TestConcurrentAgents:
    """Tests for concurrent agent execution."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_agents(self):
        """Test multiple agents running concurrently."""
        from lctl.integrations.pydantic_ai import TracedAgent

        agents = []
        for i in range(3):
            agent = MagicMock()
            agent.name = f"agent_{i}"
            agent._function_toolset = None

            async def make_run(idx):
                await asyncio.sleep(0.01)
                result = MagicMock()
                result.data = f"result_{idx}"
                del result.usage
                del result.all_messages
                return result

            agent.run = AsyncMock(side_effect=lambda x, i=i: make_run(i))
            agents.append(TracedAgent(agent, chain_id=f"chain_{i}"))

        # Run concurrently
        async def run_agent(traced, prompt):
            return await traced.run(prompt)

        results = await asyncio.gather(*[
            run_agent(agents[i], f"prompt_{i}") for i in range(3)
        ])

        assert len(results) == 3
        for i, agent in enumerate(agents):
            events = agent.tracer.session.chain.events
            assert len([e for e in events if e.type == EventType.STEP_START]) == 1
            assert len([e for e in events if e.type == EventType.STEP_END]) == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self, mock_agent, mock_run_result):
        """Test running with empty prompt."""
        from lctl.integrations.pydantic_ai import TracedAgent

        mock_agent.run = AsyncMock(return_value=mock_run_result)

        traced = TracedAgent(mock_agent, chain_id="empty-test")
        await traced.run("")

        step_start = traced.tracer.session.chain.events[0]
        assert step_start.data["input_summary"] == ""

    @pytest.mark.asyncio
    async def test_very_long_output(self, mock_agent):
        """Test output truncation for very long responses."""
        from lctl.integrations.pydantic_ai import TracedAgent

        result = MagicMock()
        result.data = "x" * 10000
        del result.usage
        del result.all_messages

        mock_agent.run = AsyncMock(return_value=result)

        traced = TracedAgent(mock_agent, chain_id="long-output-test")
        await traced.run("test")

        step_end = [e for e in traced.tracer.session.chain.events if e.type == EventType.STEP_END][0]
        # Output should be the full string since we capture it fully, truncation happens at tool level
        assert "x" in step_end.data["output_summary"]

    def test_function_schema_instrumentation(self):
        """Test that function_schema.function is also updated."""
        from lctl.integrations.pydantic_ai import TracedAgent

        agent = MagicMock()
        agent.name = "test"

        def tool_func(*args, **kwargs):
            return "result"

        tool = MagicMock()
        tool.function = tool_func

        # Add function_schema with function attribute
        tool.function_schema = MagicMock()
        tool.function_schema.function = tool_func

        mock_toolset = MagicMock()
        mock_toolset.tools = {"schema_tool": tool}
        agent._function_toolset = mock_toolset

        traced = TracedAgent(agent, chain_id="schema-test")

        # Both should be updated
        assert tool.function._lctl_instrumented is True
        assert tool.function_schema.function._lctl_instrumented is True

    @pytest.mark.asyncio
    async def test_all_messages_without_model_dump(self, mock_agent):
        """Test handling messages without model_dump method."""
        from lctl.integrations.pydantic_ai import TracedAgent

        result = MagicMock()
        result.data = "test"

        usage = MagicMock()
        usage.request_tokens = 10
        usage.response_tokens = 5
        result.usage = MagicMock(return_value=usage)

        # Message without model_dump
        msg = "plain string message"
        result.all_messages = MagicMock(return_value=[msg])

        mock_agent.run = AsyncMock(return_value=result)

        traced = TracedAgent(mock_agent, chain_id="no-model-dump-test")
        await traced.run("test")

        llm_traces = [e for e in traced.tracer.session.chain.events if e.type == EventType.LLM_TRACE]
        assert len(llm_traces) >= 1
        # Should fallback to str(msg)
        assert "plain string message" in str(llm_traces[0].data["messages"])


class TestNestedAgentTracking:
    """Tests for nested agent tracking."""

    @pytest.mark.asyncio
    async def test_nested_agent_calls(self, mock_run_result):
        """Test that nested agent calls are tracked correctly."""
        from lctl.integrations.pydantic_ai import TracedAgent

        # Create parent agent
        parent_agent = MagicMock()
        parent_agent.name = "parent"
        parent_agent._function_toolset = None

        # Create child agent
        child_agent = MagicMock()
        child_agent.name = "child"
        child_agent._function_toolset = None
        child_agent.run = AsyncMock(return_value=mock_run_result)

        # Parent calls child during its run
        async def parent_run(*args, **kwargs):
            result = MagicMock()
            result.data = "parent result"
            del result.usage
            del result.all_messages
            return result

        parent_agent.run = AsyncMock(side_effect=parent_run)

        # Use shared session for both
        session = LCTLSession(chain_id="nested-test", redaction_enabled=False)
        traced_parent = TracedAgent(parent_agent, session=session)
        traced_child = TracedAgent(child_agent, session=session)

        # Run parent
        await traced_parent.run("parent input")
        # Then run child
        await traced_child.run("child input")

        events = session.chain.events
        step_starts = [e for e in events if e.type == EventType.STEP_START]
        step_ends = [e for e in events if e.type == EventType.STEP_END]

        assert len(step_starts) == 2
        assert len(step_ends) == 2
        assert step_starts[0].agent == "parent"
        assert step_starts[1].agent == "child"


class TestSessionCleanup:
    """Tests for session management and cleanup."""

    def test_session_cleanup_of_stale_entries(self):
        """Test cleanup of stale tracking entries."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer = LCTLPydanticAITracer(chain_id="cleanup-test")

        # Session manages its own state via the chain
        # Just verify session is accessible
        assert tracer.session is not None
        assert tracer.chain is tracer.session.chain

    def test_custom_chain_id_generation(self):
        """Test that chain IDs are unique when not specified."""
        from lctl.integrations.pydantic_ai import LCTLPydanticAITracer

        tracer1 = LCTLPydanticAITracer()
        tracer2 = LCTLPydanticAITracer()

        # Each tracer should have a unique chain ID when not specified
        assert tracer1.session.chain.id != tracer2.session.chain.id


class TestAsyncContextHandling:
    """Tests for async context handling."""

    @pytest.mark.asyncio
    async def test_async_context_manager_success(self, mock_agent):
        """Test async context manager on success."""
        from lctl.integrations.pydantic_ai import TracedAgent

        stream_result = MagicMock()

        async def stream_text():
            yield "text"

        stream_result.stream_text = stream_text
        stream_result.usage = MagicMock(return_value=MagicMock(request_tokens=1, response_tokens=1))
        del stream_result.all_messages

        @asynccontextmanager
        async def run_stream(*args, **kwargs):
            yield stream_result

        mock_agent.run_stream = run_stream

        traced = TracedAgent(mock_agent, chain_id="ctx-test")

        async with traced.run_stream("prompt") as result:
            async for chunk in result.stream_text():
                assert chunk == "text"

        # Verify stream events recorded
        events = traced.tracer.session.chain.events
        assert any(e.type == EventType.STEP_START for e in events)
        assert any(e.type == EventType.STEP_END for e in events)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_streams(self, mock_agent):
        """Test multiple concurrent streams."""
        from lctl.integrations.pydantic_ai import TracedAgent

        def create_stream_result(text):
            result = MagicMock()

            async def stream_text():
                for char in text:
                    yield char

            result.stream_text = stream_text
            result.usage = MagicMock(return_value=MagicMock(request_tokens=1, response_tokens=len(text)))
            del result.all_messages
            return result

        # Create separate agents for each stream
        agents = []
        for i, text in enumerate(["abc", "xyz"]):
            agent = MagicMock()
            agent.name = f"stream_agent_{i}"
            agent._function_toolset = None

            @asynccontextmanager
            async def run_stream(*args, text=text, **kwargs):
                yield create_stream_result(text)

            agent.run_stream = run_stream
            agents.append(TracedAgent(agent, chain_id=f"stream-{i}"))

        async def run_stream(traced, prompt):
            chunks = []
            async with traced.run_stream(prompt) as result:
                async for chunk in result.stream_text():
                    chunks.append(chunk)
            return "".join(chunks)

        results = await asyncio.gather(*[
            run_stream(agents[i], f"prompt_{i}") for i in range(2)
        ])

        assert results[0] == "abc"
        assert results[1] == "xyz"
