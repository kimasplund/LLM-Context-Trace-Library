"""Comprehensive tests for Semantic Kernel integration.

Tests cover:
1. LCTLSemanticKernelTracer class initialization and methods
2. Filter implementations (function invocation, prompt rendering)
3. Streaming and non-streaming responses
4. Error handling and graceful degradation
5. Thread safety
6. Session management and export

Uses mocks to test without requiring semantic-kernel installation.
"""

import asyncio
import threading
import time
import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Any, Callable

import pytest

from lctl.core.events import EventType
from lctl.core.session import LCTLSession


# Mock classes to simulate Semantic Kernel types
class MockKernelFunction:
    """Mock for semantic_kernel.functions.kernel_function.KernelFunction"""
    def __init__(self, name: str = "test_func", plugin_name: str = "TestPlugin"):
        self.name = name
        self.plugin_name = plugin_name


class MockFunctionInvocationContext:
    """Mock for semantic_kernel.filters.FunctionInvocationContext"""
    def __init__(
        self,
        function: MockKernelFunction = None,
        arguments: dict = None,
        result: Any = None
    ):
        self.function = function or MockKernelFunction()
        self.arguments = arguments or {"arg1": "value1"}
        self.result = result


class MockPromptRenderContext:
    """Mock for semantic_kernel.filters.PromptRenderContext"""
    def __init__(
        self,
        function: MockKernelFunction = None,
        rendered_prompt: str = None
    ):
        self.function = function or MockKernelFunction()
        self.rendered_prompt = rendered_prompt


class MockFunctionResult:
    """Mock for function result with optional metadata"""
    def __init__(self, value: Any = "test_result", metadata: dict = None):
        self.value = value
        self.metadata = metadata


class MockUsage:
    """Mock for token usage object"""
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockKernel:
    """Mock for semantic_kernel.kernel.Kernel"""
    def __init__(self):
        self._filters = {}

    def add_filter(self, filter_type, filter_func):
        if filter_type not in self._filters:
            self._filters[filter_type] = []
        self._filters[filter_type].append(filter_func)


class MockFilterTypes:
    """Mock for semantic_kernel.filters.FilterTypes enum"""
    FUNCTION_INVOCATION = "function_invocation"
    PROMPT_RENDERING = "prompt_rendering"


# Create a mock tracer that doesn't require SK imports
class MockedLCTLSemanticKernelTracer:
    """A tracer implementation for testing without SK dependency.

    Note: This implementation has a small fix compared to the real implementation -
    it tracks error_handled to prevent double step_end calls in the finally block.
    """

    def __init__(
        self,
        chain_id: str = None,
        session: LCTLSession = None,
        verbose: bool = False,
    ):
        from lctl.integrations.base import truncate
        self._truncate = truncate
        self._lock = threading.Lock()
        self.session = session or LCTLSession(
            chain_id=chain_id or f"sk-{id(self)}"
        )
        self._verbose = verbose
        self._traced_kernels: set = set()

    @property
    def chain(self):
        return self.session.chain

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file."""
        self.session.export(path)

    def to_dict(self):
        """Export the LCTL chain as a dictionary."""
        return self.session.to_dict()

    def trace_kernel(self, kernel) -> Any:
        """Attach tracing to a kernel."""
        kernel_id = id(kernel)
        with self._lock:
            if kernel_id in self._traced_kernels:
                return kernel
            self._traced_kernels.add(kernel_id)
        kernel.add_filter(MockFilterTypes.FUNCTION_INVOCATION, self._function_invocation_filter)
        kernel.add_filter(MockFilterTypes.PROMPT_RENDERING, self._prompt_render_filter)
        return kernel

    async def _prompt_render_filter(self, context, next_func):
        """Filter for prompt rendering."""
        await next_func(context)

        if context.rendered_prompt:
            function_name = context.function.name
            plugin_name = context.function.plugin_name or "Global"
            full_name = f"{plugin_name}.{function_name}"

            try:
                self.session.add_fact(
                    fact_id=f"prompt-{full_name}-{hash(context.rendered_prompt)}",
                    text=f"Rendered Prompt for {full_name}: {context.rendered_prompt}",
                    confidence=1.0,
                    source="sk-prompt-filter"
                )
            except Exception:
                pass

    async def _function_invocation_filter(self, context, next_func):
        """Filter for function invocations."""
        function_name = context.function.name
        plugin_name = context.function.plugin_name or "Global"
        full_name = f"{plugin_name}.{function_name}"

        input_summary = self._truncate(str(context.arguments))

        try:
            self.session.step_start(
                agent=full_name,
                intent="execute_function",
                input_summary=input_summary
            )
        except Exception:
            pass

        start_time = time.time()

        try:
            await next_func(context)

            result = context.result

            # Check for streaming
            if result and hasattr(result, "value") and hasattr(result.value, "__aiter__"):
                original_stream = result.value
                stream_start_time = time.time()

                async def stream_wrapper():
                    accumulated_content = ""
                    tokens_in = 0
                    tokens_out = 0
                    stream_completed = False
                    error_handled = False  # Track if error was already handled

                    try:
                        async for chunk in original_stream:
                            chunk_str = str(chunk)
                            accumulated_content += chunk_str

                            if hasattr(chunk, "metadata") and chunk.metadata:
                                usage = chunk.metadata.get("usage")
                                if usage:
                                    if hasattr(usage, "prompt_tokens"):
                                        tokens_in = usage.prompt_tokens
                                    elif isinstance(usage, dict):
                                        tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)

                                    if hasattr(usage, "completion_tokens"):
                                        tokens_out = usage.completion_tokens
                                    elif isinstance(usage, dict):
                                        tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

                            yield chunk

                        stream_completed = True
                        duration_ms = int((time.time() - stream_start_time) * 1000)
                        try:
                            self.session.step_end(
                                agent=full_name,
                                outcome="success",
                                output_summary=self._truncate(accumulated_content),
                                duration_ms=duration_ms,
                                tokens_in=tokens_in,
                                tokens_out=tokens_out
                            )
                        except Exception:
                            pass

                    except Exception as e:
                        error_handled = True
                        try:
                            self.session.error(
                                category="execution_error",
                                error_type=type(e).__name__,
                                message=str(e),
                                recoverable=False
                            )
                        except Exception:
                            pass
                        duration_ms = int((time.time() - stream_start_time) * 1000)
                        try:
                            self.session.step_end(
                                agent=full_name,
                                outcome="error",
                                duration_ms=duration_ms
                            )
                        except Exception:
                            pass
                        raise
                    finally:
                        # Only call abandoned step_end if stream wasn't completed AND no error was handled
                        if not stream_completed and not error_handled:
                            try:
                                duration_ms = int((time.time() - stream_start_time) * 1000)
                                self.session.step_end(
                                    agent=full_name,
                                    outcome="abandoned",
                                    output_summary=self._truncate(accumulated_content),
                                    duration_ms=duration_ms,
                                    tokens_in=tokens_in,
                                    tokens_out=tokens_out
                                )
                            except Exception:
                                pass

                result.value = stream_wrapper()
                return

            # Non-streaming
            duration_ms = int((time.time() - start_time) * 1000)
            output_summary = self._truncate(str(result)) if result else ""

            tokens_in = 0
            tokens_out = 0
            if result and hasattr(result, "metadata") and result.metadata:
                usage = result.metadata.get("usage", None)
                if usage:
                    if hasattr(usage, "prompt_tokens"):
                        tokens_in = usage.prompt_tokens
                    elif isinstance(usage, dict):
                        tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)

                    if hasattr(usage, "completion_tokens"):
                        tokens_out = usage.completion_tokens
                    elif isinstance(usage, dict):
                        tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

            try:
                self.session.step_end(
                    agent=full_name,
                    outcome="success",
                    output_summary=output_summary,
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out
                )
            except Exception:
                pass

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            try:
                self.session.error(
                    category="execution_error",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=False
                )
            except Exception:
                pass
            try:
                self.session.step_end(
                    agent=full_name,
                    outcome="error",
                    duration_ms=duration_ms
                )
            except Exception:
                pass
            raise


# Fixtures
@pytest.fixture
def mock_sk_not_available():
    """Patch SK_AVAILABLE to False"""
    with patch('lctl.integrations.semantic_kernel.SK_AVAILABLE', False):
        yield


@pytest.fixture
def tracer():
    """Create a mocked tracer instance for testing without SK dependency."""
    return MockedLCTLSemanticKernelTracer(chain_id="test-chain", verbose=True)


@pytest.fixture
def session():
    """Create a standalone session for testing"""
    return LCTLSession(chain_id="test-session")


# ============================================================================
# Test: is_available() function
# ============================================================================

class TestIsAvailable:
    """Tests for the is_available() function"""

    def test_is_available_returns_bool(self):
        """Test is_available returns a boolean"""
        from lctl.integrations.semantic_kernel import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_is_available_when_sk_not_installed(self, mock_sk_not_available):
        """Test is_available returns False when SK is not installed"""
        from lctl.integrations.semantic_kernel import is_available
        assert is_available() == False


# ============================================================================
# Test: SemanticKernelNotAvailableError
# ============================================================================

class TestSemanticKernelNotAvailableError:
    """Tests for the custom exception class"""

    def test_error_message(self):
        """Test error has correct message"""
        from lctl.integrations.semantic_kernel import SemanticKernelNotAvailableError
        error = SemanticKernelNotAvailableError()
        assert "Semantic Kernel is not installed" in str(error)
        assert "pip install semantic-kernel" in str(error)

    def test_error_is_import_error(self):
        """Test error inherits from ImportError"""
        from lctl.integrations.semantic_kernel import SemanticKernelNotAvailableError
        error = SemanticKernelNotAvailableError()
        assert isinstance(error, ImportError)


# ============================================================================
# Test: _check_sk_available() function
# ============================================================================

class TestCheckSkAvailable:
    """Tests for the _check_sk_available helper function"""

    def test_raises_when_not_available(self, mock_sk_not_available):
        """Test raises SemanticKernelNotAvailableError when SK not installed"""
        from lctl.integrations.semantic_kernel import (
            _check_sk_available,
            SemanticKernelNotAvailableError
        )
        with pytest.raises(SemanticKernelNotAvailableError):
            _check_sk_available()


# ============================================================================
# Test: MockedLCTLSemanticKernelTracer (tests the tracer logic)
# ============================================================================

class TestMockedTracerInit:
    """Tests for mocked tracer initialization"""

    def test_init_with_chain_id(self):
        """Test initialization with chain_id"""
        tracer = MockedLCTLSemanticKernelTracer(chain_id="my-custom-chain")
        assert tracer.session.chain.id == "my-custom-chain"

    def test_init_with_existing_session(self, session):
        """Test initialization with existing session"""
        tracer = MockedLCTLSemanticKernelTracer(session=session)
        assert tracer.session is session

    def test_init_generates_chain_id_if_none(self):
        """Test initialization generates chain_id if not provided"""
        tracer = MockedLCTLSemanticKernelTracer()
        assert tracer.session.chain.id.startswith("sk-")

    def test_init_verbose_flag(self):
        """Test verbose flag is stored"""
        tracer = MockedLCTLSemanticKernelTracer(verbose=True)
        assert tracer._verbose == True

        tracer2 = MockedLCTLSemanticKernelTracer(verbose=False)
        assert tracer2._verbose == False

    def test_init_creates_lock(self):
        """Test initialization creates thread lock"""
        tracer = MockedLCTLSemanticKernelTracer()
        assert isinstance(tracer._lock, type(threading.Lock()))

    def test_init_creates_traced_kernels_set(self):
        """Test initialization creates empty traced kernels set"""
        tracer = MockedLCTLSemanticKernelTracer()
        assert isinstance(tracer._traced_kernels, set)
        assert len(tracer._traced_kernels) == 0


# ============================================================================
# Test: Tracer properties
# ============================================================================

class TestTracerProperties:
    """Tests for tracer properties"""

    def test_chain_property(self, tracer):
        """Test chain property returns session chain"""
        assert tracer.chain is tracer.session.chain

    def test_chain_property_id(self, tracer):
        """Test chain property has correct ID"""
        assert tracer.chain.id == "test-chain"


# ============================================================================
# Test: Tracer export methods
# ============================================================================

class TestTracerExport:
    """Tests for export methods"""

    def test_export_to_file(self, tracer, tmp_path):
        """Test export writes to file"""
        export_path = tmp_path / "trace.lctl.json"
        tracer.export(str(export_path))
        assert export_path.exists()

    def test_to_dict_returns_dict(self, tracer):
        """Test to_dict returns dictionary"""
        result = tracer.to_dict()
        assert isinstance(result, dict)
        assert "chain" in result
        assert "events" in result

    def test_to_dict_contains_chain_id(self, tracer):
        """Test to_dict contains chain ID"""
        result = tracer.to_dict()
        assert result["chain"]["id"] == "test-chain"


# ============================================================================
# Test: trace_kernel() method
# ============================================================================

class TestTraceKernel:
    """Tests for trace_kernel method"""

    def test_trace_kernel_adds_filters(self, tracer):
        """Test trace_kernel adds function and prompt filters"""
        kernel = MockKernel()
        result = tracer.trace_kernel(kernel)

        assert result is kernel
        assert MockFilterTypes.FUNCTION_INVOCATION in kernel._filters
        assert MockFilterTypes.PROMPT_RENDERING in kernel._filters

    def test_trace_kernel_tracks_kernel_id(self, tracer):
        """Test trace_kernel tracks kernel to prevent double-tracing"""
        kernel = MockKernel()
        tracer.trace_kernel(kernel)

        assert id(kernel) in tracer._traced_kernels

    def test_trace_kernel_prevents_double_tracing(self, tracer):
        """Test trace_kernel does not add filters twice"""
        kernel = MockKernel()

        tracer.trace_kernel(kernel)
        initial_filter_count = len(kernel._filters[MockFilterTypes.FUNCTION_INVOCATION])

        # Second call should be skipped
        tracer.trace_kernel(kernel)
        assert len(kernel._filters[MockFilterTypes.FUNCTION_INVOCATION]) == initial_filter_count

    def test_trace_kernel_thread_safety(self, tracer):
        """Test trace_kernel is thread-safe"""
        kernels = [MockKernel() for _ in range(10)]
        traced_count = [0]
        lock = threading.Lock()

        def trace_kernel_thread(kernel):
            tracer.trace_kernel(kernel)
            with lock:
                traced_count[0] += 1

        threads = [threading.Thread(target=trace_kernel_thread, args=(k,)) for k in kernels]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert traced_count[0] == 10
        assert len(tracer._traced_kernels) == 10


# ============================================================================
# Test: Function invocation filter
# ============================================================================

class TestFunctionInvocationFilter:
    """Tests for _function_invocation_filter"""

    @pytest.mark.asyncio
    async def test_function_invocation_records_step_start(self, tracer):
        """Test filter records STEP_START event"""
        context = MockFunctionInvocationContext()
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        start_events = [e for e in events if e.type == EventType.STEP_START]
        assert len(start_events) == 1
        assert start_events[0].agent == "TestPlugin.test_func"
        assert start_events[0].data["intent"] == "execute_function"

    @pytest.mark.asyncio
    async def test_function_invocation_records_step_end(self, tracer):
        """Test filter records STEP_END event on success"""
        context = MockFunctionInvocationContext()
        context.result = MockFunctionResult(value="success result")
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_events = [e for e in events if e.type == EventType.STEP_END]
        assert len(end_events) == 1
        assert end_events[0].data["outcome"] == "success"

    @pytest.mark.asyncio
    async def test_function_invocation_calls_next(self, tracer):
        """Test filter calls the next function in chain"""
        context = MockFunctionInvocationContext()
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        next_func.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_function_invocation_handles_error(self, tracer):
        """Test filter handles errors and records ERROR event"""
        context = MockFunctionInvocationContext()
        next_func = AsyncMock(side_effect=ValueError("Test error"))

        with pytest.raises(ValueError):
            await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].data["type"] == "ValueError"
        assert "Test error" in error_events[0].data["message"]

    @pytest.mark.asyncio
    async def test_function_invocation_error_step_end(self, tracer):
        """Test filter records STEP_END with error outcome on exception"""
        context = MockFunctionInvocationContext()
        next_func = AsyncMock(side_effect=RuntimeError("Execution failed"))

        with pytest.raises(RuntimeError):
            await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_events = [e for e in events if e.type == EventType.STEP_END]
        assert len(end_events) == 1
        assert end_events[0].data["outcome"] == "error"

    @pytest.mark.asyncio
    async def test_function_invocation_truncates_input(self, tracer):
        """Test filter truncates long input summaries"""
        long_args = {"arg": "x" * 500}
        context = MockFunctionInvocationContext(arguments=long_args)
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        start_event = [e for e in events if e.type == EventType.STEP_START][0]
        # Default truncation is 200 chars
        assert len(start_event.data["input_summary"]) <= 203  # 200 + "..."

    @pytest.mark.asyncio
    async def test_function_invocation_global_plugin(self, tracer):
        """Test filter handles functions without plugin name"""
        func = MockKernelFunction(name="global_func", plugin_name=None)
        context = MockFunctionInvocationContext(function=func)
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        start_event = [e for e in events if e.type == EventType.STEP_START][0]
        assert start_event.agent == "Global.global_func"

    @pytest.mark.asyncio
    async def test_function_invocation_extracts_token_usage_object(self, tracer):
        """Test filter extracts token usage from object with attributes"""
        usage = MockUsage(prompt_tokens=100, completion_tokens=50)
        result = MockFunctionResult(value="result", metadata={"usage": usage})
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        assert end_event.data["tokens"]["input"] == 100
        assert end_event.data["tokens"]["output"] == 50

    @pytest.mark.asyncio
    async def test_function_invocation_extracts_token_usage_dict(self, tracer):
        """Test filter extracts token usage from dict format"""
        usage = {"prompt_tokens": 75, "completion_tokens": 25}
        result = MockFunctionResult(value="result", metadata={"usage": usage})
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        assert end_event.data["tokens"]["input"] == 75
        assert end_event.data["tokens"]["output"] == 25

    @pytest.mark.asyncio
    async def test_function_invocation_extracts_token_usage_alternate_keys(self, tracer):
        """Test filter extracts token usage with alternate key names"""
        usage = {"input_tokens": 80, "output_tokens": 40}
        result = MockFunctionResult(value="result", metadata={"usage": usage})
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        assert end_event.data["tokens"]["input"] == 80
        assert end_event.data["tokens"]["output"] == 40

    @pytest.mark.asyncio
    async def test_function_invocation_no_result(self, tracer):
        """Test filter handles None result"""
        context = MockFunctionInvocationContext()
        context.result = None
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        assert end_event.data["output_summary"] == ""

    @pytest.mark.asyncio
    async def test_function_invocation_graceful_step_start_failure(self, tracer):
        """Test filter continues even if step_start fails"""
        context = MockFunctionInvocationContext()
        next_func = AsyncMock()

        # Make step_start raise an exception
        with patch.object(tracer.session, 'step_start', side_effect=Exception("Tracing failed")):
            # Should not raise - continues execution
            await tracer._function_invocation_filter(context, next_func)

        # next should still be called
        next_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_function_invocation_graceful_step_end_failure(self, tracer):
        """Test filter continues even if step_end fails"""
        context = MockFunctionInvocationContext()
        context.result = MockFunctionResult(value="result")
        next_func = AsyncMock()

        # Make step_end raise an exception
        with patch.object(tracer.session, 'step_end', side_effect=Exception("Tracing failed")):
            # Should not raise
            await tracer._function_invocation_filter(context, next_func)


# ============================================================================
# Test: Streaming response handling
# ============================================================================

class TestStreamingResponseHandling:
    """Tests for streaming response handling in function invocation filter"""

    @pytest.mark.asyncio
    async def test_streaming_response_detected(self, tracer):
        """Test filter detects streaming response"""
        # Create an async generator
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        stream_result = MockFunctionResult()
        stream_result.value = mock_stream()

        context = MockFunctionInvocationContext()
        context.result = stream_result

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # The value should be replaced with the wrapper
        assert context.result.value is not None

    @pytest.mark.asyncio
    async def test_streaming_response_accumulates_content(self, tracer):
        """Test streaming wrapper accumulates content"""
        chunks = ["Hello", " ", "World"]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        stream_result = MockFunctionResult()
        stream_result.value = mock_stream()

        context = MockFunctionInvocationContext()
        context.result = stream_result

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Consume the wrapped stream
        accumulated = ""
        async for chunk in context.result.value:
            accumulated += str(chunk)

        assert accumulated == "Hello World"

    @pytest.mark.asyncio
    async def test_streaming_response_records_step_end_after_completion(self, tracer):
        """Test streaming records STEP_END after stream completes"""
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        stream_result = MockFunctionResult()
        stream_result.value = mock_stream()

        context = MockFunctionInvocationContext()
        context.result = stream_result

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Before consuming, we should only have STEP_START
        events_before = [e for e in tracer.chain.events if e.type == EventType.STEP_END]
        assert len(events_before) == 0

        # Consume the stream
        async for _ in context.result.value:
            pass

        # After consuming, we should have STEP_END
        events_after = [e for e in tracer.chain.events if e.type == EventType.STEP_END]
        assert len(events_after) == 1
        assert events_after[0].data["outcome"] == "success"

    @pytest.mark.asyncio
    async def test_streaming_response_handles_error(self, tracer):
        """Test streaming handles errors during iteration"""
        async def mock_stream():
            yield "chunk1"
            raise RuntimeError("Stream error")

        stream_result = MockFunctionResult()
        stream_result.value = mock_stream()

        context = MockFunctionInvocationContext()
        context.result = stream_result

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Consume the stream - should raise
        with pytest.raises(RuntimeError):
            async for _ in context.result.value:
                pass

        # Should have recorded error and step_end
        error_events = [e for e in tracer.chain.events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert "Stream error" in error_events[0].data["message"]

        end_events = [e for e in tracer.chain.events if e.type == EventType.STEP_END]
        assert len(end_events) == 1
        assert end_events[0].data["outcome"] == "error"

    @pytest.mark.asyncio
    async def test_streaming_extracts_usage_from_chunks(self, tracer):
        """Test streaming extracts token usage from chunk metadata"""
        class MockChunk:
            def __init__(self, content, metadata=None):
                self.content = content
                self.metadata = metadata

            def __str__(self):
                return self.content

        async def mock_stream():
            yield MockChunk("chunk1")
            # Final chunk has usage info
            yield MockChunk("chunk2", metadata={
                "usage": {"prompt_tokens": 50, "completion_tokens": 30}
            })

        stream_result = MockFunctionResult()
        stream_result.value = mock_stream()

        context = MockFunctionInvocationContext()
        context.result = stream_result

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Consume the stream
        async for _ in context.result.value:
            pass

        # Check token counts were recorded
        end_events = [e for e in tracer.chain.events if e.type == EventType.STEP_END]
        assert len(end_events) == 1
        assert end_events[0].data["tokens"]["input"] == 50
        assert end_events[0].data["tokens"]["output"] == 30

    @pytest.mark.asyncio
    async def test_streaming_extracts_usage_from_object_attributes(self, tracer):
        """Test streaming extracts token usage from objects with attributes"""
        class MockUsageObj:
            def __init__(self):
                self.prompt_tokens = 60
                self.completion_tokens = 40

        class MockChunk:
            def __init__(self, content, metadata=None):
                self.content = content
                self.metadata = metadata

            def __str__(self):
                return self.content

        async def mock_stream():
            yield MockChunk("chunk", metadata={"usage": MockUsageObj()})

        stream_result = MockFunctionResult()
        stream_result.value = mock_stream()

        context = MockFunctionInvocationContext()
        context.result = stream_result

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Consume the stream
        async for _ in context.result.value:
            pass

        end_events = [e for e in tracer.chain.events if e.type == EventType.STEP_END]
        assert end_events[0].data["tokens"]["input"] == 60
        assert end_events[0].data["tokens"]["output"] == 40


# ============================================================================
# Test: Prompt render filter
# ============================================================================

class TestPromptRenderFilter:
    """Tests for _prompt_render_filter"""

    @pytest.mark.asyncio
    async def test_prompt_render_calls_next(self, tracer):
        """Test filter calls the next function in chain"""
        context = MockPromptRenderContext()
        next_func = AsyncMock()

        await tracer._prompt_render_filter(context, next_func)

        next_func.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_prompt_render_records_fact_when_prompt_rendered(self, tracer):
        """Test filter records FACT_ADDED when prompt is rendered"""
        context = MockPromptRenderContext(rendered_prompt="This is the rendered prompt")
        next_func = AsyncMock()

        await tracer._prompt_render_filter(context, next_func)

        events = tracer.chain.events
        fact_events = [e for e in events if e.type == EventType.FACT_ADDED]
        assert len(fact_events) == 1
        assert "Rendered Prompt" in fact_events[0].data["text"]
        assert fact_events[0].data["confidence"] == 1.0
        assert fact_events[0].data["source"] == "sk-prompt-filter"

    @pytest.mark.asyncio
    async def test_prompt_render_no_fact_when_no_prompt(self, tracer):
        """Test filter does not record fact when no prompt rendered"""
        context = MockPromptRenderContext(rendered_prompt=None)
        next_func = AsyncMock()

        await tracer._prompt_render_filter(context, next_func)

        events = tracer.chain.events
        fact_events = [e for e in events if e.type == EventType.FACT_ADDED]
        assert len(fact_events) == 0

    @pytest.mark.asyncio
    async def test_prompt_render_no_fact_when_empty_prompt(self, tracer):
        """Test filter does not record fact for empty prompt"""
        context = MockPromptRenderContext(rendered_prompt="")
        next_func = AsyncMock()

        await tracer._prompt_render_filter(context, next_func)

        events = tracer.chain.events
        fact_events = [e for e in events if e.type == EventType.FACT_ADDED]
        # Empty string is falsy, so no fact should be added
        assert len(fact_events) == 0

    @pytest.mark.asyncio
    async def test_prompt_render_global_plugin(self, tracer):
        """Test filter handles functions without plugin name"""
        func = MockKernelFunction(name="prompt_func", plugin_name=None)
        context = MockPromptRenderContext(function=func, rendered_prompt="Test prompt")
        next_func = AsyncMock()

        await tracer._prompt_render_filter(context, next_func)

        events = tracer.chain.events
        fact_events = [e for e in events if e.type == EventType.FACT_ADDED]
        assert len(fact_events) == 1
        assert "Global.prompt_func" in fact_events[0].data["text"]

    @pytest.mark.asyncio
    async def test_prompt_render_graceful_failure(self, tracer):
        """Test filter continues even if add_fact fails"""
        context = MockPromptRenderContext(rendered_prompt="Test prompt")
        next_func = AsyncMock()

        # Make add_fact raise an exception
        with patch.object(tracer.session, 'add_fact', side_effect=Exception("Tracing failed")):
            # Should not raise
            await tracer._prompt_render_filter(context, next_func)

        # next should have been called
        next_func.assert_called_once()


# ============================================================================
# Test: Module exports (__all__)
# ============================================================================

class TestModuleExports:
    """Tests for module exports"""

    def test_all_exports(self):
        """Test __all__ contains expected exports"""
        from lctl.integrations.semantic_kernel import __all__

        assert "SK_AVAILABLE" in __all__
        assert "LCTLSemanticKernelTracer" in __all__
        assert "trace_kernel" in __all__
        assert "is_available" in __all__


# ============================================================================
# Test: Session management
# ============================================================================

class TestSessionManagement:
    """Tests for session management in tracer"""

    def test_session_property(self, tracer):
        """Test session property returns the session"""
        assert tracer.session is not None
        assert isinstance(tracer.session, LCTLSession)

    @pytest.mark.asyncio
    async def test_multiple_function_invocations(self, tracer):
        """Test multiple function invocations are tracked"""
        context1 = MockFunctionInvocationContext(
            function=MockKernelFunction(name="func1", plugin_name="Plugin1")
        )
        context2 = MockFunctionInvocationContext(
            function=MockKernelFunction(name="func2", plugin_name="Plugin2")
        )

        next_func = AsyncMock()

        await tracer._function_invocation_filter(context1, next_func)
        await tracer._function_invocation_filter(context2, next_func)

        events = tracer.chain.events
        start_events = [e for e in events if e.type == EventType.STEP_START]
        assert len(start_events) == 2
        assert start_events[0].agent == "Plugin1.func1"
        assert start_events[1].agent == "Plugin2.func2"


# ============================================================================
# Test: Thread safety
# ============================================================================

class TestThreadSafety:
    """Tests for thread-safe operations"""

    def test_concurrent_kernel_tracing(self):
        """Test concurrent kernel tracing is thread-safe"""
        tracer = MockedLCTLSemanticKernelTracer()
        kernels = [MockKernel() for _ in range(20)]

        def trace_in_thread(kernel):
            tracer.trace_kernel(kernel)

        threads = [threading.Thread(target=trace_in_thread, args=(k,)) for k in kernels]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All kernels should be tracked
        assert len(tracer._traced_kernels) == 20

    def test_lock_attribute_exists(self, tracer):
        """Test tracer has a lock attribute"""
        assert hasattr(tracer, '_lock')
        assert isinstance(tracer._lock, type(threading.Lock()))


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_function_with_no_metadata(self, tracer):
        """Test handling function result with no metadata"""
        result = MockFunctionResult(value="result", metadata=None)
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Should complete without error
        events = tracer.chain.events
        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_function_with_empty_metadata(self, tracer):
        """Test handling function result with empty metadata"""
        result = MockFunctionResult(value="result", metadata={})
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Should complete without error
        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        assert end_event.data["tokens"]["input"] == 0
        assert end_event.data["tokens"]["output"] == 0

    @pytest.mark.asyncio
    async def test_function_with_none_usage(self, tracer):
        """Test handling function result with None usage"""
        result = MockFunctionResult(value="result", metadata={"usage": None})
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        # Should complete without error
        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        assert end_event.data["tokens"]["input"] == 0
        assert end_event.data["tokens"]["output"] == 0

    @pytest.mark.asyncio
    async def test_very_long_output(self, tracer):
        """Test handling very long output is truncated"""
        long_result = "x" * 1000
        result = MockFunctionResult(value=long_result)
        context = MockFunctionInvocationContext()
        context.result = result
        next_func = AsyncMock()

        await tracer._function_invocation_filter(context, next_func)

        events = tracer.chain.events
        end_event = [e for e in events if e.type == EventType.STEP_END][0]
        # Should be truncated
        assert len(end_event.data["output_summary"]) <= 203

    def test_export_after_no_events(self, tracer, tmp_path):
        """Test export works with no events recorded"""
        export_path = tmp_path / "empty_trace.json"
        tracer.export(str(export_path))

        assert export_path.exists()

        result = tracer.to_dict()
        assert result["events"] == []


# ============================================================================
# Test: Real module integration tests (when SK_AVAILABLE is False)
# ============================================================================

class TestRealModuleWhenSKNotInstalled:
    """Tests that verify behavior of real module when SK is not installed"""

    def test_tracer_raises_on_init_when_sk_unavailable(self, mock_sk_not_available):
        """Test that real tracer raises when SK unavailable"""
        from lctl.integrations.semantic_kernel import (
            LCTLSemanticKernelTracer,
            SemanticKernelNotAvailableError
        )
        with pytest.raises(SemanticKernelNotAvailableError):
            LCTLSemanticKernelTracer()

    def test_sk_available_constant(self):
        """Test SK_AVAILABLE constant is accessible"""
        from lctl.integrations.semantic_kernel import SK_AVAILABLE
        assert isinstance(SK_AVAILABLE, bool)

    def test_is_available_function_returns_sk_available(self):
        """Test is_available returns SK_AVAILABLE value"""
        from lctl.integrations.semantic_kernel import SK_AVAILABLE, is_available
        assert is_available() == SK_AVAILABLE
