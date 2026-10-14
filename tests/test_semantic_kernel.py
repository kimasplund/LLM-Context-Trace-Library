import pytest

from lctl.core.events import EventType
from lctl.integrations.semantic_kernel import SK_AVAILABLE, LCTLSemanticKernelTracer, trace_kernel

if SK_AVAILABLE:
    from semantic_kernel.functions.kernel_arguments import KernelArguments
    from semantic_kernel.functions.kernel_function_decorator import kernel_function
    from semantic_kernel.kernel import Kernel


@pytest.mark.skipif(not SK_AVAILABLE, reason="Semantic Kernel not installed")
class TestLCTLSemanticKernelTracer:
    @pytest.fixture
    def kernel(self):
        return Kernel()

    @pytest.fixture
    def tracer(self):
        return LCTLSemanticKernelTracer(verbose=True)

    @pytest.mark.asyncio
    async def test_tracer_creation(self, tracer):
        assert tracer.session is not None
        assert tracer.chain is not None

    @pytest.mark.asyncio
    async def test_trace_kernel_function(self, kernel, tracer):
        # 1. Define a plugin with a native function
        class EchoPlugin:
            @kernel_function(name="echo", description="Echoes input")
            def echo(self, text: str) -> str:
                return f"Echo: {text}"

        # 2. Register plugin
        kernel.add_plugin(EchoPlugin(), plugin_name="TestPlugin")

        # 3. Attach tracer
        tracer.trace_kernel(kernel)

        # 4. Invoke function
        func = kernel.get_function(plugin_name="TestPlugin", function_name="echo")
        result = await kernel.invoke(func, KernelArguments(text="Hello"))

        # 5. Verify result
        assert str(result) == "Echo: Hello"

        # 6. Verify events
        events = tracer.chain.events
        assert len(events) >= 2

        # Check START
        start = events[0]
        assert start.type == EventType.STEP_START
        assert start.agent == "TestPlugin.echo"
        assert "Hello" in str(start.data["input_summary"])

        # Check END
        end = events[-1]
        assert end.type == EventType.STEP_END
        assert end.data["outcome"] == "success"
        assert "Echo: Hello" in end.data["output_summary"]

    @pytest.mark.asyncio
    async def test_trace_kernel_error(self, kernel, tracer):
        from semantic_kernel.exceptions import KernelInvokeException

        # 1. Define a plugin that errors
        class ErrorPlugin:
            @kernel_function(name="fail")
            def fail(self) -> None:
                raise ValueError("Boom!")

        kernel.add_plugin(ErrorPlugin(), plugin_name="ErrorPlugin")
        tracer.trace_kernel(kernel)

        func = kernel.get_function(plugin_name="ErrorPlugin", function_name="fail")

        # 2. Expect error (wrapped in KernelInvokeException)
        with pytest.raises(KernelInvokeException):
            await kernel.invoke(func)

        # 3. Verify events
        events = tracer.chain.events

        # Should have START, ERROR, END
        types = [e.type for e in events]
        assert EventType.STEP_START in types
        assert EventType.ERROR in types
        assert EventType.STEP_END in types

        error_event = [e for e in events if e.type == EventType.ERROR][0]
        # The tracer sees the original ValueError before SK wraps it
        assert error_event.data["type"] == "ValueError"
        assert "Boom!" in error_event.data["message"]

    @pytest.mark.asyncio
    async def test_trace_convenience_function(self, kernel):
        # Test the convenience function wrapper
        tracer = trace_kernel(kernel)
        assert isinstance(tracer, LCTLSemanticKernelTracer)

        # Verify functionality can still run
        class TestPlugin:
            @kernel_function(name="test")
            def test(self) -> str:
                return "ok"

        kernel.add_plugin(TestPlugin(), plugin_name="Test")
        func = kernel.get_function("Test", "test")
        await kernel.invoke(func)

        assert len(tracer.chain.events) > 0
