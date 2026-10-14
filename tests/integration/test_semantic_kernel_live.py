"""Live Semantic Kernel smoke tests: a @kernel_function plugin traced via the tracer's filter."""

import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

from lctl.core.events import EventType
from lctl.integrations.semantic_kernel import LCTLSemanticKernelTracer

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _MathPlugin:
    @kernel_function(name="add", description="Add two integers")
    def add(self, a: int, b: int) -> int:
        return a + b

    @kernel_function(name="boom", description="Always fails")
    def boom(self) -> str:
        raise RuntimeError("kaboom")


def _traced_kernel():
    kernel = Kernel()
    kernel.add_plugin(_MathPlugin(), plugin_name="Math")
    tracer = LCTLSemanticKernelTracer(chain_id="sk-live")
    tracer.trace_kernel(kernel)
    return kernel, tracer


async def test_plugin_function_step_is_recorded():
    kernel, tracer = _traced_kernel()
    result = await kernel.invoke(plugin_name="Math", function_name="add", a=2, b=3)

    assert str(result) == "5"
    events = tracer.session.chain.events
    starts = [e for e in events if e.type == EventType.STEP_START]
    ends = [e for e in events if e.type == EventType.STEP_END]
    assert [e.agent for e in starts] == ["Math.add"]
    assert [e.agent for e in ends] == ["Math.add"]
    assert ends[0].data["outcome"] == "success"


async def test_exception_in_function_is_recorded_as_error():
    kernel, tracer = _traced_kernel()
    with pytest.raises(Exception, match="Math-boom"):
        await kernel.invoke(plugin_name="Math", function_name="boom")

    events = tracer.session.chain.events
    errors = [e for e in events if e.type == EventType.ERROR]
    assert errors
    assert "kaboom" in errors[0].data["message"]
    ends = [e for e in events if e.type == EventType.STEP_END]
    assert ends[-1].data["outcome"] == "error"
