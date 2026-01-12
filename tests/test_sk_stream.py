
import pytest
import asyncio
from unittest.mock import MagicMock

# Skip entire module if semantic_kernel not installed
semantic_kernel = pytest.importorskip("semantic_kernel")
Kernel = semantic_kernel.Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.filters import FilterTypes

from lctl.core.session import LCTLSession
from lctl.integrations.semantic_kernel import trace_kernel

# Mock Plugin
class StreamingPlugin:
    @kernel_function(name="stream_data")
    async def stream_data(self):
        yield "chunk1"
        yield "chunk2"
        # Simulate a chunk with metadata if possible (hard to mock internals of SK yielding)
        # But we can verify accumulation

@pytest.mark.asyncio
async def test_sk_streaming_trace():
    # Setup
    kernel = Kernel()
    kernel.add_plugin(StreamingPlugin(), plugin_name="TestPlugin")
    
    # Trace
    session = LCTLSession(chain_id="sk-stream-test")
    session.step_start = MagicMock()
    session.step_end = MagicMock()
    
    tracer = trace_kernel(kernel, session=session)
    
    # Execute stream
    result_stream = kernel.invoke_stream(
        function_name="stream_data",
        plugin_name="TestPlugin"
    )
    
    chunks = []
    async for chunk in result_stream:
        chunks.append(str(chunk))
    
    # Verify execution
    assert "".join(chunks) == "chunk1chunk2"
    
    # Verify trace calls
    session.step_start.assert_called_with(
        agent="TestPlugin.stream_data",
        intent="execute_function",
        input_summary="{}" # Arguments empty
    )
    
    session.step_end.assert_called_with(
        outcome="success",
        output_summary="chunk1chunk2",
        tokens_in=0, # Usage 0 for simple non-LLM function
        tokens_out=0
    )
    print("SK Streaming test passed verification")
