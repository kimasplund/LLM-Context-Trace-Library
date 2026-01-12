
import pytest
from unittest.mock import MagicMock, AsyncMock

# Skip entire module if pydantic_ai not installed
pytest.importorskip("pydantic_ai")

from lctl.integrations.pydantic_ai import trace_agent
from lctl.core.session import LCTLSession

# Mock classes to simulate PydanticAI streaming
class MockUsage:
    def __init__(self, req, resp):
        self.request_tokens = req
        self.response_tokens = resp

class MockStreamResult:
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks
    
    async def stream_text(self, *args, **kwargs):
        for chunk in self.text_chunks:
            yield chunk

    def usage(self):
        return MockUsage(10, 20)

class AsyncContextManagerMock:
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

@pytest.mark.asyncio
async def test_trace_agent_run_stream():
    # Setup
    mock_agent = MagicMock()
    mock_agent.name = "test_agent"
    
    # Mock run_stream to return our mock result context
    mock_result = MockStreamResult(["Hello", " ", "World"])
    mock_agent.run_stream.return_value = AsyncContextManagerMock(mock_result)

    # Trace agent
    session = LCTLSession(chain_id="test-chain")
    # Mock session methods to verify calls
    session.step_start = MagicMock()
    session.step_end = MagicMock()
    
    traced_agent = trace_agent(mock_agent, session=session)

    # Execute
    chunks = []
    async with traced_agent.run_stream("High five") as result:
        async for chunk in result.stream_text():
            chunks.append(chunk)

    # Verify
    assert "".join(chunks) == "Hello World"
    
    # Check start called
    session.step_start.assert_called_with(
        agent="test_agent",
        intent="run_stream",
        input_summary="High five"
    )
    
    # Check end called with usage
    session.step_end.assert_called_with(
        outcome="success",
        output_summary="Hello World",
        tokens_in=10,
        tokens_out=20
    )
