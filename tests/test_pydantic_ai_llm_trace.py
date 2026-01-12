
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Skip entire module if pydantic_ai not installed
pydantic_ai = pytest.importorskip("pydantic_ai")
Agent = pydantic_ai.Agent
RunContext = pydantic_ai.RunContext
from pydantic_ai.models.test import TestModel

from lctl.integrations.pydantic_ai import trace_agent
from lctl.core.events import EventType

@pytest.fixture
def mock_session():
    session = MagicMock()
    session.chain = MagicMock()
    session.chain.id = "test-chain"
    # Mock step_start/end to return a sequence number
    session.step_start.return_value = 1
    session.step_end.return_value = 2
    session.llm_trace.return_value = 3
    return session

@pytest.mark.asyncio
async def test_pydantic_ai_llm_trace(mock_session):
    """Test that llm_trace is called with messages."""
    
    # Setup agent with TestModel which generates real messages internally
    agent = Agent(TestModel(), deps_type=str)
    
    traced_agent = trace_agent(agent, session=mock_session)
    
    # Run agent
    await traced_agent.run("Hello", deps="test_deps")
    
    # Verify llm_trace was called
    assert mock_session.llm_trace.called
    call_args = mock_session.llm_trace.call_args[1]
    
    assert "messages" in call_args
    assert isinstance(call_args["messages"], list)
    assert len(call_args["messages"]) > 0
    # TestModel usually produces ModelRequest and ModelResponse
    assert "response" in call_args
    assert "usage" in call_args

@pytest.mark.asyncio
async def test_pydantic_ai_stream_llm_trace(mock_session):
    """Test that llm_trace is called after streaming."""
    
    agent = Agent(TestModel())
    traced_agent = trace_agent(agent, session=mock_session)
    
    async with traced_agent.run_stream("Hello Stream") as result:
        async for chunk in result.stream_text():
            pass
            
    # Verify llm_trace was called
    assert mock_session.llm_trace.called
    call_args = mock_session.llm_trace.call_args[1]
    
    assert "messages" in call_args
    assert len(call_args["messages"]) > 0
