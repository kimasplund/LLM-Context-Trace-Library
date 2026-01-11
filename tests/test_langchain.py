"""Tests for LangChain integration."""

from unittest.mock import MagicMock, patch
from uuid import uuid4
import time
import pytest

from lctl.core.events import EventType
from lctl.integrations.langchain import (
    LCTLCallbackHandler,
    LCTLChain,
    trace_chain,
    is_available,
    _truncate,
    _extract_token_counts
)

# Mock classes for Type checking/Import
class MockLLMResult:
    def __init__(self, generations, llm_output=None):
        self.generations = generations
        self.llm_output = llm_output

class MockGeneration:
    def __init__(self, text):
        self.text = text

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockAgentAction:
    def __init__(self, tool, tool_input, log):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

class MockAgentFinish:
    def __init__(self, return_values, log):
        self.return_values = return_values
        self.log = log

@pytest.fixture
def mock_langchain_available():
    """Mock LangChain availability."""
    with patch("lctl.integrations.langchain.LANGCHAIN_AVAILABLE", True):
        yield

@pytest.fixture
def callback_handler(mock_langchain_available):
    """Create a callback handler for testing."""
    return LCTLCallbackHandler(chain_id="test-chain")

def test_handler_init(mock_langchain_available):
    """Test handler initialization."""
    handler = LCTLCallbackHandler(chain_id="my-chain")
    assert handler.session.chain.id == "my-chain"
    assert handler.chain.id == "my-chain"

def test_truncate():
    """Test string truncation helper."""
    assert _truncate("hello", 10) == "hello"
    assert _truncate("hello world", 5) == "he..."

def test_extract_token_counts():
    """Test token count extraction."""
    # Case 1: Standard usage
    resp1 = MagicMock()
    resp1.llm_output = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    assert _extract_token_counts(resp1) == {"input": 10, "output": 20}

    # Case 2: New usage format
    resp2 = MagicMock()
    resp2.llm_output = {"usage": {"input_tokens": 5, "output_tokens": 15}}
    assert _extract_token_counts(resp2) == {"input": 5, "output": 15}

    # Case 3: No usage info
    resp3 = MagicMock()
    resp3.llm_output = {}
    assert _extract_token_counts(resp3) == {"input": 0, "output": 0}

def test_on_llm_start_end(callback_handler):
    """Test LLM start and end callbacks."""
    run_id = uuid4()
    
    # Start
    callback_handler.on_llm_start(
        serialized={"name": "gpt-4"},
        prompts=["Hello world"],
        run_id=run_id
    )
    
    events = callback_handler.session.chain.events
    assert len(events) == 1
    assert events[0].type == EventType.STEP_START
    assert events[0].agent == "gpt-4"
    assert events[0].data["intent"] == "llm_call"

    # End
    generations = [[MockGeneration("Hi there")]]
    result = MockLLMResult(generations, {"token_usage": {"prompt_tokens": 2, "completion_tokens": 2}})
    
    callback_handler.on_llm_end(result, run_id=run_id)
    
    assert len(events) == 2
    assert events[-1].type == EventType.STEP_END
    assert events[-1].data["outcome"] == "success"
    assert events[-1].data["tokens"]["input"] == 2
    assert events[-1].data["tokens"]["output"] == 2

def test_on_llm_error(callback_handler):
    """Test LLM error callback."""
    run_id = uuid4()
    callback_handler.on_llm_start({}, ["prompt"], run_id=run_id)
    
    error = ValueError("Model failure")
    callback_handler.on_llm_error(error, run_id=run_id)
    
    events = callback_handler.session.chain.events
    # step_start, step_end (error), error_event
    
    step_end = [e for e in events if e.type == EventType.STEP_END][0]
    assert step_end.data["outcome"] == "error"
    
    error_event = [e for e in events if e.type == EventType.ERROR][0]
    assert error_event.data["message"] == "Model failure"

def test_on_chain_start_end(callback_handler):
    """Test Chain callbacks."""
    run_id = uuid4()
    
    callback_handler.on_chain_start(
        serialized={"name": "AnalysisChain"},
        inputs={"text": "analyze this"},
        run_id=run_id
    )
    
    events = callback_handler.session.chain.events
    assert callback_handler._chain_depth == 1
    assert events[0].agent == "AnalysisChain"
    
    # Nested chain
    child_id = uuid4()
    callback_handler.on_chain_start(
        serialized={"name": "SubChain"},
        inputs={},
        run_id=child_id
    )
    assert callback_handler._chain_depth == 2
    
    callback_handler.on_chain_end({}, run_id=child_id)
    assert callback_handler._chain_depth == 1
    
    callback_handler.on_chain_end({"output": "result"}, run_id=run_id)
    assert callback_handler._chain_depth == 0

def test_on_tool_usage(callback_handler):
    """Test tool start and end callbacks."""
    run_id = uuid4()
    
    callback_handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="2 + 2",
        run_id=run_id
    )
    
    callback_handler.on_tool_end("4", run_id=run_id)
    
    tool_calls = [e for e in callback_handler.session.chain.events if e.type == EventType.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].data["tool"] == "calculator"
    assert tool_calls[0].data["input"] == "2 + 2"
    assert tool_calls[0].data["output"] == "4"

def test_on_agent_action(callback_handler):
    """Test agent action callback."""
    run_id = uuid4()
    action = MockAgentAction("search", "python", "Searching...")
    
    callback_handler.on_agent_action(action, run_id=run_id)
    
    facts = [e for e in callback_handler.session.chain.events if e.type == EventType.FACT_ADDED]
    assert len(facts) == 1
    assert "Agent action: search" in facts[0].data["text"]

def test_lctl_chain_wrapper(mock_langchain_available):
    """Test LCTLChain wrapper."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Chain Output"
    
    wrapper = LCTLChain(mock_chain, chain_id="wrapped-chain")
    result = wrapper.invoke({"input": "test"})
    
    assert result == "Chain Output"
    # Verify handler was injected
    call_args = mock_chain.invoke.call_args
    assert "config" in call_args.kwargs
    assert wrapper.handler in call_args.kwargs["config"]["callbacks"]

def test_trace_chain_helper(mock_langchain_available):
    """Test trace_chain convenience function."""
    mock_chain = MagicMock()
    wrapper = trace_chain(mock_chain)
    assert isinstance(wrapper, LCTLChain)
