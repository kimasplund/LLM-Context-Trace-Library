"""Tests for LlamaIndex integration (lctl/integrations/llamaindex.py)."""

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from lctl.core.events import Chain, Event, EventType, ReplayEngine
from lctl.core.session import LCTLSession


class MockCBEventType:
    """Mock LlamaIndex CBEventType enum."""

    LLM = "LLM"
    QUERY = "QUERY"
    RETRIEVE = "RETRIEVE"
    EMBEDDING = "EMBEDDING"
    SYNTHESIZE = "SYNTHESIZE"
    TEMPLATING = "TEMPLATING"
    CHUNKING = "CHUNKING"
    RERANKING = "RERANKING"
    OTHER = "OTHER"

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        return self is other

    @property
    def name(self) -> str:
        return str(self)


MockCBEventType.LLM = MockCBEventType()
MockCBEventType.QUERY = MockCBEventType()
MockCBEventType.RETRIEVE = MockCBEventType()
MockCBEventType.EMBEDDING = MockCBEventType()
MockCBEventType.SYNTHESIZE = MockCBEventType()
MockCBEventType.TEMPLATING = MockCBEventType()
MockCBEventType.CHUNKING = MockCBEventType()
MockCBEventType.RERANKING = MockCBEventType()
MockCBEventType.OTHER = MockCBEventType()


class MockEventPayload:
    """Mock LlamaIndex EventPayload."""

    MESSAGES = "messages"
    PROMPT = "prompt"
    RESPONSE = "response"
    COMPLETION = "completion"
    QUERY_STR = "query_str"
    NODES = "nodes"
    CHUNKS = "chunks"
    TEMPLATE = "template"


class MockCallbackManager:
    """Mock LlamaIndex CallbackManager."""

    def __init__(self, handlers: Optional[List[Any]] = None):
        self.handlers = handlers or []


class MockNode:
    """Mock LlamaIndex Node."""

    def __init__(self, text: str = "Sample node text"):
        self.text = text

    def get_content(self) -> str:
        return self.text


class MockUsage:
    """Mock token usage."""

    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockRaw:
    """Mock raw response with usage."""

    def __init__(self, usage: Optional[MockUsage] = None):
        self.usage = usage or MockUsage()


class MockResponse:
    """Mock LlamaIndex Response."""

    def __init__(self, text: str = "Sample response"):
        self.text = text
        self.response = text
        self.raw = MockRaw()


class MockQueryEngine:
    """Mock LlamaIndex QueryEngine."""

    def __init__(self, response_text: str = "Query response"):
        self.callback_manager: Optional[MockCallbackManager] = None
        self._response_text = response_text

    def query(self, query_str: str, **kwargs: Any) -> MockResponse:
        return MockResponse(self._response_text)

    async def aquery(self, query_str: str, **kwargs: Any) -> MockResponse:
        return MockResponse(self._response_text)


class MockChatEngine:
    """Mock LlamaIndex ChatEngine."""

    def __init__(self, response_text: str = "Chat response"):
        self.callback_manager: Optional[MockCallbackManager] = None
        self._response_text = response_text
        self._history: List[str] = []

    def chat(self, message: str, **kwargs: Any) -> MockResponse:
        self._history.append(message)
        return MockResponse(self._response_text)

    async def achat(self, message: str, **kwargs: Any) -> MockResponse:
        self._history.append(message)
        return MockResponse(self._response_text)

    def stream_chat(self, message: str, **kwargs: Any) -> MockResponse:
        self._history.append(message)
        return MockResponse(self._response_text)

    async def astream_chat(self, message: str, **kwargs: Any) -> MockResponse:
        self._history.append(message)
        return MockResponse(self._response_text)

    def reset(self) -> None:
        self._history = []


mock_llama_index_core_callbacks = MagicMock()
mock_llama_index_core_callbacks.CallbackManager = MockCallbackManager

mock_llama_index_core_callbacks_base_handler = MagicMock()
mock_llama_index_core_callbacks_base_handler.BaseCallbackHandler = object

mock_llama_index_core_callbacks_schema = MagicMock()
mock_llama_index_core_callbacks_schema.CBEventType = MockCBEventType
mock_llama_index_core_callbacks_schema.EventPayload = MockEventPayload

sys.modules["llama_index"] = MagicMock()
sys.modules["llama_index.core"] = MagicMock()
sys.modules["llama_index.core.callbacks"] = mock_llama_index_core_callbacks
sys.modules["llama_index.core.callbacks.base_handler"] = mock_llama_index_core_callbacks_base_handler
sys.modules["llama_index.core.callbacks.schema"] = mock_llama_index_core_callbacks_schema

if "lctl.integrations.llamaindex" in sys.modules:
    del sys.modules["lctl.integrations.llamaindex"]

import lctl.integrations.llamaindex as llamaindex_module
importlib.reload(llamaindex_module)


class TestLlamaIndexAvailability:
    """Tests for LlamaIndex availability checking."""

    def test_is_available_function_exists(self):
        """Test that is_available function exists."""
        assert hasattr(llamaindex_module, "is_available")
        assert callable(llamaindex_module.is_available)

    def test_llamaindex_available(self):
        """Test that LlamaIndex is marked as available after mocking."""
        assert llamaindex_module.LLAMAINDEX_AVAILABLE is True

    def test_llamaindex_not_available_error(self):
        """Test LlamaIndexNotAvailableError is defined."""
        error = llamaindex_module.LlamaIndexNotAvailableError()
        assert "LlamaIndex is not installed" in str(error)


class TestLCTLLlamaIndexCallbackBasics:
    """Tests for basic LCTLLlamaIndexCallback functionality."""

    def test_callback_creation(self):
        """Test basic callback creation."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()
        assert callback.session is not None
        assert callback.chain is not None
        assert len(callback._event_stack) == 0

    def test_callback_creation_with_chain_id(self):
        """Test callback creation with custom chain ID."""
        callback = llamaindex_module.LCTLLlamaIndexCallback(chain_id="custom-chain")
        assert callback.session.chain.id == "custom-chain"

    def test_callback_creation_with_existing_session(self):
        """Test callback creation with existing session."""
        session = LCTLSession(chain_id="existing-session")
        callback = llamaindex_module.LCTLLlamaIndexCallback(session=session)
        assert callback.session is session
        assert callback.session.chain.id == "existing-session"

    def test_get_callback_manager(self):
        """Test get_callback_manager returns a CallbackManager."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()
        manager = callback.get_callback_manager()

        assert isinstance(manager, MockCallbackManager)
        assert callback in manager.handlers


class TestLCTLLlamaIndexCallbackEvents:
    """Tests for callback event handling."""

    def test_on_event_start_llm(self):
        """Test LLM start event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        event_id = callback.on_event_start(
            event_type=MockCBEventType.LLM,
            payload={MockEventPayload.PROMPT: "Test prompt"},
            event_id="event-1",
        )

        assert event_id == "event-1"
        assert "event-1" in callback._event_stack

        step_starts = [
            e for e in callback.chain.events if e.type == EventType.STEP_START
        ]
        assert len(step_starts) == 1
        assert step_starts[0].agent == "llm"
        assert step_starts[0].data["intent"] == "llm_call"

    def test_on_event_end_llm(self):
        """Test LLM end event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.LLM,
            payload={MockEventPayload.PROMPT: "Test prompt"},
            event_id="event-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.LLM,
            payload={MockEventPayload.RESPONSE: MockResponse("LLM response")},
            event_id="event-1",
        )

        assert "event-1" not in callback._event_stack

        step_ends = [
            e for e in callback.chain.events if e.type == EventType.STEP_END
        ]
        assert len(step_ends) == 1
        assert step_ends[0].data["outcome"] == "success"

    def test_on_event_start_query(self):
        """Test query start event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.QUERY,
            payload={MockEventPayload.QUERY_STR: "What is AI?"},
            event_id="query-1",
        )

        assert callback._query_depth == 1

        step_starts = [
            e for e in callback.chain.events if e.type == EventType.STEP_START
        ]
        assert len(step_starts) == 1
        assert step_starts[0].agent == "query_engine"
        assert "What is AI?" in step_starts[0].data["input_summary"]

    def test_on_event_end_query(self):
        """Test query end event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.QUERY,
            payload={MockEventPayload.QUERY_STR: "What is AI?"},
            event_id="query-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.QUERY,
            payload={MockEventPayload.RESPONSE: MockResponse("AI is...")},
            event_id="query-1",
        )

        assert callback._query_depth == 0

    def test_on_event_start_retrieve(self):
        """Test retrieval start event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.RETRIEVE,
            payload={MockEventPayload.QUERY_STR: "search query"},
            event_id="retrieve-1",
        )

        step_starts = [
            e for e in callback.chain.events if e.type == EventType.STEP_START
        ]
        assert len(step_starts) == 1
        assert step_starts[0].agent == "retriever"

    def test_on_event_end_retrieve_with_nodes(self):
        """Test retrieval end event with nodes."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.RETRIEVE,
            payload={MockEventPayload.QUERY_STR: "search query"},
            event_id="retrieve-1",
        )

        nodes = [MockNode("Node 1 text"), MockNode("Node 2 text")]
        callback.on_event_end(
            event_type=MockCBEventType.RETRIEVE,
            payload={MockEventPayload.NODES: nodes},
            event_id="retrieve-1",
        )

        tool_calls = [
            e for e in callback.chain.events if e.type == EventType.TOOL_CALL
        ]
        assert len(tool_calls) == 1
        assert "2 nodes" in tool_calls[0].data["output"]

        fact_events = [
            e for e in callback.chain.events if e.type == EventType.FACT_ADDED
        ]
        assert len(fact_events) == 2

    def test_on_event_embedding(self):
        """Test embedding event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.EMBEDDING,
            payload={},
            event_id="embed-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.EMBEDDING,
            payload={MockEventPayload.CHUNKS: ["chunk1", "chunk2", "chunk3"]},
            event_id="embed-1",
        )

        step_ends = [
            e for e in callback.chain.events if e.type == EventType.STEP_END
        ]
        assert len(step_ends) == 1
        assert "3 chunks" in step_ends[0].data["output_summary"]

    def test_on_event_synthesize(self):
        """Test synthesize event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.SYNTHESIZE,
            payload={MockEventPayload.QUERY_STR: "synthesize query"},
            event_id="synth-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.SYNTHESIZE,
            payload={MockEventPayload.RESPONSE: MockResponse("Synthesized answer")},
            event_id="synth-1",
        )

        step_starts = [
            e for e in callback.chain.events if e.type == EventType.STEP_START
        ]
        step_ends = [
            e for e in callback.chain.events if e.type == EventType.STEP_END
        ]

        assert any(s.agent == "synthesizer" for s in step_starts)
        assert any(s.agent == "synthesizer" for s in step_ends)

    def test_on_event_reranking(self):
        """Test reranking event handling."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.RERANKING,
            payload={MockEventPayload.QUERY_STR: "rerank query"},
            event_id="rerank-1",
        )

        nodes = [MockNode("text1"), MockNode("text2")]
        callback.on_event_end(
            event_type=MockCBEventType.RERANKING,
            payload={MockEventPayload.NODES: nodes},
            event_id="rerank-1",
        )

        step_starts = [
            e for e in callback.chain.events if e.type == EventType.STEP_START
        ]
        step_ends = [
            e for e in callback.chain.events if e.type == EventType.STEP_END
        ]

        assert any(s.agent == "reranker" for s in step_starts)
        assert any("2 nodes" in s.data.get("output_summary", "") for s in step_ends)

    def test_on_event_unknown_type(self):
        """Test handling of unknown event types."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.on_event_start(
            event_type=MockCBEventType.OTHER,
            payload={},
            event_id="other-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.OTHER,
            payload={},
            event_id="other-1",
        )

        step_starts = [
            e for e in callback.chain.events if e.type == EventType.STEP_START
        ]
        step_ends = [
            e for e in callback.chain.events if e.type == EventType.STEP_END
        ]

        assert len(step_starts) >= 1
        assert len(step_ends) >= 1


class TestLCTLLlamaIndexCallbackExport:
    """Tests for export functionality."""

    def test_export_json(self, tmp_path: Path):
        """Test exporting callback to JSON file."""
        callback = llamaindex_module.LCTLLlamaIndexCallback(chain_id="export-test")

        callback.on_event_start(
            event_type=MockCBEventType.LLM,
            payload={},
            event_id="event-1",
        )
        callback.on_event_end(
            event_type=MockCBEventType.LLM,
            payload={},
            event_id="event-1",
        )

        export_path = tmp_path / "export.lctl.json"
        callback.export(str(export_path))

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "export-test"
        assert len(data["events"]) >= 2

    def test_to_dict(self):
        """Test to_dict returns chain dictionary."""
        callback = llamaindex_module.LCTLLlamaIndexCallback(chain_id="dict-test")

        result = callback.to_dict()
        assert result["chain"]["id"] == "dict-test"
        assert "events" in result


class TestLCTLQueryEngine:
    """Tests for LCTLQueryEngine wrapper."""

    def test_query_engine_creation(self):
        """Test basic query engine wrapper creation."""
        engine = MockQueryEngine()
        traced = llamaindex_module.LCTLQueryEngine(engine)

        assert traced.query_engine is engine
        assert traced.session is not None
        assert traced.callback is not None

    def test_query_engine_with_chain_id(self):
        """Test query engine with custom chain ID."""
        engine = MockQueryEngine()
        traced = llamaindex_module.LCTLQueryEngine(engine, chain_id="custom-query")

        assert traced.session.chain.id == "custom-query"

    def test_query_engine_query(self):
        """Test query method."""
        engine = MockQueryEngine(response_text="Test answer")
        traced = llamaindex_module.LCTLQueryEngine(engine)

        response = traced.query("What is AI?")

        assert response.text == "Test answer"

    def test_query_engine_export(self, tmp_path: Path):
        """Test query engine export."""
        engine = MockQueryEngine()
        traced = llamaindex_module.LCTLQueryEngine(engine, chain_id="export-query")

        traced.query("test query")

        export_path = tmp_path / "query.lctl.json"
        traced.export(str(export_path))

        assert export_path.exists()

    def test_query_engine_to_dict(self):
        """Test query engine to_dict."""
        engine = MockQueryEngine()
        traced = llamaindex_module.LCTLQueryEngine(engine, chain_id="dict-query")

        result = traced.to_dict()
        assert result["chain"]["id"] == "dict-query"


class TestLCTLChatEngine:
    """Tests for LCTLChatEngine wrapper."""

    def test_chat_engine_creation(self):
        """Test basic chat engine wrapper creation."""
        engine = MockChatEngine()
        traced = llamaindex_module.LCTLChatEngine(engine)

        assert traced.chat_engine is engine
        assert traced.session is not None
        assert traced.callback is not None

    def test_chat_engine_with_chain_id(self):
        """Test chat engine with custom chain ID."""
        engine = MockChatEngine()
        traced = llamaindex_module.LCTLChatEngine(engine, chain_id="custom-chat")

        assert traced.session.chain.id == "custom-chat"

    def test_chat_engine_chat(self):
        """Test chat method."""
        engine = MockChatEngine(response_text="Hello back!")
        traced = llamaindex_module.LCTLChatEngine(engine)

        response = traced.chat("Hello!")

        assert response.text == "Hello back!"

    def test_chat_engine_reset(self):
        """Test chat reset method."""
        engine = MockChatEngine()
        traced = llamaindex_module.LCTLChatEngine(engine)

        traced.chat("Hello!")
        assert len(engine._history) == 1

        traced.reset()
        assert len(engine._history) == 0

    def test_chat_engine_export(self, tmp_path: Path):
        """Test chat engine export."""
        engine = MockChatEngine()
        traced = llamaindex_module.LCTLChatEngine(engine, chain_id="export-chat")

        traced.chat("test message")

        export_path = tmp_path / "chat.lctl.json"
        traced.export(str(export_path))

        assert export_path.exists()

    def test_chat_engine_to_dict(self):
        """Test chat engine to_dict."""
        engine = MockChatEngine()
        traced = llamaindex_module.LCTLChatEngine(engine, chain_id="dict-chat")

        result = traced.to_dict()
        assert result["chain"]["id"] == "dict-chat"


class TestTraceQueryEngineFunction:
    """Tests for trace_query_engine helper function."""

    def test_trace_query_engine_returns_wrapper(self):
        """Test trace_query_engine returns a wrapper."""
        engine = MockQueryEngine()
        traced = llamaindex_module.trace_query_engine(engine)

        assert isinstance(traced, llamaindex_module.LCTLQueryEngine)
        assert traced.query_engine is engine

    def test_trace_query_engine_with_chain_id(self):
        """Test trace_query_engine with custom chain ID."""
        engine = MockQueryEngine()
        traced = llamaindex_module.trace_query_engine(engine, chain_id="custom-trace")

        assert traced.session.chain.id == "custom-trace"


class TestTraceChatEngineFunction:
    """Tests for trace_chat_engine helper function."""

    def test_trace_chat_engine_returns_wrapper(self):
        """Test trace_chat_engine returns a wrapper."""
        engine = MockChatEngine()
        traced = llamaindex_module.trace_chat_engine(engine)

        assert isinstance(traced, llamaindex_module.LCTLChatEngine)
        assert traced.chat_engine is engine

    def test_trace_chat_engine_with_chain_id(self):
        """Test trace_chat_engine with custom chain ID."""
        engine = MockChatEngine()
        traced = llamaindex_module.trace_chat_engine(engine, chain_id="custom-chat-trace")

        assert traced.session.chain.id == "custom-chat-trace"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        """Test truncate with short text."""
        text = "Short text"
        result = llamaindex_module._truncate(text, max_length=200)
        assert result == text

    def test_truncate_long_text(self):
        """Test truncate with long text."""
        text = "A" * 300
        result = llamaindex_module._truncate(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_get_event_name_with_name_attr(self):
        """Test _get_event_name with name attribute."""

        class MockEvent:
            name = "TEST_EVENT"

        result = llamaindex_module._get_event_name(MockEvent())
        assert result == "test_event"

    def test_get_event_name_without_name_attr(self):
        """Test _get_event_name without name attribute."""
        result = llamaindex_module._get_event_name("STRING_EVENT")
        assert result == "string_event"

    def test_get_event_name_none(self):
        """Test _get_event_name with None."""
        result = llamaindex_module._get_event_name(None)
        assert result == "unknown"


class TestIntegration:
    """Integration tests for the LlamaIndex integration."""

    def test_full_query_workflow(self, tmp_path: Path):
        """Test complete query workflow."""
        callback = llamaindex_module.LCTLLlamaIndexCallback(chain_id="integration-test")

        callback.on_event_start(
            event_type=MockCBEventType.QUERY,
            payload={MockEventPayload.QUERY_STR: "What is machine learning?"},
            event_id="query-1",
        )

        callback.on_event_start(
            event_type=MockCBEventType.RETRIEVE,
            payload={MockEventPayload.QUERY_STR: "machine learning"},
            event_id="retrieve-1",
            parent_id="query-1",
        )

        nodes = [MockNode("ML is a subset of AI"), MockNode("ML uses data")]
        callback.on_event_end(
            event_type=MockCBEventType.RETRIEVE,
            payload={MockEventPayload.NODES: nodes},
            event_id="retrieve-1",
        )

        callback.on_event_start(
            event_type=MockCBEventType.LLM,
            payload={MockEventPayload.PROMPT: "Answer based on context"},
            event_id="llm-1",
            parent_id="query-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.LLM,
            payload={MockEventPayload.RESPONSE: MockResponse("ML is...")},
            event_id="llm-1",
        )

        callback.on_event_end(
            event_type=MockCBEventType.QUERY,
            payload={MockEventPayload.RESPONSE: MockResponse("Machine learning is...")},
            event_id="query-1",
        )

        export_path = tmp_path / "integration.lctl.json"
        callback.export(str(export_path))

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "integration-test"
        assert len(data["events"]) >= 6

        engine = ReplayEngine(callback.chain)
        state = engine.replay_all()
        assert state.metrics["event_count"] >= 6

    def test_chat_workflow(self, tmp_path: Path):
        """Test chat workflow with multiple messages."""
        engine = MockChatEngine()
        traced = llamaindex_module.LCTLChatEngine(engine, chain_id="chat-workflow")

        traced.chat("Hello!")
        traced.chat("How are you?")
        traced.chat("Tell me a joke")

        assert len(engine._history) == 3

        export_path = tmp_path / "chat_workflow.lctl.json"
        traced.export(str(export_path))

        assert export_path.exists()

    def test_trace_interface_methods(self):
        """Test start_trace and end_trace interface methods."""
        callback = llamaindex_module.LCTLLlamaIndexCallback()

        callback.start_trace("trace-1")
        callback.end_trace("trace-1", {"trace-1": []})


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_exist(self):
        """Test all items in __all__ exist in module."""
        for name in llamaindex_module.__all__:
            assert hasattr(llamaindex_module, name), f"Missing export: {name}"

    def test_expected_exports(self):
        """Test expected items are exported."""
        expected = [
            "LLAMAINDEX_AVAILABLE",
            "LlamaIndexNotAvailableError",
            "LCTLLlamaIndexCallback",
            "LCTLQueryEngine",
            "LCTLChatEngine",
            "trace_query_engine",
            "trace_chat_engine",
            "is_available",
        ]
        for name in expected:
            assert name in llamaindex_module.__all__, f"Missing in __all__: {name}"
