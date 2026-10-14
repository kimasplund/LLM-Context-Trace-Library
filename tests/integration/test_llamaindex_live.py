"""Live LlamaIndex smoke tests with MockLLM/MockEmbedding (no API keys)."""

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from lctl.core.events import EventType
from lctl.integrations.llamaindex import (
    LCTLLlamaIndexCallback,
    trace_chat_engine,
    trace_query_engine,
)

pytestmark = [
    pytest.mark.integration,
    # Upstream: llama_index.core.async_utils.asyncio_run calls asyncio.get_event_loop()
    # when building chat memory, which warns on Python 3.12+ with no running loop.
    pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning"),
]


@pytest.fixture
def build_index():
    """Return a factory building an index from the current ``Settings``."""
    saved = (Settings._llm, Settings._embed_model, Settings._callback_manager)
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)

    def build():
        return VectorStoreIndex.from_documents(
            [
                Document(text="Paris is the capital of France."),
                Document(text="Berlin is in Germany."),
            ]
        )

    try:
        yield build
    finally:
        Settings._llm, Settings._embed_model, Settings._callback_manager = saved


def test_handler_constructs():
    cb = LCTLLlamaIndexCallback()
    assert not cb.event_starts_to_ignore
    assert not cb.event_ends_to_ignore


def test_settings_callback_manager_records_query(build_index):
    cb = LCTLLlamaIndexCallback(chain_id="li-live")
    Settings.callback_manager = cb.get_callback_manager()
    # The index (and its retriever) pick up Settings.callback_manager at build time.
    build_index().as_query_engine().query("q")

    events = cb.session.chain.events
    assert any(e.type == EventType.STEP_START and e.agent == "query_engine" for e in events)
    assert any(e.type == EventType.STEP_END and e.agent == "query_engine" for e in events)
    assert any(e.type == EventType.STEP_START and e.agent == "retriever" for e in events)
    assert any(e.type == EventType.TOOL_CALL and e.data["tool"] == "retriever" for e in events)
    assert any(e.type == EventType.LLM_TRACE for e in events)


def test_trace_query_engine_wraps_and_records(build_index):
    traced = trace_query_engine(build_index().as_query_engine(), chain_id="li-q")
    response = traced.query("q")

    assert str(response)
    events = traced.session.chain.events
    assert any(e.agent == "query_engine" for e in events)
    assert any(e.agent == "retriever" for e in events)
    assert any(e.type == EventType.LLM_TRACE for e in events)


def test_trace_chat_engine_wraps_and_records(build_index):
    traced = trace_chat_engine(build_index().as_chat_engine(chat_mode="context"), chain_id="li-c")
    response = traced.chat("hello")

    assert str(response)
    assert len(traced.session.chain.events) > 0
