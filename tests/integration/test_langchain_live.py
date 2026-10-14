"""Live LangChain smoke tests: a real fake chat model driven through LCTLCallbackHandler."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from lctl.core.events import EventType
from lctl.integrations.langchain import LCTLCallbackHandler, LCTLChain

pytestmark = pytest.mark.integration


def _chain():
    prompt = ChatPromptTemplate.from_messages([("system", "s"), ("human", "{input}")])
    return prompt | FakeListChatModel(responses=["Paris", "Berlin"])


def test_handler_records_balanced_steps_for_a_real_chain():
    handler = LCTLCallbackHandler(chain_id="lc-live")
    result = _chain().invoke({"input": "capital?"}, config={"callbacks": [handler]})

    assert result.content == "Paris"
    events = handler.chain.events
    starts = [e for e in events if e.type == EventType.STEP_START]
    ends = [e for e in events if e.type == EventType.STEP_END]
    assert starts
    assert len(starts) == len(ends)
    assert all(e.data.get("outcome") == "success" for e in ends)
    assert any(e.data.get("intent") == "chat_model_call" for e in starts)
    assert any("Paris" in str(e.data.get("output_summary", "")) for e in ends)


def test_lctlchain_wrapper_invokes_and_records():
    traced = LCTLChain(_chain(), chain_id="lc-wrapper")
    assert traced.invoke({"input": "x"}).content == "Paris"
    assert any(e.type == EventType.STEP_START for e in traced.chain.events)
