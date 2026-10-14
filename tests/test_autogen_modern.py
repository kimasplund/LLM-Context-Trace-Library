import pytest
from unittest.mock import MagicMock
from lctl.integrations.autogen import LCTLAutogenHandler, LCTLSession
import logging

try:
    from autogen_core import logging as ag_logging

    # Create simple dummy classes if needed, or instantiate if possible
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False


@pytest.mark.skipif(not AUTOGEN_AVAILABLE, reason="autogen-core not installed")
def test_handle_llm_call_event():
    # Setup
    session = MagicMock(spec=LCTLSession)
    handler = LCTLAutogenHandler(session)

    # Create event
    # LLMCallEvent signature: messages, response, prompt_tokens, completion_tokens
    event = ag_logging.LLMCallEvent(
        messages=[{"role": "user", "content": "hello"}],
        response={"choices": [{"message": {"content": "world"}}]},
        prompt_tokens=10,
        completion_tokens=5,
    )

    # Create log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg=event,
        args=(),
        exc_info=None,
    )

    # Emit
    handler.emit(record)

    # Verify
    session.step_start.assert_called_with(
        agent="llm", intent="llm_call", input_summary="1 messages"
    )
    session.step_end.assert_called_with(
        agent="llm",
        outcome="success",
        output_summary="{'choices': [{'message': {'content': 'world'}}]}",
        tokens_in=10,
        tokens_out=5,
    )


@pytest.mark.skipif(not AUTOGEN_AVAILABLE, reason="autogen-core not installed")
def test_handle_tool_call_event():
    session = MagicMock(spec=LCTLSession)
    handler = LCTLAutogenHandler(session)

    event = ag_logging.ToolCallEvent(tool_name="calculator", arguments={"a": 1, "b": 2}, result="3")

    record = logging.LogRecord("test", logging.INFO, __file__, 10, event, (), None)
    handler.emit(record)

    session.tool_call.assert_called_with(
        tool="calculator", input_data="{'a': 1, 'b': 2}", output_data="3", duration_ms=0
    )


@pytest.mark.skipif(not AUTOGEN_AVAILABLE, reason="autogen-core not installed")
def test_handle_message_event():
    session = MagicMock(spec=LCTLSession)
    handler = LCTLAutogenHandler(session)

    # Need generic objects for AgentId if possible or mocks
    # Inspect showed AgentId is a class. Let's try to mock str() behavior or instantiate
    class MockAgentId:
        def __str__(self):
            return "user_proxy"

    class MockReceiverId:
        def __str__(self):
            return "assistant"

    # MessageEvent signature: payload, sender, receiver, kind, delivery_stage
    # We might need actual enum values for kind/delivery_stage if type checking is strict
    # in constructor.
    # But python at runtime might be lenient if we pass mocks unless validated.
    # autogen_core uses Pydantic/dataclasses usually so it might validate type.
    # Let's try using valid enums if imports allow.

    from autogen_core.logging import MessageKind, DeliveryStage

    # Mocks for AgentId might be tricky if it expects specific class.
    # Let's assume we can mock it or use the real one.
    from autogen_core._agent_id import AgentId

    sender = AgentId("user_proxy", "group")
    receiver = AgentId("assistant", "group")

    event = ag_logging.MessageEvent(
        payload="Hello there",
        sender=sender,
        receiver=receiver,
        kind=MessageKind.RESPOND,
        delivery_stage=DeliveryStage.DELIVER,
    )

    record = logging.LogRecord("test", logging.INFO, __file__, 10, event, (), None)
    handler.emit(record)

    session.step_start.assert_called_with(
        agent=str(sender),  # "user_proxy" (roughly)
        intent="send_message",
        input_summary="Hello there",
    )
    session.step_end.assert_called_once()
