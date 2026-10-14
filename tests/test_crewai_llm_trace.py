import uuid
from unittest.mock import MagicMock

import pytest

from lctl.integrations.crewai import CREWAI_AVAILABLE, _agent_id_session_map, _llm_event_handler

# Skip if CrewAI not available (should be available in dev env)
pytestmark = pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")

if CREWAI_AVAILABLE:
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType


def test_crewai_llm_trace_handler():
    """Test that the global event handler routes events to correct session."""

    # Setup mock session
    mock_session = MagicMock()

    # Generate random agent ID
    agent_id = str(uuid.uuid4())

    # Register agent in global map (simulating LCTLCrew.kickoff)
    _agent_id_session_map[agent_id] = mock_session

    try:
        # Create event
        event = LLMCallCompletedEvent(
            messages=[{"role": "user", "content": "test prompt"}],
            response="test completion",
            call_type=LLMCallType.LLM_CALL,
            agent_id=agent_id,
            agent_role="Test Role",
            model="gpt-4",
        )

        # Trigger handler manually (simulating bus emission)
        _llm_event_handler(None, event)

        # Verify llm_trace called
        assert mock_session.llm_trace.called
        call_args = mock_session.llm_trace.call_args[1]

        assert call_args["messages"] == [{"role": "user", "content": "test prompt"}]
        assert call_args["response"] == "test completion"
        assert call_args["model"] == "gpt-4"

    finally:
        # Cleanup
        if agent_id in _agent_id_session_map:
            del _agent_id_session_map[agent_id]


def test_crewai_llm_trace_handler_missing_agent():
    """Test handler ignores events for unknown agents."""
    mock_session = MagicMock()
    agent_id = str(uuid.uuid4())
    # distinct ID
    unknown_id = str(uuid.uuid4())

    _agent_id_session_map[agent_id] = mock_session

    try:
        event = LLMCallCompletedEvent(
            messages=[],
            response="test",
            call_type=LLMCallType.LLM_CALL,
            agent_id=unknown_id,  # UNKNOWN
            agent_role="Test Role",
            model="gpt-4",
        )

        _llm_event_handler(None, event)

        assert not mock_session.llm_trace.called

    finally:
        if agent_id in _agent_id_session_map:
            del _agent_id_session_map[agent_id]
