"""Tests for AutoGen/AG2 integration (lctl/integrations/autogen.py)."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from lctl.core.events import Chain, Event, EventType, ReplayEngine
from lctl.core.session import LCTLSession

sys.modules["autogen"] = MagicMock()
sys.modules["ag2"] = MagicMock()


class MockAgent:
    """Mock AutoGen agent for testing."""

    def __init__(self, name: str = "mock_agent"):
        self.name = name
        self._hooks: Dict[str, List[Any]] = {}

    def register_hook(self, hook_name: str, hook_func: Any) -> None:
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(hook_func)

    def get_hooks(self, hook_name: str) -> List[Any]:
        return self._hooks.get(hook_name, [])


class MockGroupChat:
    """Mock AutoGen GroupChat for testing."""

    def __init__(self, agents: List[MockAgent], max_round: int = 10):
        self.agents = agents
        self.max_round = max_round
        self.messages: List[Dict[str, Any]] = []


class MockGroupChatManager(MockAgent):
    """Mock AutoGen GroupChatManager for testing."""

    def __init__(self, groupchat: MockGroupChat, name: str = "chat_manager"):
        super().__init__(name)
        self.groupchat = groupchat


@pytest.fixture
def mock_autogen():
    """Fixture to mock AutoGen imports."""
    with patch.dict(
        "sys.modules",
        {
            "autogen": MagicMock(
                Agent=MockAgent,
                ConversableAgent=MockAgent,
                GroupChat=MockGroupChat,
                GroupChatManager=MockGroupChatManager,
            ),
        },
    ):
        yield


class TestAutogenAvailability:
    """Tests for AutoGen availability checking."""

    def test_is_available_function_exists(self):
        """Test that is_available function exists."""
        import lctl.integrations.autogen as autogen_module

        assert hasattr(autogen_module, "is_available")
        assert callable(autogen_module.is_available)

    def test_autogen_not_available_error(self):
        """Test AutogenNotAvailableError is defined."""
        import lctl.integrations.autogen as autogen_module

        error = autogen_module.AutogenNotAvailableError()
        assert "AutoGen is not installed" in str(error)


class TestLCTLAutogenCallbackBasics:
    """Tests for basic LCTLAutogenCallback functionality."""

    def test_callback_creation(self, mock_autogen):
        """Test basic callback creation."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            assert callback.session is not None
            assert callback.chain is not None
            assert len(callback._attached_agents) == 0

    def test_callback_creation_with_chain_id(self, mock_autogen):
        """Test callback creation with custom chain ID."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback(chain_id="custom-chain")
            assert callback.session.chain.id == "custom-chain"

    def test_callback_creation_with_existing_session(self, mock_autogen):
        """Test callback creation with existing session."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            session = LCTLSession(chain_id="existing-session")
            callback = LCTLAutogenCallback(session=session)
            assert callback.session is session
            assert callback.session.chain.id == "existing-session"


class TestLCTLAutogenCallbackAttach:
    """Tests for attaching callbacks to agents."""

    def test_attach_agent(self, mock_autogen):
        """Test attaching callback to an agent."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            agent = MockAgent(name="test_agent")

            callback.attach(agent)

            assert agent in callback._attached_agents
            assert len(agent.get_hooks("process_message_before_send")) == 1
            assert len(agent.get_hooks("process_all_messages_before_reply")) == 1
            assert len(agent.get_hooks("update_agent_state")) == 1

    def test_attach_agent_twice_no_duplicate(self, mock_autogen):
        """Test that attaching same agent twice doesn't duplicate hooks."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            agent = MockAgent(name="test_agent")

            callback.attach(agent)
            callback.attach(agent)

            assert callback._attached_agents.count(agent) == 1
            assert len(agent.get_hooks("process_message_before_send")) == 1

    def test_attach_records_fact(self, mock_autogen):
        """Test that attaching agent records a fact."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            agent = MockAgent(name="my_agent")

            callback.attach(agent)

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            assert len(fact_events) == 1
            assert "my_agent" in fact_events[0].data["text"]


class TestLCTLAutogenCallbackGroupChat:
    """Tests for GroupChat tracing."""

    def test_attach_group_chat(self, mock_autogen):
        """Test attaching callback to a GroupChat."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ), patch(
            "lctl.integrations.autogen.GroupChat", MockGroupChat
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            agent1 = MockAgent(name="agent1")
            agent2 = MockAgent(name="agent2")
            group_chat = MockGroupChat(agents=[agent1, agent2], max_round=5)

            callback.attach_group_chat(group_chat)

            assert agent1 in callback._attached_agents
            assert agent2 in callback._attached_agents

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            config_fact = [
                e for e in fact_events if "groupchat-config" in e.data.get("id", "")
            ]
            assert len(config_fact) == 1
            assert "2 agents" in config_fact[0].data["text"]

    def test_attach_group_chat_with_manager(self, mock_autogen):
        """Test attaching GroupChat with manager."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ), patch(
            "lctl.integrations.autogen.GroupChat", MockGroupChat
        ), patch(
            "lctl.integrations.autogen.GroupChatManager", MockGroupChatManager
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            agent1 = MockAgent(name="agent1")
            agent2 = MockAgent(name="agent2")
            group_chat = MockGroupChat(agents=[agent1, agent2])
            manager = MockGroupChatManager(groupchat=group_chat)

            callback.attach_group_chat(group_chat, manager)

            assert manager in callback._attached_agents


class TestLCTLAutogenCallbackNestedChats:
    """Tests for nested conversation tracking."""

    def test_start_nested_chat(self, mock_autogen):
        """Test starting a nested chat."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            callback.start_nested_chat("parent_agent", "Testing nested chat")

            assert callback._nested_depth == 1
            assert len(callback._conversation_stack) == 1

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            assert len(step_starts) == 1
            assert step_starts[0].agent == "parent_agent"
            assert step_starts[0].data["intent"] == "start_nested_chat"

    def test_end_nested_chat(self, mock_autogen):
        """Test ending a nested chat."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            callback.start_nested_chat("parent_agent", "Testing nested chat")
            callback.end_nested_chat("Completed successfully")

            assert callback._nested_depth == 0
            assert len(callback._conversation_stack) == 0

            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_ends) == 1
            assert step_ends[0].data["outcome"] == "success"

    def test_nested_chat_depth_tracking(self, mock_autogen):
        """Test nested chat depth is tracked correctly."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            callback.start_nested_chat("level1", "First level")
            assert callback._nested_depth == 1

            callback.start_nested_chat("level2", "Second level")
            assert callback._nested_depth == 2

            callback.end_nested_chat("Level 2 done")
            assert callback._nested_depth == 1

            callback.end_nested_chat("Level 1 done")
            assert callback._nested_depth == 0

    def test_end_nested_chat_without_start(self, mock_autogen):
        """Test ending nested chat without start is safe."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            callback.end_nested_chat()

            assert callback._nested_depth == 0


class TestLCTLAutogenCallbackToolTracking:
    """Tests for tool call tracking."""

    def test_record_tool_result(self, mock_autogen):
        """Test recording a tool result."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            callback.record_tool_result(
                tool_name="calculator",
                result={"answer": 42},
                duration_ms=100,
                agent="math_agent",
            )

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            assert len(fact_events) == 1
            assert "calculator" in fact_events[0].data["text"]
            assert "42" in fact_events[0].data["text"]


class TestLCTLAutogenCallbackErrorHandling:
    """Tests for error handling."""

    def test_record_error(self, mock_autogen):
        """Test recording an error."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()

            callback.record_error(
                error=ValueError("Test error"),
                agent="failing_agent",
                recoverable=True,
            )

            error_events = [
                e for e in callback.chain.events if e.type == EventType.ERROR
            ]
            assert len(error_events) == 1
            assert error_events[0].data["type"] == "ValueError"
            assert error_events[0].data["message"] == "Test error"
            assert error_events[0].data["recoverable"] is True


class TestLCTLAutogenCallbackExport:
    """Tests for export functionality."""

    def test_export_json(self, tmp_path: Path, mock_autogen):
        """Test exporting callback to JSON file."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback(chain_id="export-test")
            agent = MockAgent(name="test_agent")
            callback.attach(agent)

            export_path = tmp_path / "export.lctl.json"
            callback.export(str(export_path))

            assert export_path.exists()
            data = json.loads(export_path.read_text())
            assert data["chain"]["id"] == "export-test"
            assert len(data["events"]) >= 1

    def test_to_dict(self, mock_autogen):
        """Test to_dict returns chain dictionary."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback(chain_id="dict-test")

            result = callback.to_dict()
            assert result["chain"]["id"] == "dict-test"
            assert "events" in result


class TestLCTLAutogenCallbackHooks:
    """Tests for hook functionality."""

    def test_before_send_hook_creates_events(self, mock_autogen):
        """Test that before_send hook creates appropriate events."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            agent = MockAgent(name="sender")
            callback.attach(agent)

            hooks = agent.get_hooks("process_message_before_send")
            assert len(hooks) == 1

            recipient = MockAgent(name="recipient")
            message = "Hello, recipient!"

            result = hooks[0](agent, message, recipient, False)

            assert result == message

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]

            send_starts = [s for s in step_starts if s.data.get("intent") == "send_message"]
            assert len(send_starts) >= 1

    def test_before_send_hook_with_tool_call(self, mock_autogen):
        """Test that before_send hook handles tool calls."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            agent = MockAgent(name="sender")
            callback.attach(agent)

            hooks = agent.get_hooks("process_message_before_send")
            recipient = MockAgent(name="recipient")

            message = {
                "content": "Let me calculate that",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": '{"a": 1, "b": 2}',
                        }
                    }
                ],
            }

            hooks[0](agent, message, recipient, False)

            tool_events = [
                e for e in callback.chain.events if e.type == EventType.TOOL_CALL
            ]
            assert len(tool_events) == 1
            assert tool_events[0].data["tool"] == "calculator"

    def test_before_reply_hook_processes_messages(self, mock_autogen):
        """Test that before_reply hook processes messages."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback()
            agent = MockAgent(name="responder")
            callback.attach(agent)

            hooks = agent.get_hooks("process_all_messages_before_reply")
            assert len(hooks) == 1

            messages = [
                {"role": "user", "name": "user1", "content": "Hello!"},
                {"role": "assistant", "name": "assistant1", "content": "Hi there!"},
            ]

            result = hooks[0](messages)

            assert result == messages

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            reply_starts = [
                s for s in step_starts if s.data.get("intent") == "generate_reply"
            ]
            assert len(reply_starts) >= 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        """Test truncate with short text."""
        from lctl.integrations.autogen import _truncate

        text = "Short text"
        result = _truncate(text, max_length=200)
        assert result == text

    def test_truncate_long_text(self):
        """Test truncate with long text."""
        from lctl.integrations.autogen import _truncate

        text = "A" * 300
        result = _truncate(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_extract_message_content_string(self):
        """Test extract_message_content with string."""
        from lctl.integrations.autogen import _extract_message_content

        result = _extract_message_content("Hello")
        assert result == "Hello"

    def test_extract_message_content_dict(self):
        """Test extract_message_content with dict."""
        from lctl.integrations.autogen import _extract_message_content

        result = _extract_message_content({"content": "Hello from dict"})
        assert result == "Hello from dict"

    def test_extract_message_content_dict_with_list(self):
        """Test extract_message_content with dict containing list content."""
        from lctl.integrations.autogen import _extract_message_content

        result = _extract_message_content(
            {"content": [{"text": "Part 1"}, {"text": "Part 2"}]}
        )
        assert "Part 1" in result
        assert "Part 2" in result

    def test_extract_message_content_none(self):
        """Test extract_message_content with None."""
        from lctl.integrations.autogen import _extract_message_content

        result = _extract_message_content(None)
        assert result == ""

    def test_get_agent_name_with_name(self):
        """Test get_agent_name with named agent."""
        from lctl.integrations.autogen import _get_agent_name

        agent = MockAgent(name="My Agent")
        result = _get_agent_name(agent)
        assert result == "my-agent"

    def test_get_agent_name_without_name(self):
        """Test get_agent_name with unnamed agent."""
        from lctl.integrations.autogen import _get_agent_name

        class NoNameAgent:
            pass

        agent = NoNameAgent()
        result = _get_agent_name(agent)
        assert result == "unknown-agent"


class TestTraceAgentFunction:
    """Tests for trace_agent helper function."""

    def test_trace_agent_returns_callback(self, mock_autogen):
        """Test trace_agent returns a callback."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import trace_agent

            agent = MockAgent(name="test_agent")
            callback = trace_agent(agent)

            assert callback is not None
            assert agent in callback._attached_agents

    def test_trace_agent_with_chain_id(self, mock_autogen):
        """Test trace_agent with custom chain ID."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import trace_agent

            agent = MockAgent(name="test_agent")
            callback = trace_agent(agent, chain_id="custom-trace")

            assert callback.session.chain.id == "custom-trace"


class TestTraceGroupChatFunction:
    """Tests for trace_group_chat helper function."""

    def test_trace_group_chat_returns_callback(self, mock_autogen):
        """Test trace_group_chat returns a callback."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ), patch(
            "lctl.integrations.autogen.GroupChat", MockGroupChat
        ):
            from lctl.integrations.autogen import trace_group_chat

            agent1 = MockAgent(name="agent1")
            agent2 = MockAgent(name="agent2")
            group_chat = MockGroupChat(agents=[agent1, agent2])

            callback = trace_group_chat(group_chat)

            assert callback is not None
            assert agent1 in callback._attached_agents
            assert agent2 in callback._attached_agents

    def test_trace_group_chat_with_manager(self, mock_autogen):
        """Test trace_group_chat with manager."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ), patch(
            "lctl.integrations.autogen.GroupChat", MockGroupChat
        ), patch(
            "lctl.integrations.autogen.GroupChatManager", MockGroupChatManager
        ):
            from lctl.integrations.autogen import trace_group_chat

            agent1 = MockAgent(name="agent1")
            group_chat = MockGroupChat(agents=[agent1])
            manager = MockGroupChatManager(groupchat=group_chat)

            callback = trace_group_chat(group_chat, manager)

            assert manager in callback._attached_agents


class TestIntegration:
    """Integration tests for the AutoGen integration."""

    def test_full_workflow(self, tmp_path: Path, mock_autogen):
        """Test complete workflow: create, attach, record, export."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback(chain_id="integration-test")

            agent1 = MockAgent(name="agent1")
            agent2 = MockAgent(name="agent2")
            callback.attach(agent1)
            callback.attach(agent2)

            send_hook = agent1.get_hooks("process_message_before_send")[0]
            send_hook(agent1, "Hello agent2!", agent2, False)

            reply_hook = agent2.get_hooks("process_all_messages_before_reply")[0]
            reply_hook([{"role": "user", "name": "agent1", "content": "Hello agent2!"}])

            callback.record_tool_result("search", {"results": ["a", "b"]}, 50)

            export_path = tmp_path / "workflow.lctl.json"
            callback.export(str(export_path))

            assert export_path.exists()
            data = json.loads(export_path.read_text())
            assert data["chain"]["id"] == "integration-test"
            assert len(data["events"]) >= 4

    def test_nested_workflow(self, mock_autogen):
        """Test workflow with nested conversations."""
        with patch(
            "lctl.integrations.autogen.AUTOGEN_AVAILABLE", True
        ), patch(
            "lctl.integrations.autogen.AUTOGEN_MODE", "legacy"
        ), patch(
            "lctl.integrations.autogen.ConversableAgent", MockAgent
        ):
            from lctl.integrations.autogen import LCTLAutogenCallback

            callback = LCTLAutogenCallback(chain_id="nested-test")

            callback.start_nested_chat("orchestrator", "Main task")
            callback.session.step_start("worker1", "subtask1", "Processing")
            callback.session.step_end("worker1", outcome="success")

            callback.start_nested_chat("worker1", "Sub-delegation")
            callback.session.step_start("worker2", "subtask2", "Deep work")
            callback.session.step_end("worker2", outcome="success")
            callback.end_nested_chat("Sub-delegation complete")

            callback.end_nested_chat("Main task complete")

            assert callback._nested_depth == 0
            assert len(callback.chain.events) >= 6

            engine = ReplayEngine(callback.chain)
            state = engine.replay_all()
            assert state.metrics["event_count"] >= 6
