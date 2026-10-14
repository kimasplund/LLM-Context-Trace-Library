"""AutoGen (AG2) integration for LCTL.

Provides automatic tracing of AutoGen agent conversations, tool calls,
and group chats with LCTL for time-travel debugging.

Usage:
    from lctl.integrations.autogen import LCTLAutogenCallback, trace_agent, trace_group_chat

    # For legacy AutoGen (<0.4):
    callback = LCTLAutogenCallback()
    callback.attach(agent1)

    # For modern AutoGen (0.4+):
    callback = trace_agent(agent1)
    # (Tracing is often global/contextual in 0.4+, attaching to one agent creates the handler)
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any
from uuid import uuid4

from ..core.session import LCTLSession
from .base import BaseTracer, truncate

# Metadata for availability
AUTOGEN_MODE = "none"  # none, legacy, modern

try:
    # Try modern AutoGen (0.4+)
    import autogen_core  # noqa: F401 - imported for availability check
    from autogen_core import EVENT_LOGGER_NAME
    from autogen_core import logging as ag_logging

    AUTOGEN_MODE = "modern"
    AUTOGEN_AVAILABLE = True
    # Define legacy names as None for compatibility
    Agent: Any = None
    ConversableAgent: Any = None
    GroupChat: Any = None
    GroupChatManager: Any = None
except ImportError:
    try:
        # Try legacy AutoGen / AG2
        from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager  # type: ignore[no-redef]

        AUTOGEN_MODE = "legacy"
        AUTOGEN_AVAILABLE = True
        # Create Dummy references for type hinting if needed
        ag_logging = None
        EVENT_LOGGER_NAME = None
    except ImportError:
        try:
            # Fallback for ag2 renaming of legacy
            from ag2 import Agent, ConversableAgent, GroupChat, GroupChatManager  # type: ignore[no-redef]  # noqa: F401  # availability probe

            AUTOGEN_MODE = "legacy"
            AUTOGEN_AVAILABLE = True
            ag_logging = None
            EVENT_LOGGER_NAME = None
        except ImportError:
            AUTOGEN_AVAILABLE = False
            AUTOGEN_MODE = "none"


class AutogenNotAvailableError(ImportError):
    """Raised when AutoGen/AG2 is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "AutoGen is not installed. Install with: pip install autogen-agentchat "
            "or pip install ag2"
        )


def _check_autogen_available() -> None:
    """Check if AutoGen is available, raise error if not."""
    if not AUTOGEN_AVAILABLE:
        raise AutogenNotAvailableError()


def _extract_message_content(message: str | dict[str, Any] | None) -> str:
    """Extract string content from a message."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return " ".join(text_parts)
    return str(message)


def _get_agent_name(agent: Any) -> str:
    """Get a readable name for an agent."""
    if hasattr(agent, "name") and agent.name:
        return str(agent.name).replace(" ", "-").lower()[:30]
    return "unknown-agent"


class LCTLAutogenHandler(logging.Handler):
    """Logging handler for Modern AutoGen (0.4+) events.

    Thread-safe handler that captures AutoGen events and records them
    to an LCTL session.
    """

    def __init__(self, session: LCTLSession):
        super().__init__()
        self.session = session
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = record.msg
            # In AutoGen 0.4+, specific event classes are used
            if not event or not ag_logging:
                return

            if isinstance(event, ag_logging.LLMCallEvent):
                self._handle_llm_call(event)
            elif isinstance(event, ag_logging.ToolCallEvent):
                self._handle_tool_call(event)
            elif isinstance(event, ag_logging.MessageEvent):
                self._handle_message(event)
            # Add more event handlers as needed

        except Exception:
            self.handleError(record)

    def _get_event_field(self, event: Any, field: str, default: Any = None) -> Any:
        """Helper to safely get field from event object or its kwargs."""
        if hasattr(event, field):
            return getattr(event, field)
        if hasattr(event, "kwargs") and isinstance(event.kwargs, dict):
            return event.kwargs.get(field, default)
        return default

    def _handle_llm_call(self, event: ag_logging.LLMCallEvent) -> None:
        # LLMCallEvent -> messages, response, prompt_tokens, completion_tokens
        # Try to extract agent name from event
        agent_name = self._get_event_field(event, "agent_name")
        if not agent_name:
            agent_name = self._get_event_field(event, "source")
        if not agent_name:
            # Check if there's model info we can use
            model = self._get_event_field(event, "model", "")
            agent_name = f"llm-{model}" if model else "llm"

        messages = self._get_event_field(event, "messages", [])
        response = self._get_event_field(event, "response", {})
        prompt_tokens = self._get_event_field(event, "prompt_tokens", 0)
        completion_tokens = self._get_event_field(event, "completion_tokens", 0)

        input_summary = f"{len(messages)} messages" if messages else "LLM Call"
        output_summary = truncate(str(response)) if response else ""

        with self._lock:
            try:
                self.session.step_start(
                    agent=agent_name, intent="llm_call", input_summary=input_summary
                )
                self.session.step_end(
                    agent=agent_name,
                    outcome="success",
                    output_summary=output_summary,
                    tokens_in=prompt_tokens,
                    tokens_out=completion_tokens,
                )
            except Exception:  # noqa: S110  # tracing must never break the host application
                pass  # Graceful degradation - don't break execution if tracing fails

    def _handle_tool_call(self, event: ag_logging.ToolCallEvent) -> None:
        # ToolCallEvent -> tool_name, arguments, result
        tool_name = self._get_event_field(event, "tool_name", "unknown_tool")
        args = str(self._get_event_field(event, "arguments", ""))
        result = str(self._get_event_field(event, "result", ""))

        # Graceful degradation
        with self._lock, contextlib.suppress(Exception):
            self.session.tool_call(
                tool=tool_name,
                input_data=truncate(args, 200),
                output_data=truncate(result, 200),
                duration_ms=0,
            )

    def _handle_message(self, event: ag_logging.MessageEvent) -> None:
        # MessageEvent -> payload, sender, receiver, kind, delivery_stage

        sender = str(self._get_event_field(event, "sender", "unknown"))
        payload = str(self._get_event_field(event, "payload", ""))

        with self._lock:
            try:
                self.session.step_start(
                    agent=sender, intent="send_message", input_summary=truncate(payload, 50)
                )
                self.session.step_end(
                    agent=sender, outcome="success", output_summary="Message processed"
                )
            except Exception:  # noqa: S110  # tracing must never break the host application
                pass  # Graceful degradation


class LCTLAutogenCallback(BaseTracer):
    """AutoGen callback handler that records events to LCTL.

    Supports both Legacy (hooks) and Modern (event logging) AutoGen.
    Thread-safe for concurrent agent operations.

    Extends BaseTracer for standardized session management, thread safety,
    and automatic stale entry cleanup.
    """

    def __init__(
        self,
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0,
    ) -> None:
        """Initialize the callback handler.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            auto_cleanup: Whether to auto-cleanup stale entries.
            cleanup_interval: Cleanup interval in seconds (default 1 hour).
        """
        _check_autogen_available()

        super().__init__(
            chain_id=chain_id,
            session=session,
            auto_cleanup=auto_cleanup,
            cleanup_interval=cleanup_interval,
        )
        self._attached_agents: list[Any] = []
        self._conversation_stack: list[dict[str, Any]] = []
        self._nested_depth = 0
        self._logging_handler: LCTLAutogenHandler | None = None

        # Track active steps for proper step_start/step_end pairing
        self._active_steps: dict[str, dict[str, Any]] = {}

        if AUTOGEN_MODE == "modern":
            self._setup_modern_tracing()

    def _setup_modern_tracing(self) -> None:
        """Setup global logging handler for Modern AutoGen."""
        if self._logging_handler:
            return

        logger = logging.getLogger(EVENT_LOGGER_NAME)
        self._logging_handler = LCTLAutogenHandler(self.session)
        logger.addHandler(self._logging_handler)

        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.add_fact(
                fact_id="tracing-enabled",
                text="Enabled global AutoGen event tracing",
                confidence=1.0,
                source="lctl-autogen",
            )

    def detach_all(self) -> None:
        """Clean up logging handler and detach from all agents.

        Call this method when done tracing to prevent memory leaks
        and remove the global logging handler.
        """
        with self._lock:
            # Remove modern logging handler
            if self._logging_handler and AUTOGEN_MODE == "modern":
                try:
                    logger = logging.getLogger(EVENT_LOGGER_NAME)
                    logger.removeHandler(self._logging_handler)
                except Exception:  # noqa: S110  # tracing must never break the host application
                    pass  # Ignore errors during cleanup
                self._logging_handler = None

            # Clear attached agents list
            self._attached_agents.clear()

            # Clear active steps tracking
            self._active_steps.clear()

            # Clear conversation stack
            self._conversation_stack.clear()
            self._nested_depth = 0

    def attach(self, agent: Any) -> None:
        """Attach LCTL tracing to an agent."""
        _check_autogen_available()

        if AUTOGEN_MODE == "modern":
            # Modern mode relies on global logging, attach does nothing/little
            # But we can log that we are 'watching' this agent
            name = _get_agent_name(agent)
            # Graceful degradation
            with contextlib.suppress(Exception):
                self.session.add_fact(
                    fact_id=f"agent-watched-{name}",
                    text=f"Watching agent {name}",
                    confidence=1.0,
                    source="lctl-autogen",
                )
            return

        # Legacy Mode
        with self._lock:
            if agent in self._attached_agents:
                return

        # Ensure we check for register_hook existence
        if not hasattr(agent, "register_hook"):
            # Fallback or error?
            return

        agent_name = _get_agent_name(agent)

        try:
            agent.register_hook(
                "process_message_before_send",
                self._create_before_send_hook(agent_name),
            )
            agent.register_hook(
                "process_all_messages_before_reply",
                self._create_before_reply_hook(agent_name),
            )
            agent.register_hook(
                "process_last_received_message",
                self._create_after_reply_hook(agent_name),
            )
            agent.register_hook(
                "update_agent_state",
                self._create_state_update_hook(agent_name),
            )
            with self._lock:
                self._attached_agents.append(agent)
        except Exception:  # noqa: S110  # tracing must never break the host application
            pass  # Ignore if hooks not supported on this object

        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.add_fact(
                fact_id=f"agent-attached-{agent_name}",
                text=f"Agent '{agent_name}' attached for tracing",
                confidence=1.0,
                source="lctl-autogen",
            )

    def _create_before_send_hook(self, sender_name: str):
        def hook(sender, message, recipient, silent):
            recipient_name = _get_agent_name(recipient)
            content = _extract_message_content(message)

            with self._lock:
                try:
                    self.session.step_start(
                        agent=sender_name,
                        intent="send_message",
                        input_summary=f"To {recipient_name}: {truncate(content, 100)}",
                    )

                    if isinstance(message, dict) and (
                        "tool_calls" in message or "function_call" in message
                    ):
                        tool_calls = message.get("tool_calls", [])
                        if not tool_calls and "function_call" in message:
                            tool_calls = [{"function": message["function_call"]}]

                        for tool_call in tool_calls:
                            func_info = tool_call.get("function", {})
                            tool_name = func_info.get("name", "unknown_tool")
                            tool_args = func_info.get("arguments", "")

                            self.session.tool_call(
                                tool=tool_name,
                                input_data=truncate(str(tool_args), 200),
                                output_data="(pending)",
                                duration_ms=0,
                            )

                    self.session.step_end(
                        agent=sender_name,
                        outcome="success",
                        output_summary=f"Message sent to {recipient_name}",
                    )
                except Exception:  # noqa: S110  # tracing must never break the host application
                    pass  # Graceful degradation

            return message

        return hook

    def _create_before_reply_hook(self, agent_name: str):
        """Create hook for before reply generation.

        Tracks the step start so it can be properly ended in after_reply_hook.
        """

        def hook(messages):
            if messages:
                step_key = f"{agent_name}-generate_reply"
                with self._lock:
                    try:
                        self.session.step_start(
                            agent=agent_name,
                            intent="generate_reply",
                            input_summary=f"Processing {len(messages)} messages",
                        )
                        self._active_steps[step_key] = {
                            "agent": agent_name,
                            "start_time": time.time(),
                        }
                    except Exception:  # noqa: S110  # tracing must never break the host application
                        pass  # Graceful degradation
            return messages

        return hook

    def _create_after_reply_hook(self, agent_name: str):
        """Create hook for after reply generation.

        Ends the step that was started in before_reply_hook.
        """

        def hook(message):
            step_key = f"{agent_name}-generate_reply"
            with self._lock:
                step_info = self._active_steps.pop(step_key, None)
                if step_info:
                    try:
                        duration_ms = int((time.time() - step_info["start_time"]) * 1000)
                        content = _extract_message_content(message)
                        self.session.step_end(
                            agent=agent_name,
                            outcome="success",
                            output_summary=truncate(content, 100),
                            duration_ms=duration_ms,
                        )
                    except Exception:  # noqa: S110  # tracing must never break the host application
                        pass  # Graceful degradation
            return message

        return hook

    def _create_state_update_hook(self, agent_name: str):
        def hook(agent, messages):
            pass

        return hook

    def attach_group_chat(self, group_chat: Any, manager: Any | None = None) -> None:
        """Attach LCTL tracing to a GroupChat."""
        if AUTOGEN_MODE == "modern":
            # Modern tracing is global
            return

        for agent in group_chat.agents:
            self.attach(agent)

        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.add_fact(
                fact_id="groupchat-config",
                text=f"GroupChat configured with {len(group_chat.agents)} agents, "
                f"max_round={group_chat.max_round}",
                confidence=1.0,
                source="lctl-autogen",
            )

        if manager is not None:
            self.attach(manager)

    def start_nested_chat(self, parent_agent: str, description: str = "") -> None:
        """Record the start of a nested conversation."""
        with self._lock:
            self._nested_depth += 1
            self._conversation_stack.append(
                {
                    "parent": parent_agent,
                    "depth": self._nested_depth,
                    "start_time": time.time(),
                }
            )

        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.step_start(
                agent=parent_agent,
                intent="start_nested_chat",
                input_summary=description or f"Nested chat at depth {self._nested_depth}",
            )

    def end_nested_chat(self, result_summary: str = "", outcome: str = "success") -> None:
        """Record the end of a nested conversation."""
        with self._lock:
            if not self._conversation_stack:
                return

            context = self._conversation_stack.pop()
            parent_agent = context["parent"]
            start_time = context["start_time"]
            duration_ms = int((time.time() - start_time) * 1000)
            current_depth = self._nested_depth
            self._nested_depth = max(0, self._nested_depth - 1)

        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.step_end(
                agent=parent_agent,
                outcome=outcome,
                output_summary=result_summary or f"Nested chat completed at depth {current_depth}",
                duration_ms=duration_ms,
            )

    def record_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: int = 0,
        agent: str | None = None,
    ) -> None:
        """Manually record a tool call result.

        Args:
            tool_name: Name of the tool that was called
            result: The result returned by the tool
            duration_ms: How long the tool call took in milliseconds
            agent: Optional agent name that invoked the tool
        """
        result_str = truncate(str(result), 500) if result else "(no result)"

        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.tool_call(
                tool=tool_name,
                input_data="(recorded manually)",
                output_data=result_str,
                duration_ms=duration_ms,
                agent=agent or "tool-executor",
            )

    def record_error(
        self,
        error: Exception,
        agent: str | None = None,
        recoverable: bool = True,
    ) -> None:
        """Record an error during conversation."""
        # Graceful degradation
        with contextlib.suppress(Exception):
            self.session.error(
                category="autogen_error",
                error_type=type(error).__name__,
                message=str(error),
                recoverable=recoverable,
                suggested_action="Check agent configuration and message format",
            )


# Convenience wrappers


def trace_agent(agent: Any, chain_id: str | None = None) -> LCTLAutogenCallback:
    callback = LCTLAutogenCallback(chain_id=chain_id)
    callback.attach(agent)
    return callback


def trace_group_chat(
    group_chat: Any, manager: Any | None = None, chain_id: str | None = None
) -> LCTLAutogenCallback:
    callback = LCTLAutogenCallback(chain_id=chain_id)
    callback.attach_group_chat(group_chat, manager)
    return callback


class LCTLConversableAgent:
    """Wrapper around AutoGen ConversableAgent with built-in LCTL tracing.

    Example:
        agent = LCTLConversableAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config={"model": "gpt-4"}
        )
        agent.initiate_chat(other_agent, message="Hello")
        agent.export_trace("trace.lctl.json")
    """

    def __init__(
        self,
        name: str,
        system_message: str | None = None,
        llm_config: dict[str, Any] | None = None,
        chain_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an LCTL-traced ConversableAgent."""
        _check_autogen_available()

        if AUTOGEN_MODE == "modern":
            # In modern mode, we can't easily inherit or wrap if ConversableAgent doesn't exist.
            # But we can try to use trace_agent on the *modern equivalent* if user passed it?
            # Actually, this class assumes creating a NEW agent.
            # If ConversableAgent class is missing, we can't create it.
            raise AutogenNotAvailableError()

        self._callback = LCTLAutogenCallback(
            chain_id=chain_id or f"agent-{name}-{str(uuid4())[:8]}"
        )

        self._agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )

        self._callback.attach(self._agent)

        self._metadata = {
            "name": name,
            "system_message": system_message[:200] if system_message else None,
            "has_llm_config": llm_config is not None,
        }

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def session(self) -> LCTLSession:
        return self._callback.session

    @property
    def callback(self) -> LCTLAutogenCallback:
        return self._callback

    def initiate_chat(
        self,
        recipient: Any,
        message: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        recipient_agent = (
            recipient.agent if isinstance(recipient, LCTLConversableAgent) else recipient
        )
        recipient_name = _get_agent_name(recipient_agent)

        if (
            isinstance(recipient, LCTLConversableAgent)
            and recipient_agent not in self._callback._attached_agents
        ):
            self._callback.attach(recipient_agent)

        # Graceful degradation
        with contextlib.suppress(Exception):
            self._callback.session.step_start(
                agent=self._agent.name,
                intent="initiate_chat",
                input_summary=f"Starting chat with {recipient_name}",
            )

        start_time = time.time()
        try:
            result = self._agent.initiate_chat(
                recipient=recipient_agent,
                message=message,
                **kwargs,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Graceful degradation
            with contextlib.suppress(Exception):
                self._callback.session.step_end(
                    agent=self._agent.name,
                    outcome="success",
                    output_summary=f"Chat with {recipient_name} completed",
                    duration_ms=duration_ms,
                )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            try:
                self._callback.session.step_end(
                    agent=self._agent.name,
                    outcome="error",
                    output_summary=f"Chat with {recipient_name} failed: {str(e)[:100]}",
                    duration_ms=duration_ms,
                )
                self._callback.session.error(
                    category="autogen_error",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=False,
                )
            except Exception:  # noqa: S110  # tracing must never break the host application
                pass  # Graceful degradation
            raise

    def export_trace(self, path: str) -> None:
        self._callback.export(path)

    def get_trace(self) -> dict[str, Any]:
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


class LCTLGroupChatManager:
    """Wrapper around AutoGen GroupChatManager with built-in LCTL tracing."""

    def __init__(
        self,
        groupchat: Any,
        name: str = "chat_manager",
        chain_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        _check_autogen_available()

        if AUTOGEN_MODE == "modern":
            raise AutogenNotAvailableError()

        self._callback = LCTLAutogenCallback(chain_id=chain_id or f"groupchat-{str(uuid4())[:8]}")

        self._manager = GroupChatManager(
            groupchat=groupchat,
            name=name,
            **kwargs,
        )

        self._groupchat = groupchat
        self._callback.attach_group_chat(groupchat, self._manager)

        agent_names = [_get_agent_name(a) for a in groupchat.agents]
        # Graceful degradation
        with contextlib.suppress(Exception):
            self._callback.session.add_fact(
                fact_id="groupchat-agents",
                text=f"GroupChat agents: {', '.join(agent_names)}",
                confidence=1.0,
                source="lctl-autogen",
            )

    @property
    def manager(self) -> Any:
        return self._manager

    @property
    def groupchat(self) -> Any:
        return self._groupchat

    @property
    def session(self) -> LCTLSession:
        return self._callback.session

    @property
    def callback(self) -> LCTLAutogenCallback:
        return self._callback

    def export_trace(self, path: str) -> None:
        self._callback.export(path)

    def get_trace(self) -> dict[str, Any]:
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


def is_available() -> bool:
    return AUTOGEN_AVAILABLE


__all__ = [
    "AUTOGEN_AVAILABLE",
    "AUTOGEN_MODE",
    "LCTLAutogenCallback",
    "LCTLAutogenHandler",
    "LCTLConversableAgent",
    "LCTLGroupChatManager",
    "is_available",
    "trace_agent",
    "trace_group_chat",
]
