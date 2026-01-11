"""AutoGen (AG2) integration for LCTL.

Provides automatic tracing of AutoGen agent conversations, tool calls,
and group chats with LCTL for time-travel debugging.

Usage:
    from lctl.integrations.autogen import LCTLAutogenCallback, trace_agent

    # Create callback and attach to agents
    callback = LCTLAutogenCallback()
    callback.attach(agent1)
    callback.attach(agent2)

    # Run conversation
    agent1.initiate_chat(agent2, message="Hello")

    # Export trace
    callback.export("trace.lctl.json")
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..core.events import EventType
from ..core.session import LCTLSession

try:
    from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from ag2 import Agent, ConversableAgent, GroupChat, GroupChatManager
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False
        Agent = None
        ConversableAgent = None
        GroupChat = None
        GroupChatManager = None


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


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _extract_message_content(message: Union[str, Dict[str, Any], None]) -> str:
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


class LCTLAutogenCallback:
    """AutoGen callback handler that records events to LCTL.

    Captures:
    - Agent-to-agent message passing
    - Tool/function calls and responses
    - GroupChat conversations
    - Nested conversation tracking
    - Errors

    Example:
        callback = LCTLAutogenCallback(chain_id="my-conversation")
        callback.attach(agent1)
        callback.attach(agent2)
        agent1.initiate_chat(agent2, message="Hello")
        callback.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
    ) -> None:
        """Initialize the callback handler.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
        """
        _check_autogen_available()

        self.session = session or LCTLSession(chain_id=chain_id)
        self._attached_agents: List[Any] = []
        self._conversation_stack: List[Dict[str, Any]] = []
        self._message_times: Dict[str, float] = {}
        self._nested_depth = 0

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self.session.chain

    def attach(self, agent: "ConversableAgent") -> None:
        """Attach LCTL tracing to an agent.

        Registers hooks for message processing and state updates.

        Args:
            agent: The AutoGen agent to attach tracing to.
        """
        _check_autogen_available()

        if agent in self._attached_agents:
            return

        agent_name = _get_agent_name(agent)

        agent.register_hook(
            "process_message_before_send",
            self._create_before_send_hook(agent_name),
        )

        agent.register_hook(
            "process_all_messages_before_reply",
            self._create_before_reply_hook(agent_name),
        )

        agent.register_hook(
            "update_agent_state",
            self._create_state_update_hook(agent_name),
        )

        self._attached_agents.append(agent)

        self.session.add_fact(
            fact_id=f"agent-attached-{agent_name}",
            text=f"Agent '{agent_name}' attached for tracing",
            confidence=1.0,
            source="lctl-autogen",
        )

    def attach_group_chat(
        self,
        group_chat: "GroupChat",
        manager: Optional["GroupChatManager"] = None,
    ) -> None:
        """Attach LCTL tracing to a GroupChat.

        Args:
            group_chat: The GroupChat to trace.
            manager: Optional GroupChatManager to also attach.
        """
        _check_autogen_available()

        for agent in group_chat.agents:
            self.attach(agent)

        self.session.add_fact(
            fact_id="groupchat-config",
            text=f"GroupChat configured with {len(group_chat.agents)} agents, "
            f"max_round={group_chat.max_round}",
            confidence=1.0,
            source="lctl-autogen",
        )

        if manager is not None:
            self.attach(manager)

    def _create_before_send_hook(
        self, sender_name: str
    ) -> Callable[
        ["ConversableAgent", Union[Dict[str, Any], str], "Agent", bool],
        Union[Dict[str, Any], str],
    ]:
        """Create a hook for process_message_before_send."""

        def hook(
            sender: "ConversableAgent",
            message: Union[Dict[str, Any], str],
            recipient: "Agent",
            silent: bool,
        ) -> Union[Dict[str, Any], str]:
            recipient_name = _get_agent_name(recipient)
            content = _extract_message_content(message)

            message_id = f"{sender_name}->{recipient_name}-{time.time()}"
            self._message_times[message_id] = time.time()

            self.session.step_start(
                agent=sender_name,
                intent="send_message",
                input_summary=f"To {recipient_name}: {_truncate(content, 100)}",
            )

            if isinstance(message, dict):
                if "tool_calls" in message or "function_call" in message:
                    tool_calls = message.get("tool_calls", [])
                    if not tool_calls and "function_call" in message:
                        tool_calls = [{"function": message["function_call"]}]

                    for tool_call in tool_calls:
                        func_info = tool_call.get("function", {})
                        tool_name = func_info.get("name", "unknown_tool")
                        tool_args = func_info.get("arguments", "")

                        self.session.tool_call(
                            tool=tool_name,
                            input_data=_truncate(str(tool_args), 200),
                            output_data="(pending)",
                            duration_ms=0,
                        )

            self.session.step_end(
                agent=sender_name,
                outcome="success",
                output_summary=f"Message sent to {recipient_name}",
            )

            return message

        return hook

    def _create_before_reply_hook(
        self, agent_name: str
    ) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Create a hook for process_all_messages_before_reply."""

        def hook(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not messages:
                return messages

            last_message = messages[-1]
            sender = last_message.get("name", last_message.get("role", "unknown"))
            content = _extract_message_content(last_message)

            self.session.step_start(
                agent=agent_name,
                intent="generate_reply",
                input_summary=f"From {sender}: {_truncate(content, 100)}",
            )

            if last_message.get("role") == "tool" or "tool_call_id" in last_message:
                tool_response = _truncate(content, 200)
                self.session.add_fact(
                    fact_id=f"tool-response-{len(self.session.chain.events)}",
                    text=f"Tool response: {tool_response}",
                    confidence=1.0,
                    source=agent_name,
                )

            return messages

        return hook

    def _create_state_update_hook(
        self, agent_name: str
    ) -> Callable[["ConversableAgent", List[Dict[str, Any]]], None]:
        """Create a hook for update_agent_state."""

        def hook(agent: "ConversableAgent", messages: List[Dict[str, Any]]) -> None:
            if self._nested_depth > 0:
                self.session.add_fact(
                    fact_id=f"nested-context-{len(self.session.chain.events)}",
                    text=f"Agent '{agent_name}' state update at nested depth {self._nested_depth}",
                    confidence=1.0,
                    source=agent_name,
                )

        return hook

    def start_nested_chat(self, parent_agent: str, description: str = "") -> None:
        """Record the start of a nested conversation.

        Call this when initiating a nested chat to track conversation hierarchy.

        Args:
            parent_agent: Name of the agent starting the nested chat.
            description: Optional description of the nested chat purpose.
        """
        self._nested_depth += 1
        self._conversation_stack.append(
            {
                "parent": parent_agent,
                "depth": self._nested_depth,
                "start_time": time.time(),
            }
        )

        self.session.step_start(
            agent=parent_agent,
            intent="start_nested_chat",
            input_summary=description or f"Nested chat at depth {self._nested_depth}",
        )

    def end_nested_chat(
        self, result_summary: str = "", outcome: str = "success"
    ) -> None:
        """Record the end of a nested conversation.

        Args:
            result_summary: Summary of the nested chat result.
            outcome: Outcome of the nested chat ('success', 'error', etc.).
        """
        if not self._conversation_stack:
            return

        context = self._conversation_stack.pop()
        parent_agent = context["parent"]
        start_time = context["start_time"]
        duration_ms = int((time.time() - start_time) * 1000)

        self.session.step_end(
            agent=parent_agent,
            outcome=outcome,
            output_summary=result_summary or f"Nested chat completed at depth {self._nested_depth}",
            duration_ms=duration_ms,
        )

        self._nested_depth = max(0, self._nested_depth - 1)

    def record_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: int = 0,
        agent: Optional[str] = None,
    ) -> None:
        """Manually record a tool call result.

        Use this when tool results need to be recorded outside of message hooks.

        Args:
            tool_name: Name of the tool.
            result: The tool result.
            duration_ms: Execution duration in milliseconds.
            agent: Optional agent name that invoked the tool.
        """
        result_str = _truncate(str(result), 500) if result else "(no result)"

        self.session.add_fact(
            fact_id=f"tool-result-{tool_name}-{len(self.session.chain.events)}",
            text=f"Tool '{tool_name}' returned: {result_str}",
            confidence=1.0,
            source=agent or "tool-executor",
        )

    def record_error(
        self,
        error: Exception,
        agent: Optional[str] = None,
        recoverable: bool = True,
    ) -> None:
        """Record an error during conversation.

        Args:
            error: The exception that occurred.
            agent: Optional agent name where error occurred.
            recoverable: Whether the error is recoverable.
        """
        self.session.error(
            category="autogen_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=recoverable,
            suggested_action="Check agent configuration and message format",
        )

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self.session.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL chain as a dictionary."""
        return self.session.to_dict()


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
        system_message: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        chain_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an LCTL-traced ConversableAgent.

        Args:
            name: Agent name.
            system_message: System message for the agent.
            llm_config: LLM configuration dictionary.
            chain_id: Optional chain ID for LCTL session.
            **kwargs: Additional arguments passed to ConversableAgent.
        """
        _check_autogen_available()

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
    def agent(self) -> "ConversableAgent":
        """Get the underlying ConversableAgent."""
        return self._agent

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session."""
        return self._callback.session

    @property
    def callback(self) -> LCTLAutogenCallback:
        """Get the LCTL callback handler."""
        return self._callback

    def initiate_chat(
        self,
        recipient: Union["LCTLConversableAgent", "ConversableAgent"],
        message: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Initiate a chat with another agent.

        Args:
            recipient: The agent to chat with.
            message: Initial message.
            **kwargs: Additional arguments passed to initiate_chat.

        Returns:
            The chat result.
        """
        recipient_agent = (
            recipient.agent if isinstance(recipient, LCTLConversableAgent) else recipient
        )
        recipient_name = _get_agent_name(recipient_agent)

        if isinstance(recipient, LCTLConversableAgent):
            if recipient_agent not in self._callback._attached_agents:
                self._callback.attach(recipient_agent)

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

            self._callback.session.step_end(
                agent=self._agent.name,
                outcome="success",
                output_summary=f"Chat with {recipient_name} completed",
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            self._callback.session.step_end(
                agent=self._agent.name,
                outcome="error",
                output_summary=f"Chat with {recipient_name} failed: {str(e)[:100]}",
                duration_ms=duration_ms,
            )

            self._callback.record_error(e, agent=self._agent.name, recoverable=False)
            raise

    def export_trace(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self._callback.export(path)

    def get_trace(self) -> Dict[str, Any]:
        """Get the LCTL trace as a dictionary."""
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying agent."""
        return getattr(self._agent, name)


class LCTLGroupChatManager:
    """Wrapper around AutoGen GroupChatManager with built-in LCTL tracing.

    Example:
        group_chat = GroupChat(agents=[agent1, agent2], messages=[], max_round=10)
        manager = LCTLGroupChatManager(
            groupchat=group_chat,
            chain_id="my-group-chat"
        )
        agent1.initiate_chat(manager, message="Let's discuss the project")
        manager.export_trace("groupchat.lctl.json")
    """

    def __init__(
        self,
        groupchat: "GroupChat",
        name: str = "chat_manager",
        chain_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an LCTL-traced GroupChatManager.

        Args:
            groupchat: The GroupChat to manage.
            name: Manager agent name.
            chain_id: Optional chain ID for LCTL session.
            **kwargs: Additional arguments passed to GroupChatManager.
        """
        _check_autogen_available()

        self._callback = LCTLAutogenCallback(
            chain_id=chain_id or f"groupchat-{str(uuid4())[:8]}"
        )

        self._manager = GroupChatManager(
            groupchat=groupchat,
            name=name,
            **kwargs,
        )

        self._groupchat = groupchat
        self._callback.attach_group_chat(groupchat, self._manager)

        agent_names = [_get_agent_name(a) for a in groupchat.agents]
        self._callback.session.add_fact(
            fact_id="groupchat-agents",
            text=f"GroupChat agents: {', '.join(agent_names)}",
            confidence=1.0,
            source="lctl-autogen",
        )

    @property
    def manager(self) -> "GroupChatManager":
        """Get the underlying GroupChatManager."""
        return self._manager

    @property
    def groupchat(self) -> "GroupChat":
        """Get the underlying GroupChat."""
        return self._groupchat

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session."""
        return self._callback.session

    @property
    def callback(self) -> LCTLAutogenCallback:
        """Get the LCTL callback handler."""
        return self._callback

    def export_trace(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self._callback.export(path)

    def get_trace(self) -> Dict[str, Any]:
        """Get the LCTL trace as a dictionary."""
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying manager."""
        return getattr(self._manager, name)


def trace_agent(
    agent: "ConversableAgent",
    chain_id: Optional[str] = None,
) -> LCTLAutogenCallback:
    """Attach LCTL tracing to an existing AutoGen agent.

    Args:
        agent: The agent to trace.
        chain_id: Optional chain ID for tracing.

    Returns:
        The LCTLAutogenCallback instance for exporting traces.

    Example:
        from autogen import ConversableAgent
        from lctl.integrations.autogen import trace_agent

        agent = ConversableAgent(name="assistant", ...)
        callback = trace_agent(agent)
        agent.initiate_chat(other_agent, message="Hello")
        callback.export("trace.lctl.json")
    """
    _check_autogen_available()

    callback = LCTLAutogenCallback(
        chain_id=chain_id or f"trace-{_get_agent_name(agent)}-{str(uuid4())[:8]}"
    )
    callback.attach(agent)
    return callback


def trace_group_chat(
    group_chat: "GroupChat",
    manager: Optional["GroupChatManager"] = None,
    chain_id: Optional[str] = None,
) -> LCTLAutogenCallback:
    """Attach LCTL tracing to an existing GroupChat.

    Args:
        group_chat: The GroupChat to trace.
        manager: Optional GroupChatManager to also trace.
        chain_id: Optional chain ID for tracing.

    Returns:
        The LCTLAutogenCallback instance for exporting traces.

    Example:
        from autogen import GroupChat, GroupChatManager
        from lctl.integrations.autogen import trace_group_chat

        group_chat = GroupChat(agents=[...], messages=[], max_round=10)
        manager = GroupChatManager(groupchat=group_chat)

        callback = trace_group_chat(group_chat, manager)
        agent1.initiate_chat(manager, message="Let's discuss")
        callback.export("groupchat.lctl.json")
    """
    _check_autogen_available()

    callback = LCTLAutogenCallback(
        chain_id=chain_id or f"groupchat-{str(uuid4())[:8]}"
    )
    callback.attach_group_chat(group_chat, manager)
    return callback


def is_available() -> bool:
    """Check if AutoGen integration is available.

    Returns:
        True if AutoGen/AG2 is installed, False otherwise.
    """
    return AUTOGEN_AVAILABLE


__all__ = [
    "AUTOGEN_AVAILABLE",
    "AutogenNotAvailableError",
    "LCTLAutogenCallback",
    "LCTLConversableAgent",
    "LCTLGroupChatManager",
    "trace_agent",
    "trace_group_chat",
    "is_available",
]
