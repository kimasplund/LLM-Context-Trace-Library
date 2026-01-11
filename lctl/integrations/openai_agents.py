"""OpenAI Agents SDK integration for LCTL.

Provides automatic tracing of OpenAI Agents SDK agent runs, tool calls,
and handoffs with LCTL for time-travel debugging.

Usage:
    from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

    tracer = LCTLOpenAIAgentTracer(chain_id="my-agent")

    # Use with Runner
    result = await Runner.run(agent, input="Hello", run_config=tracer.run_config)

    tracer.export("trace.lctl.json")
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from uuid import uuid4

from ..core.session import LCTLSession

try:
    from agents import (
        Agent,
        RunContextWrapper,
        RunHooks,
        Tool,
        Usage,
    )
    from agents.tracing import (
        Trace,
        TracingProcessor,
    )

    OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    OPENAI_AGENTS_AVAILABLE = False
    Agent = None
    RunContextWrapper = None
    RunHooks = object  # Base class for when SDK is not installed
    Tool = None
    Usage = None
    Trace = None
    TracingProcessor = object  # Base class for when SDK is not installed


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _extract_usage(usage: Any) -> Dict[str, int]:
    """Extract token usage from Usage object."""
    tokens = {"input": 0, "output": 0}

    if usage is None:
        return tokens

    if hasattr(usage, "input_tokens"):
        tokens["input"] = usage.input_tokens or 0
    if hasattr(usage, "output_tokens"):
        tokens["output"] = usage.output_tokens or 0

    return tokens


class OpenAIAgentsNotAvailableError(ImportError):
    """Raised when OpenAI Agents SDK is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "OpenAI Agents SDK is not installed. Install with: pip install openai-agents"
        )


def _check_openai_agents_available() -> None:
    """Check if OpenAI Agents SDK is available, raise error if not."""
    if not OPENAI_AGENTS_AVAILABLE:
        raise OpenAIAgentsNotAvailableError()


class LCTLRunHooks(RunHooks):
    """Run hooks implementation that records events to LCTL.

    Captures:
    - Agent run start/end
    - Tool calls
    - Handoffs between agents
    - Streaming events
    - Errors
    """

    def __init__(self, session: LCTLSession, verbose: bool = False) -> None:
        """Initialize the run hooks.

        Args:
            session: LCTL session for recording events.
            verbose: Enable verbose output.
        """
        self._session = session
        self._verbose = verbose
        self._run_stack: Dict[str, Dict[str, Any]] = {}
        self._tool_stack: Dict[str, Dict[str, Any]] = {}
        self._current_agent: Optional[str] = None
        self._stream_buffer: Dict[str, str] = {}

    def _get_agent_name(self, agent: Any) -> str:
        """Extract agent name from agent object."""
        if agent is None:
            return "unknown"
        if hasattr(agent, "name") and agent.name:
            return str(agent.name)
        if hasattr(agent, "__class__"):
            return agent.__class__.__name__
        return "agent"

    async def on_agent_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
    ) -> None:
        """Called when an agent starts execution."""
        agent_name = self._get_agent_name(agent)
        run_id = str(id(context))

        self._run_stack[run_id] = {
            "agent": agent_name,
            "start_time": time.time(),
        }
        self._current_agent = agent_name

        input_summary = ""
        if hasattr(context, "input") and context.input:
            input_text = str(context.input)
            input_summary = _truncate(input_text)

        self._session.step_start(
            agent=agent_name,
            intent="agent_run",
            input_summary=input_summary,
        )

        if hasattr(agent, "instructions") and agent.instructions:
            self._session.add_fact(
                fact_id=f"agent_{agent_name}_instructions",
                text=f"Agent instructions: {_truncate(agent.instructions, 300)}",
                confidence=1.0,
                source=agent_name,
            )

        if self._verbose:
            print(f"[LCTL] Agent started: {agent_name}")

    async def on_agent_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        output: Any,
    ) -> None:
        """Called when an agent completes execution."""
        agent_name = self._get_agent_name(agent)
        run_id = str(id(context))

        run_info = self._run_stack.pop(run_id, {})
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        output_summary = ""
        if output is not None:
            if hasattr(output, "final_output"):
                output_summary = _truncate(str(output.final_output))
            else:
                output_summary = _truncate(str(output))

        tokens = {"input": 0, "output": 0}
        if hasattr(output, "usage"):
            tokens = _extract_usage(output.usage)

        self._session.step_end(
            agent=agent_name,
            outcome="success",
            output_summary=output_summary,
            duration_ms=duration_ms,
            tokens_in=tokens["input"],
            tokens_out=tokens["output"],
        )

        self._current_agent = None

        if self._verbose:
            print(f"[LCTL] Agent ended: {agent_name} ({duration_ms}ms)")

    async def on_tool_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        input_data: Any,
    ) -> None:
        """Called when a tool invocation starts."""
        tool_name = getattr(tool, "name", str(tool))
        tool_id = f"{tool_name}_{id(input_data)}"

        self._tool_stack[tool_id] = {
            "tool": tool_name,
            "start_time": time.time(),
            "input": input_data,
        }

        if self._verbose:
            agent_name = self._get_agent_name(agent)
            print(f"[LCTL] Tool started: {tool_name} (agent: {agent_name})")

    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        output: Any,
    ) -> None:
        """Called when a tool invocation completes."""
        tool_name = getattr(tool, "name", str(tool))

        tool_info = None
        for tid, info in list(self._tool_stack.items()):
            if info["tool"] == tool_name:
                tool_info = self._tool_stack.pop(tid)
                break

        if tool_info is None:
            tool_info = {"tool": tool_name, "start_time": time.time(), "input": ""}

        start_time = tool_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)
        input_data = tool_info.get("input", "")

        input_str = _truncate(str(input_data)) if input_data else ""
        output_str = _truncate(str(output)) if output else ""

        self._session.tool_call(
            tool=tool_name,
            input_data=input_str,
            output_data=output_str,
            duration_ms=duration_ms,
        )

        if self._verbose:
            print(f"[LCTL] Tool ended: {tool_name} ({duration_ms}ms)")

    async def on_handoff(
        self,
        context: RunContextWrapper,
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        """Called when control is handed off between agents."""
        from_name = self._get_agent_name(from_agent)
        to_name = self._get_agent_name(to_agent)

        self._session.add_fact(
            fact_id=f"handoff_{from_name}_to_{to_name}_{int(time.time() * 1000)}",
            text=f"Handoff from {from_name} to {to_name}",
            confidence=1.0,
            source=from_name,
        )

        if self._verbose:
            print(f"[LCTL] Handoff: {from_name} -> {to_name}")

    async def on_error(
        self,
        context: RunContextWrapper,
        agent: Agent,
        error: BaseException,
    ) -> None:
        """Called when an error occurs during agent execution."""
        agent_name = self._get_agent_name(agent)
        run_id = str(id(context))

        run_info = self._run_stack.pop(run_id, {})
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        self._session.step_end(
            agent=agent_name,
            outcome="error",
            output_summary=f"Error: {type(error).__name__}: {str(error)[:100]}",
            duration_ms=duration_ms,
        )

        self._session.error(
            category="agent_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=False,
            suggested_action="Check agent configuration and input",
        )

        if self._verbose:
            print(f"[LCTL] Error in agent {agent_name}: {error}")


class LCTLTracingProcessor(TracingProcessor):
    """Tracing processor that records traces to LCTL.

    This provides lower-level tracing that captures the full trace
    structure from the OpenAI Agents SDK.
    """

    def __init__(self, session: LCTLSession, verbose: bool = False) -> None:
        """Initialize the tracing processor.

        Args:
            session: LCTL session for recording events.
            verbose: Enable verbose output.
        """
        self._session = session
        self._verbose = verbose

    def process_trace(self, trace: Trace) -> None:
        """Process a completed trace from the SDK."""
        if trace is None:
            return

        trace_data = {}
        if hasattr(trace, "to_dict"):
            trace_data = trace.to_dict()
        elif hasattr(trace, "__dict__"):
            trace_data = {
                k: v for k, v in trace.__dict__.items()
                if not k.startswith("_")
            }

        self._session.add_fact(
            fact_id=f"trace_{int(time.time() * 1000)}",
            text=f"Trace completed: {_truncate(str(trace_data), 500)}",
            confidence=1.0,
            source="tracing_processor",
        )

        if self._verbose:
            print("[LCTL] Trace processed")


class LCTLOpenAIAgentTracer:
    """Main tracer class for OpenAI Agents SDK integration.

    Provides convenient methods to trace agent runs with LCTL,
    supporting both sync and async execution modes.

    Example:
        tracer = LCTLOpenAIAgentTracer(chain_id="my-agent")

        # Option 1: Use run_config with Runner
        result = await Runner.run(
            agent,
            input="Hello",
            run_config=tracer.run_config
        )

        # Option 2: Use hooks directly
        hooks = tracer.create_hooks()
        # Pass hooks to your runner configuration

        tracer.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the OpenAI Agents SDK tracer.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            verbose: Enable verbose output.
        """
        _check_openai_agents_available()

        self._session = session or LCTLSession(chain_id=chain_id or f"openai-agent-{str(uuid4())[:8]}")
        self._verbose = verbose
        self._hooks: Optional[LCTLRunHooks] = None
        self._tracing_processor: Optional[LCTLTracingProcessor] = None

    @property
    def session(self) -> LCTLSession:
        """Access the LCTL session."""
        return self._session

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self._session.chain

    def create_hooks(self) -> LCTLRunHooks:
        """Create run hooks for the tracer.

        Returns:
            LCTLRunHooks instance configured with the session.
        """
        if self._hooks is None:
            self._hooks = LCTLRunHooks(self._session, verbose=self._verbose)
        return self._hooks

    def create_tracing_processor(self) -> LCTLTracingProcessor:
        """Create a tracing processor for the tracer.

        Returns:
            LCTLTracingProcessor instance configured with the session.
        """
        if self._tracing_processor is None:
            self._tracing_processor = LCTLTracingProcessor(
                self._session, verbose=self._verbose
            )
        return self._tracing_processor

    @property
    def run_config(self) -> Dict[str, Any]:
        """Get run configuration with LCTL hooks.

        Returns a configuration dictionary that can be passed to Runner.run().

        Returns:
            Dictionary with hooks configuration.
        """
        return {
            "hooks": self.create_hooks(),
        }

    def trace_agent_run(
        self,
        agent_name: str,
        input_summary: str = "",
    ) -> "AgentRunContext":
        """Context manager for manually tracing an agent run.

        Useful when you need more control over tracing or when
        using custom agent implementations.

        Args:
            agent_name: Name of the agent being traced.
            input_summary: Summary of the input to the agent.

        Returns:
            AgentRunContext that can be used as a context manager.

        Example:
            with tracer.trace_agent_run("my_agent", "Process data"):
                result = agent.process(data)
        """
        return AgentRunContext(
            self._session,
            agent_name,
            input_summary,
            verbose=self._verbose,
        )

    def record_tool_call(
        self,
        tool: str,
        input_data: Any,
        output_data: Any,
        duration_ms: int = 0,
    ) -> None:
        """Manually record a tool call.

        Args:
            tool: Name of the tool.
            input_data: Input to the tool.
            output_data: Output from the tool.
            duration_ms: Duration of the tool call in milliseconds.
        """
        self._session.tool_call(
            tool=tool,
            input_data=_truncate(str(input_data)),
            output_data=_truncate(str(output_data)),
            duration_ms=duration_ms,
        )

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
    ) -> None:
        """Manually record an agent handoff.

        Args:
            from_agent: Name of the agent handing off.
            to_agent: Name of the agent receiving control.
        """
        self._session.add_fact(
            fact_id=f"handoff_{from_agent}_to_{to_agent}_{int(time.time() * 1000)}",
            text=f"Handoff from {from_agent} to {to_agent}",
            confidence=1.0,
            source=from_agent,
        )

    def record_error(
        self,
        agent_name: str,
        error: BaseException,
        recoverable: bool = False,
    ) -> None:
        """Manually record an error.

        Args:
            agent_name: Name of the agent where error occurred.
            error: The exception that was raised.
            recoverable: Whether the error is recoverable.
        """
        self._session.error(
            category="agent_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=recoverable,
            suggested_action="Check agent configuration and input",
        )

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self._session.export(path)
        if self._verbose:
            event_count = len(self._session.chain.events)
            print(f"[LCTL] Exported {event_count} events to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL trace as a dictionary.

        Returns:
            The trace data as a dictionary.
        """
        return self._session.to_dict()


class AgentRunContext:
    """Context manager for tracing individual agent runs."""

    def __init__(
        self,
        session: LCTLSession,
        agent_name: str,
        input_summary: str = "",
        verbose: bool = False,
    ) -> None:
        """Initialize the agent run context.

        Args:
            session: LCTL session for recording events.
            agent_name: Name of the agent.
            input_summary: Summary of the input.
            verbose: Enable verbose output.
        """
        self._session = session
        self._agent_name = agent_name
        self._input_summary = input_summary
        self._verbose = verbose
        self._start_time: float = 0
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._output_summary: str = ""

    def __enter__(self) -> "AgentRunContext":
        """Start tracing the agent run."""
        self._start_time = time.time()
        self._session.step_start(
            agent=self._agent_name,
            intent="agent_run",
            input_summary=self._input_summary,
        )
        if self._verbose:
            print(f"[LCTL] Agent run started: {self._agent_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracing the agent run."""
        duration_ms = int((time.time() - self._start_time) * 1000)

        if exc_type is not None:
            self._session.step_end(
                agent=self._agent_name,
                outcome="error",
                output_summary=f"Error: {exc_type.__name__}: {str(exc_val)[:100]}",
                duration_ms=duration_ms,
                tokens_in=self._tokens_in,
                tokens_out=self._tokens_out,
            )
            self._session.error(
                category="agent_error",
                error_type=exc_type.__name__,
                message=str(exc_val),
                recoverable=False,
            )
        else:
            self._session.step_end(
                agent=self._agent_name,
                outcome="success",
                output_summary=self._output_summary,
                duration_ms=duration_ms,
                tokens_in=self._tokens_in,
                tokens_out=self._tokens_out,
            )

        if self._verbose:
            status = "error" if exc_type else "success"
            print(f"[LCTL] Agent run ended: {self._agent_name} ({status}, {duration_ms}ms)")

        return False

    async def __aenter__(self) -> "AgentRunContext":
        """Async version of __enter__."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async version of __exit__."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    def set_output(self, output_summary: str) -> None:
        """Set the output summary for the run.

        Args:
            output_summary: Summary of the output.
        """
        self._output_summary = _truncate(output_summary)

    def set_usage(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Set token usage for the run.

        Args:
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.
        """
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out

    def record_tool_call(
        self,
        tool: str,
        input_data: Any,
        output_data: Any,
        duration_ms: int = 0,
    ) -> None:
        """Record a tool call within this run.

        Args:
            tool: Name of the tool.
            input_data: Input to the tool.
            output_data: Output from the tool.
            duration_ms: Duration of the tool call.
        """
        self._session.tool_call(
            tool=tool,
            input_data=_truncate(str(input_data)),
            output_data=_truncate(str(output_data)),
            duration_ms=duration_ms,
        )


def trace_agent(
    agent: "Agent",
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False,
) -> "TracedAgent":
    """Wrap an OpenAI Agent with LCTL tracing.

    This returns a TracedAgent that automatically traces all runs.

    Args:
        agent: The OpenAI Agent to wrap.
        chain_id: Optional chain ID for the LCTL session.
        session: Optional existing LCTL session to use.
        verbose: Enable verbose output.

    Returns:
        TracedAgent wrapper with tracing enabled.

    Example:
        from agents import Agent
        from lctl.integrations.openai_agents import trace_agent

        agent = Agent(name="assistant", instructions="You are helpful")
        traced = trace_agent(agent, chain_id="my-agent")

        # Use traced.tracer to access LCTL functionality
        traced.tracer.export("trace.lctl.json")
    """
    _check_openai_agents_available()
    return TracedAgent(agent, chain_id=chain_id, session=session, verbose=verbose)


class TracedAgent:
    """Wrapper around an OpenAI Agent with LCTL tracing.

    Provides access to the underlying agent while maintaining
    a tracer for recording events.
    """

    def __init__(
        self,
        agent: "Agent",
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the traced agent.

        Args:
            agent: The OpenAI Agent to wrap.
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            verbose: Enable verbose output.
        """
        _check_openai_agents_available()

        self._agent = agent
        self._tracer = LCTLOpenAIAgentTracer(
            chain_id=chain_id,
            session=session,
            verbose=verbose,
        )

        agent_name = getattr(agent, "name", "agent")
        if hasattr(agent, "instructions") and agent.instructions:
            self._tracer.session.add_fact(
                fact_id=f"agent_{agent_name}_config",
                text=f"Agent '{agent_name}' configured with instructions: {_truncate(agent.instructions, 300)}",
                confidence=1.0,
                source="setup",
            )

    @property
    def agent(self) -> "Agent":
        """Access the underlying OpenAI Agent."""
        return self._agent

    @property
    def tracer(self) -> LCTLOpenAIAgentTracer:
        """Access the LCTL tracer."""
        return self._tracer

    @property
    def run_config(self) -> Dict[str, Any]:
        """Get run configuration with LCTL hooks.

        Returns:
            Dictionary with hooks configuration.
        """
        return self._tracer.run_config

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self._tracer.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL trace as a dictionary.

        Returns:
            The trace data as a dictionary.
        """
        return self._tracer.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying agent."""
        return getattr(self._agent, name)


def is_available() -> bool:
    """Check if OpenAI Agents SDK integration is available.

    Returns:
        True if OpenAI Agents SDK is installed, False otherwise.
    """
    return OPENAI_AGENTS_AVAILABLE


__all__ = [
    "OPENAI_AGENTS_AVAILABLE",
    "OpenAIAgentsNotAvailableError",
    "LCTLRunHooks",
    "LCTLTracingProcessor",
    "LCTLOpenAIAgentTracer",
    "AgentRunContext",
    "TracedAgent",
    "trace_agent",
    "is_available",
]
