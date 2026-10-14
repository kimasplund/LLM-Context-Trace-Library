"""PydanticAI integration for LCTL.

This module provides automatic tracing for PydanticAI agents.
It wraps the agent run methods to capture input/output and tool calls.

Usage:
    from lctl.integrations.pydantic_ai import trace_agent

    traced = trace_agent(my_agent, chain_id="my-workflow")
    result = await traced.run("Analyze this data")
    traced.export("trace.lctl.json")
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from ..core.session import LCTLSession
from .base import BaseTracer, install_command, truncate

try:
    from pydantic_ai import Agent
    from pydantic_ai.run import AgentRunResult

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.run import AgentRunResult


class LCTLPydanticAITracer(BaseTracer):
    """Tracer for PydanticAI agents.

    Captures agent execution, tool calls, and LLM interactions
    as LCTL events for debugging and analysis.

    Extends BaseTracer for standardized session management, thread safety,
    and automatic stale entry cleanup.

    Example:
        tracer = LCTLPydanticAITracer(chain_id="my-agent")
        # Use with TracedAgent for automatic instrumentation
        tracer.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        verbose: bool = False,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0,
    ):
        """Initialize the tracer.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            verbose: Whether to enable verbose logging.
            auto_cleanup: Whether to auto-cleanup stale entries.
            cleanup_interval: Cleanup interval in seconds (default 1 hour).

        Raises:
            ImportError: If PydanticAI is not installed.
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                f"PydanticAI is not installed. Install with: {install_command('pydantic-ai')}"
            )
        super().__init__(
            chain_id=chain_id,
            session=session,
            auto_cleanup=auto_cleanup,
            cleanup_interval=cleanup_interval,
        )
        self._verbose = verbose


_TOKEN_FIELDS = ("input_tokens", "request_tokens", "output_tokens", "response_tokens")


def _has_token_counts(usage: Any) -> bool:
    return any(isinstance(getattr(usage, a, None), int) for a in _TOKEN_FIELDS)


def _first_int(obj: Any, *names: str) -> int:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int):
            return value
    return 0


def _model_name(agent: Any) -> str:
    """Best-effort model name of an agent (``Model`` instance, ``"provider:model"`` string or None)."""
    model = getattr(agent, "model", None)
    if model is None:
        return "unknown"
    if isinstance(model, str):
        return model or "unknown"
    for attr in ("model_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    legacy = getattr(agent, "model_name", None)  # wrappers that expose the name directly
    if isinstance(legacy, str) and legacy:
        return legacy
    return type(model).__name__


def _usage_tokens(result: Any) -> tuple[int, int]:
    """Return (input, output) tokens from a run result across pydantic-ai versions.

    ``usage`` is a method on 1.x and a property on 2.x; the token fields were
    renamed ``request_tokens``/``response_tokens`` -> ``input_tokens``/``output_tokens``.
    """
    try:
        usage = result.usage
        # Old releases: usage() is a method. Newer 1.x keep a deprecated __call__
        # on the usage object itself, so only call it when it is not usable as is.
        if callable(usage) and not _has_token_counts(usage):
            usage = usage()
        tokens_in = _first_int(usage, "input_tokens", "request_tokens")
        tokens_out = _first_int(usage, "output_tokens", "response_tokens")
        return tokens_in, tokens_out
    except Exception:
        return 0, 0


class TracedAgent:
    """Wrapper for PydanticAI Agent with LCTL tracing.

    Automatically instruments the agent to capture:
    - Agent run start/end with input/output
    - Tool invocations with timing
    - Token usage statistics
    - Streaming responses
    - Errors and exceptions

    Example:
        from pydantic_ai import Agent
        from lctl.integrations.pydantic_ai import TracedAgent

        agent = Agent("openai:gpt-4")
        traced = TracedAgent(agent, chain_id="my-workflow")
        result = await traced.run("Analyze this")
        traced.export("trace.lctl.json")
    """

    def __init__(
        self,
        agent: Agent,
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        verbose: bool = False,
    ):
        """Initialize the traced agent wrapper.

        Args:
            agent: The PydanticAI agent to wrap.
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            verbose: Whether to enable verbose logging.
        """
        self.agent = agent
        self.tracer = LCTLPydanticAITracer(chain_id=chain_id, session=session, verbose=verbose)
        self._instrument_tools()

    def _instrument_tools(self):
        """Instrument agent tools to trace execution."""
        # PydanticAI v0.0.x / v1.x internal structure
        # Tools are stored in _function_toolset.tools
        if not hasattr(self.agent, "_function_toolset") or not self.agent._function_toolset:
            return

        if not hasattr(self.agent._function_toolset, "tools"):
            return

        for name, tool in self.agent._function_toolset.tools.items():
            if not hasattr(tool, "function"):
                continue

            # Avoid double instrumentation
            func = tool.function
            # Use strict check to handle Mocks
            if getattr(func, "_lctl_instrumented", False) is True:
                continue

            original_func = tool.function
            # Handle AsyncMock alongside normal coroutines
            is_async = inspect.iscoroutinefunction(original_func) or (
                isinstance(original_func, MagicMock)
                and getattr(original_func, "_is_coroutine", False)
            )

            if is_async:
                # Use default arguments to capture loop variables by value
                @wraps(original_func)
                async def wrapper(*args, _name=name, _func=original_func, **kwargs):
                    return await self._trace_tool_execution(_name, _func, *args, **kwargs)

            else:
                # Use default arguments to capture loop variables by value
                @wraps(original_func)
                def wrapper(*args, _name=name, _func=original_func, **kwargs):
                    return self._trace_tool_execution_sync(_name, _func, *args, **kwargs)

            wrapper._lctl_instrumented = True
            tool.function = wrapper

            # Also update function_schema.function if present
            if hasattr(tool, "function_schema") and hasattr(tool.function_schema, "function"):
                tool.function_schema.function = wrapper

    async def _trace_tool_execution(self, name: str, func: Callable, *args, **kwargs):
        """Trace execution of an async tool.

        Args:
            name: The tool name.
            func: The original tool function.
            *args: Positional arguments to the tool.
            **kwargs: Keyword arguments to the tool.

        Returns:
            The tool's return value.
        """
        # Extract input summary
        # args[0] might be RunContext if takes_ctx=True
        input_data = str(args) + str(kwargs)

        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)

            self.tracer.session.tool_call(
                tool=name,
                input_data=truncate(str(input_data)),
                output_data=truncate(str(result), max_length=500),
                duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.tool_call(
                tool=name,
                input_data=truncate(str(input_data)),
                output_data=f"Error: {e!s}",
                duration_ms=duration_ms,
            )
            raise

    def _trace_tool_execution_sync(self, name: str, func: Callable, *args, **kwargs):
        """Trace execution of a sync tool.

        Args:
            name: The tool name.
            func: The original tool function.
            *args: Positional arguments to the tool.
            **kwargs: Keyword arguments to the tool.

        Returns:
            The tool's return value.
        """
        input_data = str(args) + str(kwargs)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)

            self.tracer.session.tool_call(
                tool=name,
                input_data=truncate(str(input_data)),
                output_data=truncate(str(result), max_length=500),
                duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.tool_call(
                tool=name,
                input_data=truncate(str(input_data)),
                output_data=f"Error: {e!s}",
                duration_ms=duration_ms,
            )
            raise

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file.

        Args:
            path: File path to export to.
        """
        self.tracer.export(path)

    def to_dict(self) -> dict[str, Any]:
        """Export the LCTL trace as a dictionary.

        Returns:
            Dictionary representation of the trace.
        """
        return self.tracer.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes (``name``, ``override``, ...) to the wrapped agent."""
        # Look up `agent` via __dict__ so a half-initialised or unpickled
        # instance raises AttributeError instead of recursing forever.
        agent = self.__dict__.get("agent")
        if agent is None:
            raise AttributeError(name)
        return getattr(agent, name)

    def _agent_name(self) -> str:
        return getattr(self.agent, "name", "pydantic_ai_agent") or "pydantic_ai_agent"

    def _begin(self, intent: str, args: tuple, kwargs: dict) -> float:
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")
        self.tracer.session.step_start(
            agent=self._agent_name(),
            intent=intent,
            input_summary=truncate(str(user_prompt)),
        )
        return time.time()

    def _record_success(self, result: Any, start_time: float) -> None:
        """Record step_end and the LLM trace for a finished run (1.x and 2.x results)."""
        duration_ms = int((time.time() - start_time) * 1000)

        # `.output` on current releases, `.data` on old ones.
        if hasattr(result, "output"):
            output_summary = str(result.output)
        elif hasattr(result, "data"):
            output_summary = str(result.data)
        else:
            output_summary = str(result)

        tokens_in, tokens_out = _usage_tokens(result)

        # llm_trace first: it resolves its agent/span from the still-open step.
        if hasattr(result, "all_messages"):
            try:
                messages = []
                for msg in result.all_messages():
                    if hasattr(msg, "model_dump"):
                        messages.append(msg.model_dump(mode="json"))
                    else:
                        messages.append(str(msg))
                self.tracer.session.llm_trace(
                    messages=messages,
                    response=output_summary,
                    model=_model_name(self.agent),
                    usage={"input": tokens_in, "output": tokens_out},
                )
            except Exception:  # noqa: S110  # tracing must never break the host application
                pass

        self.tracer.session.step_end(
            outcome="success",
            output_summary=output_summary,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def _record_failure(self, exc: Exception, start_time: float) -> None:
        duration_ms = int((time.time() - start_time) * 1000)
        self.tracer.session.error(
            category="execution_error",
            error_type=type(exc).__name__,
            message=str(exc),
            recoverable=False,
        )
        self.tracer.session.step_end(outcome="error", duration_ms=duration_ms)

    async def run(
        self,
        *args,
        **kwargs,
    ) -> AgentRunResult:
        """Run the agent with tracing.

        Args:
            *args: Positional arguments passed to agent.run().
            **kwargs: Keyword arguments passed to agent.run().

        Returns:
            The agent's run result.

        Raises:
            Any exception raised by the agent.
        """
        start_time = self._begin("run", args, kwargs)
        try:
            result = await self.agent.run(*args, **kwargs)
        except Exception as e:
            self._record_failure(e, start_time)
            raise
        self._record_success(result, start_time)
        return result

    def run_sync(
        self,
        *args,
        **kwargs,
    ) -> AgentRunResult:
        """Run the agent synchronously with tracing.

        Delegates to the agent's own ``run_sync`` and records it like ``run``.

        Args:
            *args: Positional arguments passed to agent.run_sync().
            **kwargs: Keyword arguments passed to agent.run_sync().

        Returns:
            The agent's run result.

        Raises:
            Any exception raised by the agent.
        """
        start_time = self._begin("run_sync", args, kwargs)
        try:
            result = self.agent.run_sync(*args, **kwargs)
        except Exception as e:
            self._record_failure(e, start_time)
            raise
        self._record_success(result, start_time)
        return result

    @asynccontextmanager
    async def run_stream(self, *args, **kwargs):
        """Run the agent with streaming and tracing.

        Args:
            *args: Positional arguments passed to agent.run_stream().
            **kwargs: Keyword arguments passed to agent.run_stream().

        Yields:
            TracedStreamedRunResult wrapper for the stream.

        Raises:
            Any exception raised by the agent.
        """
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")
        agent_name = getattr(self.agent, "name", "pydantic_ai_agent") or "pydantic_ai_agent"

        start_time = time.time()

        self.tracer.session.step_start(
            agent=agent_name,
            intent="run_stream",
            input_summary=truncate(str(user_prompt)),
        )

        traced_result = None
        try:
            async with self.agent.run_stream(*args, **kwargs) as result:
                traced_result = TracedStreamedRunResult(
                    result, self.tracer, agent_name, start_time, _model_name(self.agent)
                )
                yield traced_result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.error(
                category="execution_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False,
            )
            self.tracer.session.step_end(outcome="error", duration_ms=duration_ms)
            raise
        finally:
            # Ensure step_end is called even if consumer abandons stream early
            if traced_result is not None and not traced_result._finalized:
                traced_result._finalize_trace()


class TracedStreamedRunResult:
    """Wrapper for StreamedRunResult to trace streaming.

    Automatically captures stream events including:
    - Stream start/end
    - Individual chunks
    - Token usage
    - LLM trace with message history

    This wrapper ensures proper cleanup even if the consumer
    abandons the stream early.
    """

    def __init__(
        self,
        result: Any,
        tracer: LCTLPydanticAITracer,
        agent_name: str,
        start_time: float,
        model_name: str = "unknown",
    ):
        """Initialize the traced stream result.

        Args:
            result: The underlying StreamedRunResult.
            tracer: The LCTL tracer instance.
            agent_name: Name of the agent for event recording.
            start_time: Start time for duration calculation.
            model_name: Model name recorded on the LLM trace.
        """
        self._model_name = model_name
        self._result = result
        self._tracer = tracer
        self._agent_name = agent_name
        self._start_time = start_time
        self._accumulated_content = ""
        self._stream_id = f"stream-{id(result)}"
        self._chunk_index = 0
        self._finalized = False

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying result."""
        return getattr(self._result, name)

    async def stream_text(self, *args, **kwargs) -> AsyncIterator[str]:
        """Trace stream_text iteration with chunk-level events.

        Args:
            *args: Positional arguments to stream_text.
            **kwargs: Keyword arguments to stream_text.

        Yields:
            Text chunks from the stream.
        """
        # Record stream start
        self._tracer.session.stream_start(
            stream_id=self._stream_id,
            description=f"Streaming response from {self._agent_name}",
        )

        try:
            is_delta = bool(kwargs.get("delta", False))
            async for chunk in self._result.stream_text(*args, **kwargs):
                # stream_text yields cumulative text unless delta=True; record
                # only the new part so the total is the final text exactly once.
                if is_delta:
                    new_text = chunk
                    self._accumulated_content += chunk
                elif chunk.startswith(self._accumulated_content):
                    new_text = chunk[len(self._accumulated_content) :]
                    self._accumulated_content = chunk
                else:  # text was rewritten rather than extended
                    new_text = chunk
                    self._accumulated_content = chunk
                self._tracer.session.stream_chunk(
                    stream_id=self._stream_id, content=new_text, index=self._chunk_index
                )
                self._chunk_index += 1
                yield chunk
        finally:
            # Ensure finalization happens even if iteration is interrupted
            if not self._finalized:
                self._finalize_trace()

    def _finalize_trace(self):
        """Record final step end with usage."""
        if self._finalized:
            return
        self._finalized = True

        duration_ms = int((time.time() - self._start_time) * 1000)

        tokens_in, tokens_out = _usage_tokens(self._result)

        # Record stream end
        self._tracer.session.stream_end(
            stream_id=self._stream_id,
            total_content=self._accumulated_content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # Record the LLM trace before step_end so it lands inside the step's span.
        if hasattr(self._result, "all_messages"):
            try:
                messages = []
                for msg in self._result.all_messages():
                    if hasattr(msg, "model_dump"):
                        messages.append(msg.model_dump(mode="json"))
                    else:
                        messages.append(str(msg))
                self._tracer.session.llm_trace(
                    messages=messages,
                    response=self._accumulated_content,
                    model=self._model_name,
                    usage={"input": tokens_in, "output": tokens_out},
                )
            except Exception:  # noqa: S110  # tracing must never break the host application
                pass

        self._tracer.session.step_end(
            outcome="success",
            output_summary=self._accumulated_content,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def trace_agent(
    agent: Agent,
    chain_id: str | None = None,
    session: LCTLSession | None = None,
    verbose: bool = False,
) -> TracedAgent:
    """Wrap a PydanticAI agent with LCTL tracing.

    This is the main entry point for adding LCTL tracing to a PydanticAI agent.
    The returned TracedAgent can be used exactly like the original agent,
    but all operations will be traced.

    Args:
        agent: The PydanticAI agent to wrap.
        chain_id: Optional chain ID for the LCTL session.
        session: Optional existing LCTL session to use.
        verbose: Whether to enable verbose logging.

    Returns:
        TracedAgent wrapper with tracing enabled.

    Example:
        from pydantic_ai import Agent
        from lctl.integrations.pydantic_ai import trace_agent

        agent = Agent("openai:gpt-4")
        traced = trace_agent(agent, chain_id="my-workflow")
        result = await traced.run("Analyze this")
        traced.export("trace.lctl.json")
    """
    return TracedAgent(agent, chain_id, session, verbose)


def is_available() -> bool:
    """Check if PydanticAI integration is available.

    Returns:
        True if PydanticAI is installed, False otherwise.
    """
    return PYDANTIC_AI_AVAILABLE


__all__ = [
    "LCTLPydanticAITracer",
    "TracedAgent",
    "TracedStreamedRunResult",
    "is_available",
    "trace_agent",
]
