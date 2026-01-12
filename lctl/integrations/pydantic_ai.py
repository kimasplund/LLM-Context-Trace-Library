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

import asyncio
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Optional
from unittest.mock import MagicMock

from ..core.session import LCTLSession

try:
    from pydantic_ai import Agent
    from pydantic_ai.run import AgentRunResult

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.run import AgentRunResult


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries.

    Args:
        text: The text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        The original text if within limit, otherwise truncated with ellipsis.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class LCTLPydanticAITracer:
    """Tracer for PydanticAI agents.

    Captures agent execution, tool calls, and LLM interactions
    as LCTL events for debugging and analysis.

    Example:
        tracer = LCTLPydanticAITracer(chain_id="my-agent")
        # Use with TracedAgent for automatic instrumentation
        tracer.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ):
        """Initialize the tracer.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            verbose: Whether to enable verbose logging.

        Raises:
            ImportError: If PydanticAI is not installed.
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "PydanticAI is not installed. Install with: pip install lctl[pydantic-ai]"
            )
        self.session = session or LCTLSession(chain_id=chain_id)
        self._verbose = verbose

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self.session.chain

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self.session.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL chain as a dictionary.

        Returns:
            Dictionary representation of the chain.
        """
        return self.session.to_dict()


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
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
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
            is_async = asyncio.iscoroutinefunction(original_func) or (
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
                input_data=_truncate(str(input_data)),
                output_data=_truncate(str(result), max_length=500),
                duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.tool_call(
                tool=name,
                input_data=_truncate(str(input_data)),
                output_data=f"Error: {str(e)}",
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
                input_data=_truncate(str(input_data)),
                output_data=_truncate(str(result), max_length=500),
                duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.tool_call(
                tool=name,
                input_data=_truncate(str(input_data)),
                output_data=f"Error: {str(e)}",
                duration_ms=duration_ms,
            )
            raise

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file.

        Args:
            path: File path to export to.
        """
        self.tracer.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL trace as a dictionary.

        Returns:
            Dictionary representation of the trace.
        """
        return self.tracer.to_dict()

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
        # Check signature of agent.run
        # Typically run(user_prompt, ...)
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")
        agent_name = getattr(self.agent, "name", "pydantic_ai_agent") or "pydantic_ai_agent"

        start_time = time.time()

        self.tracer.session.step_start(
            agent=agent_name,
            intent="run",
            input_summary=_truncate(str(user_prompt)),
        )

        try:
            result = await self.agent.run(*args, **kwargs)

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract output
            # result.data exists for structured results, result.output might be for unstructured?
            # Adjust based on observed structure
            if hasattr(result, "data"):
                output_summary = str(result.data)
            elif hasattr(result, "output"):
                output_summary = str(result.output)
            else:
                output_summary = str(result)

            # Extract usage if available
            tokens_in = 0
            tokens_out = 0

            # Currently PydanticAI might expose usage in different ways
            # Check result.usage() method or attribute
            if hasattr(result, "usage"):
                usage = result.usage()
                tokens_in = usage.request_tokens or 0
                tokens_out = usage.response_tokens or 0

            self.tracer.session.step_end(
                outcome="success",
                output_summary=output_summary,
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

            # Record LLM Trace
            # Capture full message history
            if hasattr(result, "all_messages"):
                messages = []
                try:
                    for msg in result.all_messages():
                        # PydanticAI messages are Pydantic models
                        if hasattr(msg, "model_dump"):
                            messages.append(msg.model_dump(mode="json"))
                        else:
                            messages.append(str(msg))

                    self.tracer.session.llm_trace(
                        messages=messages,
                        response=output_summary,
                        model=getattr(self.agent, "model_name", "unknown"),
                        usage={"input": tokens_in, "output": tokens_out},
                    )
                except Exception:
                    pass

            return result

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
            input_summary=_truncate(str(user_prompt)),
        )

        traced_result = None
        try:
            async with self.agent.run_stream(*args, **kwargs) as result:
                traced_result = TracedStreamedRunResult(result, self.tracer, agent_name, start_time)
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
    ):
        """Initialize the traced stream result.

        Args:
            result: The underlying StreamedRunResult.
            tracer: The LCTL tracer instance.
            agent_name: Name of the agent for event recording.
            start_time: Start time for duration calculation.
        """
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
            async for chunk in self._result.stream_text(*args, **kwargs):
                self._accumulated_content += chunk
                # Emit stream chunk event
                self._tracer.session.stream_chunk(
                    stream_id=self._stream_id, content=chunk, index=self._chunk_index
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

        try:
            usage = self._result.usage()
            tokens_in = usage.request_tokens or 0
            tokens_out = usage.response_tokens or 0
        except Exception:
            tokens_in = 0
            tokens_out = 0

        # Record stream end
        self._tracer.session.stream_end(
            stream_id=self._stream_id,
            total_content=self._accumulated_content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        self._tracer.session.step_end(
            outcome="success",
            output_summary=self._accumulated_content,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # Record LLM Trace
        if hasattr(self._result, "all_messages"):
            messages = []
            try:
                for msg in self._result.all_messages():
                    if hasattr(msg, "model_dump"):
                        messages.append(msg.model_dump(mode="json"))
                    else:
                        messages.append(str(msg))

                # Get model name from the agent if accessible through tracer
                model_name = "unknown"
                if hasattr(self._tracer, "session") and hasattr(self._tracer.session, "chain"):
                    # Try to get model from chain metadata if available
                    chain = self._tracer.session.chain
                    if hasattr(chain, "metadata") and isinstance(chain.metadata, dict):
                        model_name = chain.metadata.get("model", "unknown")

                self._tracer.session.llm_trace(
                    messages=messages,
                    response=self._accumulated_content,
                    model=model_name,
                    usage={"input": tokens_in, "output": tokens_out},
                )
            except Exception:
                pass


def trace_agent(
    agent: Agent,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
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
    "trace_agent",
    "is_available",
]
