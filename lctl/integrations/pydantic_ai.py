"""PydanticAI integration for LCTL.

This module provides automatic tracing for PydanticAI agents.
It captures agent run execution, tool calls, and LLM interactions by
wrapping the agent's run method and instrumenting dependencies.

Requires:
    - pydantic-ai>=0.0.14
    - nest_asyncio (for event loop handling)
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from ..core.events import EventType
from ..core.session import LCTLSession

try:
    import pydantic_ai
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models import Model
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        UserPromptPart,
        ToolCallPart,
        ToolReturnPart,
        RetryPromptPart,
    )
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False


class PydanticAINotAvailableError(ImportError):
    """Raised when PydanticAI is not installed."""
    def __init__(self) -> None:
        super().__init__(
            "PydanticAI is not installed. Install with: pip install pydantic-ai"
        )


def _check_pydantic_ai_available() -> None:
    if not PYDANTIC_AI_AVAILABLE:
        raise PydanticAINotAvailableError()


T = TypeVar("T")


class LCTLPydanticAITracer:
    """Tracer for PydanticAI agents."""

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ):
        """Initialize tracer.

        Args:
            chain_id: Optional unique identifier for the trace chain
            session: Optional existing LCTLSession
            verbose: Enable verbose logging
        """
        _check_pydantic_ai_available()
        self.session = session or LCTLSession(
            chain_id=chain_id or f"pydantic-ai-{id(self)}"
        )
        self._verbose = verbose

    @property
    def chain(self):
        return self.session.chain

    def trace_agent(self, agent: Agent) -> Agent:
        """Attach tracing to a PydanticAI agent.
        
        This wraps the agent's .run() and .run_sync() methods.
        """
        
        # Wrap run (async)
        original_run = agent.run
        
        @functools.wraps(original_run)
        async def traced_run(user_prompt: str, **kwargs: Any) -> Any:
            with self.trace_run_context(agent.name or "agent", user_prompt) as ctx:
                try:
                    result = await original_run(user_prompt, **kwargs)
                    usage = result.usage()
                    ctx.record_success(
                        output_summary=str(result.data),
                        tokens_in=usage.input_tokens or 0,
                        tokens_out=usage.output_tokens or 0
                    )
                    
                    # Record tool calls if accessible from result
                    # Note: PydanticAI result structure might vary, 
                    # we attempt to inspect messages if available
                    if hasattr(result, 'all_messages'):
                        self._record_messages(ctx, result.all_messages())
                        
                    return result
                except Exception as e:
                    ctx.record_error(e)
                    raise

        agent.run = traced_run

        # Wrap run_sync
        original_run_sync = agent.run_sync

        @functools.wraps(original_run_sync)
        def traced_run_sync(user_prompt: str, **kwargs: Any) -> Any:
            # We can reuse the async context manager even in sync code 
            # if we are careful, but safer to use manual start/end
            event_id = self.session.step_start(
                agent=agent.name or "agent",
                intent="run_sync",
                input_summary=user_prompt
            )
            try:
                result = original_run_sync(user_prompt, **kwargs)
                
                usage = result.usage()
                tokens_in = usage.input_tokens or 0
                tokens_out = usage.output_tokens or 0
                
                output = str(result.data)
                
                self.session.step_end(
                    outcome="success",
                    output_summary=output,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out
                )

                if hasattr(result, 'all_messages'):
                    # Manually record intermediate steps as facts/tools
                    self._record_messages_sync(result.all_messages())

                return result
            except Exception as e:
                self.session.error(
                    category="execution_error",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=False
                )
                self.session.step_end(outcome="error")
                raise

        agent.run_sync = traced_run_sync
        
        return agent

    def _record_messages(self, ctx: AgentRunContext, messages: List[Any]) -> None:
        """Record tool calls and system prompts from message history."""
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        # We might not have the result yet, or it's in the next message
                        # For simplicity, we log the call now
                        ctx.record_tool_call(
                            tool=part.tool_name,
                            input_data=part.args,
                            output_data="<pending>", # PydanticAI separates call and return
                        )
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        # This is a return value from a tool
                        # We could link it to the call if we tracked IDs, 
                        # for now just log as a fact or update
                         self.session.add_fact(
                            fact_id=f"tool_return_{id(part)}",
                            text=f"Tool {part.tool_name} returned: {part.content}",
                            source="tool"
                        )

    def _record_messages_sync(self, messages: List[Any]) -> None:
        """Sync version of message recording."""
        # Reuse logic, just without context wrapper methods if needed
        # But here we can use session directly
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        self.session.tool_call(
                            tool=part.tool_name,
                            input_data=part.args,
                            output_data="<pending>"
                        )

    def trace_run_context(self, agent_name: str, input_summary: str = "") -> AgentRunContext:
        """Get a context manager for tracing a run."""
        return AgentRunContext(self.session, agent_name, input_summary, self._verbose)

    def export(self, path: str) -> None:
        """Export trace to file."""
        self.session.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Get trace as dictionary."""
        return self.session.to_dict()


class AgentRunContext:
    """Context manager for tracing an agent run."""

    def __init__(
        self,
        session: LCTLSession,
        agent_name: str,
        input_summary: str = "",
        verbose: bool = False,
    ):
        self.session = session
        self.agent_name = agent_name
        self.input_summary = input_summary
        self.verbose = verbose
        self._event_id = None

    def __enter__(self) -> "AgentRunContext":
        self._event_id = self.session.step_start(
            agent=self.agent_name,
            intent="run",
            input_summary=self.input_summary,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Error is handled by record_error usually, but if not:
            if self.session.chain.events[-1].type != EventType.ERROR:
                 self.session.error(
                    category="execution_error",
                    error_type=exc_type.__name__,
                    message=str(exc_val),
                    recoverable=False,
                )
            self.session.step_end(outcome="error")
        else:
            # step_end called by set_output/usage or here if not called yet
            # We check if the last event was already a step_end for this agent?
            # LCTLSession.step_end doesn't easily let us check "is closed".
            # But usually we call step_end explicitly.
            # If default exit, we close it.
            # TO avoid double closing, we can track state locally.
            pass

    def record_success(self, output_summary: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record success outcome with output and usage (triggers step_end)."""
        self.session.step_end(
            outcome="success",
            output_summary=output_summary,
            tokens_in=tokens_in,
            tokens_out=tokens_out
        )

    def set_output(self, output_summary: str) -> None:
        """Set the output summary for the run (triggers step_end).
        
        Deprecated: Use record_success for consistent usage tracking.
        """
        self.session.step_end(
            outcome="success",
            output_summary=output_summary
        )

    def set_usage(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Set usage metrics (adds fact).
        
        Note: If record_success was called, step_end is already emitted.
        This allows recording usage if it comes separately.
        """
        self.session.add_fact(
            fact_id=f"rec_usage_{id(self)}",
            text=f"Token Usage: {tokens_in} in, {tokens_out} out",
            source="system"
        )

    def record_tool_call(
        self,
        tool: str,
        input_data: Any,
        output_data: Any,
        duration_ms: int = 0,
    ) -> None:
        """Record a tool call."""
        self.session.tool_call(
            tool=tool,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
        )

    def record_error(self, error: Exception) -> None:
        """Record an error."""
        self.session.error(
            category="execution_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=False,
        )


def trace_agent(
    agent: Agent,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False,
) -> LCTLPydanticAITracer:
    """Convenience function to trace a PydanticAI agent.
    
    Args:
        agent: The PydanticAI Agent instance.
        chain_id: Optional chain ID.
        session: Optional session.
        verbose: Enable verbose logging.
        
    Returns:
        The tracer instance (which holds the session/chain). 
        The agent is modified in-place.
    """
    tracer = LCTLPydanticAITracer(chain_id, session, verbose)
    tracer.trace_agent(agent)
    return tracer
