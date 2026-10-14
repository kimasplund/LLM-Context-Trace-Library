"""LCTL Session - Context manager for manual instrumentation."""

import contextvars
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from .events import Chain, Event, EventType
from .ids import new_id
from .redaction import get_redactor

if TYPE_CHECKING:
    from ..streaming.emitter import EventEmitter


# Sentinel: "attach to the innermost open span in the current context".
_UNSET: Any = object()


class LCTLSession:
    """Session for recording LCTL events.

    Usage:
        with LCTLSession() as session:
            result = llm.complete("Analyze this")
            session.add_fact("F1", "Analysis complete", confidence=0.9)

        session.export("trace.lctl.json")

    With real-time streaming:
        from lctl.streaming import EventEmitter

        emitter = EventEmitter()
        session = LCTLSession(emitter=emitter)

        @emitter.on("event")
        def handle_event(event):
            print(f"New event: {event.type}")
    """

    def __init__(
        self,
        chain_id: str | None = None,
        emitter: Optional["EventEmitter"] = None,
        redaction_enabled: bool = True,
    ):
        """Initialize the session.

        Args:
            chain_id: Optional chain ID. Generated if not provided.
            emitter: Optional EventEmitter for real-time streaming.
            redaction_enabled: Whether to redact sensitive data from events.
        """
        self.chain = Chain(id=chain_id or str(uuid4())[:8])
        self._seq = 0
        self._current_agent: str | None = None
        # Open spans per execution context, innermost last: ((span_id, agent), ...).
        # Immutable tuples, so tasks that copy the context never share a stack.
        self._span_stack: contextvars.ContextVar[tuple[tuple[str, str], ...]] = (
            contextvars.ContextVar(f"lctl_span_stack_{id(self)}", default=())
        )
        self._span_agents: dict[str, str] = {}
        self._emitter: EventEmitter | None = emitter
        self._event_listeners: list[Callable[[Event], None]] = []
        self._redaction_enabled = redaction_enabled

        if self._emitter is not None:
            self._emitter.chain_id = self.chain.id

    def __enter__(self) -> "LCTLSession":
        if self._emitter is not None:
            self._emitter.start_chain(self.chain.id)
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb):
        if exc_type is not None:
            with suppress(Exception):
                self.error(
                    category="execution_error",
                    error_type=exc_type.__name__,
                    message=str(exc_val),
                    recoverable=False,
                )
        if self._emitter is not None:
            self._emitter.end_chain()
        return False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _add_event(
        self,
        event_type: EventType,
        agent: str,
        data: dict[str, Any],
        *,
        span_id: str | None = _UNSET,
        parent_span_id: str | None = None,
    ) -> Event:
        if self._redaction_enabled:
            redactor = get_redactor()
            data = redactor.redact_event_data(data)

        event = Event(
            seq=self._next_seq(),
            type=event_type,
            timestamp=self._now(),
            agent=agent,
            data=data,
            id=new_id(),
            span_id=self.current_span_id if span_id is _UNSET else span_id,
            parent_span_id=parent_span_id,
        )
        self.chain.add_event(event)

        if self._emitter is not None:
            self._emitter.emit_lctl_event(event)

        for listener in self._event_listeners:
            with suppress(Exception):
                listener(event)

        return event

    @property
    def emitter(self) -> Optional["EventEmitter"]:
        """Get the associated EventEmitter."""
        return self._emitter

    @emitter.setter
    def emitter(self, value: Optional["EventEmitter"]) -> None:
        """Set the EventEmitter for real-time streaming."""
        self._emitter = value
        if self._emitter is not None:
            self._emitter.chain_id = self.chain.id

    def add_event_listener(self, listener: Callable[[Event], None]) -> None:
        """Add a listener for events.

        Args:
            listener: Function to call when an event is recorded.
        """
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[Event], None]) -> bool:
        """Remove an event listener.

        Args:
            listener: The listener to remove.

        Returns:
            True if the listener was found and removed.
        """
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)
            return True
        return False

    @property
    def current_span_id(self) -> str | None:
        """The innermost open span in the current execution context, if any."""
        stack = self._span_stack.get()
        return stack[-1][0] if stack else None

    def _resolve_agent(self, span_id: str | None = None) -> str:
        """Agent for an event: the given span's agent, else the innermost open span's."""
        if span_id is not None and span_id in self._span_agents:
            return self._span_agents[span_id]
        stack = self._span_stack.get()
        if stack:
            return stack[-1][1]
        return self._current_agent or "unknown"

    def _innermost_span(self, agent: str | None = None) -> str | None:
        """Innermost open span, preferring one owned by ``agent`` when given."""
        stack = self._span_stack.get()
        if agent is not None:
            for span_id, span_agent in reversed(stack):
                if span_agent == agent:
                    return span_id
        return stack[-1][0] if stack else None

    def step_start(
        self,
        agent: str,
        intent: str,
        input_summary: str = "",
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> int:
        """Record start of an agent step (opens a span). Returns sequence number."""
        new_span = span_id or new_id()
        parent = parent_span_id if parent_span_id is not None else self.current_span_id
        self._current_agent = agent
        self._span_agents[new_span] = agent
        event = self._add_event(
            EventType.STEP_START,
            agent,
            {"intent": intent, "input_summary": input_summary},
            span_id=new_span,
            parent_span_id=parent,
        )
        self._span_stack.set((*self._span_stack.get(), (new_span, agent)))
        return event.seq

    def step_end(
        self,
        agent: str | None = None,
        outcome: str = "success",
        output_summary: str = "",
        duration_ms: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        *,
        span_id: str | None = None,
    ) -> int:
        """Record end of an agent step (closes ``span_id`` or the innermost open span)."""
        target = span_id if span_id is not None else self._innermost_span(agent)
        resolved_agent = (
            agent
            or (self._span_agents.get(target) if target else None)
            or self._current_agent
            or "unknown"
        )
        event = self._add_event(
            EventType.STEP_END,
            resolved_agent,
            {
                "outcome": outcome,
                "output_summary": output_summary,
                "duration_ms": duration_ms,
                "tokens": {"input": tokens_in, "output": tokens_out},
            },
            span_id=target,
        )
        if target is not None:
            self._span_stack.set(tuple(e for e in self._span_stack.get() if e[0] != target))
        self._current_agent = None
        return event.seq

    @contextmanager
    def span(
        self,
        agent: str,
        intent: str,
        input_summary: str = "",
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ):
        """Trace a block as a span; yields its span_id.

        Usage:
            with session.span("analyzer", "analyze", "code.py") as span_id:
                do_work()

        Ends with outcome "success", or "error" followed by an ERROR event for the
        span when an exception escapes. The exception is always re-raised, and a
        failure inside tracing never masks or replaces it.
        """
        start = time.time()
        try:
            self.step_start(
                agent, intent, input_summary, span_id=span_id, parent_span_id=parent_span_id
            )
            opened = self.current_span_id
        except Exception:
            opened = None

        if opened is None:
            # Tracing could not start: run the block untraced (outside any handler,
            # so a user exception isn't chained to the tracing failure).
            yield span_id or ""
            return

        try:
            yield opened
        except BaseException as exc:
            duration_ms = int((time.time() - start) * 1000)
            with suppress(Exception):
                self.step_end(outcome="error", duration_ms=duration_ms, span_id=opened)
            if isinstance(exc, Exception):
                with suppress(Exception):
                    self.error(
                        category="execution_error",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        recoverable=False,
                        span_id=opened,
                    )
            raise
        else:
            with suppress(Exception):
                self.step_end(
                    outcome="success",
                    duration_ms=int((time.time() - start) * 1000),
                    span_id=opened,
                )

    def add_fact(
        self, fact_id: str, text: str, confidence: float = 1.0, source: str | None = None
    ) -> int:
        """Add a new fact."""
        agent = source or self._resolve_agent()
        event = self._add_event(
            EventType.FACT_ADDED,
            agent,
            {"id": fact_id, "text": text, "confidence": confidence, "source": source or agent},
        )
        return event.seq

    def modify_fact(
        self,
        fact_id: str,
        text: str | None = None,
        confidence: float | None = None,
        reason: str = "",
    ) -> int:
        """Modify an existing fact."""
        agent = self._resolve_agent()
        data: dict[str, Any] = {"id": fact_id, "reason": reason}
        if text is not None:
            data["text"] = text
        if confidence is not None:
            data["confidence"] = confidence

        event = self._add_event(EventType.FACT_MODIFIED, agent, data)
        return event.seq

    def tool_call(
        self,
        tool: str,
        input_data: Any,
        output_data: Any,
        duration_ms: int = 0,
        agent: str | None = None,
        *,
        span_id: str | None = None,
    ) -> int:
        """Record a tool invocation."""
        agent = agent or self._resolve_agent(span_id)
        event = self._add_event(
            EventType.TOOL_CALL,
            agent,
            {"tool": tool, "input": input_data, "output": output_data, "duration_ms": duration_ms},
            span_id=_UNSET if span_id is None else span_id,
        )
        return event.seq

    def llm_trace(
        self,
        messages: list[dict[str, Any]],
        response: str,
        model: str,
        usage: dict[str, int] | None = None,
        duration_ms: int = 0,
        *,
        span_id: str | None = None,
    ) -> int:
        """Record an LLM trace."""
        agent = self._resolve_agent(span_id)
        event = self._add_event(
            EventType.LLM_TRACE,
            agent,
            {
                "messages": messages,
                "response": response,
                "model": model,
                "usage": usage or {},
                "duration_ms": duration_ms,
            },
            span_id=_UNSET if span_id is None else span_id,
        )
        return event.seq

    def error(
        self,
        category: str,
        error_type: str,
        message: str,
        recoverable: bool = True,
        suggested_action: str = "",
        *,
        span_id: str | None = None,
    ) -> int:
        """Record an error."""
        agent = self._resolve_agent(span_id)
        event = self._add_event(
            EventType.ERROR,
            agent,
            {
                "category": category,
                "type": error_type,
                "message": message,
                "recoverable": recoverable,
                "suggested_action": suggested_action,
            },
            span_id=_UNSET if span_id is None else span_id,
        )
        return event.seq

    def checkpoint(self, facts_snapshot: dict | None = None) -> int:
        """Create a checkpoint for fast replay."""
        from .events import ReplayEngine

        # If no snapshot provided, compute current state
        if facts_snapshot is None:
            engine = ReplayEngine(self.chain)
            state = engine.replay_all()
            facts_snapshot = state.facts

        event = self._add_event(
            EventType.CHECKPOINT,
            "system",
            {"state_hash": str(hash(str(facts_snapshot)))[:8], "facts_snapshot": facts_snapshot},
        )
        return event.seq

    def stream_start(self, stream_id: str, description: str = "") -> int:
        """Start a streaming response.

        Args:
            stream_id: Unique identifier for this stream
            description: Optional description of what's being streamed

        Returns:
            Event sequence number
        """
        agent = self._resolve_agent()
        event = self._add_event(
            EventType.STREAM_START, agent, {"stream_id": stream_id, "description": description}
        )
        return event.seq

    def stream_chunk(self, stream_id: str, content: str, index: int = 0) -> int:
        """Record a streaming chunk.

        Args:
            stream_id: Identifier of the stream this chunk belongs to
            content: The chunk content
            index: Optional chunk index for ordering

        Returns:
            Event sequence number
        """
        agent = self._resolve_agent()
        event = self._add_event(
            EventType.STREAM_CHUNK,
            agent,
            {"stream_id": stream_id, "content": content, "index": index},
        )
        return event.seq

    def stream_end(
        self, stream_id: str, total_content: str = "", tokens_in: int = 0, tokens_out: int = 0
    ) -> int:
        """End a streaming response.

        Args:
            stream_id: Identifier of the stream
            total_content: Complete accumulated content
            tokens_in: Input tokens used
            tokens_out: Output tokens generated

        Returns:
            Event sequence number
        """
        agent = self._resolve_agent()
        event = self._add_event(
            EventType.STREAM_END,
            agent,
            {
                "stream_id": stream_id,
                "total_content": total_content,
                "tokens": {"input": tokens_in, "output": tokens_out},
            },
        )
        return event.seq

    def export(self, path: str) -> None:
        """Export chain to file.

        Raises:
            PermissionError: If write permission is denied.
            OSError: If the file cannot be written.
        """
        export_path = Path(path)

        # Check if parent directory exists
        if not export_path.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {export_path.parent}")

        try:
            self.chain.save(export_path)
        except PermissionError as err:
            raise PermissionError(f"Permission denied writing to: {path}") from err
        except OSError as e:
            raise OSError(f"Failed to export chain to {path}: {e}") from e

    def to_dict(self) -> dict[str, Any]:
        """Export chain as dictionary."""
        return self.chain.to_dict()


@contextmanager
def traced_step(session: LCTLSession, agent: str, intent: str, input_summary: str = ""):
    """Context manager for tracing a step.

    Usage:
        with traced_step(session, "analyzer", "analyze", "code.py"):
            result = do_analysis()

    Note: If tracing itself fails, the original exception is still raised.
    """
    with session.span(agent, intent, input_summary):
        yield
