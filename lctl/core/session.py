"""LCTL Session - Context manager for manual instrumentation."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from uuid import uuid4

from .events import Chain, Event, EventType

if TYPE_CHECKING:
    from ..streaming.emitter import EventEmitter


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
        chain_id: Optional[str] = None,
        emitter: Optional["EventEmitter"] = None
    ):
        """Initialize the session.

        Args:
            chain_id: Optional chain ID. Generated if not provided.
            emitter: Optional EventEmitter for real-time streaming.
        """
        self.chain = Chain(id=chain_id or str(uuid4())[:8])
        self._seq = 0
        self._current_agent: Optional[str] = None
        self._emitter: Optional["EventEmitter"] = emitter
        self._event_listeners: list[Callable[[Event], None]] = []

        if self._emitter is not None:
            self._emitter.chain_id = self.chain.id

    def __enter__(self) -> "LCTLSession":
        if self._emitter is not None:
            self._emitter.start_chain(self.chain.id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            try:
                self.error(
                    category="execution_error",
                    error_type=exc_type.__name__,
                    message=str(exc_val),
                    recoverable=False
                )
            except Exception:
                pass
        if self._emitter is not None:
            self._emitter.end_chain()
        return False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _add_event(self, event_type: EventType, agent: str, data: Dict[str, Any]) -> Event:
        event = Event(
            seq=self._next_seq(),
            type=event_type,
            timestamp=self._now(),
            agent=agent,
            data=data
        )
        self.chain.add_event(event)

        if self._emitter is not None:
            self._emitter.emit_lctl_event(event)

        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception:
                pass

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

    def step_start(self, agent: str, intent: str, input_summary: str = "") -> int:
        """Record start of an agent step. Returns sequence number."""
        self._current_agent = agent
        event = self._add_event(EventType.STEP_START, agent, {
            "intent": intent,
            "input_summary": input_summary
        })
        return event.seq

    def step_end(
        self,
        agent: Optional[str] = None,
        outcome: str = "success",
        output_summary: str = "",
        duration_ms: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0
    ) -> int:
        """Record end of an agent step."""
        agent = agent or self._current_agent or "unknown"
        event = self._add_event(EventType.STEP_END, agent, {
            "outcome": outcome,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
            "tokens": {"input": tokens_in, "output": tokens_out}
        })
        self._current_agent = None
        return event.seq

    def add_fact(
        self,
        fact_id: str,
        text: str,
        confidence: float = 1.0,
        source: Optional[str] = None
    ) -> int:
        """Add a new fact."""
        agent = source or self._current_agent or "unknown"
        event = self._add_event(EventType.FACT_ADDED, agent, {
            "id": fact_id,
            "text": text,
            "confidence": confidence,
            "source": source or agent
        })
        return event.seq

    def modify_fact(
        self,
        fact_id: str,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        reason: str = ""
    ) -> int:
        """Modify an existing fact."""
        agent = self._current_agent or "unknown"
        data = {"id": fact_id, "reason": reason}
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
        duration_ms: int = 0
    ) -> int:
        """Record a tool invocation."""
        agent = self._current_agent or "unknown"
        event = self._add_event(EventType.TOOL_CALL, agent, {
            "tool": tool,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms
        })
        return event.seq

    def error(
        self,
        category: str,
        error_type: str,
        message: str,
        recoverable: bool = True,
        suggested_action: str = ""
    ) -> int:
        """Record an error."""
        agent = self._current_agent or "unknown"
        event = self._add_event(EventType.ERROR, agent, {
            "category": category,
            "type": error_type,
            "message": message,
            "recoverable": recoverable,
            "suggested_action": suggested_action
        })
        return event.seq

    def checkpoint(self, facts_snapshot: Optional[Dict] = None) -> int:
        """Create a checkpoint for fast replay."""
        from .events import ReplayEngine

        # If no snapshot provided, compute current state
        if facts_snapshot is None:
            engine = ReplayEngine(self.chain)
            state = engine.replay_all()
            facts_snapshot = state.facts

        event = self._add_event(EventType.CHECKPOINT, "system", {
            "state_hash": str(hash(str(facts_snapshot)))[:8],
            "facts_snapshot": facts_snapshot
        })
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
        except PermissionError:
            raise PermissionError(f"Permission denied writing to: {path}")
        except OSError as e:
            raise OSError(f"Failed to export chain to {path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
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
    import time
    start = time.time()

    try:
        session.step_start(agent, intent, input_summary)
    except Exception:
        # If we can't even start tracing, just execute without tracing
        yield
        return

    try:
        yield
        duration_ms = int((time.time() - start) * 1000)
        try:
            session.step_end(agent, outcome="success", duration_ms=duration_ms)
        except Exception:
            pass  # Don't fail if we can't record the end
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        try:
            session.step_end(agent, outcome="error", duration_ms=duration_ms)
            session.error(
                category="execution_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False
            )
        except Exception:
            pass  # Don't suppress original exception if tracing fails
        raise
