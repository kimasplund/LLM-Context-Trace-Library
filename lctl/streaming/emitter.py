"""EventEmitter - Publish/subscribe pattern for real-time event distribution."""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..core.events import Event


class StreamingEventType(str, Enum):
    """Types of streaming events."""
    EVENT = "event"
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamingEvent:
    """A streaming event wrapper for real-time transmission."""
    id: str
    type: StreamingEventType
    timestamp: datetime
    chain_id: Optional[str]
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "chain_id": self.chain_id,
            "payload": self.payload
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_lctl_event(cls, event: Event, chain_id: str) -> "StreamingEvent":
        """Create a StreamingEvent from an LCTL Event."""
        return cls(
            id=str(uuid4()),
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=chain_id,
            payload=event.to_dict()
        )

    @classmethod
    def chain_started(cls, chain_id: str) -> "StreamingEvent":
        """Create a chain start event."""
        return cls(
            id=str(uuid4()),
            type=StreamingEventType.CHAIN_START,
            timestamp=datetime.now(timezone.utc),
            chain_id=chain_id,
            payload={"chain_id": chain_id}
        )

    @classmethod
    def chain_ended(cls, chain_id: str, event_count: int) -> "StreamingEvent":
        """Create a chain end event."""
        return cls(
            id=str(uuid4()),
            type=StreamingEventType.CHAIN_END,
            timestamp=datetime.now(timezone.utc),
            chain_id=chain_id,
            payload={"chain_id": chain_id, "event_count": event_count}
        )

    @classmethod
    def connected(cls, client_id: str) -> "StreamingEvent":
        """Create a connected event."""
        return cls(
            id=str(uuid4()),
            type=StreamingEventType.CONNECTED,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={"client_id": client_id}
        )

    @classmethod
    def heartbeat(cls) -> "StreamingEvent":
        """Create a heartbeat event."""
        return cls(
            id=str(uuid4()),
            type=StreamingEventType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={}
        )

    @classmethod
    def error_event(cls, message: str, chain_id: Optional[str] = None) -> "StreamingEvent":
        """Create an error event."""
        return cls(
            id=str(uuid4()),
            type=StreamingEventType.ERROR,
            timestamp=datetime.now(timezone.utc),
            chain_id=chain_id,
            payload={"message": message}
        )


EventHandler = Callable[[StreamingEvent], None]
AsyncEventHandler = Callable[[StreamingEvent], Any]


class EventEmitter:
    """Event emitter for real-time event distribution.

    Supports both synchronous and asynchronous event handlers with
    multiple subscription patterns.

    Usage:
        emitter = EventEmitter()

        @emitter.on("event")
        def handle_event(event):
            print(f"Event: {event.type}")

        # Or with specific event types
        @emitter.on("step_start")
        def handle_step_start(event):
            print(f"Step started: {event.payload}")

        # Emit an event
        emitter.emit(StreamingEvent.chain_started("chain-1"))
    """

    def __init__(self, max_history: int = 100):
        """Initialize the EventEmitter.

        Args:
            max_history: Maximum number of events to keep in history.
        """
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._async_handlers: Dict[str, List[AsyncEventHandler]] = {}
        self._once_handlers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.RLock()
        self._history: List[StreamingEvent] = []
        self._max_history = max_history
        self._chain_id: Optional[str] = None
        self._event_count = 0

    @property
    def chain_id(self) -> Optional[str]:
        """Get the current chain ID."""
        return self._chain_id

    @chain_id.setter
    def chain_id(self, value: Optional[str]) -> None:
        """Set the current chain ID."""
        self._chain_id = value

    @property
    def event_count(self) -> int:
        """Get the total number of events emitted."""
        return self._event_count

    @property
    def history(self) -> List[StreamingEvent]:
        """Get the event history."""
        with self._lock:
            return list(self._history)

    def on(
        self,
        event_type: Union[str, StreamingEventType],
        handler: Optional[EventHandler] = None
    ) -> Callable:
        """Register an event handler.

        Can be used as a decorator or called directly:
            @emitter.on("event")
            def handler(event):
                pass

            # Or
            emitter.on("event", handler)

        Args:
            event_type: The event type to listen for. Use "all" for all events.
            handler: Optional handler function.

        Returns:
            The handler function (for decorator usage).
        """
        event_key = event_type.value if isinstance(event_type, Enum) else event_type

        def decorator(fn: EventHandler) -> EventHandler:
            with self._lock:
                if event_key not in self._handlers:
                    self._handlers[event_key] = []
                self._handlers[event_key].append(fn)
            return fn

        if handler is not None:
            return decorator(handler)
        return decorator

    def on_async(
        self,
        event_type: Union[str, StreamingEventType],
        handler: Optional[AsyncEventHandler] = None
    ) -> Callable:
        """Register an async event handler.

        Args:
            event_type: The event type to listen for.
            handler: Optional async handler function.

        Returns:
            The handler function (for decorator usage).
        """
        event_key = event_type.value if isinstance(event_type, Enum) else event_type

        def decorator(fn: AsyncEventHandler) -> AsyncEventHandler:
            with self._lock:
                if event_key not in self._async_handlers:
                    self._async_handlers[event_key] = []
                self._async_handlers[event_key].append(fn)
            return fn

        if handler is not None:
            return decorator(handler)
        return decorator

    def once(
        self,
        event_type: Union[str, StreamingEventType],
        handler: Optional[EventHandler] = None
    ) -> Callable:
        """Register a one-time event handler.

        The handler will be automatically removed after first invocation.

        Args:
            event_type: The event type to listen for.
            handler: Optional handler function.

        Returns:
            The handler function (for decorator usage).
        """
        event_key = event_type.value if isinstance(event_type, Enum) else event_type

        def decorator(fn: EventHandler) -> EventHandler:
            with self._lock:
                if event_key not in self._once_handlers:
                    self._once_handlers[event_key] = []
                self._once_handlers[event_key].append(fn)
            return fn

        if handler is not None:
            return decorator(handler)
        return decorator

    def off(
        self,
        event_type: Union[str, StreamingEventType],
        handler: EventHandler
    ) -> bool:
        """Remove an event handler.

        Args:
            event_type: The event type.
            handler: The handler to remove.

        Returns:
            True if handler was found and removed.
        """
        event_key = event_type.value if isinstance(event_type, Enum) else event_type

        with self._lock:
            removed = False
            if event_key in self._handlers and handler in self._handlers[event_key]:
                self._handlers[event_key].remove(handler)
                removed = True
            if event_key in self._async_handlers and handler in self._async_handlers[event_key]:
                self._async_handlers[event_key].remove(handler)
                removed = True
            if event_key in self._once_handlers and handler in self._once_handlers[event_key]:
                self._once_handlers[event_key].remove(handler)
                removed = True
            return removed

    def off_all(self, event_type: Optional[Union[str, StreamingEventType]] = None) -> None:
        """Remove all handlers for an event type, or all handlers if no type specified.

        Args:
            event_type: Optional event type. If None, removes all handlers.
        """
        with self._lock:
            if event_type is None:
                self._handlers.clear()
                self._async_handlers.clear()
                self._once_handlers.clear()
            else:
                event_key = event_type.value if isinstance(event_type, Enum) else event_type
                self._handlers.pop(event_key, None)
                self._async_handlers.pop(event_key, None)
                self._once_handlers.pop(event_key, None)

    def emit(self, event: StreamingEvent) -> None:
        """Emit an event to all registered handlers.

        Args:
            event: The event to emit.
        """
        with self._lock:
            self._event_count += 1
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            event_key = event.type.value

            handlers_to_call: List[EventHandler] = []
            handlers_to_call.extend(self._handlers.get(event_key, []))
            handlers_to_call.extend(self._handlers.get("all", []))

            if event.type == StreamingEventType.EVENT and event.payload.get("type"):
                lctl_event_type = event.payload["type"]
                handlers_to_call.extend(self._handlers.get(lctl_event_type, []))

            once_handlers = self._once_handlers.pop(event_key, [])
            handlers_to_call.extend(once_handlers)
            handlers_to_call.extend(self._once_handlers.pop("all", []))

        for handler in handlers_to_call:
            try:
                handler(event)
            except Exception as e:
                self._emit_error(f"Handler error: {e}")

    async def emit_async(self, event: StreamingEvent) -> None:
        """Emit an event to all registered handlers asynchronously.

        Args:
            event: The event to emit.
        """
        self.emit(event)

        with self._lock:
            event_key = event.type.value

            async_handlers: List[AsyncEventHandler] = []
            async_handlers.extend(self._async_handlers.get(event_key, []))
            async_handlers.extend(self._async_handlers.get("all", []))

            if event.type == StreamingEventType.EVENT and event.payload.get("type"):
                lctl_event_type = event.payload["type"]
                async_handlers.extend(self._async_handlers.get(lctl_event_type, []))

        for handler in async_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._emit_error(f"Async handler error: {e}")

    def emit_lctl_event(self, event: Event) -> None:
        """Emit an LCTL Event as a StreamingEvent.

        Args:
            event: The LCTL Event to emit.
        """
        if self._chain_id is None:
            raise ValueError("chain_id must be set before emitting LCTL events")

        streaming_event = StreamingEvent.from_lctl_event(event, self._chain_id)
        self.emit(streaming_event)

    def start_chain(self, chain_id: str) -> None:
        """Signal the start of a new chain.

        Args:
            chain_id: The chain ID.
        """
        self._chain_id = chain_id
        self._event_count = 0
        self.emit(StreamingEvent.chain_started(chain_id))

    def end_chain(self) -> None:
        """Signal the end of the current chain."""
        if self._chain_id:
            self.emit(StreamingEvent.chain_ended(self._chain_id, self._event_count))
            self._chain_id = None

    def _emit_error(self, message: str) -> None:
        """Emit an error event.

        Args:
            message: The error message.
        """
        error_event = StreamingEvent.error_event(message, self._chain_id)
        with self._lock:
            error_handlers = self._handlers.get("error", [])
            error_handlers.extend(self._handlers.get(StreamingEventType.ERROR.value, []))

        for handler in error_handlers:
            try:
                handler(error_event)
            except Exception:
                pass

    def handler_count(self, event_type: Optional[Union[str, StreamingEventType]] = None) -> int:
        """Get the number of registered handlers.

        Args:
            event_type: Optional event type. If None, returns total count.

        Returns:
            Number of handlers.
        """
        with self._lock:
            if event_type is None:
                count = sum(len(h) for h in self._handlers.values())
                count += sum(len(h) for h in self._async_handlers.values())
                count += sum(len(h) for h in self._once_handlers.values())
                return count

            event_key = event_type.value if isinstance(event_type, Enum) else event_type
            count = len(self._handlers.get(event_key, []))
            count += len(self._async_handlers.get(event_key, []))
            count += len(self._once_handlers.get(event_key, []))
            return count

    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._history.clear()
