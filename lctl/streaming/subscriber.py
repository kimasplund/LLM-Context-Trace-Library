"""Event subscriber patterns for flexible event handling."""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from .emitter import EventEmitter, StreamingEvent, StreamingEventType


class EventSubscriber(ABC):
    """Base class for event subscribers.

    Provides a common interface for subscribing to events from an EventEmitter.
    """

    def __init__(self, emitter: EventEmitter):
        """Initialize the subscriber.

        Args:
            emitter: The EventEmitter to subscribe to.
        """
        self._emitter = emitter
        self._subscribed = False

    @property
    def emitter(self) -> EventEmitter:
        """Get the associated emitter."""
        return self._emitter

    @property
    def is_subscribed(self) -> bool:
        """Check if currently subscribed."""
        return self._subscribed

    def subscribe(self) -> "EventSubscriber":
        """Subscribe to events.

        Returns:
            Self for method chaining.
        """
        if not self._subscribed:
            self._emitter.on("all", self._handle_event)
            self._subscribed = True
        return self

    def unsubscribe(self) -> "EventSubscriber":
        """Unsubscribe from events.

        Returns:
            Self for method chaining.
        """
        if self._subscribed:
            self._emitter.off("all", self._handle_event)
            self._subscribed = False
        return self

    @abstractmethod
    def _handle_event(self, event: StreamingEvent) -> None:
        """Handle an incoming event.

        Args:
            event: The event to handle.
        """
        pass

    def __enter__(self) -> "EventSubscriber":
        """Context manager entry."""
        return self.subscribe()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.unsubscribe()


class FilteredSubscriber(EventSubscriber):
    """A subscriber that filters events before passing them to a handler.

    Supports filtering by:
    - Event types
    - Chain IDs
    - LCTL event types
    - Custom filter functions

    Usage:
        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: print(e),
            event_types=["step_start", "step_end"],
            chain_ids=["my-chain"]
        )
        subscriber.subscribe()
    """

    def __init__(
        self,
        emitter: EventEmitter,
        handler: Callable[[StreamingEvent], None],
        event_types: Optional[List[str]] = None,
        chain_ids: Optional[List[str]] = None,
        lctl_event_types: Optional[List[str]] = None,
        filter_fn: Optional[Callable[[StreamingEvent], bool]] = None
    ):
        """Initialize the filtered subscriber.

        Args:
            emitter: The EventEmitter to subscribe to.
            handler: Function to call for matching events.
            event_types: List of StreamingEventType values to accept.
            chain_ids: List of chain IDs to accept.
            lctl_event_types: List of LCTL event types to accept.
            filter_fn: Custom filter function (returns True to accept).
        """
        super().__init__(emitter)
        self._handler = handler
        self._event_types: Optional[Set[str]] = set(event_types) if event_types else None
        self._chain_ids: Optional[Set[str]] = set(chain_ids) if chain_ids else None
        self._lctl_event_types: Optional[Set[str]] = (
            set(lctl_event_types) if lctl_event_types else None
        )
        self._filter_fn = filter_fn

    def _handle_event(self, event: StreamingEvent) -> None:
        """Handle and filter an incoming event."""
        if self._matches_filters(event):
            self._handler(event)

    def _matches_filters(self, event: StreamingEvent) -> bool:
        """Check if an event matches all filters."""
        if self._event_types and event.type.value not in self._event_types:
            return False

        if self._chain_ids and event.chain_id not in self._chain_ids:
            return False

        if self._lctl_event_types:
            if event.type != StreamingEventType.EVENT:
                return False
            lctl_type = event.payload.get("type")
            if lctl_type not in self._lctl_event_types:
                return False

        if self._filter_fn and not self._filter_fn(event):
            return False

        return True

    def add_event_type(self, event_type: str) -> "FilteredSubscriber":
        """Add an event type to the filter.

        Args:
            event_type: The event type to add.

        Returns:
            Self for method chaining.
        """
        if self._event_types is None:
            self._event_types = set()
        self._event_types.add(event_type)
        return self

    def remove_event_type(self, event_type: str) -> "FilteredSubscriber":
        """Remove an event type from the filter.

        Args:
            event_type: The event type to remove.

        Returns:
            Self for method chaining.
        """
        if self._event_types:
            self._event_types.discard(event_type)
        return self

    def add_chain_id(self, chain_id: str) -> "FilteredSubscriber":
        """Add a chain ID to the filter.

        Args:
            chain_id: The chain ID to add.

        Returns:
            Self for method chaining.
        """
        if self._chain_ids is None:
            self._chain_ids = set()
        self._chain_ids.add(chain_id)
        return self

    def remove_chain_id(self, chain_id: str) -> "FilteredSubscriber":
        """Remove a chain ID from the filter.

        Args:
            chain_id: The chain ID to remove.

        Returns:
            Self for method chaining.
        """
        if self._chain_ids:
            self._chain_ids.discard(chain_id)
        return self


class BufferedSubscriber(EventSubscriber):
    """A subscriber that buffers events for batch processing.

    Events are collected in a buffer and processed when:
    - The buffer reaches max_size
    - The flush interval expires
    - flush() is called manually

    Usage:
        def process_batch(events):
            for event in events:
                print(event)

        subscriber = BufferedSubscriber(
            emitter,
            handler=process_batch,
            max_size=100,
            flush_interval=5.0
        )
        subscriber.subscribe()
    """

    def __init__(
        self,
        emitter: EventEmitter,
        handler: Callable[[List[StreamingEvent]], None],
        max_size: int = 100,
        flush_interval: float = 5.0,
        filter_fn: Optional[Callable[[StreamingEvent], bool]] = None
    ):
        """Initialize the buffered subscriber.

        Args:
            emitter: The EventEmitter to subscribe to.
            handler: Function to call with batches of events.
            max_size: Maximum buffer size before auto-flush.
            flush_interval: Interval in seconds for auto-flush.
            filter_fn: Optional filter function.
        """
        super().__init__(emitter)
        self._handler = handler
        self._max_size = max_size
        self._flush_interval = flush_interval
        self._filter_fn = filter_fn
        self._buffer: List[StreamingEvent] = []
        self._lock = threading.RLock()
        self._last_flush = time.time()
        self._flush_timer: Optional[threading.Timer] = None

    def _handle_event(self, event: StreamingEvent) -> None:
        """Handle an incoming event by buffering it."""
        if self._filter_fn and not self._filter_fn(event):
            return

        with self._lock:
            self._buffer.append(event)

            if len(self._buffer) >= self._max_size:
                self._flush_locked()
            elif self._flush_timer is None:
                self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Schedule a timed flush."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()

        self._flush_timer = threading.Timer(self._flush_interval, self.flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def flush(self) -> List[StreamingEvent]:
        """Flush the buffer and process events.

        Returns:
            The list of flushed events.
        """
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> List[StreamingEvent]:
        """Flush the buffer (must hold lock)."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

        events = self._buffer
        self._buffer = []
        self._last_flush = time.time()

        if events:
            try:
                self._handler(events)
            except Exception:
                pass

        return events

    @property
    def buffer_size(self) -> int:
        """Get the current buffer size."""
        with self._lock:
            return len(self._buffer)

    def unsubscribe(self) -> "BufferedSubscriber":
        """Unsubscribe and flush remaining events."""
        self.flush()
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
        return super().unsubscribe()


class AsyncSubscriber(EventSubscriber):
    """A subscriber that handles events asynchronously.

    Events are processed in an async context using a queue.

    Usage:
        async def async_handler(event):
            await some_async_operation(event)

        subscriber = AsyncSubscriber(emitter, handler=async_handler)

        async def main():
            subscriber.subscribe()
            await subscriber.run()
    """

    def __init__(
        self,
        emitter: EventEmitter,
        handler: Callable[[StreamingEvent], Any],
        queue_size: int = 1000,
        filter_fn: Optional[Callable[[StreamingEvent], bool]] = None
    ):
        """Initialize the async subscriber.

        Args:
            emitter: The EventEmitter to subscribe to.
            handler: Async function to call for events.
            queue_size: Maximum queue size.
            filter_fn: Optional filter function.
        """
        super().__init__(emitter)
        self._handler = handler
        self._queue_size = queue_size
        self._filter_fn = filter_fn
        self._queue: Optional[asyncio.Queue] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def _handle_event(self, event: StreamingEvent) -> None:
        """Handle an incoming event by queueing it."""
        if self._filter_fn and not self._filter_fn(event):
            return

        if self._queue is not None:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def start(self) -> None:
        """Start the async subscriber."""
        if self._running:
            return

        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._running = True
        self.subscribe()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the async subscriber."""
        self._running = False
        self.unsubscribe()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._queue = None

    async def run(self) -> None:
        """Run the subscriber until stopped."""
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                try:
                    result = self._handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @property
    def queue_size(self) -> int:
        """Get the current queue size."""
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if the subscriber is running."""
        return self._running

    async def __aenter__(self) -> "AsyncSubscriber":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self.stop()


@dataclass
class EventAggregation:
    """Aggregated event statistics."""
    count: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_agent: Dict[str, int] = field(default_factory=dict)
    chain_ids: Set[str] = field(default_factory=set)


class AggregatingSubscriber(EventSubscriber):
    """A subscriber that aggregates events for statistical analysis.

    Collects statistics about events including counts by type,
    agent, and chain.

    Usage:
        subscriber = AggregatingSubscriber(emitter)
        subscriber.subscribe()

        # ... events occur ...

        stats = subscriber.get_aggregation()
        print(f"Total events: {stats.count}")
    """

    def __init__(
        self,
        emitter: EventEmitter,
        filter_fn: Optional[Callable[[StreamingEvent], bool]] = None
    ):
        """Initialize the aggregating subscriber.

        Args:
            emitter: The EventEmitter to subscribe to.
            filter_fn: Optional filter function.
        """
        super().__init__(emitter)
        self._filter_fn = filter_fn
        self._aggregation = EventAggregation()
        self._lock = threading.RLock()

    def _handle_event(self, event: StreamingEvent) -> None:
        """Handle an incoming event by updating aggregations."""
        if self._filter_fn and not self._filter_fn(event):
            return

        with self._lock:
            self._aggregation.count += 1

            if self._aggregation.first_timestamp is None:
                self._aggregation.first_timestamp = event.timestamp
            self._aggregation.last_timestamp = event.timestamp

            event_type = event.type.value
            if event.type == StreamingEventType.EVENT and event.payload.get("type"):
                event_type = event.payload["type"]
            self._aggregation.events_by_type[event_type] = (
                self._aggregation.events_by_type.get(event_type, 0) + 1
            )

            if event.type == StreamingEventType.EVENT:
                agent = event.payload.get("agent")
                if agent:
                    self._aggregation.events_by_agent[agent] = (
                        self._aggregation.events_by_agent.get(agent, 0) + 1
                    )

            if event.chain_id:
                self._aggregation.chain_ids.add(event.chain_id)

    def get_aggregation(self) -> EventAggregation:
        """Get the current aggregation.

        Returns:
            A copy of the current aggregation.
        """
        with self._lock:
            return EventAggregation(
                count=self._aggregation.count,
                first_timestamp=self._aggregation.first_timestamp,
                last_timestamp=self._aggregation.last_timestamp,
                events_by_type=dict(self._aggregation.events_by_type),
                events_by_agent=dict(self._aggregation.events_by_agent),
                chain_ids=set(self._aggregation.chain_ids)
            )

    def reset(self) -> None:
        """Reset the aggregation."""
        with self._lock:
            self._aggregation = EventAggregation()

    @property
    def count(self) -> int:
        """Get the total event count."""
        with self._lock:
            return self._aggregation.count
