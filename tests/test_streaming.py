"""Tests for LCTL streaming module."""

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lctl.core.events import Event, EventType
from lctl.core.session import LCTLSession
from lctl.streaming import (
    AsyncSubscriber,
    BufferedSubscriber,
    EventEmitter,
    EventSubscriber,
    FilteredSubscriber,
    StreamingEvent,
)
from lctl.streaming.emitter import StreamingEventType
from lctl.streaming.websocket import SSEHandler, WebSocketServer


class TestStreamingEvent:
    """Tests for StreamingEvent class."""

    def test_create_streaming_event(self):
        """Test creating a basic streaming event."""
        event = StreamingEvent(
            id="test-id",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-1",
            payload={"test": "data"}
        )

        assert event.id == "test-id"
        assert event.type == StreamingEventType.EVENT
        assert event.chain_id == "chain-1"
        assert event.payload == {"test": "data"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ts = datetime.now(timezone.utc)
        event = StreamingEvent(
            id="test-id",
            type=StreamingEventType.EVENT,
            timestamp=ts,
            chain_id="chain-1",
            payload={"test": "data"}
        )

        d = event.to_dict()
        assert d["id"] == "test-id"
        assert d["type"] == "event"
        assert d["chain_id"] == "chain-1"
        assert d["payload"] == {"test": "data"}
        assert d["timestamp"] == ts.isoformat()

    def test_to_json(self):
        """Test JSON serialization."""
        event = StreamingEvent(
            id="test-id",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-1",
            payload={}
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "test-id"
        assert parsed["type"] == "event"

    def test_from_lctl_event(self):
        """Test creating StreamingEvent from LCTL Event."""
        lctl_event = Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=datetime.now(timezone.utc),
            agent="test-agent",
            data={"intent": "test"}
        )

        streaming_event = StreamingEvent.from_lctl_event(lctl_event, "chain-1")

        assert streaming_event.type == StreamingEventType.EVENT
        assert streaming_event.chain_id == "chain-1"
        assert streaming_event.payload["seq"] == 1
        assert streaming_event.payload["agent"] == "test-agent"

    def test_chain_started(self):
        """Test chain start event creation."""
        event = StreamingEvent.chain_started("chain-1")

        assert event.type == StreamingEventType.CHAIN_START
        assert event.chain_id == "chain-1"
        assert event.payload["chain_id"] == "chain-1"

    def test_chain_ended(self):
        """Test chain end event creation."""
        event = StreamingEvent.chain_ended("chain-1", 10)

        assert event.type == StreamingEventType.CHAIN_END
        assert event.chain_id == "chain-1"
        assert event.payload["event_count"] == 10

    def test_connected(self):
        """Test connected event creation."""
        event = StreamingEvent.connected("client-1")

        assert event.type == StreamingEventType.CONNECTED
        assert event.payload["client_id"] == "client-1"

    def test_heartbeat(self):
        """Test heartbeat event creation."""
        event = StreamingEvent.heartbeat()

        assert event.type == StreamingEventType.HEARTBEAT

    def test_error_event(self):
        """Test error event creation."""
        event = StreamingEvent.error_event("Something went wrong", "chain-1")

        assert event.type == StreamingEventType.ERROR
        assert event.payload["message"] == "Something went wrong"
        assert event.chain_id == "chain-1"


class TestEventEmitter:
    """Tests for EventEmitter class."""

    def test_create_emitter(self):
        """Test creating an emitter."""
        emitter = EventEmitter()
        assert emitter.event_count == 0
        assert emitter.chain_id is None

    def test_on_decorator(self):
        """Test using on() as a decorator."""
        emitter = EventEmitter()
        received = []

        @emitter.on("event")
        def handler(event):
            received.append(event)

        event = StreamingEvent.heartbeat()
        event = StreamingEvent(
            id="test",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={}
        )
        emitter.emit(event)

        assert len(received) == 1
        assert received[0].id == "test"

    def test_on_direct_call(self):
        """Test using on() with direct handler registration."""
        emitter = EventEmitter()
        received = []

        def handler(event):
            received.append(event)

        emitter.on("all", handler)

        event = StreamingEvent.heartbeat()
        emitter.emit(event)

        assert len(received) == 1

    def test_on_all_events(self):
        """Test subscribing to all events."""
        emitter = EventEmitter()
        received = []

        @emitter.on("all")
        def handler(event):
            received.append(event)

        emitter.emit(StreamingEvent.heartbeat())
        emitter.emit(StreamingEvent.connected("client-1"))

        assert len(received) == 2

    def test_once_handler(self):
        """Test one-time event handler."""
        emitter = EventEmitter()
        received = []

        @emitter.once("event")
        def handler(event):
            received.append(event)

        event = StreamingEvent(
            id="test",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={}
        )

        emitter.emit(event)
        emitter.emit(event)

        assert len(received) == 1

    def test_off_handler(self):
        """Test removing an event handler."""
        emitter = EventEmitter()
        received = []

        def handler(event):
            received.append(event)

        emitter.on("all", handler)
        emitter.emit(StreamingEvent.heartbeat())
        assert len(received) == 1

        removed = emitter.off("all", handler)
        assert removed is True

        emitter.emit(StreamingEvent.heartbeat())
        assert len(received) == 1

    def test_off_all(self):
        """Test removing all handlers."""
        emitter = EventEmitter()
        received = []

        @emitter.on("all")
        def handler(event):
            received.append(event)

        emitter.off_all()
        emitter.emit(StreamingEvent.heartbeat())

        assert len(received) == 0

    def test_start_end_chain(self):
        """Test chain lifecycle events."""
        emitter = EventEmitter()
        received = []

        @emitter.on("all")
        def handler(event):
            received.append(event)

        emitter.start_chain("chain-1")
        assert emitter.chain_id == "chain-1"
        assert len(received) == 1
        assert received[0].type == StreamingEventType.CHAIN_START

        emitter.end_chain()
        assert emitter.chain_id is None
        assert len(received) == 2
        assert received[1].type == StreamingEventType.CHAIN_END

    def test_emit_lctl_event(self):
        """Test emitting LCTL events."""
        emitter = EventEmitter()
        emitter.chain_id = "chain-1"
        received = []

        @emitter.on("all")
        def handler(event):
            received.append(event)

        lctl_event = Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=datetime.now(timezone.utc),
            agent="test-agent",
            data={"intent": "test"}
        )

        emitter.emit_lctl_event(lctl_event)

        assert len(received) == 1
        assert received[0].type == StreamingEventType.EVENT
        assert received[0].payload["seq"] == 1

    def test_emit_lctl_event_without_chain_id(self):
        """Test that emitting LCTL events requires chain_id."""
        emitter = EventEmitter()

        lctl_event = Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=datetime.now(timezone.utc),
            agent="test-agent",
            data={}
        )

        with pytest.raises(ValueError, match="chain_id must be set"):
            emitter.emit_lctl_event(lctl_event)

    def test_history(self):
        """Test event history."""
        emitter = EventEmitter(max_history=3)

        for i in range(5):
            emitter.emit(StreamingEvent.heartbeat())

        history = emitter.history
        assert len(history) == 3

    def test_clear_history(self):
        """Test clearing event history."""
        emitter = EventEmitter()

        emitter.emit(StreamingEvent.heartbeat())
        assert len(emitter.history) == 1

        emitter.clear_history()
        assert len(emitter.history) == 0

    def test_handler_count(self):
        """Test handler count."""
        emitter = EventEmitter()

        @emitter.on("event")
        def handler1(e):
            pass

        @emitter.on("all")
        def handler2(e):
            pass

        assert emitter.handler_count() == 2
        assert emitter.handler_count("event") == 1
        assert emitter.handler_count("all") == 1

    def test_handler_error_handling(self):
        """Test that handler errors don't break other handlers."""
        emitter = EventEmitter()
        received = []

        @emitter.on("all")
        def bad_handler(event):
            raise RuntimeError("Handler error")

        @emitter.on("all")
        def good_handler(event):
            received.append(event)

        emitter.emit(StreamingEvent.heartbeat())

        assert len(received) == 1


class TestFilteredSubscriber:
    """Tests for FilteredSubscriber class."""

    def test_filter_by_event_type(self):
        """Test filtering by event type."""
        emitter = EventEmitter()
        received = []

        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: received.append(e),
            event_types=["event"]
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={}
        ))
        emitter.emit(StreamingEvent.heartbeat())

        assert len(received) == 1
        assert received[0].id == "1"

    def test_filter_by_chain_id(self):
        """Test filtering by chain ID."""
        emitter = EventEmitter()
        received = []

        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: received.append(e),
            chain_ids=["chain-1"]
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-1",
            payload={}
        ))
        emitter.emit(StreamingEvent(
            id="2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-2",
            payload={}
        ))

        assert len(received) == 1
        assert received[0].chain_id == "chain-1"

    def test_filter_by_lctl_event_type(self):
        """Test filtering by LCTL event type."""
        emitter = EventEmitter()
        received = []

        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: received.append(e),
            lctl_event_types=["step_start"]
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={"type": "step_start"}
        ))
        emitter.emit(StreamingEvent(
            id="2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={"type": "step_end"}
        ))

        assert len(received) == 1
        assert received[0].payload["type"] == "step_start"

    def test_custom_filter_function(self):
        """Test custom filter function."""
        emitter = EventEmitter()
        received = []

        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: received.append(e),
            filter_fn=lambda e: e.payload.get("important", False)
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={"important": True}
        ))
        emitter.emit(StreamingEvent(
            id="2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={"important": False}
        ))

        assert len(received) == 1

    def test_add_remove_filters(self):
        """Test dynamically adding and removing filters."""
        emitter = EventEmitter()
        received = []

        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: received.append(e)
        )
        subscriber.subscribe()

        subscriber.add_chain_id("chain-1")
        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-2",
            payload={}
        ))
        assert len(received) == 0

        subscriber.remove_chain_id("chain-1")
        emitter.emit(StreamingEvent(
            id="2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-2",
            payload={}
        ))
        assert len(received) == 1

    def test_context_manager(self):
        """Test using subscriber as context manager."""
        emitter = EventEmitter()
        received = []

        subscriber = FilteredSubscriber(
            emitter,
            handler=lambda e: received.append(e)
        )

        with subscriber:
            emitter.emit(StreamingEvent.heartbeat())

        assert len(received) == 1

        emitter.emit(StreamingEvent.heartbeat())
        assert len(received) == 1


class TestBufferedSubscriber:
    """Tests for BufferedSubscriber class."""

    def test_buffer_until_max_size(self):
        """Test buffering until max size."""
        emitter = EventEmitter()
        batches = []

        subscriber = BufferedSubscriber(
            emitter,
            handler=lambda events: batches.append(events),
            max_size=3,
            flush_interval=60.0
        )
        subscriber.subscribe()

        for _ in range(5):
            emitter.emit(StreamingEvent.heartbeat())

        assert len(batches) == 1
        assert len(batches[0]) == 3

        subscriber.unsubscribe()

    def test_manual_flush(self):
        """Test manual flush."""
        emitter = EventEmitter()
        batches = []

        subscriber = BufferedSubscriber(
            emitter,
            handler=lambda events: batches.append(events),
            max_size=100,
            flush_interval=60.0
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent.heartbeat())
        emitter.emit(StreamingEvent.heartbeat())

        assert len(batches) == 0
        assert subscriber.buffer_size == 2

        flushed = subscriber.flush()
        assert len(flushed) == 2
        assert len(batches) == 1
        assert subscriber.buffer_size == 0

        subscriber.unsubscribe()

    def test_flush_on_unsubscribe(self):
        """Test flush on unsubscribe."""
        emitter = EventEmitter()
        batches = []

        subscriber = BufferedSubscriber(
            emitter,
            handler=lambda events: batches.append(events),
            max_size=100,
            flush_interval=60.0
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent.heartbeat())

        subscriber.unsubscribe()
        assert len(batches) == 1

    def test_filter_function(self):
        """Test buffer with filter function."""
        emitter = EventEmitter()
        batches = []

        subscriber = BufferedSubscriber(
            emitter,
            handler=lambda events: batches.append(events),
            max_size=2,
            flush_interval=60.0,
            filter_fn=lambda e: e.type == StreamingEventType.EVENT
        )
        subscriber.subscribe()

        emitter.emit(StreamingEvent.heartbeat())
        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={}
        ))
        emitter.emit(StreamingEvent(
            id="2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id=None,
            payload={}
        ))

        assert len(batches) == 1
        assert len(batches[0]) == 2

        subscriber.unsubscribe()


class TestAsyncSubscriber:
    """Tests for AsyncSubscriber class."""

    def test_async_handler(self):
        """Test async event handler."""
        import anyio

        emitter = EventEmitter()
        received = []

        async def async_handler(event):
            await anyio.sleep(0.01)
            received.append(event)

        async def run_test():
            subscriber = AsyncSubscriber(
                emitter,
                handler=async_handler
            )

            await subscriber.start()

            emitter.emit(StreamingEvent.heartbeat())

            await anyio.sleep(0.1)

            await subscriber.stop()

            assert len(received) == 1

        anyio.run(run_test)

    def test_async_context_manager(self):
        """Test async context manager."""
        import anyio

        emitter = EventEmitter()
        received = []

        async def async_handler(event):
            received.append(event)

        async def run_test():
            subscriber = AsyncSubscriber(
                emitter,
                handler=async_handler
            )

            async with subscriber:
                emitter.emit(StreamingEvent.heartbeat())
                await anyio.sleep(0.1)

            assert len(received) == 1

        anyio.run(run_test)

    def test_queue_size(self):
        """Test queue size tracking."""
        import anyio

        emitter = EventEmitter()

        async def slow_handler(event):
            await anyio.sleep(0.5)

        async def run_test():
            subscriber = AsyncSubscriber(
                emitter,
                handler=slow_handler,
                queue_size=10
            )

            await subscriber.start()

            for _ in range(5):
                emitter.emit(StreamingEvent.heartbeat())

            initial_size = subscriber.queue_size
            assert initial_size > 0

            await subscriber.stop()

        anyio.run(run_test)


class TestLCTLSessionIntegration:
    """Tests for LCTLSession integration with EventEmitter."""

    def test_session_with_emitter(self):
        """Test session with emitter integration."""
        emitter = EventEmitter()
        received = []

        @emitter.on("all")
        def handler(event):
            received.append(event)

        with LCTLSession(chain_id="test-chain", emitter=emitter) as session:
            session.step_start("agent-1", "Test step")
            session.step_end("agent-1", "Done")

        assert len(received) >= 4

        event_types = [e.type for e in received]
        assert StreamingEventType.CHAIN_START in event_types
        assert StreamingEventType.CHAIN_END in event_types
        assert StreamingEventType.EVENT in event_types

    def test_session_emitter_property(self):
        """Test setting emitter via property."""
        emitter = EventEmitter()
        session = LCTLSession(chain_id="test-chain")

        session.emitter = emitter
        assert session.emitter is emitter
        assert emitter.chain_id == "test-chain"

    def test_event_listeners(self):
        """Test event listeners on session."""
        received = []

        def listener(event):
            received.append(event)

        with LCTLSession(chain_id="test-chain") as session:
            session.add_event_listener(listener)
            session.step_start("agent-1", "Test step")

            assert len(received) == 1

            removed = session.remove_event_listener(listener)
            assert removed is True

            session.step_end("agent-1", "Done")
            assert len(received) == 1


class TestWebSocketServer:
    """Tests for WebSocketServer class."""

    def test_register_client(self):
        """Test client registration."""
        import anyio

        async def run_test():
            emitter = EventEmitter()
            server = WebSocketServer(emitter)

            client = await server.register_client()
            assert client.id is not None
            assert server.client_count == 1

            await server.unregister_client(client.id)
            assert server.client_count == 0

        anyio.run(run_test)

    def test_register_client_with_filters(self):
        """Test client registration with filters."""
        import anyio

        async def run_test():
            emitter = EventEmitter()
            server = WebSocketServer(emitter)

            client = await server.register_client(
                client_id="custom-id",
                filters={"chain_id": "chain-1"}
            )

            assert client.id == "custom-id"
            assert client.filters == {"chain_id": "chain-1"}

            await server.unregister_client(client.id)

        anyio.run(run_test)

    def test_max_clients(self):
        """Test max clients limit."""
        import anyio

        async def run_test():
            emitter = EventEmitter()
            server = WebSocketServer(emitter, max_clients=2)

            await server.register_client()
            await server.register_client()

            with pytest.raises(ConnectionError, match="Maximum client"):
                await server.register_client()

        anyio.run(run_test)

    def test_update_client_filters(self):
        """Test updating client filters."""
        import anyio

        async def run_test():
            emitter = EventEmitter()
            server = WebSocketServer(emitter)

            client = await server.register_client()

            await server.update_client_filters(
                client.id,
                {"event_types": ["step_start"]}
            )

            updated_client = server._clients[client.id]
            assert updated_client.filters == {"event_types": ["step_start"]}

            await server.unregister_client(client.id)

        anyio.run(run_test)


class TestSSEHandler:
    """Tests for SSEHandler class."""

    def test_event_stream(self):
        """Test SSE event stream generation."""
        import anyio

        async def run_test():
            emitter = EventEmitter()
            handler = SSEHandler(emitter, heartbeat_interval=0.1)

            events_received = []

            async def collect_events():
                async for event_str in handler.event_stream():
                    events_received.append(event_str)
                    if len(events_received) >= 2:
                        break

            async with anyio.create_task_group() as tg:
                tg.start_soon(collect_events)

                await anyio.sleep(0.05)
                emitter.emit(StreamingEvent.heartbeat())

                await anyio.sleep(0.2)
                tg.cancel_scope.cancel()

            assert len(events_received) >= 1

        anyio.run(run_test)

    def test_format_sse(self):
        """Test SSE formatting."""
        emitter = EventEmitter()
        handler = SSEHandler(emitter)

        result = handler._format_sse("test", {"key": "value"})

        assert result.startswith("event: test\n")
        assert "data:" in result
        assert result.endswith("\n\n")


class TestAggregatingSubscriber:
    """Tests for AggregatingSubscriber class."""

    def test_aggregation(self):
        """Test event aggregation."""
        from lctl.streaming.subscriber import AggregatingSubscriber

        emitter = EventEmitter()
        subscriber = AggregatingSubscriber(emitter)
        subscriber.subscribe()

        emitter.emit(StreamingEvent(
            id="1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-1",
            payload={"type": "step_start", "agent": "agent-1"}
        ))
        emitter.emit(StreamingEvent(
            id="2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-1",
            payload={"type": "step_end", "agent": "agent-1"}
        ))
        emitter.emit(StreamingEvent(
            id="3",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(timezone.utc),
            chain_id="chain-2",
            payload={"type": "step_start", "agent": "agent-2"}
        ))

        agg = subscriber.get_aggregation()

        assert agg.count == 3
        assert "chain-1" in agg.chain_ids
        assert "chain-2" in agg.chain_ids
        assert agg.events_by_type.get("step_start", 0) == 2
        assert agg.events_by_type.get("step_end", 0) == 1
        assert agg.events_by_agent.get("agent-1", 0) == 2
        assert agg.events_by_agent.get("agent-2", 0) == 1

        subscriber.unsubscribe()

    def test_reset(self):
        """Test reset aggregation."""
        from lctl.streaming.subscriber import AggregatingSubscriber

        emitter = EventEmitter()
        subscriber = AggregatingSubscriber(emitter)
        subscriber.subscribe()

        emitter.emit(StreamingEvent.heartbeat())

        assert subscriber.count == 1

        subscriber.reset()

        assert subscriber.count == 0

        subscriber.unsubscribe()
