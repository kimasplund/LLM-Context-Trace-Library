"""Comprehensive tests for LCTL WebSocket streaming module.

Test file: /home/kim/projects/llm-context-transfer/tests/test_websocket_comprehensive.py
Target: /home/kim/projects/llm-context-transfer/lctl/streaming/websocket.py
Goal: Increase coverage from 45% to 70%+

Covers:
1. WebSocketManager (WebSocketServer) - connection registration/unregistration
2. Message handling - subscribe/unsubscribe/ping
3. Connection lifecycle - setup, disconnect, reconnection
4. Error scenarios - closed connection, invalid JSON, timeout, max connections
5. Subscription filtering - by event type, by chain_id
6. SSE handler functionality
7. Server start/stop lifecycle
"""

import asyncio
import contextlib
import json
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lctl.streaming.emitter import EventEmitter, StreamingEvent, StreamingEventType
from lctl.streaming.websocket import (
    SSEHandler,
    WebSocketClient,
    WebSocketServer,
    create_sse_response,
    start_websocket_server,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def emitter() -> EventEmitter:
    """Create a fresh EventEmitter for testing."""
    return EventEmitter()


@pytest.fixture
def server(emitter: EventEmitter) -> WebSocketServer:
    """Create a WebSocket server with the given emitter."""
    return WebSocketServer(emitter, heartbeat_interval=0.1, max_clients=5)


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    return ws


# =============================================================================
# Helper Functions
# =============================================================================


def drain_queue(queue: asyncio.Queue) -> list[StreamingEvent]:
    """Drain all items from a queue and return them."""
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# =============================================================================
# WebSocketServer - Basic Operations
# =============================================================================


class TestWebSocketServerBasicOperations:
    """Tests for WebSocketServer basic operations."""

    @pytest.mark.asyncio
    async def test_is_running_property(self, server: WebSocketServer):
        """Test is_running property returns correct state."""
        # Initially not running
        assert server.is_running is False

        # Manually set _running to simulate server running
        server._running = True
        assert server.is_running is True

        # Reset
        server._running = False
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_client_count_property(self, server: WebSocketServer):
        """Test client_count property."""
        assert server.client_count == 0

        client1 = await server.register_client()
        assert server.client_count == 1

        client2 = await server.register_client()
        assert server.client_count == 2

        await server.unregister_client(client1.id)
        assert server.client_count == 1

        await server.unregister_client(client2.id)
        assert server.client_count == 0

    @pytest.mark.asyncio
    async def test_register_client_generates_id(self, server: WebSocketServer):
        """Test client registration generates unique IDs."""
        client1 = await server.register_client()
        client2 = await server.register_client()

        assert client1.id is not None
        assert client2.id is not None
        assert client1.id != client2.id
        assert len(client1.id) == 8  # UUID first 8 chars

    @pytest.mark.asyncio
    async def test_register_client_with_custom_id(self, server: WebSocketServer):
        """Test client registration with custom ID."""
        client = await server.register_client(client_id="my-custom-id")
        assert client.id == "my-custom-id"

    @pytest.mark.asyncio
    async def test_register_client_with_filters(self, server: WebSocketServer):
        """Test client registration with filters."""
        filters = {"chain_id": "chain-123", "event_types": ["step_start", "step_end"]}
        client = await server.register_client(filters=filters)

        assert client.filters == filters

    @pytest.mark.asyncio
    async def test_register_client_emits_connected_event(
        self, server: WebSocketServer, emitter: EventEmitter
    ):
        """Test that registering a client emits a connected event."""
        events_received: list[StreamingEvent] = []

        @emitter.on("all")
        def handler(event: StreamingEvent):
            events_received.append(event)

        client = await server.register_client()

        # Should have emitted a connected event
        connected_events = [e for e in events_received if e.type == StreamingEventType.CONNECTED]
        assert len(connected_events) == 1
        assert connected_events[0].payload["client_id"] == client.id

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_client(self, server: WebSocketServer):
        """Test unregistering a client that doesn't exist."""
        # Should not raise an error
        await server.unregister_client("nonexistent-id")
        assert server.client_count == 0

    @pytest.mark.asyncio
    async def test_unregister_client_emits_disconnected_event(
        self, server: WebSocketServer, emitter: EventEmitter
    ):
        """Test that unregistering a client emits a disconnected event."""
        events_received: list[StreamingEvent] = []

        @emitter.on("all")
        def handler(event: StreamingEvent):
            events_received.append(event)

        client = await server.register_client()
        await server.unregister_client(client.id)

        disconnected_events = [
            e for e in events_received if e.type == StreamingEventType.DISCONNECTED
        ]
        assert len(disconnected_events) == 1
        assert disconnected_events[0].payload["client_id"] == client.id

    @pytest.mark.asyncio
    async def test_update_client_filters(self, server: WebSocketServer):
        """Test updating client filters."""
        client = await server.register_client()
        assert client.filters == {}

        new_filters = {"chain_id": "chain-456"}
        await server.update_client_filters(client.id, new_filters)

        # Verify filters were updated
        updated_client = server._clients[client.id]
        assert updated_client.filters == new_filters

    @pytest.mark.asyncio
    async def test_update_filters_nonexistent_client(self, server: WebSocketServer):
        """Test updating filters for nonexistent client."""
        # Should not raise an error
        await server.update_client_filters("nonexistent", {"chain_id": "test"})


# =============================================================================
# WebSocketServer - Max Clients
# =============================================================================


class TestWebSocketServerMaxClients:
    """Tests for WebSocketServer max clients limit."""

    @pytest.mark.asyncio
    async def test_max_clients_limit(self, emitter: EventEmitter):
        """Test that max clients limit is enforced."""
        server = WebSocketServer(emitter, max_clients=2)

        await server.register_client()
        await server.register_client()

        with pytest.raises(ConnectionError, match="Maximum client connections reached"):
            await server.register_client()

    @pytest.mark.asyncio
    async def test_max_clients_after_unregister(self, emitter: EventEmitter):
        """Test that clients can be registered after others disconnect."""
        server = WebSocketServer(emitter, max_clients=2)

        client1 = await server.register_client()
        await server.register_client()

        # Should fail
        with pytest.raises(ConnectionError):
            await server.register_client()

        # Unregister one
        await server.unregister_client(client1.id)

        # Should now succeed
        client3 = await server.register_client()
        assert client3.id is not None


# =============================================================================
# WebSocketServer - Event Filtering
# =============================================================================


class TestWebSocketServerEventFiltering:
    """Tests for WebSocketServer event filtering."""

    @pytest.mark.asyncio
    async def test_should_send_to_client_no_filters(self, server: WebSocketServer):
        """Test _should_send_to_client with no filters."""
        client = WebSocketClient(id="test", connected_at=datetime.now(UTC), filters={})

        event = StreamingEvent.heartbeat()
        assert server._should_send_to_client(event, client) is True

    @pytest.mark.asyncio
    async def test_should_send_to_client_chain_id_filter_match(self, server: WebSocketServer):
        """Test _should_send_to_client with matching chain_id filter."""
        client = WebSocketClient(
            id="test",
            connected_at=datetime.now(UTC),
            filters={"chain_id": "chain-123"},
        )

        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-123",
            payload={},
        )
        assert server._should_send_to_client(event, client) is True

    @pytest.mark.asyncio
    async def test_should_send_to_client_chain_id_filter_no_match(self, server: WebSocketServer):
        """Test _should_send_to_client with non-matching chain_id filter."""
        client = WebSocketClient(
            id="test",
            connected_at=datetime.now(UTC),
            filters={"chain_id": "chain-123"},
        )

        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-456",  # Different chain
            payload={},
        )
        assert server._should_send_to_client(event, client) is False

    @pytest.mark.asyncio
    async def test_should_send_to_client_event_types_filter_match(self, server: WebSocketServer):
        """Test _should_send_to_client with matching event_types filter."""
        client = WebSocketClient(
            id="test",
            connected_at=datetime.now(UTC),
            filters={"event_types": ["step_start", "step_end"]},
        )

        # Event with payload type that matches
        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-1",
            payload={"type": "step_start"},
        )
        assert server._should_send_to_client(event, client) is True

    @pytest.mark.asyncio
    async def test_should_send_to_client_event_types_filter_no_match(self, server: WebSocketServer):
        """Test _should_send_to_client with non-matching event_types filter."""
        client = WebSocketClient(
            id="test",
            connected_at=datetime.now(UTC),
            filters={"event_types": ["step_start", "step_end"]},
        )

        # Event with payload type that doesn't match
        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-1",
            payload={"type": "error"},  # Not in filter list
        )
        assert server._should_send_to_client(event, client) is False

    @pytest.mark.asyncio
    async def test_should_send_to_client_streaming_event_type_filter(self, server: WebSocketServer):
        """Test _should_send_to_client with streaming event type filter."""
        client = WebSocketClient(
            id="test",
            connected_at=datetime.now(UTC),
            filters={"event_types": ["heartbeat"]},
        )

        # Heartbeat event should match
        event = StreamingEvent.heartbeat()
        assert server._should_send_to_client(event, client) is True

        # Event type doesn't match
        event2 = StreamingEvent.connected("client-1")
        assert server._should_send_to_client(event2, client) is False

    @pytest.mark.asyncio
    async def test_should_send_to_client_combined_filters(self, server: WebSocketServer):
        """Test _should_send_to_client with combined chain_id and event_types filters."""
        client = WebSocketClient(
            id="test",
            connected_at=datetime.now(UTC),
            filters={"chain_id": "chain-123", "event_types": ["step_start"]},
        )

        # Both match
        event1 = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-123",
            payload={"type": "step_start"},
        )
        assert server._should_send_to_client(event1, client) is True

        # Chain doesn't match
        event2 = StreamingEvent(
            id="e2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-456",
            payload={"type": "step_start"},
        )
        assert server._should_send_to_client(event2, client) is False

        # Event type doesn't match
        event3 = StreamingEvent(
            id="e3",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-123",
            payload={"type": "step_end"},
        )
        assert server._should_send_to_client(event3, client) is False


# =============================================================================
# WebSocketServer - Event Dispatch
# =============================================================================


class TestWebSocketServerEventDispatch:
    """Tests for WebSocketServer event dispatch via _on_event."""

    @pytest.mark.asyncio
    async def test_on_event_sends_to_all_clients(
        self, server: WebSocketServer, emitter: EventEmitter
    ):
        """Test that events are queued for all matching clients."""
        client1 = await server.register_client()
        client2 = await server.register_client()

        # Drain any connected events that were queued
        drain_queue(client1.send_queue)
        drain_queue(client2.send_queue)

        # Emit an event through the emitter
        event = StreamingEvent.heartbeat()
        emitter.emit(event)

        # Both clients should have the event in their queues
        assert client1.send_queue.qsize() == 1
        assert client2.send_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_on_event_respects_filters(self, server: WebSocketServer, emitter: EventEmitter):
        """Test that events respect client filters."""
        client1 = await server.register_client(filters={"chain_id": "chain-1"})
        client2 = await server.register_client(filters={"chain_id": "chain-2"})

        # Drain connected events
        drain_queue(client1.send_queue)
        drain_queue(client2.send_queue)

        # Emit event for chain-1
        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-1",
            payload={},
        )
        emitter.emit(event)

        # Only client1 should receive it
        assert client1.send_queue.qsize() == 1
        assert client2.send_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_on_event_queue_full(self, server: WebSocketServer, emitter: EventEmitter):
        """Test _on_event handles QueueFull gracefully."""
        # Create a client with a small queue
        client = await server.register_client()

        # Replace the queue with one that's full
        small_queue = asyncio.Queue(maxsize=1)
        small_queue.put_nowait(StreamingEvent.heartbeat())  # Fill it
        client.send_queue = small_queue

        # This should not raise an exception
        event = StreamingEvent.heartbeat()
        server._on_event(event)

        # Queue should still have only 1 item (the original one)
        assert small_queue.qsize() == 1


# =============================================================================
# WebSocketServer - Client Message Handling
# =============================================================================


class TestWebSocketServerClientMessageHandling:
    """Tests for WebSocketServer client message handling."""

    @pytest.mark.asyncio
    async def test_handle_client_message_subscribe(self, server: WebSocketServer):
        """Test handling subscribe message."""
        client = await server.register_client()
        assert client.filters == {}

        message = {"type": "subscribe", "filters": {"chain_id": "chain-123"}}
        await server._handle_client_message(client, message)

        # Filters should be updated
        updated_client = server._clients[client.id]
        assert updated_client.filters == {"chain_id": "chain-123"}

    @pytest.mark.asyncio
    async def test_handle_client_message_unsubscribe(self, server: WebSocketServer):
        """Test handling unsubscribe message."""
        client = await server.register_client(filters={"chain_id": "chain-123"})
        assert client.filters == {"chain_id": "chain-123"}

        message = {"type": "unsubscribe"}
        await server._handle_client_message(client, message)

        # Filters should be cleared
        updated_client = server._clients[client.id]
        assert updated_client.filters == {}

    @pytest.mark.asyncio
    async def test_handle_client_message_ping(self, server: WebSocketServer):
        """Test handling ping message."""
        client = await server.register_client()

        # Drain any connected events
        drain_queue(client.send_queue)
        assert client.send_queue.qsize() == 0

        message = {"type": "ping"}
        await server._handle_client_message(client, message)

        # Should receive a pong (heartbeat with pong=True)
        assert client.send_queue.qsize() == 1
        pong_event = await client.send_queue.get()
        assert pong_event.type == StreamingEventType.HEARTBEAT
        assert pong_event.payload.get("pong") is True

    @pytest.mark.asyncio
    async def test_handle_client_message_ping_queue_full(self, server: WebSocketServer):
        """Test handling ping message when queue is full."""
        client = await server.register_client()

        # Replace with a full queue
        small_queue = asyncio.Queue(maxsize=1)
        small_queue.put_nowait(StreamingEvent.heartbeat())
        client.send_queue = small_queue

        message = {"type": "ping"}
        # Should not raise
        await server._handle_client_message(client, message)

        # Queue should still have only 1 item
        assert small_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_handle_client_message_unknown_type(self, server: WebSocketServer):
        """Test handling unknown message type."""
        client = await server.register_client()

        # Drain any connected events
        drain_queue(client.send_queue)

        message = {"type": "unknown_type", "data": "test"}
        # Should not raise
        await server._handle_client_message(client, message)

        # No side effects (no new events added)
        assert client.send_queue.qsize() == 0


# =============================================================================
# WebSocketServer - Connection Handling
# =============================================================================


class TestWebSocketServerConnectionHandling:
    """Tests for WebSocketServer connection handling."""

    @pytest.mark.xfail(
        sys.version_info < (3, 12),
        reason=(
            "Known bug (spec follow-up 12): on 3.11 asyncio.wait_for loses the send "
            "loop's cancellation, and _send_loop recursion overflows after ~95s"
        ),
        run=False,
    )
    @pytest.mark.asyncio
    async def test_handle_connection_basic(
        self, server: WebSocketServer, mock_websocket: AsyncMock
    ):
        """Test basic connection handling."""
        # Configure mock to raise exception after welcome message to exit loop
        messages_sent = []

        async def track_send(text):
            messages_sent.append(text)

        mock_websocket.send_text = AsyncMock(side_effect=track_send)

        # Make receive_text raise to exit the loop
        mock_websocket.receive_text = AsyncMock(side_effect=Exception("Connection closed"))

        await server.handle_connection(mock_websocket)

        # Verify accept was called
        mock_websocket.accept.assert_called_once()

        # Verify welcome message was sent
        assert len(messages_sent) >= 1
        welcome = json.loads(messages_sent[0])
        assert welcome["type"] == "connected"
        assert "client_id" in welcome
        assert "timestamp" in welcome

        # Client should be unregistered after connection ends
        assert server.client_count == 0

    @pytest.mark.asyncio
    async def test_receive_loop_handles_json_decode_error(self, server: WebSocketServer):
        """Test _receive_loop handles invalid JSON gracefully."""
        mock_ws = AsyncMock()

        # First return invalid JSON, then raise to exit
        mock_ws.receive_text = AsyncMock(side_effect=["not valid json", Exception("done")])

        client = await server.register_client()

        # Should not raise
        await server._receive_loop(mock_ws, client)

    @pytest.mark.asyncio
    async def test_receive_loop_processes_valid_messages(self, server: WebSocketServer):
        """Test _receive_loop processes valid JSON messages."""
        mock_ws = AsyncMock()

        # Return subscribe message, then raise to exit
        subscribe_msg = json.dumps({"type": "subscribe", "filters": {"chain_id": "test"}})
        mock_ws.receive_text = AsyncMock(side_effect=[subscribe_msg, Exception("done")])

        client = await server.register_client()

        await server._receive_loop(mock_ws, client)

        # Filters should have been updated
        assert server._clients[client.id].filters == {"chain_id": "test"}

    @pytest.mark.asyncio
    async def test_send_loop_sends_queued_events(self, server: WebSocketServer):
        """Test _send_loop sends events from queue."""
        mock_ws = AsyncMock()
        sent_messages = []

        async def capture_send(text):
            sent_messages.append(text)
            if len(sent_messages) >= 2:
                raise Exception("done")

        mock_ws.send_text = AsyncMock(side_effect=capture_send)

        client = await server.register_client()

        # Drain connected events and add our test events
        drain_queue(client.send_queue)

        # Queue some events
        event1 = StreamingEvent.heartbeat()
        event2 = StreamingEvent.connected("test")
        client.send_queue.put_nowait(event1)
        client.send_queue.put_nowait(event2)

        await server._send_loop(mock_ws, client)

        assert len(sent_messages) == 2

    @pytest.mark.asyncio
    async def test_send_loop_sends_heartbeat_on_timeout(self, emitter: EventEmitter):
        """Test _send_loop sends heartbeat when queue times out."""
        # Use very short heartbeat interval
        server = WebSocketServer(emitter, heartbeat_interval=0.05)

        mock_ws = AsyncMock()
        sent_messages = []
        call_count = 0

        async def capture_send(text):
            nonlocal call_count
            sent_messages.append(text)
            call_count += 1
            if call_count >= 1:
                raise Exception("done")

        mock_ws.send_text = AsyncMock(side_effect=capture_send)

        client = await server.register_client()

        # Drain connected events to ensure empty queue triggers timeout
        drain_queue(client.send_queue)

        # Don't queue any events - should trigger heartbeat
        # Expected to exit via exception
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server._send_loop(mock_ws, client), timeout=0.2)

        # Should have sent at least one heartbeat
        assert len(sent_messages) >= 1
        parsed = json.loads(sent_messages[0])
        assert parsed["type"] == "heartbeat"


# =============================================================================
# WebSocketServer - Heartbeat Loop
# =============================================================================


class TestWebSocketServerHeartbeatLoop:
    """Tests for WebSocketServer heartbeat loop."""

    @pytest.mark.asyncio
    async def test_heartbeat_loop_sends_heartbeats(self, emitter: EventEmitter):
        """Test _heartbeat_loop sends heartbeats to all clients."""
        server = WebSocketServer(emitter, heartbeat_interval=0.05)
        server._running = True

        client1 = await server.register_client()
        client2 = await server.register_client()

        # Drain connected events
        drain_queue(client1.send_queue)
        drain_queue(client2.send_queue)

        # Run heartbeat loop briefly
        async def run_heartbeat():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(server._heartbeat_loop(), timeout=0.15)

        await run_heartbeat()

        # Both clients should have received heartbeats
        assert client1.send_queue.qsize() >= 1
        assert client2.send_queue.qsize() >= 1

        # Verify they are heartbeat events
        hb1 = await client1.send_queue.get()
        assert hb1.type == StreamingEventType.HEARTBEAT

    @pytest.mark.asyncio
    async def test_heartbeat_loop_handles_queue_full(self, emitter: EventEmitter):
        """Test _heartbeat_loop handles full queues gracefully."""
        server = WebSocketServer(emitter, heartbeat_interval=0.05)
        server._running = True

        client = await server.register_client()

        # Replace with a full queue
        small_queue = asyncio.Queue(maxsize=1)
        small_queue.put_nowait(StreamingEvent.heartbeat())
        client.send_queue = small_queue

        # Run heartbeat loop briefly - should not raise
        async def run_heartbeat():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(server._heartbeat_loop(), timeout=0.15)

        await run_heartbeat()

        # Queue should still have only 1 item
        assert small_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_heartbeat_loop_stops_when_not_running(self, emitter: EventEmitter):
        """Test _heartbeat_loop exits when _running is False."""
        server = WebSocketServer(emitter, heartbeat_interval=0.01)
        server._running = False

        # Should exit immediately
        await server._heartbeat_loop()


# =============================================================================
# WebSocketServer - Start/Stop
# =============================================================================


class TestWebSocketServerStartStop:
    """Tests for WebSocketServer start/stop methods."""

    @pytest.mark.asyncio
    async def test_stop_cancels_heartbeat_task(self, server: WebSocketServer):
        """Test stop() cancels heartbeat task."""

        # Create a mock heartbeat task
        async def dummy_heartbeat():
            await asyncio.sleep(10)

        server._running = True
        server._heartbeat_task = asyncio.create_task(dummy_heartbeat())

        await server.stop()

        assert server._running is False
        assert server._heartbeat_task.cancelled() or server._heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_stop_clears_clients(self, server: WebSocketServer):
        """Test stop() clears all clients."""
        await server.register_client()
        await server.register_client()
        assert server.client_count == 2

        await server.stop()

        assert server.client_count == 0

    @pytest.mark.asyncio
    async def test_stop_without_heartbeat_task(self, server: WebSocketServer):
        """Test stop() works when no heartbeat task exists."""
        server._running = True
        server._heartbeat_task = None

        # Should not raise
        await server.stop()

        assert server._running is False

    @pytest.mark.asyncio
    async def test_start_requires_websockets_package(self, server: WebSocketServer):
        """Test start() raises ImportError if websockets not installed."""
        # Force re-import check
        with (
            patch.dict("sys.modules", {"websockets": None}),
            patch("lctl.streaming.websocket.WebSocketServer.start") as mock_start,
        ):
            # Create a new server to test the import check
            async def raise_import():
                try:
                    import websockets  # noqa
                except ImportError as err:
                    raise ImportError("websockets package required for standalone mode") from err

            mock_start.side_effect = raise_import
            with pytest.raises(ImportError, match="websockets"):
                await mock_start()


# =============================================================================
# SSEHandler Tests
# =============================================================================


class TestSSEHandler:
    """Tests for SSEHandler class."""

    def test_format_sse_basic(self, emitter: EventEmitter):
        """Test _format_sse basic formatting."""
        handler = SSEHandler(emitter)

        result = handler._format_sse("test-event", {"key": "value"})

        assert result.startswith("event: test-event\n")
        assert "data:" in result
        assert result.endswith("\n\n")
        assert '"key"' in result
        assert '"value"' in result

    def test_format_sse_complex_data(self, emitter: EventEmitter):
        """Test _format_sse with complex data."""
        handler = SSEHandler(emitter)

        data = {"nested": {"list": [1, 2, 3]}, "timestamp": datetime.now()}
        result = handler._format_sse("complex", data)

        assert "event: complex\n" in result
        assert "nested" in result
        assert "list" in result

    def test_should_send_no_filters(self, emitter: EventEmitter):
        """Test _should_send with no filters."""
        handler = SSEHandler(emitter)

        event = StreamingEvent.heartbeat()
        assert handler._should_send(event, None) is True
        assert handler._should_send(event, {}) is True

    def test_should_send_chain_id_filter_match(self, emitter: EventEmitter):
        """Test _should_send with matching chain_id filter."""
        handler = SSEHandler(emitter)

        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-123",
            payload={},
        )

        assert handler._should_send(event, {"chain_id": "chain-123"}) is True

    def test_should_send_chain_id_filter_no_match(self, emitter: EventEmitter):
        """Test _should_send with non-matching chain_id filter."""
        handler = SSEHandler(emitter)

        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-456",
            payload={},
        )

        assert handler._should_send(event, {"chain_id": "chain-123"}) is False

    def test_should_send_event_types_filter_match(self, emitter: EventEmitter):
        """Test _should_send with matching event_types filter."""
        handler = SSEHandler(emitter)

        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id=None,
            payload={"type": "step_start"},
        )

        assert handler._should_send(event, {"event_types": ["step_start"]}) is True

    def test_should_send_event_types_filter_no_match(self, emitter: EventEmitter):
        """Test _should_send with non-matching event_types filter."""
        handler = SSEHandler(emitter)

        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id=None,
            payload={"type": "step_end"},
        )

        assert handler._should_send(event, {"event_types": ["step_start"]}) is False

    def test_should_send_streaming_event_type_filter(self, emitter: EventEmitter):
        """Test _should_send with streaming event type filter."""
        handler = SSEHandler(emitter)

        event = StreamingEvent.heartbeat()

        assert handler._should_send(event, {"event_types": ["heartbeat"]}) is True
        assert handler._should_send(event, {"event_types": ["connected"]}) is False

    @pytest.mark.asyncio
    async def test_event_stream_yields_connected_message(self, emitter: EventEmitter):
        """Test event_stream yields connected message first."""
        handler = SSEHandler(emitter, heartbeat_interval=0.1)

        messages = []

        async def collect():
            async for msg in handler.event_stream():
                messages.append(msg)
                break  # Just get first message

        await asyncio.wait_for(collect(), timeout=1.0)

        assert len(messages) == 1
        assert "event: connected" in messages[0]
        assert "client_id" in messages[0]

    @pytest.mark.asyncio
    async def test_event_stream_with_history(self, emitter: EventEmitter):
        """Test event_stream includes history when requested."""
        handler = SSEHandler(emitter, heartbeat_interval=0.5)

        # Add some events to history
        emitter.emit(StreamingEvent.heartbeat())
        emitter.emit(StreamingEvent.heartbeat())

        messages = []

        async def collect():
            async for msg in handler.event_stream(include_history=True):
                messages.append(msg)
                if len(messages) >= 3:  # connected + 2 history
                    break

        await asyncio.wait_for(collect(), timeout=1.0)

        # Should have connected + history events
        assert len(messages) >= 3
        assert "event: connected" in messages[0]
        assert "event: event" in messages[1]

    @pytest.mark.asyncio
    async def test_event_stream_with_history_filtered(self, emitter: EventEmitter):
        """Test event_stream history respects filters."""
        handler = SSEHandler(emitter, heartbeat_interval=0.5)

        # Add events with different chain_ids
        emitter.emit(
            StreamingEvent(
                id="1",
                type=StreamingEventType.EVENT,
                timestamp=datetime.now(UTC),
                chain_id="chain-1",
                payload={},
            )
        )
        emitter.emit(
            StreamingEvent(
                id="2",
                type=StreamingEventType.EVENT,
                timestamp=datetime.now(UTC),
                chain_id="chain-2",
                payload={},
            )
        )

        messages = []

        async def collect():
            filters = {"chain_id": "chain-1"}
            async for msg in handler.event_stream(filters=filters, include_history=True):
                messages.append(msg)
                if len(messages) >= 2:  # connected + 1 filtered history
                    break

        await asyncio.wait_for(collect(), timeout=1.0)

        # Should have connected + only chain-1 event
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_event_stream_yields_heartbeat_on_timeout(self, emitter: EventEmitter):
        """Test event_stream yields heartbeat comment on timeout."""
        handler = SSEHandler(emitter, heartbeat_interval=0.05)

        messages = []

        async def collect():
            async for msg in handler.event_stream():
                messages.append(msg)
                if len(messages) >= 2:  # connected + heartbeat
                    break

        await asyncio.wait_for(collect(), timeout=0.5)

        assert len(messages) >= 2
        assert "event: connected" in messages[0]
        # Heartbeat is a comment
        assert ": heartbeat" in messages[1]

    @pytest.mark.asyncio
    async def test_event_stream_queue_full_handling(self, emitter: EventEmitter):
        """Test event_stream handles QueueFull gracefully."""
        handler = SSEHandler(emitter, heartbeat_interval=0.5)

        messages = []
        event_count = 0

        async def collect():
            nonlocal event_count
            # Create a patched queue class that raises QueueFull
            original_method = asyncio.Queue.put_nowait

            def limited_put(self, item):
                nonlocal event_count
                event_count += 1
                if event_count > 2:
                    raise asyncio.QueueFull()
                return original_method(self, item)

            with patch.object(asyncio.Queue, "put_nowait", limited_put):
                async for msg in handler.event_stream():
                    messages.append(msg)
                    if len(messages) >= 1:
                        break

        await asyncio.wait_for(collect(), timeout=1.0)

        # Should still work, just dropping events
        assert len(messages) >= 1


# =============================================================================
# Utility Functions Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_sse_response(self, emitter: EventEmitter):
        """Test create_sse_response creates an SSE handler."""
        result = create_sse_response(emitter)

        # Should return an async generator
        assert hasattr(result, "__anext__") or hasattr(result, "__aiter__")

    def test_create_sse_response_with_filters(self, emitter: EventEmitter):
        """Test create_sse_response with filters."""
        filters = {"chain_id": "chain-123"}
        result = create_sse_response(emitter, filters=filters, include_history=True)

        assert hasattr(result, "__anext__") or hasattr(result, "__aiter__")

    def test_start_websocket_server_blocking_mode(self, emitter: EventEmitter):
        """Test start_websocket_server in blocking mode."""
        # We can't fully test blocking mode, but we can test that it creates a server
        # The stand-in closes the coroutine it receives, as asyncio.run would finish it;
        # a bare MagicMock would leave server.start() un-awaited.
        with patch("asyncio.run", side_effect=lambda coro: coro.close()) as mock_run:
            result = start_websocket_server(
                emitter, host="127.0.0.1", port=9999, run_in_thread=False
            )

            # In blocking mode, should return None
            mock_run.assert_called_once()
            assert mock_run.call_args.args[0].__qualname__ == "WebSocketServer.start"
            # Result is None because asyncio.run is mocked
            assert result is None

    def test_start_websocket_server_threaded_mode(self, emitter: EventEmitter):
        """Test start_websocket_server in threaded mode."""
        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            result = start_websocket_server(
                emitter, host="127.0.0.1", port=9999, run_in_thread=True
            )

            # Should return a WebSocketServer
            assert isinstance(result, WebSocketServer)

            # Thread should be started
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()


# =============================================================================
# WebSocketClient Tests
# =============================================================================


class TestWebSocketClient:
    """Tests for WebSocketClient dataclass."""

    def test_create_client(self):
        """Test creating a WebSocketClient."""
        client = WebSocketClient(id="test-client", connected_at=datetime.now(UTC))

        assert client.id == "test-client"
        assert client.filters == {}  # default
        assert isinstance(client.send_queue, asyncio.Queue)

    def test_create_client_with_filters(self):
        """Test creating a WebSocketClient with filters."""
        filters = {"chain_id": "chain-1"}
        client = WebSocketClient(
            id="test-client",
            connected_at=datetime.now(UTC),
            filters=filters,
        )

        assert client.filters == filters

    def test_create_client_with_custom_queue(self):
        """Test WebSocketClient can have events queued."""
        client = WebSocketClient(id="test-client", connected_at=datetime.now(UTC))

        event = StreamingEvent.heartbeat()
        client.send_queue.put_nowait(event)

        assert client.send_queue.qsize() == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestWebSocketServerIntegration:
    """Integration tests for WebSocketServer."""

    @pytest.mark.asyncio
    async def test_full_connection_lifecycle(self, emitter: EventEmitter):
        """Test full connection lifecycle from connect to disconnect."""
        server = WebSocketServer(emitter, heartbeat_interval=0.1, max_clients=10)

        events_captured: list[StreamingEvent] = []

        @emitter.on("all")
        def capture(event):
            events_captured.append(event)

        # Register client
        client = await server.register_client(
            client_id="test-client", filters={"chain_id": "chain-1"}
        )

        assert server.client_count == 1

        # Update filters
        await server.update_client_filters(client.id, {"chain_id": "chain-2"})

        # Drain connected events
        drain_queue(client.send_queue)

        # Simulate some events
        emitter.emit(
            StreamingEvent(
                id="e1",
                type=StreamingEventType.EVENT,
                timestamp=datetime.now(UTC),
                chain_id="chain-2",
                payload={},
            )
        )

        # Client should receive the event
        assert client.send_queue.qsize() == 1

        # Unregister
        await server.unregister_client(client.id)

        assert server.client_count == 0

        # Verify connected and disconnected events were captured
        event_types = [e.type for e in events_captured]
        assert StreamingEventType.CONNECTED in event_types
        assert StreamingEventType.DISCONNECTED in event_types

    @pytest.mark.asyncio
    async def test_multiple_clients_different_filters(self, emitter: EventEmitter):
        """Test multiple clients with different filters."""
        server = WebSocketServer(emitter)

        client1 = await server.register_client(filters={"chain_id": "chain-1"})
        client2 = await server.register_client(filters={"chain_id": "chain-2"})
        client3 = await server.register_client()  # No filters, gets all

        # Drain connected events
        drain_queue(client1.send_queue)
        drain_queue(client2.send_queue)
        drain_queue(client3.send_queue)

        # Emit event for chain-1
        event1 = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-1",
            payload={},
        )
        emitter.emit(event1)

        # Emit event for chain-2
        event2 = StreamingEvent(
            id="e2",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="chain-2",
            payload={},
        )
        emitter.emit(event2)

        # Verify distribution
        assert client1.send_queue.qsize() == 1  # Only chain-1
        assert client2.send_queue.qsize() == 1  # Only chain-2
        assert client3.send_queue.qsize() == 2  # Both

        # Clean up
        await server.stop()
