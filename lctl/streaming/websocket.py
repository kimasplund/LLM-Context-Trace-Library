"""WebSocket and SSE server for real-time event streaming."""

import asyncio
import concurrent.futures
import contextlib
import json
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .emitter import EventEmitter, StreamingEvent, StreamingEventType


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""

    id: str
    connected_at: datetime
    filters: dict[str, Any] = field(default_factory=dict)
    send_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class _StandaloneRun:
    """State of one active start() call, read by other threads as a single snapshot."""

    loop: asyncio.AbstractEventLoop
    thread_id: int
    stop_event: asyncio.Event
    # Completed when start() has finished; awaitable from any thread or event loop.
    done: concurrent.futures.Future[None] = field(default_factory=concurrent.futures.Future)


class _WebsocketsConnection:
    """Expose a `websockets` library connection through the Starlette-style API
    (accept / send_text / receive_text) that handle_connection() speaks."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def accept(self) -> None:
        """The websockets library completes the handshake before calling the handler."""

    async def send_text(self, data: str) -> None:
        await self._connection.send(data)

    async def receive_text(self) -> str:
        message = await self._connection.recv()
        return message if isinstance(message, str) else message.decode()


class WebSocketServer:
    """WebSocket server for broadcasting events to connected clients.

    Usage:
        emitter = EventEmitter()
        server = WebSocketServer(emitter)

        # In FastAPI:
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await server.handle_connection(websocket)

        # Or run standalone:
        await server.start(host="127.0.0.1", port=8081)
    """

    def __init__(
        self, emitter: EventEmitter, heartbeat_interval: float = 30.0, max_clients: int = 100
    ):
        """Initialize the WebSocket server.

        Args:
            emitter: The EventEmitter to subscribe to.
            heartbeat_interval: Interval in seconds for heartbeat messages.
            max_clients: Maximum number of concurrent clients.
        """
        self._emitter = emitter
        self._heartbeat_interval = heartbeat_interval
        self._max_clients = max_clients
        self._clients: dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._standalone: _StandaloneRun | None = None
        self._standalone_claim = threading.Lock()

        self._emitter.on("all", self._on_event)

    @property
    def client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running

    def _on_event(self, event: StreamingEvent) -> None:
        """Handle incoming events from the emitter (called on whichever thread emits)."""
        run = self._standalone
        if run is not None and threading.get_ident() != run.thread_id:
            # asyncio.Queue is not thread-safe and a plain put would not wake the
            # standalone server's loop, so hand the enqueue to that loop.
            with contextlib.suppress(RuntimeError):  # loop already closed: server is gone
                run.loop.call_soon_threadsafe(self._enqueue_event, event)
            return
        self._enqueue_event(event)

    def _enqueue_event(self, event: StreamingEvent) -> None:
        """Queue an event for every client whose filters match."""
        for client in list(self._clients.values()):
            if self._should_send_to_client(event, client):
                with contextlib.suppress(asyncio.QueueFull):
                    client.send_queue.put_nowait(event)

    def _should_send_to_client(self, event: StreamingEvent, client: WebSocketClient) -> bool:
        """Check if an event should be sent to a specific client."""
        filters = client.filters

        if not filters:
            return True

        if "chain_id" in filters and event.chain_id != filters["chain_id"]:
            return False

        if "event_types" in filters:
            event_type = event.type.value
            if event.type == StreamingEventType.EVENT and event.payload.get("type"):
                event_type = event.payload["type"]
            if event_type not in filters["event_types"]:
                return False

        return True

    async def register_client(
        self, client_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> WebSocketClient:
        """Register a new client.

        Args:
            client_id: Optional client ID. Generated if not provided.
            filters: Optional event filters.

        Returns:
            The registered client.

        Raises:
            ConnectionError: If max clients reached.
        """
        async with self._lock:
            if len(self._clients) >= self._max_clients:
                raise ConnectionError("Maximum client connections reached")

            if client_id is None:
                client_id = str(uuid4())[:8]

            client = WebSocketClient(
                id=client_id, connected_at=datetime.now(UTC), filters=filters or {}
            )
            self._clients[client_id] = client

            self._emitter.emit(StreamingEvent.connected(client_id))
            return client

    async def unregister_client(self, client_id: str) -> None:
        """Unregister a client.

        Args:
            client_id: The client ID to unregister.
        """
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                disconnected_event = StreamingEvent(
                    id=str(uuid4()),
                    type=StreamingEventType.DISCONNECTED,
                    timestamp=datetime.now(UTC),
                    chain_id=None,
                    payload={"client_id": client_id},
                )
                self._emitter.emit(disconnected_event)

    async def update_client_filters(self, client_id: str, filters: dict[str, Any]) -> None:
        """Update filters for a client.

        Args:
            client_id: The client ID.
            filters: The new filters.
        """
        async with self._lock:
            if client_id in self._clients:
                self._clients[client_id].filters = filters

    async def handle_connection(self, websocket: Any) -> None:
        """Handle a WebSocket connection.

        Compatible with FastAPI/Starlette WebSocket.

        Args:
            websocket: The WebSocket connection.
        """
        await websocket.accept()

        client = await self.register_client()
        client_id = client.id

        try:
            welcome_msg = {
                "type": "connected",
                "client_id": client_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await websocket.send_text(json.dumps(welcome_msg))

            receive_task = asyncio.create_task(self._receive_loop(websocket, client))
            send_task = asyncio.create_task(self._send_loop(websocket, client))

            _done, pending = await asyncio.wait(
                [receive_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        finally:
            await self.unregister_client(client_id)

    async def _receive_loop(self, websocket: Any, client: WebSocketClient) -> None:
        """Receive loop for processing client messages."""
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    await self._handle_client_message(client, message)
                except json.JSONDecodeError:
                    pass
        except Exception:  # noqa: S110  # tracing must never break the host application
            pass

    async def _send_loop(self, websocket: Any, client: WebSocketClient) -> None:
        """Send loop for pushing events to client."""
        try:
            while True:
                try:
                    # asyncio.timeout (not wait_for) so cancellation is never lost on 3.11
                    async with asyncio.timeout(self._heartbeat_interval):
                        event = await client.send_queue.get()
                except TimeoutError:
                    event = StreamingEvent.heartbeat()
                await websocket.send_text(event.to_json())
        except Exception:  # noqa: S110  # send failed: the connection is gone, end this loop
            pass

    async def _handle_client_message(
        self, client: WebSocketClient, message: dict[str, Any]
    ) -> None:
        """Handle a message from a client."""
        msg_type = message.get("type")

        if msg_type == "subscribe":
            filters = message.get("filters", {})
            await self.update_client_filters(client.id, filters)

        elif msg_type == "unsubscribe":
            await self.update_client_filters(client.id, {})

        elif msg_type == "ping":
            pong_event = StreamingEvent(
                id=str(uuid4()),
                type=StreamingEventType.HEARTBEAT,
                timestamp=datetime.now(UTC),
                chain_id=None,
                payload={"pong": True},
            )
            with contextlib.suppress(asyncio.QueueFull):
                client.send_queue.put_nowait(pong_event)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to all clients."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            heartbeat = StreamingEvent.heartbeat()
            for client in list(self._clients.values()):
                with contextlib.suppress(asyncio.QueueFull):
                    client.send_queue.put_nowait(heartbeat)

    async def start(self, host: str = "127.0.0.1", port: int = 8081) -> None:
        """Start the WebSocket server (standalone mode).

        Serves until stop() is called, from this or any other thread/event loop.

        Args:
            host: Host to bind to.
            port: Port to bind to.

        Raises:
            RuntimeError: If start() is already running on this server.
        """
        try:
            import websockets
        except ImportError as err:
            raise ImportError(
                "websockets package required for standalone mode. "
                "Install with: pip install websockets"
            ) from err

        # Claim the server before the first await, so a stop() issued right after
        # create_task(start()) is seen once the listener is bound.
        with self._standalone_claim:
            if self._standalone is not None:
                raise RuntimeError("WebSocketServer.start() is already running")
            run = _StandaloneRun(
                loop=asyncio.get_running_loop(),
                thread_id=threading.get_ident(),
                stop_event=asyncio.Event(),
            )
            self._standalone = run

        self._running = True
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._heartbeat_task = heartbeat_task

        # websockets>=14 calls handler(connection); older releases also pass the request path.
        async def handler(connection: Any, path: str | None = None) -> None:
            await self.handle_connection(_WebsocketsConnection(connection))

        try:
            # Leaving the block closes the listener and every client connection.
            async with websockets.serve(handler, host, port):
                await run.stop_event.wait()
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            if self._heartbeat_task is heartbeat_task:
                self._heartbeat_task = None
            self._running = False
            self._standalone = None
            run.done.set_result(None)

    def _request_stop(self, run: _StandaloneRun | None) -> None:
        """Make the given start() run leave its serve block. Runs on that run's loop."""
        if run is None or run is self._standalone:
            self._running = False
        if run is not None:
            run.stop_event.set()

    async def stop(self) -> None:
        """Stop the WebSocket server.

        From another thread or event loop than a standalone start(), this waits
        until that server has shut down.
        """
        run = self._standalone
        if run is None:
            # Let a start() task created just before this call claim the server.
            await asyncio.sleep(0)
            run = self._standalone

        if run is not None and run.loop is not asyncio.get_running_loop():
            with contextlib.suppress(RuntimeError):  # loop already closed: start() is done
                run.loop.call_soon_threadsafe(self._request_stop, run)
            # shield: cancelling this stop() must not cancel the shared done future
            await asyncio.shield(asyncio.wrap_future(run.done))
            return

        self._request_stop(run)
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        async with self._lock:
            self._clients.clear()


class SSEHandler:
    """Server-Sent Events handler for HTTP-based event streaming.

    Usage with FastAPI:
        emitter = EventEmitter()
        sse = SSEHandler(emitter)

        @app.get("/events")
        async def events():
            return StreamingResponse(
                sse.event_stream(),
                media_type="text/event-stream"
            )
    """

    def __init__(self, emitter: EventEmitter, heartbeat_interval: float = 30.0):
        """Initialize the SSE handler.

        Args:
            emitter: The EventEmitter to subscribe to.
            heartbeat_interval: Interval for heartbeat comments.
        """
        self._emitter = emitter
        self._heartbeat_interval = heartbeat_interval

    async def event_stream(
        self, filters: dict[str, Any] | None = None, include_history: bool = False
    ) -> AsyncIterator[str]:
        """Generate an SSE event stream.

        Args:
            filters: Optional event filters.
            include_history: Whether to include historical events.

        Yields:
            SSE formatted event strings.
        """
        queue: asyncio.Queue = asyncio.Queue()
        client_id = str(uuid4())[:8]

        def handler(event: StreamingEvent) -> None:
            if self._should_send(event, filters):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

        self._emitter.on("all", handler)

        try:
            connected_data = {"type": "connected", "client_id": client_id}
            yield self._format_sse("connected", connected_data)

            if include_history:
                for event in self._emitter.history:
                    if self._should_send(event, filters):
                        yield self._format_sse("event", event.to_dict())

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=self._heartbeat_interval)
                    yield self._format_sse("event", event.to_dict())
                except TimeoutError:
                    yield ": heartbeat\n\n"

        finally:
            self._emitter.off("all", handler)

    def _should_send(self, event: StreamingEvent, filters: dict[str, Any] | None) -> bool:
        """Check if an event should be sent based on filters."""
        if not filters:
            return True

        if "chain_id" in filters and event.chain_id != filters["chain_id"]:
            return False

        if "event_types" in filters:
            event_type = event.type.value
            if event.type == StreamingEventType.EVENT and event.payload.get("type"):
                event_type = event.payload["type"]
            if event_type not in filters["event_types"]:
                return False

        return True

    def _format_sse(self, event_type: str, data: Any) -> str:
        """Format data as SSE event.

        Args:
            event_type: The event type.
            data: The event data.

        Returns:
            SSE formatted string.
        """
        json_data = json.dumps(data, default=str)
        return f"event: {event_type}\ndata: {json_data}\n\n"


def create_sse_response(
    emitter: EventEmitter, filters: dict[str, Any] | None = None, include_history: bool = False
) -> AsyncIterator[str]:
    """Create an SSE response generator.

    Convenience function for creating SSE responses.

    Args:
        emitter: The EventEmitter to stream from.
        filters: Optional event filters.
        include_history: Whether to include historical events.

    Returns:
        An async generator yielding SSE formatted strings.
    """
    handler = SSEHandler(emitter)
    return handler.event_stream(filters, include_history)


def start_websocket_server(
    emitter: EventEmitter,
    host: str = "127.0.0.1",
    port: int = 8081,
    run_in_thread: bool = False,
) -> WebSocketServer | None:
    """Start a WebSocket server for event streaming.

    Args:
        emitter: The EventEmitter to stream from.
        host: Host to bind to.
        port: Port to bind to.
        run_in_thread: Whether to run in a background thread.

    Returns:
        The WebSocketServer instance if run_in_thread is True.
    """
    server = WebSocketServer(emitter)

    if run_in_thread:

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(server.start(host, port))
            finally:
                loop.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return server
    else:
        asyncio.run(server.start(host, port))
        return None
