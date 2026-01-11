"""WebSocket and SSE server for real-time event streaming."""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from .emitter import EventEmitter, StreamingEvent, StreamingEventType


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""
    id: str
    connected_at: datetime
    filters: Dict[str, Any] = field(default_factory=dict)
    send_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


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
        await server.start(host="0.0.0.0", port=8081)
    """

    def __init__(
        self,
        emitter: EventEmitter,
        heartbeat_interval: float = 30.0,
        max_clients: int = 100
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
        self._clients: Dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

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
        """Handle incoming events from the emitter."""
        for client in list(self._clients.values()):
            if self._should_send_to_client(event, client):
                try:
                    client.send_queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

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
        self,
        client_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
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
                id=client_id,
                connected_at=datetime.now(timezone.utc),
                filters=filters or {}
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
                    timestamp=datetime.now(timezone.utc),
                    chain_id=None,
                    payload={"client_id": client_id}
                )
                self._emitter.emit(disconnected_event)

    async def update_client_filters(
        self,
        client_id: str,
        filters: Dict[str, Any]
    ) -> None:
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
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await websocket.send_text(json.dumps(welcome_msg))

            receive_task = asyncio.create_task(self._receive_loop(websocket, client))
            send_task = asyncio.create_task(self._send_loop(websocket, client))

            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

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
        except Exception:
            pass

    async def _send_loop(self, websocket: Any, client: WebSocketClient) -> None:
        """Send loop for pushing events to client."""
        try:
            while True:
                event = await asyncio.wait_for(
                    client.send_queue.get(),
                    timeout=self._heartbeat_interval
                )
                await websocket.send_text(event.to_json())
        except asyncio.TimeoutError:
            heartbeat = StreamingEvent.heartbeat()
            await websocket.send_text(heartbeat.to_json())
            await self._send_loop(websocket, client)
        except Exception:
            pass

    async def _handle_client_message(
        self,
        client: WebSocketClient,
        message: Dict[str, Any]
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
                timestamp=datetime.now(timezone.utc),
                chain_id=None,
                payload={"pong": True}
            )
            try:
                client.send_queue.put_nowait(pong_event)
            except asyncio.QueueFull:
                pass

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to all clients."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            heartbeat = StreamingEvent.heartbeat()
            for client in list(self._clients.values()):
                try:
                    client.send_queue.put_nowait(heartbeat)
                except asyncio.QueueFull:
                    pass

    async def start(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        """Start the WebSocket server (standalone mode).

        Args:
            host: Host to bind to.
            port: Port to bind to.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required for standalone mode. "
                "Install with: pip install websockets"
            )

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        async def handler(websocket: Any, path: str) -> None:
            await self.handle_connection(websocket)

        server = await websockets.serve(handler, host, port)
        try:
            await server.wait_closed()
        finally:
            self._running = False
            if self._heartbeat_task:
                self._heartbeat_task.cancel()

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

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

    def __init__(
        self,
        emitter: EventEmitter,
        heartbeat_interval: float = 30.0
    ):
        """Initialize the SSE handler.

        Args:
            emitter: The EventEmitter to subscribe to.
            heartbeat_interval: Interval for heartbeat comments.
        """
        self._emitter = emitter
        self._heartbeat_interval = heartbeat_interval

    async def event_stream(
        self,
        filters: Optional[Dict[str, Any]] = None,
        include_history: bool = False
    ) -> Iterator[str]:
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
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

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
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self._heartbeat_interval
                    )
                    yield self._format_sse("event", event.to_dict())
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"

        finally:
            self._emitter.off("all", handler)

    def _should_send(
        self,
        event: StreamingEvent,
        filters: Optional[Dict[str, Any]]
    ) -> bool:
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
    emitter: EventEmitter,
    filters: Optional[Dict[str, Any]] = None,
    include_history: bool = False
) -> Iterator[str]:
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
    host: str = "0.0.0.0",
    port: int = 8081,
    run_in_thread: bool = False
) -> Optional[WebSocketServer]:
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
