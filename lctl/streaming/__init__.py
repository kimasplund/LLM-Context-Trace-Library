"""LCTL Streaming - Real-time event streaming for live workflow updates.

This module provides real-time event streaming capabilities for LCTL:
- EventEmitter: Publish/subscribe pattern for event distribution
- WebSocket server: Live updates for dashboards and external consumers
- SSE (Server-Sent Events): HTTP-based alternative to WebSocket
- Subscriber patterns: Flexible event subscription management

Usage:
    from lctl.streaming import EventEmitter, start_websocket_server

    emitter = EventEmitter()

    @emitter.on("event")
    def handle_event(event):
        print(f"New event: {event.type}")

    # For dashboard
    start_websocket_server(emitter, port=8081)
"""

from .emitter import EventEmitter, EventHandler, StreamingEvent
from .subscriber import (
    AsyncSubscriber,
    BufferedSubscriber,
    EventSubscriber,
    FilteredSubscriber,
)
from .websocket import (
    SSEHandler,
    WebSocketServer,
    create_sse_response,
    start_websocket_server,
)

__all__ = [
    "EventEmitter",
    "EventHandler",
    "StreamingEvent",
    "EventSubscriber",
    "FilteredSubscriber",
    "BufferedSubscriber",
    "AsyncSubscriber",
    "WebSocketServer",
    "SSEHandler",
    "start_websocket_server",
    "create_sse_response",
]
