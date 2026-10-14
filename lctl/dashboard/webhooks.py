"""Webhook delivery for the dashboard: matching, HMAC signing, retries, delivery log.

Delivery must never block or break event emission, so ``dispatch`` only schedules work
and swallows every error.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any, Protocol

from ..core.ids import new_id

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 5.0
MAX_ATTEMPTS = 3
# Seconds to wait after a failed attempt (module constant so tests can patch it).
BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)
DELIVERY_LOG_SIZE = 100


class WebhookLike(Protocol):
    url: str
    events: list[str]
    chain_id: str | None
    secret: str | None


def sign(body: bytes, secret: str) -> str:
    """HMAC-SHA256 signature in the ``X-LCTL-Signature`` header format."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def event_type_of(event: dict[str, Any]) -> str:
    """The LCTL event type (``payload.type``) for wrapped events, else the stream type."""
    payload = event.get("payload")
    if event.get("type") == "event" and isinstance(payload, dict) and payload.get("type"):
        return str(payload["type"])
    return str(event.get("type", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _post(url: str, body: bytes, headers: dict[str, str]) -> int:
    """Blocking POST (run in a thread). Raises on non-2xx or transport errors."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310  # scheme validated at registration
    try:
        with _opener.open(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"HTTP {err.code}") from err


class WebhookDispatcher:
    """Matches events against registrations and delivers them in the background."""

    def __init__(self) -> None:
        self._hooks: dict[str, WebhookLike] = {}
        self._logs: dict[str, deque[dict[str, Any]]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- lifecycle -------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    async def aclose(self) -> None:
        """Cancel and await outstanding deliveries (app shutdown)."""
        self._loop = None
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- registration ----------------------------------------------------
    def register(self, webhook_id: str, reg: WebhookLike) -> None:
        self._hooks[webhook_id] = reg
        self._logs.setdefault(webhook_id, deque(maxlen=DELIVERY_LOG_SIZE))

    def unregister(self, webhook_id: str) -> None:
        self._hooks.pop(webhook_id, None)
        self._logs.pop(webhook_id, None)

    def deliveries(self, webhook_id: str) -> list[dict[str, Any]]:
        return list(self._logs.get(webhook_id, ()))

    def _record(self, webhook_id: str, record: dict[str, Any]) -> None:
        self._logs.setdefault(webhook_id, deque(maxlen=DELIVERY_LOG_SIZE)).append(record)

    # -- dispatch --------------------------------------------------------
    def dispatch(self, event: dict[str, Any]) -> None:
        """Schedule delivery to every matching webhook. Never blocks, never raises."""
        try:
            etype = event_type_of(event)
            matches = [
                (wid, reg)
                for wid, reg in list(self._hooks.items())
                if etype in reg.events
                and (reg.chain_id is None or reg.chain_id == event.get("chain_id"))
            ]
            for wid, reg in matches:
                self._schedule(self._deliver(wid, reg, event, etype))
        except Exception:
            logger.exception("webhook dispatch failed")

    def _schedule(self, coro: Coroutine[Any, Any, None]) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            self._start(coro)
            return
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._start, coro)
                return
            except RuntimeError:  # loop closed between the check and the call
                pass
        coro.close()
        logger.warning("webhook delivery dropped: no running event loop")

    def _start(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver(self, wid: str, reg: WebhookLike, event: dict[str, Any], etype: str) -> None:
        delivery_id = new_id()
        record: dict[str, Any] = {
            "delivery_id": delivery_id,
            "event_id": event.get("id"),
            "event": etype,
            "url": reg.url,
            "status": "pending",
            "attempts": 0,
            "last_error": None,
            "status_code": None,
            "created_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
        }
        self._record(wid, record)
        body = json.dumps(event, default=str).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lctl-webhook/1",
            "X-LCTL-Event": etype,
            "X-LCTL-Delivery": delivery_id,
        }
        if reg.secret:
            headers["X-LCTL-Signature"] = sign(body, reg.secret)
        try:
            for attempt in range(MAX_ATTEMPTS):
                record["attempts"] = attempt + 1
                try:
                    record["status_code"] = await asyncio.to_thread(_post, reg.url, body, headers)
                    record["status"] = "delivered"
                    record["last_error"] = None
                    return
                except Exception as err:
                    record["last_error"] = f"{type(err).__name__}: {err}"
                    logger.warning(
                        "webhook %s attempt %d/%d failed: %s",
                        wid,
                        attempt + 1,
                        MAX_ATTEMPTS,
                        record["last_error"],
                    )
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
            record["status"] = "failed"
        except asyncio.CancelledError:
            record["status"] = "failed"
            record["last_error"] = "cancelled (shutdown)"
            raise
        finally:
            record["finished_at"] = datetime.now(UTC).isoformat()
