"""Real delivery of RPA webhooks: signing, filtering, retries, non-blocking emission."""

import json
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lctl.dashboard import webhooks
from lctl.dashboard.app import create_app

HEADERS = {"X-API-Key": "k1"}


class Receiver:
    """Local HTTP receiver recording every POST it gets."""

    def __init__(self, status: int = 200, delay: float = 0.0):
        self.requests: list[tuple[dict, bytes]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # http.server API
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(({k.lower(): v for k, v in self.headers.items()}, body))
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/hook"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LCTL_API_KEYS", "k1")
    monkeypatch.setattr(webhooks, "BACKOFF_SECONDS", (0, 0, 0))
    with TestClient(create_app(tmp_path), headers=HEADERS) as c:
        yield c


def register(client, url, **extra):
    r = client.post("/api/rpa/webhooks", json={"url": url, **extra})
    assert r.status_code == 200, r.text
    return r.json()["webhook_id"]


def emit(client, event_type="error", chain_id="c1"):
    return client.post(
        "/api/events/emit",
        json={"type": event_type, "chain_id": chain_id, "payload": {"msg": "boom"}},
    )


def test_sign_format():
    sig = webhooks.sign(b"body", "s")
    assert sig.startswith("sha256=")
    assert len(sig) == 7 + 64


def test_matching_event_is_delivered_signed(client):
    with Receiver() as rx:
        register(client, rx.url, secret="s", events=["error"])
        assert emit(client).status_code == 200
        assert wait_for(lambda: rx.requests)
    headers, body = rx.requests[0]
    event = json.loads(body)
    assert event["chain_id"] == "c1"
    assert event["payload"]["type"] == "error"
    assert headers["x-lctl-event"] == "error"
    assert headers["x-lctl-signature"] == webhooks.sign(body, "s")
    import uuid

    assert uuid.UUID(headers["x-lctl-delivery"]).version == 7


def test_non_matching_event_not_delivered(client):
    with Receiver() as rx:
        wid = register(client, rx.url, events=["error"])
        emit(client, event_type="step_start")
        emit(client, event_type="error", chain_id="sentinel")
        assert wait_for(lambda: rx.requests)
        # The sentinel arrived after the non-matching event, so it would have shown up by now.
        assert [json.loads(b)["payload"]["type"] for _, b in rx.requests] == ["error"]
        client.delete(f"/api/rpa/webhooks/{wid}")


def test_chain_filter(client):
    with Receiver() as rx:
        register(client, rx.url, events=["error"], chain_id="wanted")
        emit(client, event_type="error", chain_id="other")
        emit(client, event_type="error", chain_id="wanted")
        assert wait_for(lambda: rx.requests)
        assert [json.loads(b)["chain_id"] for _, b in rx.requests] == ["wanted"]


def test_failing_receiver_retried_three_times_and_logged(client):
    with Receiver(status=500) as rx:
        wid = register(client, rx.url, events=["error"])
        emit(client)

        def done():
            d = client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"]
            return d and d[0]["status"] == "failed"

        assert wait_for(done)
        record = client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"][0]
    assert len(rx.requests) == 3
    assert record["attempts"] == 3
    assert "500" in record["last_error"]
    assert record["status_code"] == 500
    assert len({h["x-lctl-delivery"] for h, _ in rx.requests}) == 1


def test_redirect_response_counts_as_failure(client):
    with Receiver(status=302) as rx:
        wid = register(client, rx.url, events=["error"])
        emit(client)

        def done():
            d = client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"]
            return d and d[0]["status"] == "failed"

        assert wait_for(done)
        record = client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"][0]
    assert record["status_code"] == 302
    assert len(rx.requests) == 3


def test_unreachable_receiver_logged_not_raised(client):
    wid = register(client, "http://127.0.0.1:1/hook", events=["error"])
    assert emit(client).status_code == 200

    def done():
        d = client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"]
        return d and d[0]["status"] == "failed"

    assert wait_for(done)
    assert (
        client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"][0]["attempts"] == 3
    )


def test_successful_delivery_logged(client):
    with Receiver() as rx:
        wid = register(client, rx.url, events=["error"])
        emit(client)

        def done():
            d = client.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"]
            return d and d[0]["status"] == "delivered"

        assert wait_for(done)


def test_slow_receiver_does_not_block_emission(client):
    with Receiver(delay=2.0) as rx:
        register(client, rx.url, events=["error"])
        start = time.monotonic()
        assert emit(client).status_code == 200
        assert time.monotonic() - start < 0.5
        assert wait_for(lambda: rx.requests)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://",
        "nonsense",
        "http://h:notaport/x",
        "http://h:99999/x",
    ],
)
def test_only_http_urls_accepted(client, url):
    r = client.post("/api/rpa/webhooks", json={"url": url})
    assert r.status_code in (400, 422)


def test_delete_stops_delivery_and_unknown_deliveries_404(client):
    with Receiver() as rx:
        wid = register(client, rx.url, events=["error"])
        register(client, rx.url, events=["step_end"])  # sentinel hook, still registered
        assert client.delete(f"/api/rpa/webhooks/{wid}").status_code == 200
        emit(client)
        emit(client, event_type="step_end")
        assert wait_for(lambda: rx.requests)
        assert [json.loads(b)["payload"]["type"] for _, b in rx.requests] == ["step_end"]
    assert client.get(f"/api/rpa/webhooks/{wid}/deliveries").status_code == 404


def test_emit_without_running_loop_never_raises(tmp_path, monkeypatch):
    """Dispatch from a plain thread with no app loop must drop, not raise."""
    monkeypatch.setenv("LCTL_API_KEYS", "k1")
    app = create_app(tmp_path)  # no lifespan started: no loop captured
    from lctl.core.ids import new_id
    from lctl.dashboard.app import WebhookRegistration

    d = app.state.webhook_dispatcher
    d.register("w1", WebhookRegistration(url="http://127.0.0.1:1/h", events=["error"]))
    d.dispatch({"id": new_id(), "type": "event", "chain_id": "c", "payload": {"type": "error"}})
    assert d.deliveries("w1") == []


def test_delivery_log_capped_at_100():
    d = webhooks.WebhookDispatcher()
    from lctl.dashboard.app import WebhookRegistration

    d.register("w", WebhookRegistration(url="http://127.0.0.1:1/h"))
    for i in range(150):
        d._logs["w"].append({"n": i})
    assert len(d.deliveries("w")) == 100


def test_slow_receiver_does_not_block_when_lifespan_never_ran(monkeypatch, tmp_path):
    """TestClient used without `with` (or a mounted sub-app) never runs the lifespan."""
    monkeypatch.setenv("LCTL_API_KEYS", "k1")
    plain = TestClient(create_app(tmp_path), headers=HEADERS)
    with Receiver(delay=2.0) as rx:
        wid = register(plain, rx.url, events=["error"])
        start = time.monotonic()
        assert emit(plain).status_code == 200
        assert time.monotonic() - start < 0.5
        time.sleep(0.2)
        # Nothing was scheduled on a throwaway request loop, so no bogus "failed" record.
        assert plain.get(f"/api/rpa/webhooks/{wid}/deliveries").json()["deliveries"] == []


def test_emit_from_plain_thread_is_delivered(client):
    from lctl.streaming.emitter import StreamingEvent, StreamingEventType

    with Receiver() as rx:
        register(client, rx.url, events=["error"])
        event = StreamingEvent(
            id="e1",
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id="c1",
            payload={"type": "error"},
        )
        t = threading.Thread(target=client.app.state.emitter.emit, args=(event,))
        t.start()
        t.join()
        assert wait_for(lambda: rx.requests)


def test_no_orphan_log_after_unregister(client):
    from lctl.dashboard.app import WebhookRegistration

    d = client.app.state.webhook_dispatcher
    reg = WebhookRegistration(url="http://127.0.0.1:1/h", events=["error"])
    d.register("gone", reg)
    d.dispatch({"id": "e", "type": "event", "chain_id": "c", "payload": {"type": "error"}})
    d.unregister("gone")  # before the scheduled task gets to run
    time.sleep(0.3)
    assert d.deliveries("gone") == []
    assert "gone" not in d._logs
