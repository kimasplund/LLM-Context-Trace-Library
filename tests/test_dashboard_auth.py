"""Auth, bind defaults and reload for the dashboard (every route must be guarded)."""

import inspect
import os

import pytest

pytest.importorskip("fastapi")

from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lctl.dashboard.app import create_app

REMOTE = ("203.0.113.5", 50000)
PUBLIC_PATHS = {"/"}
STREAMING_PATHS = {"/api/events/stream"}


def _fill(path: str) -> str:
    out = path
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        out = out[:start] + "dummy" + out[end + 1 :]
    return out


def _guarded_routes(app):
    for route in app.routes:
        if isinstance(route, APIRoute | APIWebSocketRoute) and route.path not in PUBLIC_PATHS:
            yield route


@pytest.fixture
def keyed_app(monkeypatch, tmp_path):
    monkeypatch.setenv("LCTL_API_KEYS", "k1")
    monkeypatch.delenv("LCTL_REQUIRE_API_KEY", raising=False)
    monkeypatch.delenv("LCTL_LOCALHOST_BYPASS", raising=False)
    return create_app(tmp_path)


def _stream_status(app, method, path, headers):
    """Status of a never-ending response, read from the first ASGI message then cancelled.

    TestClient buffers whole responses, so an infinite SSE stream would hang it.
    """
    import asyncio

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": REMOTE,
        "server": ("testserver", 80),
        "scheme": "http",
        "root_path": "",
        "state": {},
    }

    async def run():
        started: asyncio.Future = asyncio.get_running_loop().create_future()

        async def receive():
            await asyncio.Event().wait()  # no body, never disconnects

        async def send(message):
            if message["type"] == "http.response.start" and not started.done():
                started.set_result(message["status"])

        task = asyncio.create_task(app(scope, receive, send))
        try:
            return await asyncio.wait_for(started, timeout=5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return asyncio.run(run())


def _request(client, route, headers=None):
    path = _fill(route.path)
    method = sorted(route.methods - {"HEAD"})[0]
    if route.path in STREAMING_PATHS:
        return _stream_status(client.app, method, path, headers or {})
    return client.request(method, path, headers=headers or {}).status_code


def test_every_data_route_requires_key_for_remote_clients(keyed_app):
    client = TestClient(keyed_app, client=REMOTE)
    seen_http = seen_ws = 0
    for route in _guarded_routes(keyed_app):
        if isinstance(route, APIWebSocketRoute):
            seen_ws += 1
            with pytest.raises(WebSocketDisconnect), client.websocket_connect(route.path):
                pass
            with client.websocket_connect(route.path + "?api_key=k1") as ws:
                assert ws.receive_json()["type"] == "connected"
            with client.websocket_connect(route.path, headers={"X-API-Key": "k1"}) as ws:
                assert ws.receive_json()["type"] == "connected"
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect(route.path + "?api_key=wrong"),
            ):
                pass
            continue
        seen_http += 1
        assert _request(client, route) in (401, 403), route.path
        assert _request(client, route, {"X-API-Key": "k1"}) not in (401, 403), route.path
    assert seen_http > 20
    assert seen_ws == 1


def test_index_and_static_stay_public(keyed_app):
    client = TestClient(keyed_app, client=REMOTE)
    assert client.get("/").status_code == 200
    static = [r for r in keyed_app.routes if getattr(r, "path", "") == "/static"]
    assert static
    assert client.get("/static/nonexistent.js").status_code == 404


def test_localhost_bypass_default_and_disable(monkeypatch, tmp_path):
    monkeypatch.setenv("LCTL_API_KEYS", "k1")
    monkeypatch.delenv("LCTL_LOCALHOST_BYPASS", raising=False)
    local = TestClient(create_app(tmp_path), client=("127.0.0.1", 50000))
    assert local.get("/api/chains").status_code == 200

    monkeypatch.setenv("LCTL_LOCALHOST_BYPASS", "false")
    strict = TestClient(create_app(tmp_path), client=("127.0.0.1", 50000))
    assert strict.get("/api/chains").status_code == 401
    assert strict.get("/api/chains", headers={"X-API-Key": "k1"}).status_code == 200


def test_bind_defaults_are_loopback():
    from click.testing import CliRunner

    from lctl.cli.main import cli
    from lctl.streaming.websocket import WebSocketServer, start_websocket_server

    help_text = CliRunner().invoke(cli, ["dashboard", "--help"]).output
    assert "127.0.0.1" in help_text
    assert inspect.signature(WebSocketServer.start).parameters["host"].default == "127.0.0.1"
    assert inspect.signature(start_websocket_server).parameters["host"].default == "127.0.0.1"


def test_run_dashboard_reload_uses_import_string(monkeypatch, tmp_path):
    import uvicorn

    from lctl.dashboard.app import run_dashboard

    monkeypatch.delenv("LCTL_DASHBOARD_DIR", raising=False)
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    run_dashboard(reload=True, working_dir=tmp_path)
    ((args, kwargs),) = calls
    assert isinstance(args[0], str)
    assert ":" in args[0]
    assert kwargs["factory"] is True
    assert kwargs["reload"] is True
    assert os.environ["LCTL_DASHBOARD_DIR"] == str(tmp_path)
    monkeypatch.delenv("LCTL_DASHBOARD_DIR")


def test_create_app_from_env(monkeypatch, tmp_path):
    from lctl.dashboard.app import create_app_from_env

    monkeypatch.setenv("LCTL_DASHBOARD_DIR", str(tmp_path))
    assert create_app_from_env().state.working_dir == tmp_path
