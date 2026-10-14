"""LCTL Dashboard - FastAPI web application for chain visualization."""

import asyncio
import hmac
import json
import os
import secrets
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .. import __version__
from ..core.events import Chain, ReplayEngine
from ..evaluation.metrics import ChainMetrics
from ..streaming.emitter import EventEmitter, StreamingEvent, StreamingEventType
import contextlib

# =============================================================================
# Request/Response Models
# =============================================================================


class ReplayRequest(BaseModel):
    """Request model for replay endpoint."""

    filename: str
    target_seq: int


class CompareRequest(BaseModel):
    """Request model for chain comparison endpoint."""

    filename1: str
    filename2: str


class BatchMetricsRequest(BaseModel):
    """Request model for batch metrics endpoint."""

    filenames: list[str] = Field(..., max_length=100)


class SearchRequest(BaseModel):
    """Request model for cross-chain search."""

    query: str
    event_types: list[str] | None = None
    agents: list[str] | None = None
    limit: int = 100


class WebhookRegistration(BaseModel):
    """Request model for webhook registration."""

    url: str
    events: list[str] = ["error", "step_end"]
    chain_id: str | None = None
    secret: str | None = None


class RpaEventSubmission(BaseModel):
    """Request model for submitting RPA workflow events."""

    chain_id: str
    agent: str
    event_type: str
    data: dict | None = None


# =============================================================================
# Module-level Helper Functions
# =============================================================================


def _should_send_to_client(event: StreamingEvent, filters: dict[str, Any]) -> bool:
    """Check if an event should be sent to a client based on filters."""
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


def _get_secure_path(app: FastAPI, filename: str) -> Path:
    """Resolve path and ensure it's within working directory."""
    working_dir = app.state.working_dir.resolve()
    try:
        file_path = (working_dir / filename).resolve()

        # Security: Check for symlinks pointing outside working dir
        if file_path.is_symlink():
            raise HTTPException(status_code=403, detail="Access denied: Symbolic links not allowed")

        if not file_path.is_relative_to(working_dir):
            raise HTTPException(
                status_code=403, detail="Access denied: Path outside working directory"
            )
        return file_path
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: Invalid path")


def _get_cached_engine(app: FastAPI, file_path: Path) -> ReplayEngine:
    """Get ReplayEngine from cache or load it."""
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Chain file not found: {file_path.name}")

    stat = file_path.stat()
    cache_key = (str(file_path), stat.st_mtime)

    if cache_key in app.state.engine_cache:
        return app.state.engine_cache[cache_key]

    # Prune cache if too big (simple strategy)
    if len(app.state.engine_cache) > 50:
        app.state.engine_cache.clear()

    try:
        chain = Chain.load(file_path)
        engine = ReplayEngine(chain)
        app.state.engine_cache[cache_key] = engine
        return engine
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid chain file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading chain: {e}")


def _is_localhost(request: Request) -> bool:
    """Check if request is from localhost."""
    client_host = request.client.host if request.client else ""
    return client_host in ("127.0.0.1", "localhost", "::1")


def _create_verify_api_key(app: FastAPI) -> Callable:
    """Create an API key verification dependency for the given app."""

    async def verify_api_key(
        request: Request, x_api_key: str | None = Header(None, alias="X-API-Key")
    ) -> bool:
        """Verify API key for protected endpoints.

        Security behavior:
        - If LCTL_REQUIRE_API_KEY=true: Always require valid API key
        - If LCTL_LOCALHOST_BYPASS=true (default): Skip auth for localhost
        - Otherwise: Require API key for non-localhost requests
        """
        # Check if we should bypass for localhost
        if app.state.localhost_bypass and _is_localhost(request):
            return True

        # If API key enforcement is disabled and no keys configured, allow
        if not app.state.require_api_key and not app.state.api_keys:
            return True

        # Require API key if enforcement enabled or keys are configured
        if app.state.require_api_key or app.state.api_keys:
            if not x_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="API key required. Set X-API-Key header.",
                    headers={"WWW-Authenticate": "ApiKey"},
                )

            # Constant-time comparison to prevent timing attacks
            key_valid = any(
                hmac.compare_digest(x_api_key, valid_key) for valid_key in app.state.api_keys
            )

            if not key_valid:
                raise HTTPException(status_code=403, detail="Invalid API key")

        return True

    return verify_api_key


# =============================================================================
# Setup Functions
# =============================================================================


def _setup_middleware(app: FastAPI) -> None:
    """Configure CORS and other middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
            "http://localhost:5000",
            "http://127.0.0.1:5000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _setup_rate_limiter(app: FastAPI) -> Limiter:
    """Setup rate limiting."""
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return limiter


def _setup_app_state(app: FastAPI, working_dir: Path | None) -> None:
    """Initialize application state."""
    # Store working directory in app state
    app.state.working_dir = working_dir or Path.cwd()
    app.state.engine_cache = {}  # Cache for ReplayEngine instances: (abs_path, mtime) -> engine
    app.state.emitter = EventEmitter(max_history=500)  # Global event emitter for streaming
    app.state.webhooks: dict[str, WebhookRegistration] = {}  # Registered webhooks by ID

    # API Key Security Configuration
    # Load API keys from environment variable (comma-separated) or generate one
    env_keys = os.environ.get("LCTL_API_KEYS", "")
    if env_keys:
        app.state.api_keys = set(k.strip() for k in env_keys.split(",") if k.strip())
    else:
        app.state.api_keys = set()

    # Security settings
    app.state.require_api_key = os.environ.get("LCTL_REQUIRE_API_KEY", "false").lower() == "true"
    app.state.localhost_bypass = os.environ.get("LCTL_LOCALHOST_BYPASS", "true").lower() == "true"


def _setup_static_files(app: FastAPI) -> None:
    """Mount static files."""
    dashboard_dir = Path(__file__).parent
    static_dir = dashboard_dir / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =============================================================================
# Endpoint Registration Functions
# =============================================================================


def _register_health_endpoints(app: FastAPI) -> None:
    """Register health check and index endpoints."""
    dashboard_dir = Path(__file__).parent
    templates_dir = dashboard_dir / "templates"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main dashboard HTML."""
        index_path = templates_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="Dashboard template not found")
        return index_path.read_text()

    @app.get("/api/health", response_class=JSONResponse)
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "version": __version__, "streaming": True}


def _register_core_endpoints(app: FastAPI, limiter: Limiter) -> None:
    """Register core chain/replay endpoints."""

    @app.get("/api/chains", response_class=JSONResponse)
    async def list_chains():
        """List available .lctl.json files in the working directory."""
        working_dir = app.state.working_dir
        chains = []

        # Search for .lctl.json and .lctl.yaml files
        for pattern in ["*.lctl.json", "*.lctl.yaml", "*.lctl.yml"]:
            for file_path in working_dir.glob(pattern):
                try:
                    chain = Chain.load(file_path)
                    chains.append(
                        {
                            "filename": file_path.name,
                            "path": str(file_path),
                            "id": chain.id,
                            "version": chain.version,
                            "event_count": len(chain.events),
                        }
                    )
                except Exception as e:
                    # Include file but mark as having errors
                    chains.append(
                        {
                            "filename": file_path.name,
                            "path": str(file_path),
                            "id": "error",
                            "version": "unknown",
                            "event_count": 0,
                            "error": str(e),
                        }
                    )

        return {"chains": chains, "working_dir": str(working_dir)}

    @app.get("/api/chain/{filename}", response_class=JSONResponse)
    async def get_chain(filename: str):
        """Load and return chain data with analysis."""
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        # Get full state
        state = engine.replay_all()

        # Get bottlenecks
        bottlenecks = engine.find_bottlenecks()

        # Get confidence timeline
        confidence_timeline = engine.get_confidence_timeline()

        # Get trace
        trace = engine.get_trace()

        # Extract unique agents
        agents = list(set(e.agent for e in chain.events))

        # Build events list with serializable data
        events = [e.to_dict() for e in chain.events]

        # Find errors
        [
            e.to_dict()
            for e in chain.events
            if (e.type.value if hasattr(e.type, "value") else e.type) == "error"
        ]

        return {
            "chain": {"id": chain.id, "version": chain.version, "filename": filename},
            "events": events,
            "agents": sorted(agents),
            "state": {"facts": state.facts, "errors": state.errors, "metrics": state.metrics},
            "analysis": {
                "bottlenecks": bottlenecks,
                "confidence_timeline": confidence_timeline,
                "trace": trace,
            },
        }

    @app.post("/api/replay", response_class=JSONResponse)
    async def replay_chain(request: ReplayRequest):
        """Replay chain to a specific sequence number."""
        file_path = _get_secure_path(app, request.filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        if request.target_seq < 1:
            raise HTTPException(status_code=400, detail="target_seq must be >= 1")

        if chain.events and request.target_seq > chain.events[-1].seq:
            raise HTTPException(
                status_code=400,
                detail=f"target_seq {request.target_seq} exceeds max seq {chain.events[-1].seq}",
            )

        state = engine.replay_to(request.target_seq)

        # Get events up to target_seq
        events_at_seq = [e.to_dict() for e in chain.events if e.seq <= request.target_seq]

        return {
            "target_seq": request.target_seq,
            "events": events_at_seq,
            "state": {
                "facts": state.facts,
                "errors": state.errors,
                "metrics": state.metrics,
                "current_agent": state.current_agent,
                "current_step": state.current_step,
            },
        }

    @app.post("/api/compare", response_class=JSONResponse)
    @limiter.limit("10/minute")
    async def compare_chains(request: Request, compare_req: CompareRequest):
        """Compare two chains and return differences."""
        path1 = _get_secure_path(app, compare_req.filename1)
        path2 = _get_secure_path(app, compare_req.filename2)

        engine1 = _get_cached_engine(app, path1)
        engine2 = _get_cached_engine(app, path2)
        chain1 = engine1.chain
        chain2 = engine2.chain

        state1 = engine1.replay_all()
        state2 = engine2.replay_all()

        diffs = engine1.diff(engine2)

        def get_chain_summary(chain: Chain, state, engine: ReplayEngine) -> dict[str, Any]:
            bottlenecks = engine.find_bottlenecks()
            return {
                "id": chain.id,
                "event_count": len(chain.events),
                "agent_count": len(set(e.agent for e in chain.events)),
                "agents": sorted(set(e.agent for e in chain.events)),
                "fact_count": len(state.facts),
                "error_count": state.metrics.get("error_count", 0),
                "total_duration_ms": state.metrics.get("total_duration_ms", 0),
                "total_tokens": state.metrics.get("total_tokens_in", 0)
                + state.metrics.get("total_tokens_out", 0),
                "top_bottleneck": bottlenecks[0] if bottlenecks else None,
            }

        fact_comparison = {"only_in_first": [], "only_in_second": [], "different": [], "same": []}

        all_fact_ids = set(state1.facts.keys()) | set(state2.facts.keys())
        for fact_id in all_fact_ids:
            fact1 = state1.facts.get(fact_id)
            fact2 = state2.facts.get(fact_id)

            if fact1 and not fact2:
                fact_comparison["only_in_first"].append({"id": fact_id, **fact1})
            elif fact2 and not fact1:
                fact_comparison["only_in_second"].append({"id": fact_id, **fact2})
            elif fact1 and fact2:
                if fact1.get("text") != fact2.get("text") or fact1.get("confidence") != fact2.get(
                    "confidence"
                ):
                    fact_comparison["different"].append(
                        {"id": fact_id, "first": fact1, "second": fact2}
                    )
                else:
                    fact_comparison["same"].append(fact_id)

        return {
            "chain1": get_chain_summary(chain1, state1, engine1),
            "chain2": get_chain_summary(chain2, state2, engine2),
            "event_diffs": diffs[:50],
            "diff_count": len(diffs),
            "fact_comparison": fact_comparison,
            "divergence_point": diffs[0]["seq"] if diffs else None,
        }


def _register_analysis_endpoints(app: FastAPI) -> None:
    """Register analysis endpoints (stats, bottlenecks, confidence)."""

    @app.get("/api/metrics/{filename}", response_class=JSONResponse)
    async def get_metrics(filename: str):
        """Get detailed metrics for a chain including per-agent breakdown."""
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        state = engine.replay_all()
        bottlenecks = engine.find_bottlenecks()

        # use shared ChainMetrics to calculate stats efficiently
        metrics = ChainMetrics.from_chain(chain, state)
        agent_metrics = metrics.agent_stats

        error_timeline = []
        for event in chain.events:
            event_type = event.type.value if hasattr(event.type, "value") else event.type
            if event_type == "error":
                error_timeline.append(
                    {
                        "seq": event.seq,
                        "timestamp": event.timestamp.isoformat(),
                        "agent": event.agent,
                        "category": event.data.get("category", "unknown"),
                        "type": event.data.get("type", "unknown"),
                        "message": event.data.get("message", ""),
                        "recoverable": event.data.get("recoverable", False),
                    }
                )

        token_distribution = {
            "input": state.metrics.get("total_tokens_in", 0),
            "output": state.metrics.get("total_tokens_out", 0),
        }

        return {
            "chain": {"id": chain.id, "filename": filename},
            "summary": {
                "total_events": metrics.total_events,
                "total_agents": metrics.agent_count,
                "total_duration_ms": metrics.total_duration_ms,
                "total_tokens": metrics.total_tokens,
                "total_errors": metrics.error_count,
                "total_facts": metrics.fact_count,
            },
            "agent_metrics": agent_metrics,
            "token_distribution": token_distribution,
            "error_timeline": error_timeline,
            "bottlenecks": bottlenecks[:10],
        }

    @app.get("/api/evaluation/{filename}", response_class=JSONResponse)
    async def get_evaluation(filename: str):
        """Get comprehensive evaluation report for a chain."""
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain
        state = engine.replay_all()
        bottlenecks = engine.find_bottlenecks()
        confidence_timeline = engine.get_confidence_timeline()

        total_duration = state.metrics.get("total_duration_ms", 0)
        total_tokens = state.metrics.get("total_tokens_in", 0) + state.metrics.get(
            "total_tokens_out", 0
        )
        error_count = state.metrics.get("error_count", 0)
        fact_count = len(state.facts)

        perf_score = 100
        if total_duration > 30000:
            perf_score -= min(30, (total_duration - 30000) // 1000)
        if total_tokens > 50000:
            perf_score -= min(20, (total_tokens - 50000) // 5000)
        perf_score = max(0, perf_score)

        reliability_score = 100
        if error_count > 0:
            reliability_score -= min(50, error_count * 10)
        recoverable_errors = sum(1 for e in state.errors if e.get("recoverable", False))
        if recoverable_errors > 0:
            reliability_score += min(20, recoverable_errors * 5)
        reliability_score = max(0, min(100, reliability_score))

        avg_confidence = 0
        if state.facts:
            confidences = [f.get("confidence", 1.0) for f in state.facts.values()]
            avg_confidence = sum(confidences) / len(confidences)

        quality_score = int(avg_confidence * 100)
        low_confidence_facts = [
            {"id": fid, **fact}
            for fid, fact in state.facts.items()
            if fact.get("confidence", 1.0) < 0.7
        ]

        fact_heatmap = []
        for fact_id, timeline in confidence_timeline.items():
            for point in timeline:
                fact_heatmap.append(
                    {
                        "fact_id": fact_id,
                        "seq": point["seq"],
                        "confidence": point["confidence"],
                        "agent": point["agent"],
                    }
                )
        fact_heatmap.sort(key=lambda x: x["seq"])

        issues = []
        warnings = []

        if error_count > 0:
            issues.append(
                {
                    "type": "errors",
                    "severity": "high" if error_count > 3 else "medium",
                    "message": f"{error_count} error(s) occurred during execution",
                    "details": state.errors[:5],
                }
            )

        if bottlenecks and bottlenecks[0].get("percentage", 0) > 50:
            warnings.append(
                {
                    "type": "bottleneck",
                    "severity": "medium",
                    "message": f"Agent '{bottlenecks[0]['agent']}' consumed {bottlenecks[0]['percentage']:.0f}% of total time",
                    "details": bottlenecks[0],
                }
            )

        if low_confidence_facts:
            warnings.append(
                {
                    "type": "low_confidence",
                    "severity": "low",
                    "message": f"{len(low_confidence_facts)} fact(s) have confidence below 70%",
                    "details": low_confidence_facts[:5],
                }
            )

        if total_tokens > 100000:
            warnings.append(
                {
                    "type": "high_token_usage",
                    "severity": "medium",
                    "message": f"High token consumption: {total_tokens:,} tokens",
                    "details": {
                        "total": total_tokens,
                        "input": state.metrics.get("total_tokens_in", 0),
                        "output": state.metrics.get("total_tokens_out", 0),
                    },
                }
            )

        overall_score = int(perf_score * 0.3 + reliability_score * 0.4 + quality_score * 0.3)

        return {
            "chain": {"id": chain.id, "filename": filename, "version": chain.version},
            "scores": {
                "overall": overall_score,
                "performance": perf_score,
                "reliability": reliability_score,
                "quality": quality_score,
            },
            "metrics": {
                "total_events": len(chain.events),
                "total_duration_ms": total_duration,
                "total_tokens": total_tokens,
                "tokens_in": state.metrics.get("total_tokens_in", 0),
                "tokens_out": state.metrics.get("total_tokens_out", 0),
                "error_count": error_count,
                "fact_count": fact_count,
                "average_confidence": round(avg_confidence, 3),
            },
            "issues": issues,
            "warnings": warnings,
            "bottlenecks": bottlenecks[:5],
            "low_confidence_facts": low_confidence_facts,
            "fact_confidence_heatmap": fact_heatmap,
            "confidence_timeline": confidence_timeline,
        }


def _register_streaming_endpoints(app: FastAPI) -> None:
    """Register WebSocket and SSE streaming endpoints."""

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time event streaming.

        Clients can send messages to filter events:
        - {"type": "subscribe", "filters": {"chain_id": "xxx", "event_types": ["step_start"]}}
        - {"type": "unsubscribe"}
        - {"type": "ping"}
        """
        await websocket.accept()

        client_id = str(uuid4())[:8]
        filters: dict[str, Any] = {}
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        welcome_msg = {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await websocket.send_text(json.dumps(welcome_msg))

        emitter = app.state.emitter

        def on_event(event: StreamingEvent) -> None:
            if _should_send_to_client(event, filters):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

        emitter.on("all", on_event)

        async def receive_loop():
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)
                        msg_type = message.get("type")

                        if msg_type == "subscribe":
                            nonlocal filters
                            filters = message.get("filters", {})
                        elif msg_type == "unsubscribe":
                            filters = {}
                        elif msg_type == "ping":
                            pong = {
                                "type": "pong",
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                            await websocket.send_text(json.dumps(pong))
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass

        async def send_loop():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        await websocket.send_text(event.to_json())
                    except TimeoutError:
                        heartbeat = {
                            "type": "heartbeat",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        await websocket.send_text(json.dumps(heartbeat))
            except Exception:
                pass

        try:
            receive_task = asyncio.create_task(receive_loop())
            send_task = asyncio.create_task(send_loop())

            done, pending = await asyncio.wait(
                [receive_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            emitter.off("all", on_event)

    @app.get("/api/events/stream")
    async def sse_endpoint(
        chain_id: str | None = Query(None, description="Filter by chain ID"),
        event_types: str | None = Query(None, description="Comma-separated event types"),
        include_history: bool = Query(False, description="Include historical events"),
    ):
        """Server-Sent Events endpoint for real-time event streaming.

        Alternative to WebSocket for HTTP-based streaming.
        """
        filters: dict[str, Any] = {}
        if chain_id:
            filters["chain_id"] = chain_id
        if event_types:
            filters["event_types"] = event_types.split(",")

        async def event_generator() -> AsyncIterator[str]:
            client_id = str(uuid4())[:8]
            queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

            emitter = app.state.emitter

            def on_event(event: StreamingEvent) -> None:
                if _should_send_to_client(event, filters):
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(event)

            emitter.on("all", on_event)

            try:
                connected_data = {"type": "connected", "client_id": client_id}
                yield f"event: connected\ndata: {json.dumps(connected_data)}\n\n"

                if include_history:
                    for event in emitter.history:
                        if _should_send_to_client(event, filters):
                            yield f"event: event\ndata: {event.to_json()}\n\n"

                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"event: event\ndata: {event.to_json()}\n\n"
                    except TimeoutError:
                        yield ": heartbeat\n\n"

            finally:
                emitter.off("all", on_event)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/streaming/status", response_class=JSONResponse)
    async def streaming_status():
        """Get streaming status including connected clients and event count."""
        emitter = app.state.emitter
        return {
            "enabled": True,
            "chain_id": emitter.chain_id,
            "event_count": emitter.event_count,
            "handler_count": emitter.handler_count(),
            "history_size": len(emitter.history),
        }


def _register_event_endpoints(app: FastAPI) -> None:
    """Register event emission endpoints."""

    @app.post("/api/events/emit", response_class=JSONResponse)
    async def emit_event(request: Request):
        """Emit a custom event to all connected clients.

        Useful for testing and manual event injection.
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        event_type = data.get("type", "custom")
        chain_id = data.get("chain_id")
        payload = data.get("payload", {})

        streaming_event = StreamingEvent(
            id=str(uuid4()),
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id=chain_id,
            payload={"type": event_type, **payload},
        )

        app.state.emitter.emit(streaming_event)

        return {"status": "emitted", "event_id": streaming_event.id}


def _register_rpa_endpoints(app: FastAPI, limiter: Limiter) -> None:
    """Register RPA/UiPath integration endpoints."""
    verify_api_key = _create_verify_api_key(app)

    @app.get("/api/rpa/auth/status", response_class=JSONResponse)
    async def rpa_auth_status(request: Request):
        """Check authentication status and requirements.

        Returns info about security configuration without requiring auth.
        """
        return {
            "require_api_key": app.state.require_api_key,
            "localhost_bypass": app.state.localhost_bypass,
            "is_localhost": _is_localhost(request),
            "api_keys_configured": len(app.state.api_keys) > 0,
            "auth_required": (
                app.state.require_api_key
                or (
                    app.state.api_keys
                    and not (app.state.localhost_bypass and _is_localhost(request))
                )
            ),
        }

    @app.post("/api/rpa/auth/generate-key", response_class=JSONResponse)
    @limiter.limit("5/minute")
    async def rpa_generate_api_key(request: Request, _auth: bool = Depends(verify_api_key)):
        """Generate a new API key (requires existing auth or localhost).

        For initial setup on localhost, this can be called without auth.
        """
        # Only allow from localhost if no keys exist yet
        if not app.state.api_keys and not _is_localhost(request):
            raise HTTPException(
                status_code=403, detail="Initial key generation only allowed from localhost"
            )

        new_key = secrets.token_urlsafe(32)
        app.state.api_keys.add(new_key)

        return {
            "api_key": new_key,
            "message": "Store this key securely. It cannot be retrieved later.",
            "usage": "Set header: X-API-Key: " + new_key,
        }

    @app.get("/api/rpa/summary/{filename}", response_class=JSONResponse)
    async def rpa_summary(filename: str, _auth: bool = Depends(verify_api_key)):
        """Get minimal summary for quick RPA checks.

        Returns flat structure optimized for UiPath DataTable mapping.
        """
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain
        state = engine.replay_all()

        error_count = state.metrics.get("error_count", 0)
        has_errors = error_count > 0

        return {
            "chain_id": chain.id,
            "filename": filename,
            "event_count": len(chain.events),
            "agent_count": len(set(e.agent for e in chain.events)),
            "error_count": error_count,
            "has_errors": has_errors,
            "fact_count": len(state.facts),
            "total_duration_ms": state.metrics.get("total_duration_ms", 0),
            "total_tokens": state.metrics.get("total_tokens_in", 0)
            + state.metrics.get("total_tokens_out", 0),
            "status": "error" if has_errors else "success",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/api/rpa/events/{filename}", response_class=JSONResponse)
    async def rpa_events(
        filename: str,
        event_type: str | None = Query(None, description="Filter by event type"),
        agent: str | None = Query(None, description="Filter by agent"),
        limit: int = Query(1000, description="Max events to return"),
        offset: int = Query(0, description="Skip first N events"),
        _auth: bool = Depends(verify_api_key),
    ):
        """Get flattened events list optimized for RPA DataTable.

        Each event is a flat dictionary with no nested objects.
        """
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        events = []
        for e in chain.events:
            # Apply filters
            e_type = e.type.value if hasattr(e.type, "value") else e.type
            if event_type and e_type != event_type:
                continue
            if agent and e.agent != agent:
                continue

            # Flatten event data
            flat_event = {
                "seq": e.seq,
                "type": e_type,
                "timestamp": e.timestamp.isoformat(),
                "agent": e.agent or "",
                "chain_id": chain.id,
            }

            # Flatten common data fields
            if e.data:
                flat_event["intent"] = e.data.get("intent", "")
                flat_event["outcome"] = e.data.get("outcome", "")
                flat_event["duration_ms"] = e.data.get("duration_ms", 0)
                flat_event["tokens_in"] = e.data.get("tokens_in", 0)
                flat_event["tokens_out"] = e.data.get("tokens_out", 0)
                flat_event["tool_name"] = e.data.get("tool_name", "")
                flat_event["fact_id"] = e.data.get("fact_id", "")
                flat_event["confidence"] = e.data.get("confidence", 0)
                flat_event["error_type"] = e.data.get("type", "")
                flat_event["error_message"] = e.data.get("message", "")
                flat_event["input_summary"] = str(e.data.get("input_summary", ""))[:200]
                flat_event["output_summary"] = str(e.data.get("output_summary", ""))[:200]

            events.append(flat_event)

        # Apply pagination
        total = len(events)
        events = events[offset : offset + limit]

        return {
            "events": events,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(events) < total,
        }

    @app.get("/api/rpa/export/{filename}")
    async def rpa_export(
        filename: str,
        format: str = Query("csv", description="Export format: csv, json"),
        event_type: str | None = Query(None, description="Filter by event type"),
        _auth: bool = Depends(verify_api_key),
    ):
        """Export chain data in RPA-friendly formats (CSV or JSON).

        CSV format is ideal for Excel/DataTable processing in UiPath.
        """
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        events = []
        for e in chain.events:
            e_type = e.type.value if hasattr(e.type, "value") else e.type
            if event_type and e_type != event_type:
                continue

            flat_event = {
                "seq": e.seq,
                "type": e_type,
                "timestamp": e.timestamp.isoformat(),
                "agent": e.agent or "",
                "intent": e.data.get("intent", "") if e.data else "",
                "outcome": e.data.get("outcome", "") if e.data else "",
                "duration_ms": e.data.get("duration_ms", 0) if e.data else 0,
                "tokens_in": e.data.get("tokens_in", 0) if e.data else 0,
                "tokens_out": e.data.get("tokens_out", 0) if e.data else 0,
                "error_message": e.data.get("message", "") if e.data else "",
            }
            events.append(flat_event)

        if format == "csv":
            import csv
            import io

            output = io.StringIO()
            if events:
                writer = csv.DictWriter(output, fieldnames=events[0].keys())
                writer.writeheader()
                writer.writerows(events)

            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"},
            )
        else:
            return JSONResponse({"events": events, "chain_id": chain.id})

    @app.post("/api/rpa/batch/metrics", response_class=JSONResponse)
    @limiter.limit("20/minute")
    async def rpa_batch_metrics(
        request: Request,
        batch_req: BatchMetricsRequest,
        _auth: bool = Depends(verify_api_key),
    ):
        """Get metrics for multiple chains in one request.

        Reduces API calls for UiPath workflows processing multiple chains.
        """
        results = []
        errors = []

        for filename in batch_req.filenames:
            try:
                file_path = _get_secure_path(app, filename)
                engine = _get_cached_engine(app, file_path)
                chain = engine.chain
                state = engine.replay_all()

                results.append(
                    {
                        "filename": filename,
                        "chain_id": chain.id,
                        "event_count": len(chain.events),
                        "error_count": state.metrics.get("error_count", 0),
                        "duration_ms": state.metrics.get("total_duration_ms", 0),
                        "tokens": state.metrics.get("total_tokens_in", 0)
                        + state.metrics.get("total_tokens_out", 0),
                        "fact_count": len(state.facts),
                        "status": "success",
                    }
                )
            except Exception as e:
                errors.append({"filename": filename, "error": str(e), "status": "error"})

        return {
            "results": results,
            "errors": errors,
            "total_processed": len(results),
            "total_errors": len(errors),
        }

    @app.post("/api/rpa/search", response_class=JSONResponse)
    @limiter.limit("30/minute")
    async def rpa_search(
        request: Request,
        search_req: SearchRequest,
        _auth: bool = Depends(verify_api_key),
    ):
        """Search across all chains for specific events or patterns.

        Useful for finding errors, specific agents, or patterns across workflows.
        """
        working_dir = app.state.working_dir
        results = []
        query_lower = search_req.query.lower()

        for pattern in ["*.lctl.json", "*.lctl.yaml", "*.lctl.yml"]:
            for file_path in working_dir.glob(pattern):
                try:
                    chain = Chain.load(file_path)

                    for event in chain.events:
                        e_type = event.type.value if hasattr(event.type, "value") else event.type

                        # Apply type filter
                        if search_req.event_types and e_type not in search_req.event_types:
                            continue

                        # Apply agent filter
                        if search_req.agents and event.agent not in search_req.agents:
                            continue

                        # Search in event data
                        match = False
                        searchable = f"{event.agent} {e_type}"
                        if event.data:
                            searchable += f" {json.dumps(event.data)}"

                        if query_lower in searchable.lower():
                            match = True

                        if match:
                            results.append(
                                {
                                    "filename": file_path.name,
                                    "chain_id": chain.id,
                                    "seq": event.seq,
                                    "type": e_type,
                                    "agent": event.agent,
                                    "timestamp": event.timestamp.isoformat(),
                                    "preview": str(event.data)[:100] if event.data else "",
                                }
                            )

                            if len(results) >= search_req.limit:
                                break

                except Exception:
                    continue

                if len(results) >= search_req.limit:
                    break

        return {
            "results": results,
            "total": len(results),
            "query": search_req.query,
            "truncated": len(results) >= search_req.limit,
        }

    @app.post("/api/rpa/webhooks", response_class=JSONResponse)
    async def register_webhook(webhook: WebhookRegistration, _auth: bool = Depends(verify_api_key)):
        """Register a webhook for event notifications.

        UiPath can receive callbacks when specific events occur.
        """
        webhook_id = str(uuid4())[:12]
        app.state.webhooks[webhook_id] = webhook

        return {
            "webhook_id": webhook_id,
            "url": webhook.url,
            "events": webhook.events,
            "status": "registered",
        }

    @app.get("/api/rpa/webhooks", response_class=JSONResponse)
    async def list_webhooks(_auth: bool = Depends(verify_api_key)):
        """List all registered webhooks."""
        webhooks = []
        for wid, webhook in app.state.webhooks.items():
            webhooks.append(
                {
                    "webhook_id": wid,
                    "url": webhook.url,
                    "events": webhook.events,
                    "chain_id": webhook.chain_id,
                }
            )
        return {"webhooks": webhooks, "total": len(webhooks)}

    @app.delete("/api/rpa/webhooks/{webhook_id}", response_class=JSONResponse)
    async def delete_webhook(webhook_id: str, _auth: bool = Depends(verify_api_key)):
        """Unregister a webhook."""
        if webhook_id in app.state.webhooks:
            del app.state.webhooks[webhook_id]
            return {"status": "deleted", "webhook_id": webhook_id}
        raise HTTPException(status_code=404, detail="Webhook not found")

    @app.post("/api/rpa/submit", response_class=JSONResponse)
    async def rpa_submit_event(event: RpaEventSubmission, _auth: bool = Depends(verify_api_key)):
        """Submit an event from RPA workflow.

        Allows UiPath to log its own workflow steps to LCTL.
        """
        streaming_event = StreamingEvent(
            id=str(uuid4()),
            type=StreamingEventType.EVENT,
            timestamp=datetime.now(UTC),
            chain_id=event.chain_id,
            payload={
                "type": event.event_type,
                "agent": event.agent,
                "source": "rpa",
                **(event.data or {}),
            },
        )

        app.state.emitter.emit(streaming_event)

        return {
            "status": "submitted",
            "event_id": streaming_event.id,
            "chain_id": event.chain_id,
            "timestamp": streaming_event.timestamp.isoformat(),
        }

    @app.get("/api/rpa/poll/{filename}", response_class=JSONResponse)
    async def rpa_poll(
        filename: str,
        since_seq: int = Query(0, description="Return events after this sequence"),
        _timeout_ms: int = Query(
            0, description="Long-poll timeout (0 = immediate, reserved for future use)"
        ),
        _auth: bool = Depends(verify_api_key),
    ):
        """Poll for new events (alternative to streaming for RPA).

        Supports long-polling with timeout for efficient polling.
        """
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        new_events = [
            {
                "seq": e.seq,
                "type": e.type.value if hasattr(e.type, "value") else e.type,
                "timestamp": e.timestamp.isoformat(),
                "agent": e.agent or "",
            }
            for e in chain.events
            if e.seq > since_seq
        ]

        return {
            "events": new_events,
            "count": len(new_events),
            "last_seq": chain.events[-1].seq if chain.events else 0,
            "has_new": len(new_events) > 0,
        }

    @app.get("/api/rpa/errors/{filename}", response_class=JSONResponse)
    async def rpa_errors(filename: str, _auth: bool = Depends(verify_api_key)):
        """Get all errors from a chain in flat format.

        Quick endpoint to check if a workflow had any issues.
        """
        file_path = _get_secure_path(app, filename)
        engine = _get_cached_engine(app, file_path)
        chain = engine.chain

        errors = []
        for e in chain.events:
            e_type = e.type.value if hasattr(e.type, "value") else e.type
            if e_type == "error":
                errors.append(
                    {
                        "seq": e.seq,
                        "timestamp": e.timestamp.isoformat(),
                        "agent": e.agent or "",
                        "category": e.data.get("category", "unknown") if e.data else "unknown",
                        "error_type": e.data.get("type", "unknown") if e.data else "unknown",
                        "message": e.data.get("message", "") if e.data else "",
                        "recoverable": e.data.get("recoverable", False) if e.data else False,
                    }
                )

        return {
            "errors": errors,
            "count": len(errors),
            "has_errors": len(errors) > 0,
            "chain_id": chain.id,
        }


# =============================================================================
# Application Factory
# =============================================================================


def create_app(working_dir: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        working_dir: Directory to search for .lctl.json files. Defaults to cwd.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="LCTL Dashboard",
        description="Web-based visualization for multi-agent LLM workflows",
        version=__version__,
    )

    # Setup middleware
    _setup_middleware(app)

    # Setup rate limiter
    limiter = _setup_rate_limiter(app)

    # Initialize app state
    _setup_app_state(app, working_dir)

    # Register endpoint groups
    _register_health_endpoints(app)
    _register_core_endpoints(app, limiter)
    _register_analysis_endpoints(app)
    _register_streaming_endpoints(app)
    _register_rpa_endpoints(app, limiter)
    _register_event_endpoints(app)

    # Mount static files
    _setup_static_files(app)

    return app


# =============================================================================
# Server Runner
# =============================================================================


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    working_dir: Path | None = None,
    reload: bool = False,
) -> None:
    """Run the dashboard server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        working_dir: Directory to search for chain files.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    # Create app with working directory
    app = create_app(working_dir)

    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
