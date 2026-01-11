"""LCTL Dashboard - FastAPI web application for chain visualization."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.events import Chain, ReplayEngine
from ..evaluation.metrics import ChainMetrics
from ..streaming.emitter import EventEmitter, StreamingEvent, StreamingEventType


class ReplayRequest(BaseModel):
    """Request model for replay endpoint."""
    filename: str
    target_seq: int


class CompareRequest(BaseModel):
    """Request model for chain comparison endpoint."""
    filename1: str
    filename2: str


def create_app(working_dir: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        working_dir: Directory to search for .lctl.json files. Defaults to cwd.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="LCTL Dashboard",
        description="Web-based visualization for multi-agent LLM workflows",
        version="4.0.0"
    )

    # Store working directory in app state
    app.state.working_dir = working_dir or Path.cwd()
    app.state.engine_cache = {}  # Cache for ReplayEngine instances: (abs_path, mtime) -> engine
    app.state.emitter = EventEmitter(max_history=500)  # Global event emitter for streaming

    # Get paths to static files and templates
    dashboard_dir = Path(__file__).parent
    static_dir = dashboard_dir / "static"
    templates_dir = dashboard_dir / "templates"

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def get_secure_path(filename: str) -> Path:
        """Resolve path and ensure it's within working directory."""
        working_dir = app.state.working_dir.resolve()
        try:
            file_path = (working_dir / filename).resolve()
            if not file_path.is_relative_to(working_dir):
                raise HTTPException(status_code=403, detail="Access denied: Path outside working directory")
            return file_path
        except (ValueError, RuntimeError):
             raise HTTPException(status_code=400, detail="Invalid filename")

    def get_cached_engine(file_path: Path) -> ReplayEngine:
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

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main dashboard HTML."""
        index_path = templates_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="Dashboard template not found")
        return index_path.read_text()

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
                    chains.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "id": chain.id,
                        "version": chain.version,
                        "event_count": len(chain.events)
                    })
                except Exception as e:
                    # Include file but mark as having errors
                    chains.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "id": "error",
                        "version": "unknown",
                        "event_count": 0,
                        "error": str(e)
                    })

        return {"chains": chains, "working_dir": str(working_dir)}

    @app.get("/api/chain/{filename}", response_class=JSONResponse)
    async def get_chain(filename: str):
        """Load and return chain data with analysis."""
        file_path = get_secure_path(filename)
        engine = get_cached_engine(file_path)
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
        [e.to_dict() for e in chain.events
                  if (e.type.value if hasattr(e.type, 'value') else e.type) == "error"]

        return {
            "chain": {
                "id": chain.id,
                "version": chain.version,
                "filename": filename
            },
            "events": events,
            "agents": sorted(agents),
            "state": {
                "facts": state.facts,
                "errors": state.errors,
                "metrics": state.metrics
            },
            "analysis": {
                "bottlenecks": bottlenecks,
                "confidence_timeline": confidence_timeline,
                "trace": trace
            }
        }

    @app.post("/api/replay", response_class=JSONResponse)
    async def replay_chain(request: ReplayRequest):
        """Replay chain to a specific sequence number."""
        file_path = get_secure_path(request.filename)
        engine = get_cached_engine(file_path)
        chain = engine.chain

        if request.target_seq < 1:
            raise HTTPException(status_code=400, detail="target_seq must be >= 1")

        if chain.events and request.target_seq > chain.events[-1].seq:
            raise HTTPException(
                status_code=400,
                detail=f"target_seq {request.target_seq} exceeds max seq {chain.events[-1].seq}"
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
                "current_step": state.current_step
            }
        }

    @app.get("/api/metrics/{filename}", response_class=JSONResponse)
    async def get_metrics(filename: str):
        """Get detailed metrics for a chain including per-agent breakdown."""
        file_path = get_secure_path(filename)
        engine = get_cached_engine(file_path)
        chain = engine.chain

        state = engine.replay_all()
        bottlenecks = engine.find_bottlenecks()

        # use shared ChainMetrics to calculate stats efficiently
        metrics = ChainMetrics.from_chain(chain, state)
        agent_metrics = metrics.agent_stats

        error_timeline = []
        for event in chain.events:
            event_type = event.type.value if hasattr(event.type, 'value') else event.type
            if event_type == "error":
                error_timeline.append({
                    "seq": event.seq,
                    "timestamp": event.timestamp.isoformat(),
                    "agent": event.agent,
                    "category": event.data.get("category", "unknown"),
                    "type": event.data.get("type", "unknown"),
                    "message": event.data.get("message", ""),
                    "recoverable": event.data.get("recoverable", False)
                })

        token_distribution = {
            "input": state.metrics.get("total_tokens_in", 0),
            "output": state.metrics.get("total_tokens_out", 0)
        }

        return {
            "chain": {
                "id": chain.id,
                "filename": filename
            },
            "summary": {
                "total_events": metrics.total_events,
                "total_agents": metrics.agent_count,
                "total_duration_ms": metrics.total_duration_ms,
                "total_tokens": metrics.total_tokens,
                "total_errors": metrics.error_count,
                "total_facts": metrics.fact_count
            },
            "agent_metrics": agent_metrics,
            "token_distribution": token_distribution,
            "error_timeline": error_timeline,
            "bottlenecks": bottlenecks[:10]
        }

    @app.post("/api/compare", response_class=JSONResponse)
    async def compare_chains(request: CompareRequest):
        """Compare two chains and return differences."""
        path1 = get_secure_path(request.filename1)
        path2 = get_secure_path(request.filename2)

        engine1 = get_cached_engine(path1)
        engine2 = get_cached_engine(path2)
        chain1 = engine1.chain
        chain2 = engine2.chain

        state1 = engine1.replay_all()
        state2 = engine2.replay_all()

        diffs = engine1.diff(engine2)

        def get_chain_summary(chain: Chain, state, engine: ReplayEngine) -> Dict[str, Any]:
            bottlenecks = engine.find_bottlenecks()
            return {
                "id": chain.id,
                "event_count": len(chain.events),
                "agent_count": len(set(e.agent for e in chain.events)),
                "agents": sorted(set(e.agent for e in chain.events)),
                "fact_count": len(state.facts),
                "error_count": state.metrics.get("error_count", 0),
                "total_duration_ms": state.metrics.get("total_duration_ms", 0),
                "total_tokens": state.metrics.get("total_tokens_in", 0) + state.metrics.get("total_tokens_out", 0),
                "top_bottleneck": bottlenecks[0] if bottlenecks else None
            }

        fact_comparison = {
            "only_in_first": [],
            "only_in_second": [],
            "different": [],
            "same": []
        }

        all_fact_ids = set(state1.facts.keys()) | set(state2.facts.keys())
        for fact_id in all_fact_ids:
            fact1 = state1.facts.get(fact_id)
            fact2 = state2.facts.get(fact_id)

            if fact1 and not fact2:
                fact_comparison["only_in_first"].append({"id": fact_id, **fact1})
            elif fact2 and not fact1:
                fact_comparison["only_in_second"].append({"id": fact_id, **fact2})
            elif fact1 and fact2:
                if fact1.get("text") != fact2.get("text") or fact1.get("confidence") != fact2.get("confidence"):
                    fact_comparison["different"].append({
                        "id": fact_id,
                        "first": fact1,
                        "second": fact2
                    })
                else:
                    fact_comparison["same"].append(fact_id)

        return {
            "chain1": get_chain_summary(chain1, state1, engine1),
            "chain2": get_chain_summary(chain2, state2, engine2),
            "event_diffs": diffs[:50],
            "diff_count": len(diffs),
            "fact_comparison": fact_comparison,
            "divergence_point": diffs[0]["seq"] if diffs else None
        }

    @app.get("/api/evaluation/{filename}", response_class=JSONResponse)
    async def get_evaluation(filename: str):
        """Get comprehensive evaluation report for a chain."""
        file_path = get_secure_path(filename)
        engine = get_cached_engine(file_path)
        chain = engine.chain
        state = engine.replay_all()
        bottlenecks = engine.find_bottlenecks()
        confidence_timeline = engine.get_confidence_timeline()

        total_duration = state.metrics.get("total_duration_ms", 0)
        total_tokens = state.metrics.get("total_tokens_in", 0) + state.metrics.get("total_tokens_out", 0)
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
                fact_heatmap.append({
                    "fact_id": fact_id,
                    "seq": point["seq"],
                    "confidence": point["confidence"],
                    "agent": point["agent"]
                })
        fact_heatmap.sort(key=lambda x: x["seq"])

        issues = []
        warnings = []

        if error_count > 0:
            issues.append({
                "type": "errors",
                "severity": "high" if error_count > 3 else "medium",
                "message": f"{error_count} error(s) occurred during execution",
                "details": state.errors[:5]
            })

        if bottlenecks and bottlenecks[0].get("percentage", 0) > 50:
            warnings.append({
                "type": "bottleneck",
                "severity": "medium",
                "message": f"Agent '{bottlenecks[0]['agent']}' consumed {bottlenecks[0]['percentage']:.0f}% of total time",
                "details": bottlenecks[0]
            })

        if low_confidence_facts:
            warnings.append({
                "type": "low_confidence",
                "severity": "low",
                "message": f"{len(low_confidence_facts)} fact(s) have confidence below 70%",
                "details": low_confidence_facts[:5]
            })

        if total_tokens > 100000:
            warnings.append({
                "type": "high_token_usage",
                "severity": "medium",
                "message": f"High token consumption: {total_tokens:,} tokens",
                "details": {"total": total_tokens, "input": state.metrics.get("total_tokens_in", 0), "output": state.metrics.get("total_tokens_out", 0)}
            })

        overall_score = int((perf_score * 0.3 + reliability_score * 0.4 + quality_score * 0.3))

        return {
            "chain": {
                "id": chain.id,
                "filename": filename,
                "version": chain.version
            },
            "scores": {
                "overall": overall_score,
                "performance": perf_score,
                "reliability": reliability_score,
                "quality": quality_score
            },
            "metrics": {
                "total_events": len(chain.events),
                "total_duration_ms": total_duration,
                "total_tokens": total_tokens,
                "tokens_in": state.metrics.get("total_tokens_in", 0),
                "tokens_out": state.metrics.get("total_tokens_out", 0),
                "error_count": error_count,
                "fact_count": fact_count,
                "average_confidence": round(avg_confidence, 3)
            },
            "issues": issues,
            "warnings": warnings,
            "bottlenecks": bottlenecks[:5],
            "low_confidence_facts": low_confidence_facts,
            "fact_confidence_heatmap": fact_heatmap,
            "confidence_timeline": confidence_timeline
        }

    @app.get("/api/health", response_class=JSONResponse)
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "version": "4.0.0", "streaming": True}

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
        filters: Dict[str, Any] = {}
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        welcome_msg = {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await websocket.send_text(json.dumps(welcome_msg))

        emitter = app.state.emitter

        def on_event(event: StreamingEvent) -> None:
            if _should_send_to_client(event, filters):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

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
                            pong = {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}
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
                    except asyncio.TimeoutError:
                        heartbeat = {"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()}
                        await websocket.send_text(json.dumps(heartbeat))
            except Exception:
                pass

        try:
            receive_task = asyncio.create_task(receive_loop())
            send_task = asyncio.create_task(send_loop())

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
            emitter.off("all", on_event)

    @app.get("/api/events/stream")
    async def sse_endpoint(
        chain_id: Optional[str] = Query(None, description="Filter by chain ID"),
        event_types: Optional[str] = Query(None, description="Comma-separated event types"),
        include_history: bool = Query(False, description="Include historical events")
    ):
        """Server-Sent Events endpoint for real-time event streaming.

        Alternative to WebSocket for HTTP-based streaming.
        """
        filters: Dict[str, Any] = {}
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
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass

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
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"

            finally:
                emitter.off("all", on_event)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

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
            timestamp=datetime.now(timezone.utc),
            chain_id=chain_id,
            payload={"type": event_type, **payload}
        )

        app.state.emitter.emit(streaming_event)

        return {"status": "emitted", "event_id": streaming_event.id}

    @app.get("/api/streaming/status", response_class=JSONResponse)
    async def streaming_status():
        """Get streaming status including connected clients and event count."""
        emitter = app.state.emitter
        return {
            "enabled": True,
            "chain_id": emitter.chain_id,
            "event_count": emitter.event_count,
            "handler_count": emitter.handler_count(),
            "history_size": len(emitter.history)
        }

    return app


def _should_send_to_client(event: StreamingEvent, filters: Dict[str, Any]) -> bool:
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


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8080,
    working_dir: Optional[Path] = None,
    reload: bool = False
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

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
