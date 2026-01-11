"""LCTL Dashboard - FastAPI web application for chain visualization."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.events import Chain, ReplayEngine


class ReplayRequest(BaseModel):
    """Request model for replay endpoint."""
    filename: str
    target_seq: int


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

    # Get paths to static files and templates
    dashboard_dir = Path(__file__).parent
    static_dir = dashboard_dir / "static"
    templates_dir = dashboard_dir / "templates"

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
        working_dir = app.state.working_dir

        # Security: prevent path traversal
        if ".." in filename or filename.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = working_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Chain file not found: {filename}")

        try:
            chain = Chain.load(file_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid chain file: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading chain: {e}")

        engine = ReplayEngine(chain)

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
        errors = [e.to_dict() for e in chain.events
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
        working_dir = app.state.working_dir

        # Security: prevent path traversal
        if ".." in request.filename or request.filename.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = working_dir / request.filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Chain file not found: {request.filename}")

        try:
            chain = Chain.load(file_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid chain file: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading chain: {e}")

        if request.target_seq < 1:
            raise HTTPException(status_code=400, detail="target_seq must be >= 1")

        if chain.events and request.target_seq > chain.events[-1].seq:
            raise HTTPException(
                status_code=400,
                detail=f"target_seq {request.target_seq} exceeds max seq {chain.events[-1].seq}"
            )

        engine = ReplayEngine(chain)
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

    @app.get("/api/health", response_class=JSONResponse)
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "version": "4.0.0"}

    return app


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
