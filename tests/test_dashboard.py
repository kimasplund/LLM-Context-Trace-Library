"""Tests for LCTL Dashboard."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from lctl.core.events import Chain, Event, EventType

# Skip tests if FastAPI is not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lctl.dashboard.app import create_app


@pytest.fixture
def sample_chain_data():
    """Create sample chain data for testing."""
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    return {
        "lctl": "4.0",
        "chain": {"id": "test-dashboard-chain"},
        "events": [
            {
                "seq": 1,
                "type": "step_start",
                "timestamp": base_time.isoformat(),
                "agent": "agent-a",
                "data": {"intent": "analyze", "input_summary": "test input"}
            },
            {
                "seq": 2,
                "type": "fact_added",
                "timestamp": (base_time + timedelta(seconds=1)).isoformat(),
                "agent": "agent-a",
                "data": {"id": "F1", "text": "Test fact", "confidence": 0.85}
            },
            {
                "seq": 3,
                "type": "tool_call",
                "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
                "agent": "agent-a",
                "data": {"tool": "search", "input": "query", "output": "results", "duration_ms": 100}
            },
            {
                "seq": 4,
                "type": "step_end",
                "timestamp": (base_time + timedelta(seconds=3)).isoformat(),
                "agent": "agent-a",
                "data": {
                    "outcome": "success",
                    "output_summary": "analysis complete",
                    "duration_ms": 3000,
                    "tokens": {"input": 100, "output": 50}
                }
            },
            {
                "seq": 5,
                "type": "step_start",
                "timestamp": (base_time + timedelta(seconds=4)).isoformat(),
                "agent": "agent-b",
                "data": {"intent": "review", "input_summary": "F1"}
            },
            {
                "seq": 6,
                "type": "error",
                "timestamp": (base_time + timedelta(seconds=5)).isoformat(),
                "agent": "agent-b",
                "data": {
                    "category": "validation",
                    "type": "ValidationError",
                    "message": "Test error",
                    "recoverable": True
                }
            },
            {
                "seq": 7,
                "type": "step_end",
                "timestamp": (base_time + timedelta(seconds=6)).isoformat(),
                "agent": "agent-b",
                "data": {
                    "outcome": "error",
                    "output_summary": "failed",
                    "duration_ms": 2000,
                    "tokens": {"input": 50, "output": 25}
                }
            }
        ]
    }


@pytest.fixture
def temp_chain_dir(sample_chain_data, tmp_path):
    """Create a temporary directory with chain files."""
    # Create main chain file
    chain_file = tmp_path / "test.lctl.json"
    chain_file.write_text(json.dumps(sample_chain_data, indent=2))

    # Create a second chain
    second_chain = sample_chain_data.copy()
    second_chain["chain"] = {"id": "second-chain"}
    second_file = tmp_path / "second.lctl.json"
    second_file.write_text(json.dumps(second_chain, indent=2))

    return tmp_path


@pytest.fixture
def client(temp_chain_dir):
    """Create a test client for the dashboard app."""
    app = create_app(working_dir=temp_chain_dir)
    return TestClient(app)


class TestDashboardApp:
    """Tests for dashboard FastAPI application."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "4.0.0"

    def test_list_chains(self, client):
        """Test listing available chains."""
        response = client.get("/api/chains")
        assert response.status_code == 200
        data = response.json()

        assert "chains" in data
        assert "working_dir" in data
        assert len(data["chains"]) == 2

        # Check chain details
        filenames = [c["filename"] for c in data["chains"]]
        assert "test.lctl.json" in filenames
        assert "second.lctl.json" in filenames

        # Check that chain info is included
        test_chain = next(c for c in data["chains"] if c["filename"] == "test.lctl.json")
        assert test_chain["id"] == "test-dashboard-chain"
        assert test_chain["event_count"] == 7

    def test_get_chain(self, client):
        """Test loading a specific chain."""
        response = client.get("/api/chain/test.lctl.json")
        assert response.status_code == 200
        data = response.json()

        # Check chain info
        assert data["chain"]["id"] == "test-dashboard-chain"
        assert data["chain"]["version"] == "4.0"
        assert data["chain"]["filename"] == "test.lctl.json"

        # Check events
        assert len(data["events"]) == 7

        # Check agents
        assert sorted(data["agents"]) == ["agent-a", "agent-b"]

        # Check state
        assert "facts" in data["state"]
        assert "F1" in data["state"]["facts"]
        assert data["state"]["facts"]["F1"]["confidence"] == 0.85

        # Check metrics
        assert data["state"]["metrics"]["total_duration_ms"] == 5100  # 3000 + 100 + 2000
        assert data["state"]["metrics"]["error_count"] == 1

        # Check analysis
        assert "bottlenecks" in data["analysis"]
        assert "confidence_timeline" in data["analysis"]
        assert "trace" in data["analysis"]

    def test_get_chain_not_found(self, client):
        """Test loading a non-existent chain."""
        response = client.get("/api/chain/nonexistent.lctl.json")
        assert response.status_code == 404

    def test_get_chain_path_traversal(self, client):
        """Test that path traversal is blocked."""
        # Test with explicit path traversal characters
        response = client.get("/api/chain/..%2F..%2Fetc%2Fpasswd")
        # Should be blocked - either 400 (invalid) or 404 (not found after sanitization)
        assert response.status_code in (400, 404)

        # Test with absolute path
        response = client.get("/api/chain/%2Fetc%2Fpasswd")
        assert response.status_code in (400, 404)

    def test_replay_chain(self, client):
        """Test replaying to a specific sequence."""
        response = client.post("/api/replay", json={
            "filename": "test.lctl.json",
            "target_seq": 4
        })
        assert response.status_code == 200
        data = response.json()

        assert data["target_seq"] == 4
        assert len(data["events"]) == 4

        # State at seq 4 should have 1 fact
        assert "F1" in data["state"]["facts"]

        # Should not have errors yet (error is at seq 6)
        assert data["state"]["metrics"]["error_count"] == 0

    def test_replay_chain_to_seq_2(self, client):
        """Test replaying to seq 2 to verify fact is present."""
        response = client.post("/api/replay", json={
            "filename": "test.lctl.json",
            "target_seq": 2
        })
        assert response.status_code == 200
        data = response.json()

        assert data["target_seq"] == 2
        assert len(data["events"]) == 2
        assert "F1" in data["state"]["facts"]

    def test_replay_invalid_seq(self, client):
        """Test replaying with invalid sequence number."""
        # Too high
        response = client.post("/api/replay", json={
            "filename": "test.lctl.json",
            "target_seq": 100
        })
        assert response.status_code == 400

        # Too low
        response = client.post("/api/replay", json={
            "filename": "test.lctl.json",
            "target_seq": 0
        })
        assert response.status_code == 400

    def test_replay_chain_not_found(self, client):
        """Test replaying a non-existent chain."""
        response = client.post("/api/replay", json={
            "filename": "nonexistent.lctl.json",
            "target_seq": 1
        })
        assert response.status_code == 404

    def test_index_page(self, client):
        """Test that index page is served."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "LCTL Dashboard" in response.text

    def test_static_files(self, client):
        """Test that static files are served."""
        # CSS
        response = client.get("/static/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

        # JS
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_websocket_connect(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert "client_id" in data

    def test_websocket_stream_events(self, temp_chain_dir, client):
        """Test streaming events via WebSocket."""
        # Note: This test simulates the server pushing events, which requires triggering
        # an event in the same process or mocking the manager.
        # Here we just verify connection and basic protocol.
        
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome message
            websocket.receive_json() 
            
            # Send a subscribe message
            websocket.send_json({"type": "subscribe", "filters": {"chain_id": "test-chain"}})
            # Expect confirmation or no error
            pass

    def test_get_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/metrics/test.lctl.json")
        assert response.status_code == 200
        data = response.json()
        
        # Check specific metrics fields
        assert "summary" in data
        assert data["summary"]["total_events"] == 7
        assert data["summary"]["total_errors"] == 1
        assert "agent_metrics" in data
        assert "agent-a" in data["agent_metrics"]
        
        # Verify Token Distribution (from refactor)
        assert "token_distribution" in data
        assert data["token_distribution"]["input"] == 150
        
    def test_get_evaluation(self, client):
        """Test evaluation endpoint."""
        response = client.get("/api/evaluation/test.lctl.json")
        assert response.status_code == 200
        data = response.json()
        
        assert "scores" in data
        assert "overall" in data["scores"]
        assert "issues" in data
        # Check that error was captured in issues
        assert any(i["type"] == "errors" for i in data["issues"])

    def test_compare_chains(self, client):
        """Test chain comparison endpoint."""
        response = client.post("/api/compare", json={
            "filename1": "test.lctl.json",
            "filename2": "second.lctl.json"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert "chain1" in data
        assert "chain2" in data
        assert "event_diffs" in data
        # Chains are identical except ID, so diffs might be minimal or just metadata
        # Or diff engine finds them same events?
        # Actually second chain was a copy, so events are same.
        assert len(data["event_diffs"]) == 0 or data["diff_count"] == 0


class TestDashboardAnalysis:
    """Tests for dashboard analysis features."""

    def test_bottleneck_detection(self, client):
        """Test that bottlenecks are correctly identified."""
        response = client.get("/api/chain/test.lctl.json")
        assert response.status_code == 200
        data = response.json()

        bottlenecks = data["analysis"]["bottlenecks"]
        assert len(bottlenecks) > 0

        # First bottleneck should be the longest step
        assert bottlenecks[0]["duration_ms"] >= bottlenecks[-1]["duration_ms"]

    def test_confidence_timeline(self, client):
        """Test that confidence timeline is generated."""
        response = client.get("/api/chain/test.lctl.json")
        assert response.status_code == 200
        data = response.json()

        timeline = data["analysis"]["confidence_timeline"]
        assert "F1" in timeline
        assert len(timeline["F1"]) >= 1
        assert timeline["F1"][0]["confidence"] == 0.85

    def test_trace_generation(self, client):
        """Test that execution trace is generated."""
        response = client.get("/api/chain/test.lctl.json")
        assert response.status_code == 200
        data = response.json()

        trace = data["analysis"]["trace"]
        # Should have step_start and step_end events
        step_types = [t["type"] for t in trace]
        assert "step_start" in step_types
        assert "step_end" in step_types


class TestDashboardEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_chain(self, tmp_path):
        """Test handling of empty chain."""
        empty_chain = {"lctl": "4.0", "chain": {"id": "empty"}, "events": []}
        chain_file = tmp_path / "empty.lctl.json"
        chain_file.write_text(json.dumps(empty_chain))

        app = create_app(working_dir=tmp_path)
        client = TestClient(app)

        response = client.get("/api/chain/empty.lctl.json")
        assert response.status_code == 200
        data = response.json()

        assert data["chain"]["id"] == "empty"
        assert len(data["events"]) == 0
        assert len(data["agents"]) == 0

    def test_invalid_json_chain(self, tmp_path):
        """Test handling of invalid JSON."""
        invalid_file = tmp_path / "invalid.lctl.json"
        invalid_file.write_text("not valid json")

        app = create_app(working_dir=tmp_path)
        client = TestClient(app)

        response = client.get("/api/chain/invalid.lctl.json")
        assert response.status_code == 400

    def test_chain_with_all_event_types(self, tmp_path):
        """Test chain with all event types."""
        base_time = datetime(2025, 1, 15, 10, 0, 0)
        chain_data = {
            "lctl": "4.0",
            "chain": {"id": "all-types"},
            "events": [
                {"seq": 1, "type": "step_start", "timestamp": base_time.isoformat(),
                 "agent": "a", "data": {"intent": "test"}},
                {"seq": 2, "type": "fact_added", "timestamp": base_time.isoformat(),
                 "agent": "a", "data": {"id": "F1", "text": "fact", "confidence": 0.9}},
                {"seq": 3, "type": "fact_modified", "timestamp": base_time.isoformat(),
                 "agent": "a", "data": {"id": "F1", "confidence": 0.95, "reason": "verified"}},
                {"seq": 4, "type": "tool_call", "timestamp": base_time.isoformat(),
                 "agent": "a", "data": {"tool": "test", "duration_ms": 50}},
                {"seq": 5, "type": "checkpoint", "timestamp": base_time.isoformat(),
                 "agent": "system", "data": {"state_hash": "abc123"}},
                {"seq": 6, "type": "error", "timestamp": base_time.isoformat(),
                 "agent": "a", "data": {"category": "test", "type": "TestError", "message": "test"}},
                {"seq": 7, "type": "step_end", "timestamp": base_time.isoformat(),
                 "agent": "a", "data": {"outcome": "error", "duration_ms": 1000}}
            ]
        }

        chain_file = tmp_path / "all-types.lctl.json"
        chain_file.write_text(json.dumps(chain_data))

        app = create_app(working_dir=tmp_path)
        client = TestClient(app)

        response = client.get("/api/chain/all-types.lctl.json")
        assert response.status_code == 200
        data = response.json()

        assert len(data["events"]) == 7
        event_types = [e["type"] for e in data["events"]]
        assert "step_start" in event_types
        assert "fact_added" in event_types
        assert "fact_modified" in event_types
        assert "tool_call" in event_types
        assert "checkpoint" in event_types
        assert "error" in event_types
        assert "step_end" in event_types

    def test_no_chains_in_directory(self, tmp_path):
        """Test when no chain files exist."""
        app = create_app(working_dir=tmp_path)
        client = TestClient(app)

        response = client.get("/api/chains")
        assert response.status_code == 200
        data = response.json()

        assert data["chains"] == []


class TestCreateApp:
    """Tests for app creation."""

    def test_create_app_default_dir(self):
        """Test creating app with default directory."""
        app = create_app()
        assert app is not None
        assert hasattr(app.state, "working_dir")

    def test_create_app_custom_dir(self, tmp_path):
        """Test creating app with custom directory."""
        app = create_app(working_dir=tmp_path)
        assert app.state.working_dir == tmp_path
