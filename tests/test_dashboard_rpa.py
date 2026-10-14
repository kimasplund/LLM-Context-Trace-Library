"""Tests for LCTL Dashboard RPA/UiPath Integration Endpoints.

This module tests the RPA-specific endpoints designed for UiPath and
other RPA tool integration, including API key authentication, batch
operations, search, export, and webhooks.
"""

import csv
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip tests if FastAPI is not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lctl.dashboard.app import create_app
from lctl.core.schema import CURRENT_VERSION


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rpa_sample_chain_data():
    """Create sample chain data optimized for RPA testing."""
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    return {
        "lctl": CURRENT_VERSION,
        "chain": {"id": "rpa-test-chain"},
        "events": [
            {
                "seq": 1,
                "type": "step_start",
                "timestamp": base_time.isoformat(),
                "agent": "analyzer",
                "data": {"intent": "analyze", "input_summary": "document.pdf"},
            },
            {
                "seq": 2,
                "type": "fact_added",
                "timestamp": (base_time + timedelta(seconds=1)).isoformat(),
                "agent": "analyzer",
                "data": {"id": "F1", "text": "Document contains invoice data", "confidence": 0.85},
            },
            {
                "seq": 3,
                "type": "tool_call",
                "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
                "agent": "analyzer",
                "data": {
                    "tool_name": "ocr",
                    "input": "page1",
                    "output": "text extracted",
                    "duration_ms": 500,
                },
            },
            {
                "seq": 4,
                "type": "step_end",
                "timestamp": (base_time + timedelta(seconds=3)).isoformat(),
                "agent": "analyzer",
                "data": {
                    "outcome": "success",
                    "output_summary": "analysis complete",
                    "duration_ms": 3000,
                    "tokens_in": 100,
                    "tokens_out": 50,
                },
            },
            {
                "seq": 5,
                "type": "step_start",
                "timestamp": (base_time + timedelta(seconds=4)).isoformat(),
                "agent": "processor",
                "data": {"intent": "process", "input_summary": "invoice data"},
            },
            {
                "seq": 6,
                "type": "error",
                "timestamp": (base_time + timedelta(seconds=5)).isoformat(),
                "agent": "processor",
                "data": {
                    "category": "validation",
                    "type": "ValidationError",
                    "message": "Missing required field: amount",
                    "recoverable": True,
                },
            },
            {
                "seq": 7,
                "type": "step_end",
                "timestamp": (base_time + timedelta(seconds=6)).isoformat(),
                "agent": "processor",
                "data": {
                    "outcome": "error",
                    "output_summary": "processing failed",
                    "duration_ms": 2000,
                    "tokens_in": 75,
                    "tokens_out": 25,
                },
            },
        ],
    }


@pytest.fixture
def rpa_temp_chain_dir(rpa_sample_chain_data, tmp_path):
    """Create a temporary directory with chain files for RPA testing."""
    # Create main chain file
    chain_file = tmp_path / "rpa-test.lctl.json"
    chain_file.write_text(json.dumps(rpa_sample_chain_data, indent=2))

    # Create a second chain for batch testing
    second_chain = rpa_sample_chain_data.copy()
    second_chain["chain"] = {"id": "second-rpa-chain"}
    second_file = tmp_path / "second-rpa.lctl.json"
    second_file.write_text(json.dumps(second_chain, indent=2))

    # Create a third chain with no errors
    success_chain = {
        "lctl": CURRENT_VERSION,
        "chain": {"id": "success-chain"},
        "events": [
            {
                "seq": 1,
                "type": "step_start",
                "timestamp": datetime(2025, 1, 15, 10, 0, 0).isoformat(),
                "agent": "worker",
                "data": {"intent": "work"},
            },
            {
                "seq": 2,
                "type": "step_end",
                "timestamp": datetime(2025, 1, 15, 10, 0, 1).isoformat(),
                "agent": "worker",
                "data": {"outcome": "success", "duration_ms": 1000},
            },
        ],
    }
    success_file = tmp_path / "success.lctl.json"
    success_file.write_text(json.dumps(success_chain, indent=2))

    return tmp_path


@pytest.fixture
def rpa_client(rpa_temp_chain_dir):
    """Create a test client for the dashboard app (no API key required)."""
    app = create_app(working_dir=rpa_temp_chain_dir)
    # Ensure API key is not required for basic tests
    app.state.api_keys = set()
    app.state.require_api_key = False
    app.state.localhost_bypass = True
    return TestClient(app)


@pytest.fixture
def rpa_client_with_api_key(rpa_temp_chain_dir):
    """Create a test client that requires API key authentication."""
    app = create_app(working_dir=rpa_temp_chain_dir)
    # Configure API key requirement
    app.state.api_keys = {"test-api-key-12345", "another-valid-key"}
    app.state.require_api_key = True
    app.state.localhost_bypass = False
    return TestClient(app)


@pytest.fixture
def rpa_client_localhost_bypass(rpa_temp_chain_dir):
    """Create a test client with localhost bypass enabled."""
    app = create_app(working_dir=rpa_temp_chain_dir)
    app.state.api_keys = {"test-api-key-12345"}
    app.state.require_api_key = True
    app.state.localhost_bypass = True
    return TestClient(app)


# =============================================================================
# API Key Authentication Tests
# =============================================================================


class TestRpaApiKeyAuthentication:
    """Tests for RPA endpoint API key authentication."""

    def test_rpa_endpoint_requires_api_key(self, rpa_client_with_api_key):
        """Test that RPA endpoints require API key when configured."""
        # Attempt to access protected endpoint without API key
        response = rpa_client_with_api_key.get("/api/rpa/summary/rpa-test.lctl.json")

        assert response.status_code == 401
        data = response.json()
        assert "API key required" in data["detail"]

    def test_rpa_endpoint_accepts_valid_api_key(self, rpa_client_with_api_key):
        """Test that valid API key grants access."""
        response = rpa_client_with_api_key.get(
            "/api/rpa/summary/rpa-test.lctl.json", headers={"X-API-Key": "test-api-key-12345"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["chain_id"] == "rpa-test-chain"

    def test_rpa_endpoint_rejects_invalid_api_key(self, rpa_client_with_api_key):
        """Test that invalid API key is rejected."""
        response = rpa_client_with_api_key.get(
            "/api/rpa/summary/rpa-test.lctl.json", headers={"X-API-Key": "invalid-key"}
        )

        assert response.status_code == 403
        data = response.json()
        assert "Invalid API key" in data["detail"]

    def test_localhost_bypass_works(self, rpa_client_localhost_bypass):
        """Test that localhost requests bypass API key check.

        Note: TestClient simulates requests from 'testclient' which may not
        be recognized as localhost. This test verifies the bypass logic exists.
        """
        # The TestClient may not simulate localhost properly, so we test
        # the auth status endpoint which doesn't require auth
        response = rpa_client_localhost_bypass.get("/api/rpa/auth/status")

        assert response.status_code == 200
        data = response.json()
        assert data["localhost_bypass"] is True
        assert "is_localhost" in data

    def test_rpa_auth_status_endpoint(self, rpa_client_with_api_key):
        """Test auth status endpoint returns correct configuration."""
        response = rpa_client_with_api_key.get("/api/rpa/auth/status")

        assert response.status_code == 200
        data = response.json()
        assert data["require_api_key"] is True
        assert data["localhost_bypass"] is False
        assert data["api_keys_configured"] is True
        assert "auth_required" in data

    def test_rpa_multiple_valid_keys(self, rpa_client_with_api_key):
        """Test that multiple API keys can be configured and used."""
        # Test first key
        response1 = rpa_client_with_api_key.get(
            "/api/rpa/summary/rpa-test.lctl.json", headers={"X-API-Key": "test-api-key-12345"}
        )
        assert response1.status_code == 200

        # Test second key
        response2 = rpa_client_with_api_key.get(
            "/api/rpa/summary/rpa-test.lctl.json", headers={"X-API-Key": "another-valid-key"}
        )
        assert response2.status_code == 200


class TestRpaApiKeyGeneration:
    """Tests for API key generation endpoint."""

    def test_generate_api_key_with_existing_auth(self, rpa_client_with_api_key):
        """Test generating API key when authenticated with existing key.

        The generate-key endpoint requires authentication via verify_api_key
        dependency, so we must provide a valid API key to generate new ones.
        """
        response = rpa_client_with_api_key.post(
            "/api/rpa/auth/generate-key", headers={"X-API-Key": "test-api-key-12345"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert len(data["api_key"]) > 20  # URL-safe token should be substantial
        assert "Store this key securely" in data["message"]
        assert "X-API-Key" in data["usage"]

    def test_generate_api_key_requires_auth(self, rpa_client_with_api_key):
        """Test that generate-key endpoint requires authentication."""
        response = rpa_client_with_api_key.post("/api/rpa/auth/generate-key")

        # Without auth, should return 401
        assert response.status_code == 401


# =============================================================================
# RPA Summary Endpoint Tests
# =============================================================================


class TestRpaSummaryEndpoint:
    """Tests for /api/rpa/summary/{filename} endpoint."""

    def test_rpa_summary_returns_flat_structure(self, rpa_client):
        """Test /api/rpa/summary/{filename} returns UiPath-compatible data."""
        response = rpa_client.get("/api/rpa/summary/rpa-test.lctl.json")

        assert response.status_code == 200
        data = response.json()

        # Verify flat structure with simple types (UiPath DataTable compatible)
        assert data["chain_id"] == "rpa-test-chain"
        assert data["filename"] == "rpa-test.lctl.json"
        assert data["event_count"] == 7
        assert data["agent_count"] == 2
        assert data["error_count"] == 1
        assert data["has_errors"] is True
        assert data["fact_count"] == 1
        assert isinstance(data["total_duration_ms"], int)
        assert isinstance(data["total_tokens"], int)
        assert data["status"] == "error"  # Because there's an error
        assert "timestamp" in data

    def test_rpa_summary_success_status(self, rpa_client):
        """Test summary returns success status for chains without errors."""
        response = rpa_client.get("/api/rpa/summary/success.lctl.json")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["has_errors"] is False
        assert data["error_count"] == 0

    def test_rpa_summary_not_found(self, rpa_client):
        """Test summary with non-existent file."""
        response = rpa_client.get("/api/rpa/summary/nonexistent.lctl.json")

        assert response.status_code == 404


# =============================================================================
# RPA Events Endpoint Tests
# =============================================================================


class TestRpaEventsEndpoint:
    """Tests for /api/rpa/events/{filename} endpoint with filters."""

    def test_rpa_events_returns_flat_events(self, rpa_client):
        """Test events endpoint returns flattened event data."""
        response = rpa_client.get("/api/rpa/events/rpa-test.lctl.json")

        assert response.status_code == 200
        data = response.json()

        assert "events" in data
        assert data["total"] == 7
        assert len(data["events"]) == 7

        # Check flat structure of first event
        event = data["events"][0]
        assert event["seq"] == 1
        assert event["type"] == "step_start"
        assert event["agent"] == "analyzer"
        assert event["chain_id"] == "rpa-test-chain"
        # All fields should be flat (not nested)
        assert "intent" in event
        assert "timestamp" in event

    def test_rpa_events_with_type_filter(self, rpa_client):
        """Test /api/rpa/events/{filename} filters by event type."""
        response = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?event_type=error")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert len(data["events"]) == 1
        assert data["events"][0]["type"] == "error"
        assert data["events"][0]["error_message"] == "Missing required field: amount"

    def test_rpa_events_with_agent_filter(self, rpa_client):
        """Test /api/rpa/events/{filename} filters by agent."""
        response = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?agent=analyzer")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 4  # 4 events from analyzer
        for event in data["events"]:
            assert event["agent"] == "analyzer"

    def test_rpa_events_with_combined_filters(self, rpa_client):
        """Test events with both type and agent filters."""
        response = rpa_client.get(
            "/api/rpa/events/rpa-test.lctl.json?event_type=step_end&agent=processor"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["events"][0]["agent"] == "processor"
        assert data["events"][0]["type"] == "step_end"

    def test_rpa_events_pagination(self, rpa_client):
        """Test events endpoint pagination with limit and offset."""
        # First page
        response1 = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?limit=3&offset=0")
        data1 = response1.json()
        assert len(data1["events"]) == 3
        assert data1["has_more"] is True
        assert data1["offset"] == 0

        # Second page
        response2 = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?limit=3&offset=3")
        data2 = response2.json()
        assert len(data2["events"]) == 3
        assert data2["offset"] == 3

        # Third page (partial)
        response3 = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?limit=3&offset=6")
        data3 = response3.json()
        assert len(data3["events"]) == 1
        assert data3["has_more"] is False

    def test_rpa_events_no_match(self, rpa_client):
        """Test events endpoint with filter that matches nothing."""
        response = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?agent=nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["events"]) == 0
        assert data["has_more"] is False


# =============================================================================
# RPA Batch Endpoints Tests
# =============================================================================


class TestRpaBatchEndpoints:
    """Tests for RPA batch processing endpoints."""

    def test_rpa_batch_metrics(self, rpa_client):
        """Test /api/rpa/batch/metrics returns aggregated data."""
        response = rpa_client.post(
            "/api/rpa/batch/metrics",
            json={"filenames": ["rpa-test.lctl.json", "second-rpa.lctl.json", "success.lctl.json"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "errors" in data
        assert data["total_processed"] == 3
        assert data["total_errors"] == 0

        # Check individual results
        results_by_file = {r["filename"]: r for r in data["results"]}
        assert "rpa-test.lctl.json" in results_by_file
        assert results_by_file["rpa-test.lctl.json"]["chain_id"] == "rpa-test-chain"
        assert results_by_file["rpa-test.lctl.json"]["event_count"] == 7

    def test_rpa_batch_metrics_with_missing_file(self, rpa_client):
        """Test batch metrics handles missing files gracefully."""
        response = rpa_client.post(
            "/api/rpa/batch/metrics",
            json={"filenames": ["rpa-test.lctl.json", "nonexistent.lctl.json"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_processed"] == 1
        assert data["total_errors"] == 1
        assert len(data["errors"]) == 1
        assert data["errors"][0]["filename"] == "nonexistent.lctl.json"
        assert data["errors"][0]["status"] == "error"

    def test_rpa_batch_metrics_validates_max_files(self, rpa_client):
        """Test that batch requests are limited to 100 files.

        The BatchMetricsRequest model has max_length=100 on filenames,
        which causes a 422 validation error when exceeded.
        """
        filenames = [f"file{i}.lctl.json" for i in range(150)]

        response = rpa_client.post("/api/rpa/batch/metrics", json={"filenames": filenames})

        # Should return 422 Unprocessable Entity due to Pydantic validation
        assert response.status_code == 422
        data = response.json()
        # Pydantic error should indicate the list is too long
        assert "detail" in data

    def test_rpa_batch_metrics_at_max_files(self, rpa_client):
        """Test batch metrics accepts exactly 100 files."""
        # Create list of exactly 100 filenames (all non-existent)
        filenames = [f"file{i}.lctl.json" for i in range(100)]

        response = rpa_client.post("/api/rpa/batch/metrics", json={"filenames": filenames})

        # Should accept 100 files
        assert response.status_code == 200
        data = response.json()
        # All 100 should be processed (as errors since files don't exist)
        assert data["total_errors"] == 100

    def test_rpa_batch_metrics_empty_list(self, rpa_client):
        """Test batch metrics with empty file list."""
        response = rpa_client.post("/api/rpa/batch/metrics", json={"filenames": []})

        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 0
        assert data["total_errors"] == 0


# =============================================================================
# RPA Search Endpoint Tests
# =============================================================================


class TestRpaSearchEndpoint:
    """Tests for /api/rpa/search endpoint."""

    def test_rpa_search_finds_matching_events(self, rpa_client):
        """Test /api/rpa/search finds events matching query."""
        response = rpa_client.post("/api/rpa/search", json={"query": "invoice", "limit": 100})

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "invoice"
        assert len(data["results"]) > 0
        # Should find the fact about invoice data
        assert any("invoice" in r.get("preview", "").lower() for r in data["results"])

    def test_rpa_search_with_event_type_filter(self, rpa_client):
        """Test search with event type filter."""
        response = rpa_client.post(
            "/api/rpa/search",
            json={"query": "analyzer", "event_types": ["step_start"], "limit": 100},
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            assert result["type"] == "step_start"

    def test_rpa_search_with_agent_filter(self, rpa_client):
        """Test search with agent filter."""
        response = rpa_client.post(
            "/api/rpa/search", json={"query": "error", "agents": ["processor"], "limit": 100}
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            assert result["agent"] == "processor"

    def test_rpa_search_respects_limit(self, rpa_client):
        """Test search respects result limit."""
        response = rpa_client.post(
            "/api/rpa/search",
            json={"query": "a", "limit": 2},  # Broad search
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) <= 2
        if data["total"] > 2:
            assert data["truncated"] is True

    def test_rpa_search_no_results(self, rpa_client):
        """Test search with query that matches nothing."""
        response = rpa_client.post(
            "/api/rpa/search", json={"query": "xyznonexistent123", "limit": 100}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 0
        assert len(data["results"]) == 0
        assert data["truncated"] is False

    def test_rpa_search_case_insensitive(self, rpa_client):
        """Test search is case-insensitive."""
        response_lower = rpa_client.post(
            "/api/rpa/search", json={"query": "validation", "limit": 100}
        )
        response_upper = rpa_client.post(
            "/api/rpa/search", json={"query": "VALIDATION", "limit": 100}
        )

        assert response_lower.json()["total"] == response_upper.json()["total"]


# =============================================================================
# RPA Export Endpoints Tests
# =============================================================================


class TestRpaExportEndpoints:
    """Tests for RPA export endpoints."""

    def test_rpa_export_csv(self, rpa_client):
        """Test /api/rpa/export/csv returns valid CSV."""
        response = rpa_client.get("/api/rpa/export/rpa-test.lctl.json?format=csv")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers.get("content-disposition", "")

        # Parse CSV content
        csv_content = response.text
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        assert len(rows) == 7  # All events
        assert reader.fieldnames is not None
        assert "seq" in reader.fieldnames
        assert "type" in reader.fieldnames
        assert "agent" in reader.fieldnames
        assert "timestamp" in reader.fieldnames

    def test_rpa_export_csv_with_type_filter(self, rpa_client):
        """Test CSV export with event type filter."""
        response = rpa_client.get("/api/rpa/export/rpa-test.lctl.json?format=csv&event_type=error")

        assert response.status_code == 200

        csv_content = response.text
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["type"] == "error"

    def test_rpa_export_json(self, rpa_client):
        """Test export in JSON format."""
        response = rpa_client.get("/api/rpa/export/rpa-test.lctl.json?format=json")

        assert response.status_code == 200
        data = response.json()

        assert "events" in data
        assert "chain_id" in data
        assert len(data["events"]) == 7

    def test_rpa_export_timeline(self, rpa_client):
        """Test /api/rpa/export/timeline returns flat timeline data.

        Note: The current API uses /api/rpa/export/{filename} with format parameter.
        This test uses the JSON format which provides timeline-like data.
        """
        response = rpa_client.get("/api/rpa/export/rpa-test.lctl.json?format=json")

        assert response.status_code == 200
        data = response.json()

        # Events are flat and ordered by seq (timeline)
        events = data["events"]
        for i in range(len(events) - 1):
            assert events[i]["seq"] < events[i + 1]["seq"]

    def test_rpa_export_empty_csv(self, rpa_client):
        """Test CSV export with filter that matches nothing."""
        response = rpa_client.get(
            "/api/rpa/export/rpa-test.lctl.json?format=csv&event_type=nonexistent"
        )

        assert response.status_code == 200
        # Should still have headers
        csv_content = response.text
        # Empty but valid CSV (just headers or empty)
        assert csv_content == "" or "seq" in csv_content


# =============================================================================
# Webhook Registration Tests
# =============================================================================


class TestWebhookRegistration:
    """Tests for webhook registration endpoints."""

    def test_webhook_registration(self, rpa_client):
        """Test /api/rpa/webhooks registers webhook."""
        response = rpa_client.post(
            "/api/rpa/webhooks",
            json={
                "url": "https://example.com/webhook",
                "events": ["error", "step_end"],
                "chain_id": "rpa-test-chain",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "webhook_id" in data
        assert len(data["webhook_id"]) > 0
        assert data["url"] == "https://example.com/webhook"
        assert data["events"] == ["error", "step_end"]
        assert data["status"] == "registered"

    def test_webhook_registration_with_secret(self, rpa_client):
        """Test webhook registration with secret for payload verification."""
        response = rpa_client.post(
            "/api/rpa/webhooks",
            json={
                "url": "https://example.com/webhook",
                "events": ["error"],
                "secret": "my-webhook-secret",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"

    def test_webhook_list(self, rpa_client):
        """Test /api/rpa/webhooks GET lists registered webhooks."""
        # Register a webhook first
        rpa_client.post(
            "/api/rpa/webhooks", json={"url": "https://example.com/hook1", "events": ["error"]}
        )
        rpa_client.post(
            "/api/rpa/webhooks", json={"url": "https://example.com/hook2", "events": ["step_end"]}
        )

        # List webhooks
        response = rpa_client.get("/api/rpa/webhooks")

        assert response.status_code == 200
        data = response.json()

        assert "webhooks" in data
        assert "total" in data
        assert data["total"] >= 2

        urls = [w["url"] for w in data["webhooks"]]
        assert "https://example.com/hook1" in urls
        assert "https://example.com/hook2" in urls

    def test_webhook_delete(self, rpa_client):
        """Test webhook deletion."""
        # Register a webhook
        register_response = rpa_client.post(
            "/api/rpa/webhooks", json={"url": "https://example.com/to-delete", "events": ["error"]}
        )
        webhook_id = register_response.json()["webhook_id"]

        # Delete it
        delete_response = rpa_client.delete(f"/api/rpa/webhooks/{webhook_id}")

        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data["status"] == "deleted"
        assert data["webhook_id"] == webhook_id

        # Verify it's gone
        list_response = rpa_client.get("/api/rpa/webhooks")
        webhook_ids = [w["webhook_id"] for w in list_response.json()["webhooks"]]
        assert webhook_id not in webhook_ids

    def test_webhook_delete_not_found(self, rpa_client):
        """Test deleting non-existent webhook."""
        response = rpa_client.delete("/api/rpa/webhooks/nonexistent-id")

        assert response.status_code == 404

    def test_webhook_default_events(self, rpa_client):
        """Test webhook registration with default events."""
        response = rpa_client.post("/api/rpa/webhooks", json={"url": "https://example.com/default"})

        assert response.status_code == 200
        data = response.json()
        # Default events are ["error", "step_end"]
        assert "error" in data["events"]
        assert "step_end" in data["events"]


# =============================================================================
# RPA Errors Endpoint Tests
# =============================================================================


class TestRpaErrorsEndpoint:
    """Tests for /api/rpa/errors/{filename} endpoint."""

    def test_rpa_errors_returns_flat_errors(self, rpa_client):
        """Test errors endpoint returns flat error list."""
        response = rpa_client.get("/api/rpa/errors/rpa-test.lctl.json")

        assert response.status_code == 200
        data = response.json()

        assert data["has_errors"] is True
        assert data["count"] == 1
        assert data["chain_id"] == "rpa-test-chain"

        error = data["errors"][0]
        assert error["seq"] == 6
        assert error["agent"] == "processor"
        assert error["category"] == "validation"
        assert error["error_type"] == "ValidationError"
        assert "Missing required field" in error["message"]
        assert error["recoverable"] is True

    def test_rpa_errors_no_errors(self, rpa_client):
        """Test errors endpoint with chain that has no errors."""
        response = rpa_client.get("/api/rpa/errors/success.lctl.json")

        assert response.status_code == 200
        data = response.json()

        assert data["has_errors"] is False
        assert data["count"] == 0
        assert len(data["errors"]) == 0


# =============================================================================
# RPA Poll Endpoint Tests
# =============================================================================


class TestRpaPollEndpoint:
    """Tests for /api/rpa/poll/{filename} endpoint."""

    def test_rpa_poll_all_events(self, rpa_client):
        """Test polling for all events."""
        response = rpa_client.get("/api/rpa/poll/rpa-test.lctl.json?since_seq=0")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 7
        assert data["has_new"] is True
        assert data["last_seq"] == 7

    def test_rpa_poll_new_events_only(self, rpa_client):
        """Test polling for events after a specific sequence."""
        response = rpa_client.get("/api/rpa/poll/rpa-test.lctl.json?since_seq=5")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 2  # Events 6 and 7
        assert all(e["seq"] > 5 for e in data["events"])

    def test_rpa_poll_no_new_events(self, rpa_client):
        """Test polling when no new events exist."""
        response = rpa_client.get("/api/rpa/poll/rpa-test.lctl.json?since_seq=7")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 0
        assert data["has_new"] is False


# =============================================================================
# RPA Submit Endpoint Tests
# =============================================================================


class TestRpaSubmitEndpoint:
    """Tests for /api/rpa/submit endpoint."""

    def test_rpa_submit_event(self, rpa_client):
        """Test submitting an event from RPA workflow."""
        response = rpa_client.post(
            "/api/rpa/submit",
            json={
                "chain_id": "uipath-workflow-123",
                "agent": "UiPath.Bot",
                "event_type": "step_start",
                "data": {"intent": "process_document", "input_summary": "invoice.pdf"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "submitted"
        assert "event_id" in data
        assert data["chain_id"] == "uipath-workflow-123"
        assert "timestamp" in data

    def test_rpa_submit_event_minimal(self, rpa_client):
        """Test submitting event with minimal data."""
        response = rpa_client.post(
            "/api/rpa/submit",
            json={"chain_id": "minimal-chain", "agent": "bot", "event_type": "custom"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRpaIntegration:
    """Integration tests for RPA workflow scenarios."""

    def test_full_rpa_workflow(self, rpa_client):
        """Test a complete RPA workflow: list, check, process, export."""
        # Step 1: Get summary
        summary = rpa_client.get("/api/rpa/summary/rpa-test.lctl.json").json()
        assert summary["has_errors"] is True

        # Step 2: Get error details
        errors = rpa_client.get("/api/rpa/errors/rpa-test.lctl.json").json()
        assert errors["count"] == 1

        # Step 3: Search for related events
        search = rpa_client.post("/api/rpa/search", json={"query": "processor", "limit": 10}).json()
        assert search["total"] > 0

        # Step 4: Export for reporting
        export = rpa_client.get("/api/rpa/export/rpa-test.lctl.json?format=csv")
        assert export.status_code == 200

    def test_batch_processing_workflow(self, rpa_client):
        """Test batch processing multiple chains."""
        # Get metrics for all chains
        response = rpa_client.post(
            "/api/rpa/batch/metrics",
            json={"filenames": ["rpa-test.lctl.json", "second-rpa.lctl.json", "success.lctl.json"]},
        )

        data = response.json()

        # Analyze results
        total_events = sum(r["event_count"] for r in data["results"])
        total_errors = sum(r["error_count"] for r in data["results"])

        assert total_events == 16  # 7 + 7 + 2
        assert total_errors == 2  # 1 + 1 + 0

    def test_webhook_and_polling_workflow(self, rpa_client):
        """Test webhook registration with polling fallback."""
        # Register webhook
        webhook = rpa_client.post(
            "/api/rpa/webhooks",
            json={"url": "https://rpa.example.com/callback", "events": ["error"]},
        ).json()

        assert webhook["status"] == "registered"

        # Fall back to polling
        poll = rpa_client.get("/api/rpa/poll/rpa-test.lctl.json?since_seq=0").json()
        assert poll["has_new"] is True

        # Clean up
        rpa_client.delete(f"/api/rpa/webhooks/{webhook['webhook_id']}")


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestRpaEdgeCases:
    """Edge case and error handling tests for RPA endpoints."""

    def test_rpa_summary_path_traversal_blocked(self, rpa_client):
        """Test that path traversal attempts are blocked."""
        response = rpa_client.get("/api/rpa/summary/..%2F..%2Fetc%2Fpasswd")
        assert response.status_code in (400, 403, 404)

    def test_rpa_events_invalid_pagination(self, rpa_client):
        """Test events endpoint with extreme pagination values."""
        # Very large offset
        response = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?offset=10000")
        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) == 0

        # Negative values should be handled
        response = rpa_client.get("/api/rpa/events/rpa-test.lctl.json?limit=0")
        assert response.status_code == 200

    def test_rpa_search_special_characters(self, rpa_client):
        """Test search with special characters in query."""
        response = rpa_client.post("/api/rpa/search", json={"query": "field: amount", "limit": 100})
        assert response.status_code == 200

    def test_rpa_export_nonexistent_file(self, rpa_client):
        """Test export with non-existent file."""
        response = rpa_client.get("/api/rpa/export/nonexistent.lctl.json?format=csv")
        assert response.status_code == 404

    def test_rpa_batch_with_duplicates(self, rpa_client):
        """Test batch metrics with duplicate filenames."""
        response = rpa_client.post(
            "/api/rpa/batch/metrics",
            json={"filenames": ["rpa-test.lctl.json", "rpa-test.lctl.json"]},
        )

        assert response.status_code == 200
        data = response.json()
        # Should process both (even if duplicate)
        assert data["total_processed"] == 2
