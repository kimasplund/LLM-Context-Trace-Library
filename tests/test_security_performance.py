
import time
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Mock fastapi if not installed, just to allow collecting verify logic if possible?
# But we need TestClient. If fastapi is missing, we can't really test the app.
try:
    from fastapi.testclient import TestClient
    from lctl.dashboard.app import create_app
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from lctl.core.events import Chain, Event, EventType

@pytest.fixture
def temp_chain_env(tmp_path):
    """Create a temporary environment with a chain file."""
    chain_data = {
        "lctl": "4.0",
        "chain": {"id": "security-test"},
        "events": [
            {
                "seq": 1,
                "type": "step_start",
                "timestamp": "2025-01-01T12:00:00",
                "agent": "test-agent",
                "data": {}
            }
        ]
    }
    
    # Valid file
    (tmp_path / "valid.lctl.json").write_text(
        '{"lctl": "4.0", "chain": {"id": "valid"}, "events": []}'
    )
    
    return tmp_path

@pytest.fixture
def client(temp_chain_env):
    app = create_app(working_dir=temp_chain_env)
    return TestClient(app)

class TestSecurity:
    def test_path_traversal_absolute(self, client):
        """Test accessing absolute path outside working dir."""
        # attempts to access /etc/passwd
        response = client.get("/api/chain/%2Fetc%2Fpasswd")
        # Should be 400 (Invalid filename) or 403 (Access denied) depending on implementation details
        # My implementation raises 400 for ValueError (if path can't be resolved relative) or 403
        assert response.status_code in (400, 403, 404)

    def test_path_traversal_relative(self, client):
        """Test accessing relative path with .."""
        response = client.get("/api/chain/..%2F..%2Fetc%2Fpasswd")
        assert response.status_code in (400, 403, 404)

    def test_valid_access(self, client):
        """Test accessing a valid file."""
        response = client.get("/api/chain/valid.lctl.json")
        assert response.status_code == 200

class TestPerformance:
    def test_caching_behavior(self, client, temp_chain_env):
        """Verify that the engine is cached."""
        file_path = temp_chain_env / "valid.lctl.json"
        
        # Patch Chain.load to count calls
        with patch("lctl.core.events.Chain.load", side_effect=Chain.load) as mock_load:
            # First call
            client.get("/api/chain/valid.lctl.json")
            assert mock_load.call_count == 1
            
            # Second call - should hit cache
            client.get("/api/chain/valid.lctl.json")
            assert mock_load.call_count == 1  # Still 1
            
            # Touch file to update mtime
            file_path.touch()
            
            # Third call - should reload due to mtime change
            client.get("/api/chain/valid.lctl.json")
            assert mock_load.call_count == 2
