"""Tests for LCTL package __init__.py public API."""

from pathlib import Path

import pytest

import lctl
from lctl import Chain, Event, EventType, LCTLSession, ReplayEngine


class TestAutoInstrument:
    """Tests for auto_instrument() function."""

    def test_auto_instrument_creates_session(self):
        """Test auto_instrument creates a global session."""
        # Clear any existing global session
        lctl._global_session = None

        session = lctl.auto_instrument()

        assert session is not None
        assert isinstance(session, LCTLSession)

        # Cleanup
        lctl._global_session = None

    def test_auto_instrument_returns_same_session(self):
        """Test auto_instrument returns the same session on subsequent calls."""
        lctl._global_session = None

        session1 = lctl.auto_instrument()
        session2 = lctl.auto_instrument()

        assert session1 is session2

        # Cleanup
        lctl._global_session = None

    def test_auto_instrument_chain_has_id(self):
        """Test auto_instrument creates session with chain id."""
        lctl._global_session = None

        session = lctl.auto_instrument()

        # Chain should have an auto-generated or default id
        assert session.chain.id is not None
        assert len(session.chain.id) > 0

        # Cleanup
        lctl._global_session = None


class TestGetSession:
    """Tests for get_session() function."""

    def test_get_session_creates_if_none(self):
        """Test get_session creates a session if none exists."""
        lctl._global_session = None

        session = lctl.get_session()

        assert session is not None
        assert isinstance(session, LCTLSession)

        # Cleanup
        lctl._global_session = None

    def test_get_session_returns_existing(self):
        """Test get_session returns existing global session."""
        lctl._global_session = None

        # First call creates session
        session1 = lctl.get_session()
        # Second call returns same session
        session2 = lctl.get_session()

        assert session1 is session2

        # Cleanup
        lctl._global_session = None


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def test_traced_decorator_basic(self):
        """Test traced decorator wraps function correctly."""
        lctl._global_session = None
        session = lctl.auto_instrument()

        @lctl.traced("test-agent", "process")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

        assert result == 10

        # Check that events were recorded
        events = session.chain.events
        step_starts = [e for e in events if e.type == EventType.STEP_START]
        step_ends = [e for e in events if e.type == EventType.STEP_END]

        assert len(step_starts) >= 1
        assert len(step_ends) >= 1

        # Cleanup
        lctl._global_session = None

    def test_traced_decorator_with_exception(self):
        """Test traced decorator handles exceptions."""
        lctl._global_session = None
        session = lctl.auto_instrument()

        @lctl.traced("error-agent", "fail")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Check that error was recorded
        events = session.chain.events
        step_ends = [e for e in events if e.type == EventType.STEP_END]

        assert len(step_ends) >= 1
        # The step_end should record the error outcome
        last_end = step_ends[-1]
        assert last_end.data.get("outcome") == "error" or "error" in str(last_end.data).lower()

        # Cleanup
        lctl._global_session = None

    def test_traced_decorator_preserves_function_metadata(self):
        """Test traced decorator preserves function name and docstring."""

        @lctl.traced("agent", "intent")
        def documented_function():
            """This is the docstring."""

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


class TestExport:
    """Tests for export() function."""

    def test_export_global_session(self, tmp_path: Path):
        """Test export() exports the global session."""
        lctl._global_session = None
        session = lctl.auto_instrument()

        # Record some events
        session.step_start("agent", "intent", "summary")
        session.step_end("agent", "success", duration_ms=100)

        # Export
        export_path = tmp_path / "exported.lctl.json"
        lctl.export(str(export_path))

        assert export_path.exists()

        # Verify exported chain
        loaded = Chain.load(export_path)
        assert loaded.id is not None  # Has some id
        assert len(loaded.events) >= 2

        # Cleanup
        lctl._global_session = None

    def test_export_without_session(self, tmp_path: Path):
        """Test export() when no session exists."""
        lctl._global_session = None

        # Export should create a session first
        export_path = tmp_path / "empty_export.lctl.json"
        lctl.export(str(export_path))

        assert export_path.exists()

        # Cleanup
        lctl._global_session = None


class TestPublicAPI:
    """Tests for public API exports."""

    def test_chain_exported(self):
        """Test Chain is exported from lctl."""
        assert hasattr(lctl, "Chain")
        assert lctl.Chain is Chain

    def test_event_exported(self):
        """Test Event is exported from lctl."""
        assert hasattr(lctl, "Event")
        assert lctl.Event is Event

    def test_event_type_exported(self):
        """Test EventType is exported from lctl."""
        assert hasattr(lctl, "EventType")
        assert lctl.EventType is EventType

    def test_replay_engine_exported(self):
        """Test ReplayEngine is exported from lctl."""
        assert hasattr(lctl, "ReplayEngine")
        assert lctl.ReplayEngine is ReplayEngine

    def test_lctl_session_exported(self):
        """Test LCTLSession is exported from lctl."""
        assert hasattr(lctl, "LCTLSession")
        assert lctl.LCTLSession is LCTLSession

    def test_version_exported(self):
        """Test __version__ is exported from lctl."""
        assert hasattr(lctl, "__version__")
        assert lctl.__version__ == "4.1.0"
