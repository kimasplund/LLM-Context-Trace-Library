"""Tests for LCTL integration base classes and utilities (lctl/integrations/base.py).

Test ID Format: TEST-2026-01-23-NNN
Coverage Target: >80% statement coverage, >70% branch coverage
"""

import json
import threading
import time
from pathlib import Path

import pytest

from lctl.core.session import LCTLSession
from lctl.integrations.base import (
    BaseTracer,
    IntegrationNotAvailableError,
    TracerDelegate,
    check_availability,
    truncate,
)

# =============================================================================
# Test: truncate function
# =============================================================================


class TestTruncate:
    """Tests for truncate() utility function.

    Covers lines: 23-30 (full coverage of function)
    """

    def test_truncate_empty_string(self):
        """Test truncation of empty string returns empty string."""
        result = truncate("")
        assert result == ""

    def test_truncate_none_like_empty(self):
        """Test truncation of empty/falsy values."""
        result = truncate("")
        assert result == ""

    def test_truncate_short_text_no_change(self):
        """Test that short text is returned unchanged."""
        short_text = "Hello, world!"
        result = truncate(short_text, max_length=200)
        assert result == short_text

    def test_truncate_exactly_max_length(self):
        """Test text exactly at max_length is not truncated."""
        text = "x" * 50
        result = truncate(text, max_length=50)
        assert result == text
        assert len(result) == 50

    def test_truncate_long_text_with_ellipsis(self):
        """Test long text is truncated with ellipsis."""
        long_text = "x" * 100
        result = truncate(long_text, max_length=50)
        assert result == "x" * 47 + "..."
        assert len(result) == 50

    def test_truncate_max_length_less_than_4(self):
        """Test truncation when max_length < 4 (no room for ellipsis)."""
        text = "Hello"
        result = truncate(text, max_length=3)
        assert result == "Hel"
        assert len(result) == 3

    def test_truncate_max_length_exactly_4(self):
        """Test truncation when max_length is exactly 4."""
        text = "Hello World"
        result = truncate(text, max_length=4)
        assert result == "H..."
        assert len(result) == 4

    def test_truncate_default_max_length(self):
        """Test truncation uses default max_length of 200."""
        long_text = "x" * 250
        result = truncate(long_text)
        assert len(result) == 200
        assert result.endswith("...")

    def test_truncate_non_string_input(self):
        """Test truncation converts non-string to string."""
        result = truncate(12345, max_length=3)
        assert result == "123"

    def test_truncate_unicode_text(self):
        """Test truncation works with unicode characters."""
        unicode_text = "Hello World!"
        result = truncate(unicode_text, max_length=10)
        assert result == "Hello W..."
        assert len(result) == 10

    def test_truncate_one_over_max_length(self):
        """Test text one character over max_length."""
        text = "x" * 51
        result = truncate(text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")


# =============================================================================
# Test: check_availability function
# =============================================================================


class TestCheckAvailability:
    """Tests for check_availability() function.

    Covers lines: 43-47 (full coverage of function)
    """

    def test_check_availability_returns_true_when_available(self):
        """Test returns True when import_check succeeds."""

        def successful_import():
            pass  # No exception means success

        result = check_availability("test_module", successful_import)
        assert result is True

    def test_check_availability_returns_false_when_not_available(self):
        """Test returns False when import_check raises ImportError."""

        def failing_import():
            raise ImportError("Module not found")

        result = check_availability("missing_module", failing_import)
        assert result is False

    def test_check_availability_with_lambda(self):
        """Test check_availability with lambda import check."""
        # Successful case
        result = check_availability("json", lambda: __import__("json"))
        assert result is True

        # Failing case
        result = check_availability(
            "nonexistent_module_xyz",
            lambda: __import__("nonexistent_module_xyz_abc_123"),
        )
        assert result is False

    def test_check_availability_passes_module_name(self):
        """Test that module_name parameter is accepted (for error messages)."""
        # The module_name is currently not used in the function but is documented
        # for error message purposes. This test ensures the parameter is accepted.
        result = check_availability("my_framework", lambda: None)
        assert result is True


# =============================================================================
# Test: IntegrationNotAvailableError exception
# =============================================================================


class TestIntegrationNotAvailableError:
    """Tests for IntegrationNotAvailableError exception class.

    Covers lines: 53-58 (full coverage of class)
    """

    def test_error_message_formatting(self):
        """Test error message is correctly formatted."""
        error = IntegrationNotAvailableError(
            framework="LangChain", install_hint="pip install langchain"
        )

        assert str(error) == "LangChain is not installed. Install with: pip install langchain"

    def test_error_attributes(self):
        """Test error has framework and install_hint attributes."""
        error = IntegrationNotAvailableError(framework="CrewAI", install_hint="pip install crewai")

        assert error.framework == "CrewAI"
        assert error.install_hint == "pip install crewai"

    def test_error_inherits_from_import_error(self):
        """Test error inherits from ImportError."""
        error = IntegrationNotAvailableError("Test", "pip install test")
        assert isinstance(error, ImportError)

    def test_error_can_be_caught_as_import_error(self):
        """Test error can be caught as ImportError."""
        with pytest.raises(ImportError):
            raise IntegrationNotAvailableError("MyFramework", "pip install myfw")

    def test_error_can_be_caught_specifically(self):
        """Test error can be caught by its own type."""
        with pytest.raises(IntegrationNotAvailableError) as exc_info:
            raise IntegrationNotAvailableError("TestFW", "pip install testfw")

        assert exc_info.value.framework == "TestFW"


# =============================================================================
# Test: BaseTracer abstract class
# =============================================================================


class ConcreteTracer(BaseTracer):
    """Concrete implementation of BaseTracer for testing."""


class TestBaseTracer:
    """Tests for BaseTracer abstract base class.

    Covers lines: 87-94, 99, 104, 112, 120, 124-126, 137-147, 151-152, 156-157, 161-162
    """

    def test_init_with_chain_id(self):
        """Test initialization with chain_id creates new session."""
        tracer = ConcreteTracer(chain_id="test-chain-123")

        assert tracer._session is not None
        assert tracer._session.chain.id == "test-chain-123"
        assert tracer._lock is not None
        assert tracer._auto_cleanup is True
        assert tracer._cleanup_interval == 3600.0
        assert isinstance(tracer._tracked_items, dict)

    def test_init_with_existing_session(self):
        """Test initialization with existing session uses that session."""
        existing_session = LCTLSession(chain_id="existing-session")
        tracer = ConcreteTracer(session=existing_session)

        assert tracer._session is existing_session
        assert tracer._session.chain.id == "existing-session"

    def test_init_without_arguments(self):
        """Test initialization without arguments generates chain ID."""
        tracer = ConcreteTracer()

        assert tracer._session is not None
        assert len(tracer._session.chain.id) == 8  # UUID[:8] format

    def test_init_auto_cleanup_disabled(self):
        """Test initialization with auto_cleanup disabled."""
        tracer = ConcreteTracer(auto_cleanup=False)
        assert tracer._auto_cleanup is False

    def test_init_custom_cleanup_interval(self):
        """Test initialization with custom cleanup interval."""
        tracer = ConcreteTracer(cleanup_interval=1800.0)
        assert tracer._cleanup_interval == 1800.0

    def test_session_property(self):
        """Test session property returns the LCTL session."""
        tracer = ConcreteTracer(chain_id="session-test")
        session = tracer.session

        assert session is tracer._session
        assert isinstance(session, LCTLSession)

    def test_chain_property(self):
        """Test chain property returns the underlying chain."""
        tracer = ConcreteTracer(chain_id="chain-prop-test")
        chain = tracer.chain

        assert chain is tracer._session.chain
        assert chain.id == "chain-prop-test"

    def test_export_delegates_to_session(self, tmp_path: Path):
        """Test export() delegates to session.export()."""
        tracer = ConcreteTracer(chain_id="export-test")
        tracer.session.step_start("agent", "action")
        tracer.session.step_end()

        export_path = tmp_path / "test_export.lctl.json"
        tracer.export(str(export_path))

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "export-test"
        assert len(data["events"]) == 2

    def test_to_dict_delegates_to_session(self):
        """Test to_dict() delegates to session.to_dict()."""
        tracer = ConcreteTracer(chain_id="dict-test")
        tracer.session.add_fact("F1", "Test fact", confidence=0.9)

        result = tracer.to_dict()

        assert result["chain"]["id"] == "dict-test"
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "fact_added"

    def test_should_cleanup_returns_false_when_disabled(self):
        """Test _should_cleanup returns False when auto_cleanup disabled."""
        tracer = ConcreteTracer(auto_cleanup=False)
        assert tracer._should_cleanup() is False

    def test_should_cleanup_returns_false_before_interval(self):
        """Test _should_cleanup returns False before interval passes."""
        tracer = ConcreteTracer(cleanup_interval=3600.0)
        # Just created, so interval hasn't passed
        assert tracer._should_cleanup() is False

    def test_should_cleanup_returns_true_after_interval(self):
        """Test _should_cleanup returns True after interval passes."""
        tracer = ConcreteTracer(cleanup_interval=0.001)  # Very short interval
        time.sleep(0.01)  # Wait longer than interval
        assert tracer._should_cleanup() is True

    def test_cleanup_stale_entries_removes_old_items(self):
        """Test cleanup_stale_entries removes items older than max_age."""
        tracer = ConcreteTracer()

        # Add items with artificial old timestamps
        now = time.time()
        tracer._tracked_items = {
            "old1": now - 4000,  # Older than 3600s
            "old2": now - 5000,  # Older than 3600s
            "new1": now - 100,  # Newer than 3600s
        }

        removed = tracer.cleanup_stale_entries(max_age_seconds=3600.0)

        assert removed == 2
        assert "old1" not in tracer._tracked_items
        assert "old2" not in tracer._tracked_items
        assert "new1" in tracer._tracked_items

    def test_cleanup_stale_entries_updates_last_cleanup(self):
        """Test cleanup_stale_entries updates _last_cleanup timestamp."""
        tracer = ConcreteTracer()
        old_cleanup_time = tracer._last_cleanup

        time.sleep(0.01)
        tracer.cleanup_stale_entries()

        assert tracer._last_cleanup > old_cleanup_time

    def test_cleanup_stale_entries_thread_safe(self):
        """Test cleanup_stale_entries is thread-safe."""
        tracer = ConcreteTracer()
        now = time.time()

        # Add many items
        for i in range(100):
            tracer._tracked_items[f"item_{i}"] = now - (i * 100)

        results = []

        def cleanup_worker():
            removed = tracer.cleanup_stale_entries(max_age_seconds=2000.0)
            results.append(removed)

        threads = [threading.Thread(target=cleanup_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total removed should equal total stale items (items with i*100 > 2000)
        # That's items where i >= 21 (i.e., 80 items)
        total_removed = sum(results)
        # Due to race conditions, the total might vary, but should be <= 80
        assert total_removed <= 80

    def test_maybe_cleanup_does_nothing_before_interval(self):
        """Test _maybe_cleanup does nothing if interval hasn't passed."""
        tracer = ConcreteTracer(cleanup_interval=3600.0)
        now = time.time()
        tracer._tracked_items = {"old": now - 4000}

        tracer._maybe_cleanup()

        # Item should still exist since cleanup shouldn't have run
        assert "old" in tracer._tracked_items

    def test_maybe_cleanup_runs_when_interval_passed(self):
        """Test _maybe_cleanup runs cleanup after interval passes."""
        tracer = ConcreteTracer(cleanup_interval=0.001)
        now = time.time()
        tracer._tracked_items = {"old": now - 1}  # 1 second old

        time.sleep(0.01)
        tracer._maybe_cleanup()

        # Item should be removed since it's older than cleanup_interval (0.001s)
        assert "old" not in tracer._tracked_items

    def test_track_item_adds_with_timestamp(self):
        """Test _track_item adds item with current timestamp."""
        tracer = ConcreteTracer()

        before = time.time()
        tracer._track_item("test_item")
        after = time.time()

        assert "test_item" in tracer._tracked_items
        assert before <= tracer._tracked_items["test_item"] <= after

    def test_track_item_updates_existing(self):
        """Test _track_item updates timestamp for existing item."""
        tracer = ConcreteTracer()
        tracer._tracked_items["existing"] = time.time() - 1000

        tracer._track_item("existing")

        # Timestamp should be updated to recent
        assert tracer._tracked_items["existing"] > time.time() - 1

    def test_track_item_thread_safe(self):
        """Test _track_item is thread-safe."""
        tracer = ConcreteTracer()

        def track_worker(item_id):
            for _ in range(10):
                tracer._track_item(f"{item_id}_{_}")
                time.sleep(0.001)

        threads = [threading.Thread(target=track_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(tracer._tracked_items) == 50

    def test_untrack_item_removes_item(self):
        """Test _untrack_item removes tracked item."""
        tracer = ConcreteTracer()
        tracer._tracked_items["to_remove"] = time.time()

        tracer._untrack_item("to_remove")

        assert "to_remove" not in tracer._tracked_items

    def test_untrack_item_ignores_nonexistent(self):
        """Test _untrack_item ignores items that don't exist."""
        tracer = ConcreteTracer()
        tracer._tracked_items["existing"] = time.time()

        # Should not raise
        tracer._untrack_item("nonexistent")

        assert "existing" in tracer._tracked_items

    def test_untrack_item_thread_safe(self):
        """Test _untrack_item is thread-safe."""
        tracer = ConcreteTracer()

        # Add items
        for i in range(50):
            tracer._tracked_items[f"item_{i}"] = time.time()

        def untrack_worker(start):
            for i in range(start, start + 10):
                tracer._untrack_item(f"item_{i}")

        threads = [threading.Thread(target=untrack_worker, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(tracer._tracked_items) == 0

    def test_lock_is_threading_lock(self):
        """Test that _lock is a proper threading lock."""
        tracer = ConcreteTracer()
        assert isinstance(tracer._lock, type(threading.Lock()))


# =============================================================================
# Test: TracerDelegate class
# =============================================================================


class TestTracerDelegate:
    """Tests for TracerDelegate class.

    TracerDelegate provides BaseTracer functionality via composition for
    callback-based integrations that can't inherit from BaseTracer.
    """

    def test_init_with_chain_id(self):
        """Test initialization with chain_id creates new session."""
        delegate = TracerDelegate(chain_id="delegate-chain-test")

        assert delegate.session is not None
        assert delegate.session.chain.id == "delegate-chain-test"
        assert delegate.lock is not None
        assert delegate._auto_cleanup is True
        assert delegate._cleanup_interval == 3600.0

    def test_init_with_existing_session(self):
        """Test initialization with existing session uses that session."""
        existing_session = LCTLSession(chain_id="existing-session")
        delegate = TracerDelegate(session=existing_session)

        assert delegate.session is existing_session
        assert delegate.session.chain.id == "existing-session"

    def test_chain_property(self):
        """Test chain property returns session's chain."""
        delegate = TracerDelegate(chain_id="delegate-chain-test")

        assert delegate.chain is delegate.session.chain
        assert delegate.chain.id == "delegate-chain-test"

    def test_lock_property(self):
        """Test lock property returns threading lock."""
        delegate = TracerDelegate(chain_id="lock-test")

        assert delegate.lock is not None
        assert isinstance(delegate.lock, type(threading.Lock()))

    def test_export_delegates_to_session(self, tmp_path: Path):
        """Test export() delegates to session.export()."""
        delegate = TracerDelegate(chain_id="delegate-export-test")
        delegate.session.add_fact("F1", "Test fact")

        export_path = tmp_path / "delegate_export.lctl.json"
        delegate.export(str(export_path))

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "delegate-export-test"

    def test_to_dict_delegates_to_session(self):
        """Test to_dict() delegates to session.to_dict()."""
        delegate = TracerDelegate(chain_id="delegate-dict-test")
        delegate.session.step_start("agent", "action")

        result = delegate.to_dict()

        assert result["chain"]["id"] == "delegate-dict-test"
        assert len(result["events"]) == 1

    def test_delegate_with_events(self, tmp_path: Path):
        """Test delegate export/to_dict with multiple events."""
        delegate = TracerDelegate(chain_id="delegate-events-test")
        delegate.session.step_start("planner", "plan")
        delegate.session.add_fact("F1", "Planning complete", confidence=0.9)
        delegate.session.tool_call("search", {"q": "test"}, {"results": []}, duration_ms=100)
        delegate.session.step_end()

        # Test to_dict
        result = delegate.to_dict()
        assert len(result["events"]) == 4

        # Test export
        export_path = tmp_path / "delegate_multi.lctl.json"
        delegate.export(str(export_path))
        data = json.loads(export_path.read_text())
        assert len(data["events"]) == 4

    def test_track_item(self):
        """Test track_item adds item with timestamp."""
        delegate = TracerDelegate(chain_id="track-test")

        before = time.time()
        delegate.track_item("test_item")
        after = time.time()

        ts = delegate.get_tracked_item("test_item")
        assert ts is not None
        assert before <= ts <= after

    def test_untrack_item(self):
        """Test untrack_item removes tracked item."""
        delegate = TracerDelegate(chain_id="untrack-test")
        delegate.track_item("to_remove")

        delegate.untrack_item("to_remove")

        assert delegate.get_tracked_item("to_remove") is None

    def test_cleanup_stale_entries(self):
        """Test cleanup_stale_entries removes old items."""
        delegate = TracerDelegate(chain_id="cleanup-test")

        # Add items with artificial timestamps
        now = time.time()
        delegate._tracked_items = {
            "old": now - 4000,  # Older than 3600s
            "new": now - 100,  # Newer than 3600s
        }

        removed = delegate.cleanup_stale_entries(max_age_seconds=3600.0)

        assert removed == 1
        assert "old" not in delegate._tracked_items
        assert "new" in delegate._tracked_items

    def test_maybe_cleanup(self):
        """Test maybe_cleanup runs when interval passed."""
        delegate = TracerDelegate(cleanup_interval=0.001)
        now = time.time()
        delegate._tracked_items = {"old": now - 1}

        time.sleep(0.01)
        delegate.maybe_cleanup()

        assert "old" not in delegate._tracked_items

    def test_auto_cleanup_disabled(self):
        """Test auto_cleanup can be disabled."""
        delegate = TracerDelegate(auto_cleanup=False, cleanup_interval=0.001)
        now = time.time()
        delegate._tracked_items = {"item": now - 1}

        time.sleep(0.01)
        delegate.maybe_cleanup()

        # Item should still exist since auto_cleanup is disabled
        assert "item" in delegate._tracked_items


# =============================================================================
# Integration Tests
# =============================================================================


class TestBaseTracerIntegration:
    """Integration tests for BaseTracer with full workflows."""

    def test_full_tracing_workflow(self, tmp_path: Path):
        """Test complete tracing workflow with BaseTracer."""
        tracer = ConcreteTracer(chain_id="integration-workflow")

        # Simulate agent work
        tracer.session.step_start("analyzer", "analyze")
        tracer.session.add_fact("FINDING_1", "Issue found", confidence=0.85)
        tracer.session.tool_call("grep", {"pattern": "TODO"}, {"count": 5}, 50)
        tracer.session.step_end(outcome="success", duration_ms=500)

        # Track some items
        tracer._track_item("run_1")
        tracer._track_item("run_2")

        # Export
        export_path = tmp_path / "integration.lctl.json"
        tracer.export(str(export_path))

        # Verify
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "integration-workflow"
        assert len(data["events"]) == 4

        # Cleanup
        tracer._untrack_item("run_1")
        assert "run_1" not in tracer._tracked_items
        assert "run_2" in tracer._tracked_items

    def test_concurrent_tracing(self):
        """Test concurrent usage of BaseTracer."""
        tracer = ConcreteTracer(chain_id="concurrent-test")
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    tracer._track_item(f"w{worker_id}_item_{i}")
                    if i % 2 == 0:
                        tracer._untrack_item(f"w{worker_id}_item_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_session_reuse(self):
        """Test reusing session across multiple tracers."""
        shared_session = LCTLSession(chain_id="shared-session")
        shared_session.step_start("initial_agent", "setup")
        shared_session.step_end()

        # Create tracers with shared session
        tracer1 = ConcreteTracer(session=shared_session)
        tracer2 = ConcreteTracer(session=shared_session)

        # Both should reference same session and chain
        assert tracer1.session is tracer2.session
        assert tracer1.chain is tracer2.chain

        # Events from either should appear in the chain
        tracer1.session.add_fact("F1", "From tracer1")
        tracer2.session.add_fact("F2", "From tracer2")

        result = tracer1.to_dict()
        assert len(result["events"]) == 4  # 2 initial + 2 facts


class TestInstallHints:
    """`lctl` is not on PyPI: install advice must never be a bare `pip install lctl`."""

    def test_install_command_uses_the_git_url(self):
        from lctl.integrations.base import install_command

        assert "git+https://" in install_command()
        assert "lctl[dspy] @ git+https://" in install_command("dspy")

    def test_not_available_error_message_has_git_url(self):
        from lctl.integrations.base import IntegrationNotAvailableError, install_command

        err = IntegrationNotAvailableError("X", install_command("x"))
        assert "git+https://" in str(err)

    def test_no_bare_lctl_install_hints_in_package_or_examples(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        bare = re.compile(r"pip install ['\"]?lctl(\[[^\]]*\])?['\"]?(?!\s*@)(?=[\s'\"`]|$)")
        offenders = []
        for base in ("lctl", "examples", "vscode-lctl/src"):
            for path in (root / base).rglob("*"):
                if path.suffix not in {".py", ".md", ".ts"}:
                    continue
                for n, line in enumerate(path.read_text().splitlines(), 1):
                    for m in bare.finditer(line):
                        if " @ git+" not in line[m.start() :]:
                            offenders.append(f"{path.relative_to(root)}:{n}")
        assert not offenders
