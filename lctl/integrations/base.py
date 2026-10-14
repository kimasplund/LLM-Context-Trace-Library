"""Base classes and utilities for LCTL integrations."""

from __future__ import annotations

import threading
import time
from abc import ABC
from collections.abc import Callable
from typing import Any

from ..core.session import LCTLSession


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length (default 200)

    Returns:
        Truncated text with '...' suffix if needed
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_length:
        return text
    if max_length < 4:
        return text[:max_length]
    return text[: max_length - 3] + "..."


def check_availability(module_name: str, import_check: Callable[[], object]) -> bool:
    """Check if a framework is available.

    Args:
        module_name: Name of the module for error messages
        import_check: Callable that attempts the import

    Returns:
        True if available, False otherwise
    """
    try:
        import_check()
        return True
    except ImportError:
        return False


REPO_URL = "git+https://github.com/kimasplund/LLM-Context-Trace-Library"


def install_command(extra: str | None = None) -> str:
    """The pip command that installs LCTL (optionally with an extra) from the repository.

    ``lctl`` is not published on PyPI, so a plain pip install by name would fetch
    whatever package anyone registers under that name.
    """
    spec = f"lctl[{extra}]" if extra else "lctl"
    return f'pip install "{spec} @ {REPO_URL}"'


class IntegrationNotAvailableError(ImportError):
    """Raised when an integration's framework is not installed."""

    def __init__(self, framework: str, install_hint: str):
        self.framework = framework
        self.install_hint = install_hint
        super().__init__(f"{framework} is not installed. Install with: {install_hint}")


class BaseTracer(ABC):  # noqa: B024  # concrete base with shared defaults; subclasses override hooks
    """Base class for all LCTL integration tracers.

    Provides common functionality:
    - Session management
    - Thread-safe operations
    - Export methods
    - Stale entry cleanup
    """

    def __init__(
        self,
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0,
    ):
        """Initialize the tracer.

        Args:
            chain_id: Optional chain ID for new session
            session: Optional existing session to use
            auto_cleanup: Whether to auto-cleanup stale entries
            cleanup_interval: Cleanup interval in seconds (default 1 hour)
        """
        self._session = session or LCTLSession(chain_id=chain_id)
        self._lock = threading.Lock()
        self._auto_cleanup = auto_cleanup
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

        # Subclasses should add their tracking dicts here
        self._tracked_items: dict[str, float] = {}  # id -> timestamp

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session."""
        return self._session

    @property
    def chain(self):
        """Get the underlying chain."""
        return self._session.chain

    def export(self, path: str) -> None:
        """Export the trace to a file.

        Args:
            path: File path to export to
        """
        self._session.export(path)

    def to_dict(self) -> dict[str, Any]:
        """Convert the trace to a dictionary.

        Returns:
            Dictionary representation of the chain
        """
        return self._session.to_dict()

    def _should_cleanup(self) -> bool:
        """Check if cleanup should run."""
        if not self._auto_cleanup:
            return False
        return time.time() - self._last_cleanup >= self._cleanup_interval

    def cleanup_stale_entries(self, max_age_seconds: float = 3600.0) -> int:
        """Remove entries older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age for entries (default 1 hour)

        Returns:
            Number of entries removed
        """
        now = time.time()
        self._last_cleanup = now

        with self._lock:
            stale = [key for key, ts in self._tracked_items.items() if now - ts > max_age_seconds]
            for key in stale:
                del self._tracked_items[key]
            return len(stale)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        if self._should_cleanup():
            self.cleanup_stale_entries(self._cleanup_interval)

    def _track_item(self, item_id: str) -> None:
        """Track an item with current timestamp."""
        with self._lock:
            self._tracked_items[item_id] = time.time()

    def _untrack_item(self, item_id: str) -> None:
        """Stop tracking an item."""
        with self._lock:
            self._tracked_items.pop(item_id, None)


class TracerDelegate:
    """Delegate class for callback-based integrations that can't inherit from BaseTracer.

    Use this when the integration must extend a framework's callback class
    but still needs BaseTracer functionality via composition.

    Example:
        class MyCallbackHandler(FrameworkCallback):
            def __init__(self, chain_id=None, session=None):
                super().__init__()
                self._tracer = TracerDelegate(chain_id=chain_id, session=session)

            @property
            def session(self):
                return self._tracer.session

            def export(self, path):
                self._tracer.export(path)
    """

    def __init__(
        self,
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0,
    ):
        """Initialize the delegate.

        Args:
            chain_id: Optional chain ID for new session
            session: Optional existing session to use
            auto_cleanup: Whether to auto-cleanup stale entries
            cleanup_interval: Cleanup interval in seconds (default 1 hour)
        """
        self._session = session or LCTLSession(chain_id=chain_id)
        self._lock = threading.Lock()
        self._auto_cleanup = auto_cleanup
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._tracked_items: dict[str, float] = {}

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session."""
        return self._session

    @property
    def chain(self):
        """Get the underlying chain."""
        return self._session.chain

    @property
    def lock(self) -> threading.Lock:
        """Get the threading lock for thread-safe operations."""
        return self._lock

    def export(self, path: str) -> None:
        """Export the trace to a file."""
        self._session.export(path)

    def to_dict(self) -> dict[str, Any]:
        """Convert the trace to a dictionary."""
        return self._session.to_dict()

    def _should_cleanup(self) -> bool:
        """Check if cleanup should run."""
        if not self._auto_cleanup:
            return False
        return time.time() - self._last_cleanup >= self._cleanup_interval

    def cleanup_stale_entries(self, max_age_seconds: float = 3600.0) -> int:
        """Remove entries older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age for entries (default 1 hour)

        Returns:
            Number of entries removed
        """
        now = time.time()
        self._last_cleanup = now

        with self._lock:
            stale = [key for key, ts in self._tracked_items.items() if now - ts > max_age_seconds]
            for key in stale:
                del self._tracked_items[key]
            return len(stale)

    def maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        if self._should_cleanup():
            self.cleanup_stale_entries(self._cleanup_interval)

    def track_item(self, item_id: str) -> None:
        """Track an item with current timestamp."""
        with self._lock:
            self._tracked_items[item_id] = time.time()

    def untrack_item(self, item_id: str) -> None:
        """Stop tracking an item."""
        with self._lock:
            self._tracked_items.pop(item_id, None)

    def get_tracked_item(self, item_id: str) -> float | None:
        """Get the timestamp for a tracked item.

        Args:
            item_id: The item ID to look up

        Returns:
            Timestamp when item was tracked, or None if not found
        """
        with self._lock:
            return self._tracked_items.get(item_id)


__all__ = [
    "BaseTracer",
    "IntegrationNotAvailableError",
    "TracerDelegate",
    "check_availability",
    "truncate",
]
