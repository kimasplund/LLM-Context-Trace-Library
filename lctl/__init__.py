"""LCTL - LLM Context Trace Library.

Time-travel debugging for multi-agent LLM workflows.

Usage:
    # Zero-config auto-instrumentation
    import lctl
    lctl.auto_instrument()

    # Manual session
    from lctl import LCTLSession
    with LCTLSession() as session:
        session.step_start("analyzer", "analyze")
        # ... do work ...
        session.add_fact("F1", "Found issue", confidence=0.9)
        session.step_end()
    session.export("trace.lctl.json")

CLI:
    lctl replay chain.lctl.json      # Time-travel
    lctl stats chain.lctl.json       # Performance
    lctl bottleneck chain.lctl.json  # Find slow steps
    lctl trace chain.lctl.json       # Execution flow
    lctl diff v1.json v2.json        # Compare chains
"""

__version__ = "4.1.0"

from .core.events import Chain, Event, EventType, ReplayEngine, State
from .core.session import LCTLSession, traced_step

# Global session for auto-instrumentation
_global_session = None


def auto_instrument():
    """Create (or return) the global session used by `@lctl.traced` and `lctl.export()`.

    It does not instrument frameworks; use the integration for your framework.
    """
    global _global_session
    if _global_session is None:
        _global_session = LCTLSession(chain_id="auto")
    print(f"[LCTL] Auto-instrumentation enabled. Chain ID: {_global_session.chain.id}")
    return _global_session


def get_session() -> LCTLSession:
    """Get the global LCTL session."""
    global _global_session
    if _global_session is None:
        _global_session = LCTLSession(chain_id="auto")
    return _global_session


def traced(agent: str, intent: str = "execute"):
    """Decorator to trace a function as an agent step.

    Usage:
        @lctl.traced("analyzer", "analyze")
        def analyze_code(code):
            return findings
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            session = get_session()
            # Build input summary from args
            input_summary = f"args={len(args)}, kwargs={list(kwargs.keys())}"
            with session.span(agent, intent, input_summary):
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def export(path: str):
    """Export the global session to file."""
    session = get_session()
    session.export(path)
    print(f"[LCTL] Exported {len(session.chain.events)} events to {path}")


__all__ = [  # noqa: RUF022  # grouped by topic; ruff's sort ignores group comments
    # Core
    "Chain",
    "Event",
    "EventType",
    "ReplayEngine",
    "State",
    # Session
    "LCTLSession",
    "traced_step",
    # Auto-instrumentation
    "auto_instrument",
    "get_session",
    "traced",
    "export",
]
