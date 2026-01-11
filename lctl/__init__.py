"""LCTL - LLM Context Transfer Language.

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

__version__ = "4.0.0"

from .core.events import Chain, Event, EventType, ReplayEngine, State
from .core.session import LCTLSession, traced_step

# Global session for auto-instrumentation
_global_session = None


def auto_instrument():
    """Enable automatic instrumentation for supported frameworks.

    Currently supports:
    - Basic Python (via decorators)

    Planned:
    - LangChain
    - CrewAI
    - AutoGen
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
            import time
            session = get_session()
            start = time.time()

            # Build input summary from args
            input_summary = f"args={len(args)}, kwargs={list(kwargs.keys())}"

            session.step_start(agent, intent, input_summary)
            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start) * 1000)
                session.step_end(agent, outcome="success", duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                session.step_end(agent, outcome="error", duration_ms=duration_ms)
                session.error(
                    category="execution_error",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=False
                )
                raise

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def export(path: str):
    """Export the global session to file."""
    session = get_session()
    session.export(path)
    print(f"[LCTL] Exported {len(session.chain.events)} events to {path}")


__all__ = [
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
