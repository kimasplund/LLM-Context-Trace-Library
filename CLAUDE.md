# CLAUDE.md - LCTL Codebase Guide

> This file helps AI assistants understand and work with the LCTL codebase.

## Project Overview

**LCTL (LLM Context Trace Library)** is a time-travel debugging and observability library for multi-agent LLM workflows. It uses event sourcing to record agent execution, enabling developers to replay workflows to any point in time, compare different runs, and analyze performance bottlenecks.

### Core Philosophy

> "Don't pitch the protocol. Pitch the tool."

The protocol succeeds when developers never think about it. They use `lctl debug` because it's 10x better than print statements. The fact that it uses LCTL format is an implementation detail.

### Key Features

- **Time-Travel Replay**: Step back through agent execution to any point
- **Chain Comparison**: Diff two workflow runs to find divergence
- **Performance Analysis**: Identify bottlenecks and optimize workflows
- **Confidence Tracking**: Monitor fact confidence as it evolves
- **Framework Integrations**: LangChain, CrewAI, AutoGen, OpenAI Agents SDK, PydanticAI, Semantic Kernel, DSPy, LlamaIndex
- **Web Dashboard**: Visual debugger with timeline and swim lanes

## Architecture

```
lctl/
├── __init__.py          # Public API, auto_instrument(), traced decorator
├── core/
│   ├── events.py        # Chain, Event, EventType, State, ReplayEngine
│   ├── session.py       # LCTLSession, traced_step context manager
│   ├── redaction.py     # Sensitive data redaction for traces
│   └── schema.py        # Schema versioning and migration
├── cli/
│   └── main.py          # Click CLI: replay, stats, bottleneck, diff, trace, debug, claude
├── dashboard/
│   └── app.py           # FastAPI web dashboard
├── streaming/
│   ├── emitter.py       # EventEmitter: publish/subscribe for real-time events
│   ├── subscriber.py    # Event subscriber patterns (buffered, async, filtered)
│   └── websocket.py     # WebSocket and SSE servers for event streaming
├── metrics/
│   ├── collectors.py    # Collect metrics from chains
│   ├── aggregators.py   # Aggregate metrics across chains
│   └── exporters.py     # Export metrics to various formats
├── evaluation/
│   ├── comparator.py    # Chain comparator: A/B testing of executions
│   ├── metrics.py       # Metrics for comparing chain executions
│   └── statistical.py   # Statistical significance tests for A/B comparisons
└── integrations/
    ├── __init__.py      # Integration exports
    ├── base.py          # BaseTracer, TracerDelegate, truncate, check_availability
    ├── claude_code.py   # LCTLClaudeCodeTracer, generate_hooks (SELF-TRACING!)
    ├── langchain.py     # LCTLCallbackHandler (TracerDelegate), LCTLChain, trace_chain
    ├── crewai.py        # LCTLCrew, LCTLAgent, LCTLTask, trace_crew
    ├── autogen.py       # LCTLAutogenCallback (BaseTracer), trace_agent, trace_group_chat
    ├── openai_agents.py # LCTLOpenAIAgentTracer (BaseTracer), LCTLRunHooks, TracedAgent
    ├── pydantic_ai.py   # LCTLPydanticAITracer (BaseTracer), TracedAgent, trace_agent
    ├── semantic_kernel.py # LCTLSemanticKernelTracer (BaseTracer), trace_kernel
    ├── dspy.py          # LCTLDSPyCallback (BaseTracer), TracedDSPyModule, trace_module
    └── llamaindex.py    # LCTLLlamaIndexCallback (TracerDelegate), trace_query_engine
```

### Self-Tracing with Claude Code

LCTL can trace its own multi-agent workflows when running in Claude Code. This enables time-travel debugging of Claude Code sessions themselves.

**Setup**:
```bash
# Initialize LCTL hooks in your project
lctl claude init

# Validate installation
lctl claude validate
```

**What's Captured**:
- Task tool invocations (agent spawning)
- Agent completions with tokens and duration
- Tool calls (Bash, Write, Edit, WebFetch, etc.)
- TodoWrite updates, Skill invocations
- MCP tool calls, Git commits
- User interactions (AskUserQuestion)

**CLI Commands**:
- `lctl claude init` - Generate hook scripts
- `lctl claude validate` - Check hook installation
- `lctl claude status` - Show active session
- `lctl claude report` - Generate HTML report
- `lctl claude clean` - Clean old traces

See `docs/tutorials/claude-code-tutorial.md` for full documentation.

## Key Concepts

### Events (Event Sourcing)

Everything is an immutable event. The event stream is the source of truth.

```python
from lctl.core.events import EventType

# Standard event types
EventType.STEP_START  # Agent begins work
EventType.STEP_END  # Agent completes (with duration, tokens)
EventType.FACT_ADDED  # New fact registered (id, text, confidence)
EventType.FACT_MODIFIED  # Fact updated (confidence change, text change)
EventType.TOOL_CALL  # Tool invocation (tool name, input, output)
EventType.ERROR  # Failure occurred (category, type, message)
EventType.CHECKPOINT  # State snapshot for fast replay
```

### Chain

A Chain is a collection of events with an ID and version:

```python
from lctl.core.events import Chain

chain = Chain(id="my-workflow")
chain.add_event(event)
chain.save(Path("workflow.lctl.json"))

# Load and inspect
loaded = Chain.load(Path("workflow.lctl.json"))
```

### State

State is derived by replaying events. Never stored directly - always computed.

```python
from lctl.core.events import State

state = State()
state.apply_event(event)  # Updates facts, metrics, errors
# state.facts: Dict[str, Dict] - fact_id -> {text, confidence, source}
# state.metrics: total_duration_ms, total_tokens_in, total_tokens_out
# state.errors: List of error dicts
```

### ReplayEngine

The core debugging tool - replay events to any sequence number:

```python
from lctl.core.events import ReplayEngine

engine = ReplayEngine(chain)
state = engine.replay_to(target_seq=10)  # State at event 10
state = engine.replay_all()  # Final state

# Analysis methods
trace = engine.get_trace()  # Step-level execution flow
bottlenecks = engine.find_bottlenecks()  # Performance analysis
diffs = engine.diff(other_engine)  # Compare two chains
timeline = engine.get_confidence_timeline()  # Fact confidence over time
```

### LCTLSession

Context manager for recording events:

```python
from lctl import LCTLSession

with LCTLSession(chain_id="my-chain") as session:
    session.step_start("analyzer", "analyze", "input data")
    session.add_fact("F1", "Found issue X", confidence=0.85)
    session.tool_call("grep", {"pattern": "x"}, {"matches": 3}, duration_ms=50)
    session.step_end("analyzer", outcome="success", duration_ms=1000)

session.export("trace.lctl.json")
```

### Facts

Facts are findings with confidence scores that can evolve:

- **Initial confidence**: 0.0-1.0 based on source reliability
- **Decay**: Confidence can decrease over hops (0.95 per hop typical)
- **Consensus**: Multiple agents confirming increases confidence
- **Modification**: Facts can be updated with new text/confidence

```python
session.add_fact("F1", "Hypothesis A", confidence=0.7)
session.modify_fact("F1", confidence=0.9, reason="verified by tests")
```

### Confidence Thresholds

| Confidence | Action |
|------------|--------|
| >= 0.80    | Proceed automatically |
| 0.60-0.79  | Warn, but proceed |
| 0.40-0.59  | Request verification |
| < 0.40     | Block, require human review |

## Directory Structure

```
/home/kim/projects/llm-context-transfer/
├── lctl/                       # Main package
│   ├── __init__.py             # Public API exports
│   ├── core/ cli/ streaming/ metrics/ evaluation/ integrations/  # see Architecture
│   └── dashboard/              # Web UI: app.py, static/ (CSS, JS), templates/
├── tests/                      # Test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_events.py          # Core events tests
│   ├── test_session.py         # Session tests
│   ├── test_cli.py             # CLI command tests
│   └── test_*.py               # Integration tests
├── examples/                   # Usage examples
│   ├── sample-chain.lctl.json  # Example chain file
│   ├── langchain_example.py    # LangChain integration demo
│   ├── crewai_example.py       # CrewAI integration demo
│   └── autogen_example.py      # AutoGen integration demo
├── scripts/check_wheel.py      # Wheel content/metadata check (CI package job)
├── vscode-lctl/                # VS Code extension (TypeScript + React webview)
├── .github/                    # CI workflow, Dependabot
├── .pre-commit-config.yaml     # Same checks as CI's lint job
├── pyproject.toml              # Project, uv dependency groups, ruff/mypy/pytest config
├── uv.lock                     # Locked dependencies (commit changes with pyproject.toml)
├── README.md                   # User documentation
└── LLM-CONTEXT-TRANSFER.md     # Full specification
```

## Development Guide

### Setup

```bash
# Dev environment: project + dev group (pytest, ruff, mypy, pre-commit, dashboard deps)
uv sync
uv run pre-commit install

# Add framework integrations
uv sync --all-extras            # or: uv sync --extra langchain

# Without uv (pip >= 25.1)
pip install -e . --group dev
```

### Running Tests

```bash
uv run pytest                                  # All tests, branch coverage (fail_under in pyproject)
uv run pytest tests/test_events.py             # One file
uv run pytest tests/test_cli.py::TestReplayCommand
uv run pytest --no-cov -x                      # Quick, stop on first failure
```

Warnings are errors (`filterwarnings = ["error"]`); fix the cause rather than ignoring it.

### Linting and Formatting

```bash
uv run ruff format .                  # Format
uv run ruff check --fix .             # Lint and fix
uv run mypy                           # Type check (lctl/)
uv run pre-commit run --all-files     # Everything CI's lint job runs
```

Intentional rule violations get a line-level `# noqa: RULE  # reason`; never file-wide ignores under `lctl/`.

### Building

```bash
uv build                                       # sdist + wheel into dist/
python3 scripts/check_wheel.py dist/*.whl      # Verify wheel contents and metadata
```

## Code Patterns

### 1. Event Creation Pattern

Always use session methods, never create Event objects directly in application code:

```python
# Good - use session methods
session.step_start("agent", "intent", "summary")
session.add_fact("F1", "text", confidence=0.8)

# Avoid - manual event creation (only for tests/internal)
event = Event(seq=1, type=EventType.STEP_START, ...)
```

### 2. Integration Pattern

All integrations use `BaseTracer` (inheritance) or `TracerDelegate` (composition) from `lctl.integrations.base`; see [Integration Guide](#integration-guide) for both templates.

**Current integration patterns:**
- BaseTracer: pydantic_ai, semantic_kernel, dspy, openai_agents, autogen
- TracerDelegate: langchain, llamaindex (must extend framework callbacks)
- Wrapper: crewai (wraps framework classes)

### 3. Error Handling Pattern

Graceful degradation - tracing should never break the main code:

```python
try:
    session.step_start(...)
except Exception:
    pass  # Don't break execution if tracing fails

# Context managers handle this automatically
with traced_step(session, "agent", "intent"):
    do_work()  # If tracing fails, work still executes
```

### 4. Chain File Loading Pattern

Always handle errors gracefully:

```python
from lctl.core.events import Chain

try:
    chain = Chain.load(Path(filepath))
except FileNotFoundError:
    print(f"File not found: {filepath}")
except ValueError as e:
    print(f"Invalid format: {e}")
```

### 5. Truncation Pattern

Summaries should be truncated for readability:

```python
def _truncate(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
```

## Testing Guide

### Test Organization

- `conftest.py`: Shared fixtures (sample_chain, base_timestamp, etc.)
- `test_events.py`: Core Chain, Event, State, ReplayEngine tests
- `test_session.py`: LCTLSession tests
- `test_cli.py`: CLI command tests using Click's CliRunner
- `test_*.py`: Integration-specific tests

### Key Fixtures

```python
@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed timestamp for reproducible tests."""
    return datetime(2025, 1, 15, 10, 0, 0)


@pytest.fixture
def sample_chain(sample_events) -> Chain:
    """Chain with standard test events."""
    chain = Chain(id="test-chain")
    for event in sample_events:
        chain.add_event(event)
    return chain


@pytest.fixture
def temp_chain_file(sample_chain, tmp_path) -> Path:
    """Temporary chain file for CLI tests."""
    file_path = tmp_path / "test.lctl.json"
    sample_chain.save(file_path)
    return file_path
```

### Writing Tests

```python
# Test core functionality
def test_replay_to_seq(sample_chain):
    engine = ReplayEngine(sample_chain)
    state = engine.replay_to(3)
    assert len(state.facts) == 1
    assert state.facts["F1"]["confidence"] == 0.9


# Test CLI commands
def test_replay_command(runner, temp_chain_file):
    result = runner.invoke(cli, ["replay", str(temp_chain_file)])
    assert result.exit_code == 0
    assert "Replaying" in result.output


# Test error handling
def test_invalid_file(runner, tmp_path):
    result = runner.invoke(cli, ["replay", str(tmp_path / "missing.json")])
    assert result.exit_code != 0
```

## Integration Guide

### Adding a New Framework Integration

1. **Create integration file**: `lctl/integrations/newframework.py`

2. **Choose the pattern** based on whether you can inherit from BaseTracer:

**Option A: BaseTracer inheritance** (preferred when possible)
```python
"""NewFramework integration for LCTL."""

from __future__ import annotations
from typing import Any, Optional
from .base import BaseTracer, IntegrationNotAvailableError, truncate

try:
    from newframework import Agent, Runner

    NEWFRAMEWORK_AVAILABLE = True
except ImportError:
    NEWFRAMEWORK_AVAILABLE = False
    Agent = None
    Runner = None


def is_available() -> bool:
    return NEWFRAMEWORK_AVAILABLE


class LCTLNewFrameworkTracer(BaseTracer):
    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional["LCTLSession"] = None,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0,
    ):
        if not NEWFRAMEWORK_AVAILABLE:
            raise IntegrationNotAvailableError("NewFramework", "pip install newframework")
        super().__init__(
            chain_id=chain_id,
            session=session,
            auto_cleanup=auto_cleanup,
            cleanup_interval=cleanup_interval,
        )

    # Implement framework-specific hooks
    def on_agent_start(self, agent: Agent):
        agent_name = getattr(agent, "name", "agent")
        self._track_item(agent_name)  # Track for cleanup
        self.session.step_start(agent_name, "execute", "")

    def on_agent_end(self, agent: Agent, result: Any):
        agent_name = getattr(agent, "name", "agent")
        self._untrack_item(agent_name)
        self.session.step_end(agent_name, outcome="success")
        self._maybe_cleanup()  # Periodic cleanup


__all__ = ["NEWFRAMEWORK_AVAILABLE", "LCTLNewFrameworkTracer", "is_available"]
```

**Option B: TracerDelegate composition** (when extending framework callbacks)
```python
from .base import TracerDelegate, IntegrationNotAvailableError


class LCTLNewFrameworkCallback(FrameworkBaseCallback):
    def __init__(self, chain_id=None, session=None, *, auto_cleanup=True, cleanup_interval=3600.0):
        super().__init__()  # Framework's __init__
        self._tracer = TracerDelegate(
            chain_id=chain_id,
            session=session,
            auto_cleanup=auto_cleanup,
            cleanup_interval=cleanup_interval,
        )

    @property
    def session(self):
        return self._tracer.session

    def export(self, path: str):
        self._tracer.export(path)
```

3. **Update integrations/__init__.py**:

```python
from .newframework import (
    LCTLNewFrameworkTracer,
    trace_agent as trace_newframework_agent,
    is_available as newframework_available,
)
```

4. **Add to pyproject.toml**:

```toml
[project.optional-dependencies]
newframework = ["newframework>=1.0"]
```

5. **Create tests**: `tests/test_newframework.py`

6. **Create example**: `examples/newframework_example.py`

## Quick Commands

```bash
# Development
uv run pytest                       # Run tests
uv run pytest --no-cov -x           # Quick test, stop on first failure
uv run ruff format .                # Format code
uv run ruff check --fix .           # Lint and fix
uv run mypy                         # Type check

# CLI usage
lctl replay chain.lctl.json         # Replay to end
lctl replay --to-seq 10 chain.json  # Replay to seq 10
lctl stats chain.lctl.json          # Show statistics
lctl stats --json chain.lctl.json   # JSON output
lctl bottleneck chain.lctl.json     # Find slow steps
lctl trace chain.lctl.json          # Show execution flow
lctl confidence chain.lctl.json     # Show fact confidence
lctl diff v1.json v2.json           # Compare chains
lctl debug chain.lctl.json          # Launch web debugger
lctl dashboard                      # Launch dashboard server

# Dashboard
lctl dashboard --port 3000          # Custom port
lctl dashboard --dir ./traces       # Specific directory
```

## File Formats

### Chain File (.lctl.json)

```json
{
  "lctl": "4.0",
  "chain": {
    "id": "my-workflow"
  },
  "events": [
    {
      "seq": 1,
      "type": "step_start",
      "timestamp": "2024-01-15T10:30:00Z",
      "agent": "analyzer",
      "data": {
        "intent": "analyze",
        "input_summary": "code.py"
      }
    }
  ]
}
```

### YAML Format (.lctl.yaml)

```yaml
lctl: "4.0"
chain:
  id: my-workflow
events:
  - seq: 1
    type: step_start
    timestamp: "2024-01-15T10:30:00Z"
    agent: analyzer
    data:
      intent: analyze
      input_summary: code.py
```

## Common Tasks

### Add a new CLI command

1. Edit `lctl/cli/main.py`
2. Add command with Click decorator:

```python
@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--option", "-o", help="Description")
def newcommand(chain_file: str, option: str):
    """Command description."""
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)
    # Implementation
```

3. Add tests in `tests/test_cli.py`

### Add a new event type

1. Add to `EventType` enum in `lctl/core/events.py`
2. Update `State.apply_event()` to handle new type
3. Update tests in `tests/test_events.py`

### Modify dashboard API

1. Edit `lctl/dashboard/app.py`
2. Add/modify endpoints using FastAPI patterns
3. Update tests in `tests/test_dashboard.py`

## Troubleshooting

### Import Errors

```python
# Check if integration is available
from lctl.integrations import langchain_available

if langchain_available():
    from lctl.integrations.langchain import LCTLCallbackHandler
```

### Dashboard Won't Start

```bash
# Install dashboard dependencies
pip install "lctl[dashboard]"
# Or manually
pip install fastapi uvicorn
```

### Tests Failing

```bash
# Ensure the dev environment is in sync with uv.lock
uv sync

# Check for import issues
python -c "from lctl import Chain, ReplayEngine"
```

## Version

- **LCTL Version**: see `__version__` in `lctl/__init__.py` (single source; packaging reads it)
- **Python**: 3.11+
- **Key Dependencies**: click>=8.0, pyyaml>=6.0
