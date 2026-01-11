# CLAUDE.md - LCTL Codebase Guide

> This file helps AI assistants understand and work with the LCTL codebase.

## Project Overview

**LCTL (LLM Context Transfer Language)** is a time-travel debugging and observability library for multi-agent LLM workflows. It uses event sourcing to record agent execution, enabling developers to replay workflows to any point in time, compare different runs, and analyze performance bottlenecks.

### Core Philosophy

> "Don't pitch the protocol. Pitch the tool."

The protocol succeeds when developers never think about it. They use `lctl debug` because it's 10x better than print statements. The fact that it uses LCTL format is an implementation detail.

### Key Features

- **Time-Travel Replay**: Step back through agent execution to any point
- **Chain Comparison**: Diff two workflow runs to find divergence
- **Performance Analysis**: Identify bottlenecks and optimize workflows
- **Confidence Tracking**: Monitor fact confidence as it evolves
- **Framework Integrations**: LangChain, CrewAI, AutoGen, OpenAI Agents SDK
- **Web Dashboard**: Visual debugger with timeline and swim lanes

## Architecture

```
lctl/
├── __init__.py          # Public API, auto_instrument(), traced decorator
├── core/
│   ├── events.py        # Chain, Event, EventType, State, ReplayEngine
│   └── session.py       # LCTLSession, traced_step context manager
├── cli/
│   └── main.py          # Click CLI: replay, stats, bottleneck, diff, trace, debug
├── dashboard/
│   └── app.py           # FastAPI web dashboard
└── integrations/
    ├── __init__.py      # Integration exports
    ├── langchain.py     # LCTLCallbackHandler, LCTLChain, trace_chain
    ├── crewai.py        # LCTLCrew, LCTLAgent, LCTLTask, trace_crew
    ├── autogen.py       # LCTLAutogenCallback, trace_agent, trace_group_chat
    └── openai_agents.py # LCTLOpenAIAgentTracer, LCTLRunHooks, TracedAgent
```

## Key Concepts

### Events (Event Sourcing)

Everything is an immutable event. The event stream is the source of truth.

```python
from lctl.core.events import EventType

# Standard event types
EventType.STEP_START      # Agent begins work
EventType.STEP_END        # Agent completes (with duration, tokens)
EventType.FACT_ADDED      # New fact registered (id, text, confidence)
EventType.FACT_MODIFIED   # Fact updated (confidence change, text change)
EventType.TOOL_CALL       # Tool invocation (tool name, input, output)
EventType.ERROR           # Failure occurred (category, type, message)
EventType.CHECKPOINT      # State snapshot for fast replay
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
state = engine.replay_all()               # Final state

# Analysis methods
trace = engine.get_trace()                # Step-level execution flow
bottlenecks = engine.find_bottlenecks()   # Performance analysis
diffs = engine.diff(other_engine)         # Compare two chains
timeline = engine.get_confidence_timeline() # Fact confidence over time
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
│   ├── core/                   # Core functionality
│   │   ├── events.py           # Chain, Event, State, ReplayEngine
│   │   └── session.py          # LCTLSession, traced_step
│   ├── cli/                    # Command-line interface
│   │   └── main.py             # Click commands
│   ├── dashboard/              # Web UI
│   │   ├── app.py              # FastAPI application
│   │   ├── static/             # CSS, JS
│   │   └── templates/          # HTML templates
│   └── integrations/           # Framework integrations
│       ├── langchain.py        # LangChain support
│       ├── crewai.py           # CrewAI support
│       ├── autogen.py          # AutoGen/AG2 support
│       └── openai_agents.py    # OpenAI Agents SDK support
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
├── pyproject.toml              # Project configuration
├── README.md                   # User documentation
└── LLM-CONTEXT-TRANSFER.md     # Full specification
```

## Development Guide

### Setup

```bash
# Clone and create virtual environment
cd /home/kim/projects/llm-context-transfer
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install with all integrations
pip install -e ".[all]"

# Install specific integration
pip install -e ".[langchain]"
pip install -e ".[crewai]"
pip install -e ".[dashboard]"
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_events.py

# Run specific test class
pytest tests/test_cli.py::TestReplayCommand

# Run with verbose output
pytest -v

# Run and show print statements
pytest -s

# Quick test (no coverage)
pytest --no-cov
```

### Linting and Formatting

```bash
# Format code
black lctl/ tests/

# Lint code
ruff check lctl/ tests/

# Fix auto-fixable lint issues
ruff check --fix lctl/ tests/
```

### Building

```bash
# Build package
python -m build

# Install locally built package
pip install dist/lctl-4.0.0-py3-none-any.whl
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

All integrations follow this structure:

```python
# 1. Check availability
try:
    from framework import SomeClass
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

# 2. Provide availability check
def is_available() -> bool:
    return FRAMEWORK_AVAILABLE

# 3. Wrapper class with session
class LCTLWrapper:
    def __init__(self, chain_id=None, session=None):
        if not FRAMEWORK_AVAILABLE:
            raise ImportError("Install with: pip install framework")
        self.session = session or LCTLSession(chain_id=chain_id)

    def export(self, path: str):
        self.session.export(path)

# 4. Convenience function
def trace_thing(thing, chain_id=None):
    return LCTLWrapper(thing, chain_id=chain_id)
```

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
    return text[:max_length - 3] + "..."
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

2. **Follow the pattern**:

```python
"""NewFramework integration for LCTL."""

from __future__ import annotations
from typing import Any, Dict, Optional
from ..core.session import LCTLSession

try:
    from newframework import Agent, Runner
    NEWFRAMEWORK_AVAILABLE = True
except ImportError:
    NEWFRAMEWORK_AVAILABLE = False
    Agent = None
    Runner = None

def is_available() -> bool:
    return NEWFRAMEWORK_AVAILABLE

class LCTLNewFrameworkTracer:
    def __init__(self, chain_id: Optional[str] = None, session: Optional[LCTLSession] = None):
        if not NEWFRAMEWORK_AVAILABLE:
            raise ImportError("Install with: pip install newframework")
        self.session = session or LCTLSession(chain_id=chain_id)

    # Implement framework-specific hooks
    def on_agent_start(self, agent: Agent):
        agent_name = getattr(agent, "name", "agent")
        self.session.step_start(agent_name, "execute", "")

    def on_agent_end(self, agent: Agent, result: Any):
        agent_name = getattr(agent, "name", "agent")
        self.session.step_end(agent_name, outcome="success")

    def export(self, path: str):
        self.session.export(path)

def trace_agent(agent: Agent, chain_id: Optional[str] = None) -> LCTLNewFrameworkTracer:
    """Convenience function to trace an agent."""
    tracer = LCTLNewFrameworkTracer(chain_id=chain_id)
    # Attach tracer to agent
    return tracer

__all__ = ["NEWFRAMEWORK_AVAILABLE", "LCTLNewFrameworkTracer", "trace_agent", "is_available"]
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
pytest                              # Run tests
pytest --no-cov -x                  # Quick test, stop on first failure
black lctl/ tests/                  # Format code
ruff check --fix lctl/              # Lint and fix

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
# Ensure dev dependencies
pip install -e ".[dev]"

# Check for import issues
python -c "from lctl import Chain, ReplayEngine"
```

## Version

- **LCTL Version**: 4.0.0
- **Python**: 3.9+
- **Key Dependencies**: click>=8.0, pyyaml>=6.0
