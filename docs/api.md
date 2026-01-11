# LCTL v4.0 API Reference

> Time-travel debugging for multi-agent LLM workflows

**Version**: 4.0.0
**Last Updated**: 2026-01-11

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Module (lctl.core)](#core-module-lctlcore)
  - [EventType](#eventtype)
  - [Event](#event)
  - [Chain](#chain)
  - [State](#state)
  - [ReplayEngine](#replayengine)
  - [LCTLSession](#lctlsession)
  - [traced_step](#traced_step)
- [Integrations (lctl.integrations)](#integrations-lctlintegrations)
  - [LangChain Integration](#langchain-integration)
  - [CrewAI Integration](#crewai-integration)
  - [AutoGen Integration](#autogen-integration)
  - [OpenAI Agents Integration](#openai-agents-integration)
  - [DSPy Integration](#dspy-integration)
  - [LlamaIndex Integration](#llamaindex-integration)
- [Dashboard (lctl.dashboard)](#dashboard-lctldashboard)
  - [create_app](#create_app)
  - [run_dashboard](#run_dashboard)
  - [API Endpoints](#api-endpoints)
- [Global Functions](#global-functions)

---

## Overview

LCTL (LLM Context Transfer Language) provides event sourcing and time-travel debugging capabilities for multi-agent LLM workflows. It captures execution traces as immutable events, enabling replay, analysis, and debugging of complex agent interactions.

### Key Features

- **Event Sourcing**: Every action is recorded as an immutable event
- **Time-Travel Debugging**: Replay execution to any point in time
- **Framework Integrations**: Native support for LangChain, CrewAI, AutoGen, and OpenAI Agents SDK
- **Performance Analysis**: Built-in bottleneck detection and metrics
- **Web Dashboard**: Visual exploration of agent workflows

---

## Quick Start

```python
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
```

---

## Core Module (lctl.core)

The core module provides the fundamental building blocks for event sourcing and replay.

### EventType

```python
from lctl import EventType
```

An enumeration of standard LCTL event types.

#### Values

| Value | Description |
|-------|-------------|
| `STEP_START` | Agent step begins |
| `STEP_END` | Agent step completes |
| `FACT_ADDED` | New fact recorded |
| `FACT_MODIFIED` | Existing fact updated |
| `TOOL_CALL` | Tool invocation |
| `ERROR` | Error occurred |
| `CHECKPOINT` | State checkpoint for fast replay |
| `STREAM_START` | Streaming response begins |
| `STREAM_CHUNK` | Streaming response chunk |
| `STREAM_END` | Streaming response ends |
| `CONTRACT_VALIDATION` | Contract validation event |
| `MODEL_ROUTING` | Model routing decision |

#### Example

```python
from lctl import EventType

if event.type == EventType.STEP_START:
    print(f"Agent {event.agent} started step")
elif event.type == EventType.ERROR:
    print(f"Error in agent {event.agent}")
```

---

### Event

```python
from lctl import Event
```

A single LCTL event representing an atomic action in the workflow.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `seq` | `int` | Sequence number (1-indexed, monotonically increasing) |
| `type` | `EventType` | The type of event |
| `timestamp` | `datetime` | When the event occurred (UTC) |
| `agent` | `str` | Name of the agent that produced the event |
| `data` | `Dict[str, Any]` | Event-specific payload data |

#### Methods

##### `to_dict() -> Dict[str, Any]`

Serialize the event to a dictionary.

**Returns**: Dictionary representation of the event.

```python
event_dict = event.to_dict()
# {
#     "seq": 1,
#     "type": "step_start",
#     "timestamp": "2026-01-11T10:30:00+00:00",
#     "agent": "analyzer",
#     "data": {"intent": "analyze", "input_summary": "code.py"}
# }
```

##### `from_dict(d: Dict[str, Any]) -> Event` (classmethod)

Create an Event from a dictionary.

**Parameters**:
- `d` (`Dict[str, Any]`): Dictionary containing event data.

**Returns**: New Event instance.

**Raises**:
- `ValueError`: If required fields are missing or invalid.

```python
event = Event.from_dict({
    "seq": 1,
    "type": "step_start",
    "timestamp": "2026-01-11T10:30:00+00:00",
    "agent": "analyzer",
    "data": {"intent": "analyze"}
})
```

---

### Chain

```python
from lctl import Chain
```

A collection of events representing a complete execution trace.

#### Constructor

```python
Chain(id: str, events: List[Event] = [], version: str = "4.0")
```

**Parameters**:
- `id` (`str`): Unique identifier for the chain.
- `events` (`List[Event]`): Initial list of events (default: empty).
- `version` (`str`): LCTL version string (default: "4.0").

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Chain identifier |
| `events` | `List[Event]` | List of events in the chain |
| `version` | `str` | LCTL format version |

#### Methods

##### `add_event(event: Event) -> None`

Add an event to the chain.

**Parameters**:
- `event` (`Event`): The event to add.

```python
chain = Chain(id="my-chain")
chain.add_event(event)
```

##### `to_dict() -> Dict[str, Any]`

Serialize the chain to a dictionary.

**Returns**: Dictionary representation of the chain.

```python
chain_dict = chain.to_dict()
# {
#     "lctl": "4.0",
#     "chain": {"id": "my-chain"},
#     "events": [...]
# }
```

##### `from_dict(d: Dict[str, Any]) -> Chain` (classmethod)

Create a Chain from a dictionary.

**Parameters**:
- `d` (`Dict[str, Any]`): Dictionary containing chain data.

**Returns**: New Chain instance.

##### `load(path: Path) -> Chain` (classmethod)

Load a chain from a JSON or YAML file.

**Parameters**:
- `path` (`Path`): Path to the chain file.

**Returns**: Loaded Chain instance.

**Raises**:
- `FileNotFoundError`: If the file does not exist.
- `ValueError`: If the file is empty, has invalid format, or contains malformed data.
- `PermissionError`: If read permission is denied.

```python
from pathlib import Path
chain = Chain.load(Path("trace.lctl.json"))
```

##### `save(path: Path) -> None`

Save the chain to a file (JSON or YAML based on extension).

**Parameters**:
- `path` (`Path`): Destination file path.

```python
chain.save(Path("trace.lctl.json"))  # JSON format
chain.save(Path("trace.lctl.yaml"))  # YAML format
```

---

### State

```python
from lctl import State
```

Materialized state derived from replaying events. Represents the world state at a particular point in the event sequence.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `facts` | `Dict[str, Dict[str, Any]]` | Known facts indexed by fact ID |
| `current_agent` | `Optional[str]` | Currently executing agent |
| `current_step` | `Optional[int]` | Current step sequence number |
| `errors` | `List[Dict[str, Any]]` | List of recorded errors |
| `metrics` | `Dict[str, Any]` | Accumulated metrics |

#### Default Metrics Structure

```python
{
    "total_duration_ms": 0,
    "total_tokens_in": 0,
    "total_tokens_out": 0,
    "event_count": 0,
    "error_count": 0
}
```

#### Methods

##### `apply_event(event: Event) -> None`

Apply an event to update the state.

**Parameters**:
- `event` (`Event`): The event to apply.

This method handles all event types:
- `STEP_START`: Sets current agent and step
- `STEP_END`: Accumulates duration and token metrics
- `FACT_ADDED`: Records new fact
- `FACT_MODIFIED`: Updates existing fact
- `ERROR`: Records error and increments error count
- `TOOL_CALL`: Accumulates tool duration

```python
state = State()
for event in chain.events:
    state.apply_event(event)
print(f"Total duration: {state.metrics['total_duration_ms']}ms")
```

---

### ReplayEngine

```python
from lctl import ReplayEngine
```

Engine for replaying events and performing time-travel debugging operations.

#### Constructor

```python
ReplayEngine(chain: Chain)
```

**Parameters**:
- `chain` (`Chain`): The chain to replay.

#### Methods

##### `replay_to(target_seq: int) -> State`

Replay events up to the target sequence number.

**Parameters**:
- `target_seq` (`int`): The sequence number to replay to (inclusive).

**Returns**: `State` at the target sequence.

```python
engine = ReplayEngine(chain)
state_at_5 = engine.replay_to(5)
print(f"Facts at seq 5: {state_at_5.facts}")
```

##### `replay_all() -> State`

Replay all events in the chain.

**Returns**: Final `State` after all events.

```python
final_state = engine.replay_all()
```

##### `get_trace() -> List[Dict[str, Any]]`

Get step-level trace for visualization.

**Returns**: List of step start/end events with their data.

```python
trace = engine.get_trace()
for step in trace:
    print(f"[{step['seq']}] {step['agent']}: {step['type']}")
```

##### `diff(other: ReplayEngine) -> List[Dict[str, Any]]`

Compare two chains and find divergence points.

**Parameters**:
- `other` (`ReplayEngine`): Another replay engine to compare against.

**Returns**: List of differences, each with type:
- `missing_in_first`: Event exists only in second chain
- `missing_in_second`: Event exists only in first chain
- `diverged`: Events differ at same sequence

```python
engine1 = ReplayEngine(chain1)
engine2 = ReplayEngine(chain2)
diffs = engine1.diff(engine2)
for d in diffs:
    print(f"Difference at seq {d['seq']}: {d['type']}")
```

##### `find_bottlenecks() -> List[Dict[str, Any]]`

Analyze performance and identify bottlenecks.

**Returns**: List of steps sorted by duration (descending), each containing:
- `agent`: Agent name
- `seq`: Sequence number
- `duration_ms`: Step duration in milliseconds
- `tokens`: Token usage (if available)
- `percentage`: Percentage of total time

```python
bottlenecks = engine.find_bottlenecks()
for b in bottlenecks[:3]:
    print(f"{b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")
```

##### `get_confidence_timeline() -> Dict[str, List[Dict[str, Any]]]`

Track confidence changes for each fact over time.

**Returns**: Dictionary mapping fact IDs to lists of confidence changes.

```python
timeline = engine.get_confidence_timeline()
for fact_id, changes in timeline.items():
    print(f"Fact {fact_id}:")
    for change in changes:
        print(f"  seq {change['seq']}: confidence={change['confidence']}")
```

---

### LCTLSession

```python
from lctl import LCTLSession
```

Session for recording LCTL events with context manager support.

#### Constructor

```python
LCTLSession(chain_id: Optional[str] = None)
```

**Parameters**:
- `chain_id` (`Optional[str]`): Identifier for the chain. Auto-generated if not provided.

#### Context Manager Usage

```python
with LCTLSession(chain_id="my-workflow") as session:
    session.step_start("agent1", "process")
    # ... do work ...
    session.step_end()
# Errors during the block are automatically recorded
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `chain` | `Chain` | The underlying chain being built |

#### Methods

##### `step_start(agent: str, intent: str, input_summary: str = "") -> int`

Record the start of an agent step.

**Parameters**:
- `agent` (`str`): Name of the agent.
- `intent` (`str`): What the agent intends to do.
- `input_summary` (`str`): Summary of the input (optional).

**Returns**: Sequence number of the event.

```python
seq = session.step_start("analyzer", "analyze", "input.txt")
```

##### `step_end(agent: Optional[str] = None, outcome: str = "success", output_summary: str = "", duration_ms: int = 0, tokens_in: int = 0, tokens_out: int = 0) -> int`

Record the end of an agent step.

**Parameters**:
- `agent` (`Optional[str]`): Agent name (uses current if not provided).
- `outcome` (`str`): "success" or "error".
- `output_summary` (`str`): Summary of the output.
- `duration_ms` (`int`): Step duration in milliseconds.
- `tokens_in` (`int`): Input tokens used.
- `tokens_out` (`int`): Output tokens generated.

**Returns**: Sequence number of the event.

```python
session.step_end(
    outcome="success",
    output_summary="Analysis complete",
    duration_ms=1500,
    tokens_in=100,
    tokens_out=250
)
```

##### `add_fact(fact_id: str, text: str, confidence: float = 1.0, source: Optional[str] = None) -> int`

Add a new fact to the knowledge base.

**Parameters**:
- `fact_id` (`str`): Unique identifier for the fact.
- `text` (`str`): The fact content.
- `confidence` (`float`): Confidence level (0.0 to 1.0).
- `source` (`Optional[str]`): Source of the fact (defaults to current agent).

**Returns**: Sequence number of the event.

```python
session.add_fact(
    fact_id="F1",
    text="The code has a memory leak in function process_data()",
    confidence=0.85,
    source="analyzer"
)
```

##### `modify_fact(fact_id: str, text: Optional[str] = None, confidence: Optional[float] = None, reason: str = "") -> int`

Modify an existing fact.

**Parameters**:
- `fact_id` (`str`): ID of the fact to modify.
- `text` (`Optional[str]`): New text (if updating).
- `confidence` (`Optional[float]`): New confidence (if updating).
- `reason` (`str`): Reason for the modification.

**Returns**: Sequence number of the event.

```python
session.modify_fact(
    fact_id="F1",
    confidence=0.95,
    reason="Confirmed by second analyzer"
)
```

##### `tool_call(tool: str, input_data: Any, output_data: Any, duration_ms: int = 0) -> int`

Record a tool invocation.

**Parameters**:
- `tool` (`str`): Name of the tool.
- `input_data` (`Any`): Input provided to the tool.
- `output_data` (`Any`): Output from the tool.
- `duration_ms` (`int`): Tool execution duration.

**Returns**: Sequence number of the event.

```python
session.tool_call(
    tool="web_search",
    input_data={"query": "Python memory leak detection"},
    output_data={"results": [...]},
    duration_ms=500
)
```

##### `error(category: str, error_type: str, message: str, recoverable: bool = True, suggested_action: str = "") -> int`

Record an error.

**Parameters**:
- `category` (`str`): Error category (e.g., "execution_error", "validation_error").
- `error_type` (`str`): Exception type name.
- `message` (`str`): Error message.
- `recoverable` (`bool`): Whether the workflow can continue.
- `suggested_action` (`str`): Suggested remediation action.

**Returns**: Sequence number of the event.

```python
session.error(
    category="api_error",
    error_type="RateLimitError",
    message="API rate limit exceeded",
    recoverable=True,
    suggested_action="Wait 60 seconds and retry"
)
```

##### `checkpoint(facts_snapshot: Optional[Dict] = None) -> int`

Create a checkpoint for fast replay.

**Parameters**:
- `facts_snapshot` (`Optional[Dict]`): Pre-computed facts snapshot. If not provided, computes current state.

**Returns**: Sequence number of the event.

```python
session.checkpoint()  # Auto-compute snapshot
```

##### `export(path: str) -> None`

Export the chain to a file.

**Parameters**:
- `path` (`str`): Destination file path.

**Raises**:
- `FileNotFoundError`: If the parent directory does not exist.
- `PermissionError`: If write permission is denied.
- `OSError`: If the file cannot be written.

```python
session.export("trace.lctl.json")
```

##### `to_dict() -> Dict[str, Any]`

Export the chain as a dictionary.

**Returns**: Dictionary representation of the chain.

```python
data = session.to_dict()
```

---

### traced_step

```python
from lctl import traced_step
```

Context manager for tracing a step with automatic timing and error handling.

#### Signature

```python
traced_step(
    session: LCTLSession,
    agent: str,
    intent: str,
    input_summary: str = ""
) -> Generator[None, None, None]
```

**Parameters**:
- `session` (`LCTLSession`): The session to record to.
- `agent` (`str`): Agent name.
- `intent` (`str`): Step intent.
- `input_summary` (`str`): Summary of input.

#### Example

```python
from lctl import LCTLSession, traced_step

session = LCTLSession()

with traced_step(session, "analyzer", "analyze", "code.py"):
    result = do_analysis()
    # Duration automatically measured
    # Errors automatically recorded
```

---

## Integrations (lctl.integrations)

LCTL provides native integrations for popular multi-agent frameworks.

---

### LangChain Integration

```python
from lctl.integrations.langchain import (
    LCTLCallbackHandler,
    LCTLChain,
    trace_chain,
    is_available
)
```

#### is_available() -> bool

Check if LangChain is installed.

```python
if is_available():
    print("LangChain integration available")
```

---

#### LCTLCallbackHandler

LangChain callback handler that records events to LCTL.

##### Constructor

```python
LCTLCallbackHandler(
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None
)
```

**Parameters**:
- `chain_id` (`Optional[str]`): Optional chain ID.
- `session` (`Optional[LCTLSession]`): Optional existing session.

**Raises**:
- `ImportError`: If LangChain is not installed.

##### Captured Events

- LLM calls (start/end with token counts)
- Chain execution (start/end)
- Tool invocations
- Agent actions
- Retriever operations
- Errors

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `chain` | `Chain` | The underlying LCTL chain |
| `session` | `LCTLSession` | The LCTL session |

##### Methods

- `export(path: str)`: Export trace to file
- `to_dict() -> Dict[str, Any]`: Get trace as dictionary

##### Example

```python
from lctl.integrations.langchain import LCTLCallbackHandler
from langchain_openai import ChatOpenAI

handler = LCTLCallbackHandler(chain_id="my-chain")
llm = ChatOpenAI()

result = llm.invoke(
    "Hello, how are you?",
    config={"callbacks": [handler]}
)

handler.export("langchain_trace.lctl.json")
```

---

#### LCTLChain

Wrapper that adds LCTL tracing to any LangChain chain.

##### Constructor

```python
LCTLChain(
    chain: Any,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None
)
```

**Parameters**:
- `chain` (`Any`): The LangChain chain to wrap.
- `chain_id` (`Optional[str]`): Optional chain ID.
- `session` (`Optional[LCTLSession]`): Optional existing session.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `chain` | `Chain` | The underlying LCTL chain |
| `session` | `LCTLSession` | The LCTL session |
| `handler` | `LCTLCallbackHandler` | The callback handler |

##### Methods

- `invoke(input, config=None, **kwargs)`: Invoke with tracing
- `ainvoke(input, config=None, **kwargs)`: Async invoke
- `stream(input, config=None, **kwargs)`: Stream with tracing
- `astream(input, config=None, **kwargs)`: Async stream
- `export(path: str)`: Export trace
- `to_dict() -> Dict[str, Any]`: Get trace as dictionary

##### Example

```python
from lctl.integrations.langchain import LCTLChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI()
chain = prompt | llm

traced = LCTLChain(chain, chain_id="topic-explainer")
result = traced.invoke({"topic": "quantum computing"})
traced.export("chain_trace.lctl.json")
```

---

#### trace_chain

Convenience function to wrap a chain with LCTL tracing.

##### Signature

```python
trace_chain(
    chain: Any,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None
) -> LCTLChain
```

**Parameters**:
- `chain` (`Any`): The LangChain chain to trace.
- `chain_id` (`Optional[str]`): Optional chain ID.
- `session` (`Optional[LCTLSession]`): Optional existing session.

**Returns**: `LCTLChain` wrapper.

##### Example

```python
from lctl.integrations.langchain import trace_chain

traced = trace_chain(my_chain, chain_id="my-workflow")
result = traced.invoke(input_data)
traced.export("trace.lctl.json")
```

---

### CrewAI Integration

```python
from lctl.integrations.crewai import (
    LCTLAgent,
    LCTLTask,
    LCTLCrew,
    trace_crew,
    CREWAI_AVAILABLE
)
```

#### LCTLAgent

Wrapper around CrewAI Agent with LCTL tracing.

##### Constructor

```python
LCTLAgent(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[Any]] = None,
    llm: Optional[Any] = None,
    verbose: bool = False,
    allow_delegation: bool = True,
    max_iter: int = 25,
    max_rpm: Optional[int] = None,
    max_execution_time: Optional[int] = None,
    **kwargs
)
```

**Parameters**:
- `role` (`str`): The agent's role description.
- `goal` (`str`): The agent's primary goal.
- `backstory` (`str`): Background context.
- `tools` (`Optional[List]`): Available tools.
- `llm` (`Optional[Any]`): Language model.
- `verbose` (`bool`): Enable verbose output.
- `allow_delegation` (`bool`): Allow task delegation.
- `max_iter` (`int`): Maximum iterations.
- `max_rpm` (`Optional[int]`): Rate limit.
- `max_execution_time` (`Optional[int]`): Timeout in seconds.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `agent` | `Agent` | Underlying CrewAI Agent |
| `metadata` | `Dict[str, Any]` | Agent metadata for tracing |

##### Example

```python
from lctl.integrations.crewai import LCTLAgent

researcher = LCTLAgent(
    role="Senior Researcher",
    goal="Research topics thoroughly",
    backstory="Expert researcher with years of experience",
    verbose=True
)
```

---

#### LCTLTask

Wrapper around CrewAI Task with LCTL tracing.

##### Constructor

```python
LCTLTask(
    description: str,
    expected_output: str,
    agent: Optional[Union[LCTLAgent, Agent]] = None,
    tools: Optional[List[Any]] = None,
    async_execution: bool = False,
    context: Optional[List[LCTLTask]] = None,
    **kwargs
)
```

**Parameters**:
- `description` (`str`): Task description.
- `expected_output` (`str`): Expected output format.
- `agent` (`Optional`): Assigned agent.
- `tools` (`Optional[List]`): Task-specific tools.
- `async_execution` (`bool`): Enable async.
- `context` (`Optional[List]`): Context from other tasks.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `task` | `Task` | Underlying CrewAI Task |
| `metadata` | `Dict[str, Any]` | Task metadata |

---

#### LCTLCrew

Wrapper around CrewAI Crew with full LCTL tracing.

##### Constructor

```python
LCTLCrew(
    agents: List[Union[LCTLAgent, Agent]],
    tasks: List[Union[LCTLTask, Task]],
    process: Optional[str] = None,
    verbose: bool = False,
    manager_llm: Optional[Any] = None,
    manager_agent: Optional[Union[LCTLAgent, Agent]] = None,
    chain_id: Optional[str] = None,
    **kwargs
)
```

**Parameters**:
- `agents` (`List`): List of agents.
- `tasks` (`List`): List of tasks.
- `process` (`Optional[str]`): "sequential" or "hierarchical".
- `verbose` (`bool`): Enable verbose output.
- `manager_llm` (`Optional`): LLM for hierarchical manager.
- `manager_agent` (`Optional`): Custom manager agent.
- `chain_id` (`Optional[str]`): LCTL chain ID.

##### Captured Events

- Crew kickoff and completion
- Agent steps
- Task execution
- Delegation events
- Tool usage
- Errors

##### Methods

- `kickoff(inputs: Optional[Dict] = None) -> Any`: Execute crew
- `kickoff_async(inputs: Optional[Dict] = None) -> Any`: Async execute
- `export_trace(path: str)`: Export trace
- `get_trace() -> Dict[str, Any]`: Get trace as dictionary

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session` | `LCTLSession` | The LCTL session |
| `crew` | `Crew` | Underlying CrewAI Crew |

##### Example

```python
from lctl.integrations.crewai import LCTLAgent, LCTLTask, LCTLCrew

researcher = LCTLAgent(
    role="Researcher",
    goal="Research topics",
    backstory="Expert researcher"
)

task = LCTLTask(
    description="Research quantum computing",
    expected_output="Comprehensive report",
    agent=researcher
)

crew = LCTLCrew(
    agents=[researcher],
    tasks=[task],
    verbose=True,
    chain_id="research-crew"
)

result = crew.kickoff()
crew.export_trace("research_trace.lctl.json")
```

---

#### trace_crew

Wrap an existing CrewAI Crew with LCTL tracing.

##### Signature

```python
trace_crew(
    crew: Crew,
    chain_id: Optional[str] = None,
    verbose: bool = False
) -> LCTLCrew
```

**Parameters**:
- `crew` (`Crew`): Existing CrewAI Crew.
- `chain_id` (`Optional[str]`): Chain ID.
- `verbose` (`bool`): Enable verbose output.

**Returns**: `LCTLCrew` wrapper.

##### Example

```python
from crewai import Crew
from lctl.integrations.crewai import trace_crew

crew = Crew(agents=[...], tasks=[...])
traced = trace_crew(crew, chain_id="my-crew")
result = traced.kickoff()
traced.export_trace("trace.lctl.json")
```

---

### AutoGen Integration

```python
from lctl.integrations.autogen import (
    LCTLAutogenCallback,
    LCTLConversableAgent,
    LCTLGroupChatManager,
    trace_agent,
    trace_group_chat,
    is_available
)
```

#### LCTLAutogenCallback

AutoGen callback handler that records events to LCTL.

##### Constructor

```python
LCTLAutogenCallback(
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None
)
```

**Parameters**:
- `chain_id` (`Optional[str]`): Optional chain ID.
- `session` (`Optional[LCTLSession]`): Optional existing session.

##### Captured Events

- Agent-to-agent messages
- Tool/function calls and responses
- GroupChat conversations
- Nested conversation tracking
- Errors

##### Methods

- `attach(agent: ConversableAgent)`: Attach tracing to agent
- `attach_group_chat(group_chat, manager=None)`: Attach to group chat
- `start_nested_chat(parent_agent, description="")`: Record nested chat start
- `end_nested_chat(result_summary="", outcome="success")`: Record nested chat end
- `record_tool_result(tool_name, result, duration_ms=0, agent=None)`: Manual tool recording
- `record_error(error, agent=None, recoverable=True)`: Record error
- `export(path: str)`: Export trace
- `to_dict() -> Dict[str, Any]`: Get trace as dictionary

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `chain` | `Chain` | The underlying LCTL chain |
| `session` | `LCTLSession` | The LCTL session |

##### Example

```python
from lctl.integrations.autogen import LCTLAutogenCallback
from autogen import ConversableAgent

callback = LCTLAutogenCallback(chain_id="my-conversation")

agent1 = ConversableAgent(name="assistant", ...)
agent2 = ConversableAgent(name="user_proxy", ...)

callback.attach(agent1)
callback.attach(agent2)

agent1.initiate_chat(agent2, message="Hello")
callback.export("autogen_trace.lctl.json")
```

---

#### LCTLConversableAgent

Wrapper around AutoGen ConversableAgent with built-in tracing.

##### Constructor

```python
LCTLConversableAgent(
    name: str,
    system_message: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    chain_id: Optional[str] = None,
    **kwargs
)
```

**Parameters**:
- `name` (`str`): Agent name.
- `system_message` (`Optional[str]`): System message.
- `llm_config` (`Optional[Dict]`): LLM configuration.
- `chain_id` (`Optional[str]`): LCTL chain ID.

##### Methods

- `initiate_chat(recipient, message=None, **kwargs)`: Start traced chat
- `export_trace(path: str)`: Export trace
- `get_trace() -> Dict[str, Any]`: Get trace

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `agent` | `ConversableAgent` | Underlying agent |
| `session` | `LCTLSession` | LCTL session |
| `callback` | `LCTLAutogenCallback` | Callback handler |

---

#### LCTLGroupChatManager

Wrapper around GroupChatManager with built-in tracing.

##### Constructor

```python
LCTLGroupChatManager(
    groupchat: GroupChat,
    name: str = "chat_manager",
    chain_id: Optional[str] = None,
    **kwargs
)
```

**Parameters**:
- `groupchat` (`GroupChat`): The GroupChat to manage.
- `name` (`str`): Manager name.
- `chain_id` (`Optional[str]`): LCTL chain ID.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `manager` | `GroupChatManager` | Underlying manager |
| `groupchat` | `GroupChat` | The group chat |
| `session` | `LCTLSession` | LCTL session |
| `callback` | `LCTLAutogenCallback` | Callback handler |

##### Example

```python
from autogen import GroupChat
from lctl.integrations.autogen import LCTLGroupChatManager

group_chat = GroupChat(agents=[agent1, agent2], messages=[], max_round=10)
manager = LCTLGroupChatManager(
    groupchat=group_chat,
    chain_id="group-discussion"
)

agent1.initiate_chat(manager.manager, message="Let's discuss")
manager.export_trace("groupchat.lctl.json")
```

---

#### trace_agent

Attach LCTL tracing to an existing AutoGen agent.

##### Signature

```python
trace_agent(
    agent: ConversableAgent,
    chain_id: Optional[str] = None
) -> LCTLAutogenCallback
```

**Returns**: `LCTLAutogenCallback` for exporting traces.

##### Example

```python
from autogen import ConversableAgent
from lctl.integrations.autogen import trace_agent

agent = ConversableAgent(name="assistant", ...)
callback = trace_agent(agent, chain_id="my-agent")

agent.initiate_chat(other_agent, message="Hello")
callback.export("trace.lctl.json")
```

---

#### trace_group_chat

Attach LCTL tracing to an existing GroupChat.

##### Signature

```python
trace_group_chat(
    group_chat: GroupChat,
    manager: Optional[GroupChatManager] = None,
    chain_id: Optional[str] = None
) -> LCTLAutogenCallback
```

**Parameters**:
- `group_chat` (`GroupChat`): The GroupChat to trace.
- `manager` (`Optional`): Optional manager to also trace.
- `chain_id` (`Optional[str]`): Chain ID.

**Returns**: `LCTLAutogenCallback` for exporting traces.

---

### OpenAI Agents Integration

```python
from lctl.integrations.openai_agents import (
    LCTLOpenAIAgentTracer,
    LCTLRunHooks,
    TracedAgent,
    trace_agent,
    is_available
)
```

#### LCTLOpenAIAgentTracer

Main tracer class for OpenAI Agents SDK integration.

##### Constructor

```python
LCTLOpenAIAgentTracer(
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False
)
```

**Parameters**:
- `chain_id` (`Optional[str]`): Optional chain ID.
- `session` (`Optional[LCTLSession]`): Optional existing session.
- `verbose` (`bool`): Enable verbose output.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session` | `LCTLSession` | The LCTL session |
| `chain` | `Chain` | The underlying chain |
| `run_config` | `Dict[str, Any]` | Config for Runner.run() |

##### Methods

###### `create_hooks() -> LCTLRunHooks`

Create run hooks for the tracer.

**Returns**: `LCTLRunHooks` instance.

###### `create_tracing_processor() -> LCTLTracingProcessor`

Create a tracing processor.

**Returns**: `LCTLTracingProcessor` instance.

###### `trace_agent_run(agent_name: str, input_summary: str = "") -> AgentRunContext`

Context manager for manually tracing an agent run.

**Parameters**:
- `agent_name` (`str`): Name of the agent.
- `input_summary` (`str`): Summary of input.

**Returns**: `AgentRunContext` context manager.

###### `record_tool_call(tool, input_data, output_data, duration_ms=0)`

Manually record a tool call.

###### `record_handoff(from_agent, to_agent)`

Manually record an agent handoff.

###### `record_error(agent_name, error, recoverable=False)`

Manually record an error.

###### `export(path: str)`

Export trace to file.

###### `to_dict() -> Dict[str, Any]`

Get trace as dictionary.

##### Example

```python
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer
from agents import Agent, Runner

tracer = LCTLOpenAIAgentTracer(chain_id="my-agent")

agent = Agent(name="assistant", instructions="You are helpful")

# Use with Runner
result = await Runner.run(
    agent,
    input="Hello",
    run_config=tracer.run_config
)

tracer.export("openai_agent_trace.lctl.json")
```

---

#### LCTLRunHooks

Run hooks implementation for automatic event capture.

##### Constructor

```python
LCTLRunHooks(session: LCTLSession, verbose: bool = False)
```

##### Captured Events

- Agent run start/end
- Tool calls
- Handoffs between agents
- Streaming events
- Errors

##### Hook Methods

All hooks are async and called automatically by the SDK:

- `on_agent_start(context, agent)`
- `on_agent_end(context, agent, output)`
- `on_tool_start(context, agent, tool, input_data)`
- `on_tool_end(context, agent, tool, output)`
- `on_handoff(context, from_agent, to_agent)`
- `on_error(context, agent, error)`

---

#### TracedAgent

Wrapper around OpenAI Agent with LCTL tracing.

##### Constructor

```python
TracedAgent(
    agent: Agent,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False
)
```

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `agent` | `Agent` | Underlying OpenAI Agent |
| `tracer` | `LCTLOpenAIAgentTracer` | The tracer |
| `run_config` | `Dict[str, Any]` | Config for Runner.run() |

##### Methods

- `export(path: str)`: Export trace
- `to_dict() -> Dict[str, Any]`: Get trace

---

#### trace_agent

Wrap an OpenAI Agent with LCTL tracing.

##### Signature

```python
trace_agent(
    agent: Agent,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False
) -> TracedAgent
```

**Returns**: `TracedAgent` wrapper.

##### Example

```python
from agents import Agent
from lctl.integrations.openai_agents import trace_agent

agent = Agent(name="assistant", instructions="You are helpful")
traced = trace_agent(agent, chain_id="my-agent")

# Access tracer for LCTL functionality
traced.tracer.export("trace.lctl.json")
```

---

#### AgentRunContext

Context manager for tracing individual agent runs.

##### Constructor

```python
AgentRunContext(
    session: LCTLSession,
    agent_name: str,
    input_summary: str = "",
    verbose: bool = False
)
```

##### Methods

- `set_output(output_summary: str)`: Set output summary
- `set_usage(tokens_in=0, tokens_out=0)`: Set token usage
- `record_tool_call(tool, input_data, output_data, duration_ms=0)`: Record tool

##### Example

```python
with tracer.trace_agent_run("my_agent", "Process data") as ctx:
    result = agent.process(data)
    ctx.set_output(str(result))
    ctx.set_usage(tokens_in=100, tokens_out=200)
```

---

### DSPy Integration

```python
from lctl.integrations.dspy import (
    LCTLDSPyCallback,
    TracedDSPyModule,
    LCTLDSPyTeleprompter,
    trace_module,
    is_available
)
```

#### `is_available() -> bool`

Check if DSPy is installed.

#### `LCTLDSPyCallback`

Callback handler for DSPy operations.

##### Constructor

```python
LCTLDSPyCallback(
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False
)
```

##### Traced Events
- Module execution (inputs/outputs)
- LLM calls (prompts/responses/tokens)
- Optimization/Compile steps (iterations/scores)

#### `trace_module`

Decorator/wrapper to trace a DSPy module.

```python
@trace_module(callback)
class MyModule(dspy.Module):
    ...

# Or wrap instance
traced_module = trace_module(module_instance, callback)
```

#### `LCTLDSPyTeleprompter`

Wrapper for teleprompters to trace optimization.

```python
tracer = LCTLDSPyTeleprompter(teleprompter, callback)
optimized_program = tracer.compile(program, trainset=trainset)
```

---

### LlamaIndex Integration

```python
from lctl.integrations.llamaindex import (
    LCTLLlamaIndexCallback,
    trace_query_engine,
    trace_chat_engine,
    is_available
)
```

#### `is_available() -> bool`

Check if LlamaIndex is installed.

#### `LCTLLlamaIndexCallback`

Callback handler for LlamaIndex.

##### Constructor

```python
callback = LCTLLlamaIndexCallback(
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None
)
```

##### Usage

```python
from llama_index.core import Settings

Settings.callback_manager = callback.get_callback_manager()
```

#### `trace_query_engine`

Wrap a query engine for simplified tracing.

```python
traced_engine = trace_query_engine(query_engine, chain_id="my-query")
response = traced_engine.query("What is LCTL?")
```

#### `trace_chat_engine`

Wrap a chat engine for simplified tracing.

```python
traced_chat = trace_chat_engine(chat_engine, chain_id="my-chat")
response = traced_chat.chat("Hello there")
```

---

## Dashboard (lctl.dashboard)

Web-based visualization for multi-agent workflows.

```python
from lctl.dashboard import create_app, run_dashboard
```

### create_app

Create and configure the FastAPI application.

#### Signature

```python
create_app(working_dir: Optional[Path] = None) -> FastAPI
```

**Parameters**:
- `working_dir` (`Optional[Path]`): Directory to search for `.lctl.json` files. Defaults to current working directory.

**Returns**: Configured FastAPI application.

#### Example

```python
from pathlib import Path
from lctl.dashboard import create_app

app = create_app(working_dir=Path("/traces"))

# Use with any ASGI server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

### run_dashboard

Run the dashboard server directly.

#### Signature

```python
run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8080,
    working_dir: Optional[Path] = None,
    reload: bool = False
) -> None
```

**Parameters**:
- `host` (`str`): Host to bind to (default: "0.0.0.0").
- `port` (`int`): Port to bind to (default: 8080).
- `working_dir` (`Optional[Path]`): Directory for chain files.
- `reload` (`bool`): Enable auto-reload for development.

#### Example

```python
from pathlib import Path
from lctl.dashboard import run_dashboard

run_dashboard(
    host="localhost",
    port=8000,
    working_dir=Path("./traces"),
    reload=True
)
```

---

### API Endpoints

The dashboard exposes the following REST API endpoints:

#### GET /

Serve the main dashboard HTML interface.

**Response**: HTML page

---

#### GET /api/chains

List available `.lctl.json` files in the working directory.

**Response**:
```json
{
  "chains": [
    {
      "filename": "trace.lctl.json",
      "path": "/path/to/trace.lctl.json",
      "id": "my-chain",
      "version": "4.0",
      "event_count": 42
    }
  ],
  "working_dir": "/path/to/working/dir"
}
```

---

#### GET /api/chain/{filename}

Load and return chain data with full analysis.

**Parameters**:
- `filename` (path): The chain file name

**Response**:
```json
{
  "chain": {
    "id": "my-chain",
    "version": "4.0",
    "filename": "trace.lctl.json"
  },
  "events": [...],
  "agents": ["agent1", "agent2"],
  "state": {
    "facts": {...},
    "errors": [...],
    "metrics": {...}
  },
  "analysis": {
    "bottlenecks": [...],
    "confidence_timeline": {...},
    "trace": [...]
  }
}
```

**Error Responses**:
- `400`: Invalid filename or chain file
- `404`: Chain file not found
- `500`: Server error

---

#### POST /api/replay

Replay chain to a specific sequence number for time-travel debugging.

**Request Body**:
```json
{
  "filename": "trace.lctl.json",
  "target_seq": 10
}
```

**Response**:
```json
{
  "target_seq": 10,
  "events": [...],
  "state": {
    "facts": {...},
    "errors": [...],
    "metrics": {...},
    "current_agent": "analyzer",
    "current_step": 10
  }
}
```

**Error Responses**:
- `400`: Invalid filename, chain file, or target_seq
- `404`: Chain file not found

---

#### GET /api/health

Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "version": "4.0.0"
}
```

---

## Global Functions

These functions are available at the `lctl` package level.

### auto_instrument

```python
import lctl
session = lctl.auto_instrument()
```

Enable automatic instrumentation for supported frameworks.

**Returns**: `LCTLSession` - The global session.

### get_session

```python
session = lctl.get_session()
```

Get the global LCTL session. Creates one if it doesn't exist.

**Returns**: `LCTLSession` - The global session.

### traced

```python
@lctl.traced("analyzer", "analyze")
def analyze_code(code):
    return findings
```

Decorator to trace a function as an agent step.

**Parameters**:
- `agent` (`str`): Agent name.
- `intent` (`str`): Step intent (default: "execute").

### export

```python
lctl.export("trace.lctl.json")
```

Export the global session to file.

**Parameters**:
- `path` (`str`): Destination file path.

---

## CLI Commands

LCTL also provides a command-line interface:

```bash
lctl replay chain.lctl.json      # Time-travel debugging
lctl stats chain.lctl.json       # Performance statistics
lctl bottleneck chain.lctl.json  # Find slow steps
lctl trace chain.lctl.json       # Execution flow visualization
lctl diff v1.json v2.json        # Compare two chains
```

---

## File Formats

LCTL supports both JSON and YAML formats:

### JSON Format

```json
{
  "lctl": "4.0",
  "chain": {
    "id": "my-chain"
  },
  "events": [
    {
      "seq": 1,
      "type": "step_start",
      "timestamp": "2026-01-11T10:30:00+00:00",
      "agent": "analyzer",
      "data": {
        "intent": "analyze",
        "input_summary": "code.py"
      }
    }
  ]
}
```

### YAML Format

```yaml
lctl: "4.0"
chain:
  id: my-chain
events:
  - seq: 1
    type: step_start
    timestamp: "2026-01-11T10:30:00+00:00"
    agent: analyzer
    data:
      intent: analyze
      input_summary: code.py
```

---

## Error Handling

All LCTL components use standard Python exceptions:

| Exception | When Raised |
|-----------|-------------|
| `FileNotFoundError` | Chain file does not exist |
| `ValueError` | Invalid file format, empty file, missing fields |
| `PermissionError` | Read/write permission denied |
| `ImportError` | Framework integration not installed |

Integration-specific exceptions:
- `CrewAINotAvailableError`: CrewAI not installed
- `AutogenNotAvailableError`: AutoGen not installed
- `OpenAIAgentsNotAvailableError`: OpenAI Agents SDK not installed

---

## Best Practices

1. **Use chain IDs**: Always provide meaningful chain IDs for easier identification
2. **Add facts liberally**: Facts provide valuable context for debugging
3. **Include confidence levels**: Track certainty in agent conclusions
4. **Export traces**: Save traces for post-hoc analysis
5. **Use checkpoints**: Create checkpoints for faster replay of long chains
6. **Handle errors gracefully**: LCTL automatically captures exceptions in context managers

---

*This documentation was generated for LCTL v4.0.0*
