# LLM Context Transfer Language (LCTL) v4.0

A tool-first observability protocol enabling time-travel debugging for multi-agent LLM workflows.

## Philosophy

> **"Don't pitch the protocol. Pitch the tool. The protocol is an implementation detail."**

LCTL v4.0 is designed around a killer feature: **Time-Travel Replay**. The protocol exists to enable tools that let developers step back through agent execution, modify state, and replay to understand "what if?"

## Core Design Principles

1. **Tool-First**: Protocol serves tooling, not the other way around
2. **Zero-Config**: Auto-instrumentation, not manual annotation
3. **One-Page Core**: Essential spec fits on one page
4. **Event-Sourced**: Store events, derive any view
5. **Pull Model**: Agents query what they need

---

## Part 1: Core Protocol (The One-Pager)

### Minimal Schema

```yaml
lctl: "4.0"

# Required: Chain identification
chain:
  id: "uuid-or-descriptive-id"

# Required: Event stream (append-only)
events:
  - seq: 1
    type: step_start | step_end | fact_added | fact_modified | tool_call | error | checkpoint
    timestamp: "2024-01-15T10:30:00Z"
    agent: "agent-name"
    data: {}  # Type-specific payload

# Optional: Materialized state (for fast replay)
state:
  facts: {}
  checkpoint_seq: 0
```

### Event Types

| Type | Purpose | Data Payload |
|------|---------|--------------|
| `step_start` | Agent begins work | `{intent, input_summary}` |
| `step_end` | Agent completes | `{outcome, output_summary, duration_ms, tokens}` |
| `fact_added` | New fact registered | `{id, text, confidence, source}` |
| `fact_modified` | Fact updated | `{id, text, confidence, reason}` |
| `tool_call` | Tool invocation | `{tool, input, output, duration_ms}` |
| `error` | Failure occurred | `{category, type, message, recoverable}` |
| `checkpoint` | State snapshot | `{state_hash, facts_snapshot}` |

### That's It

The core protocol is just **chain ID + event stream**. Everything else (confidence systems, handoffs, visualizations) is derived from this event log.

---

## Part 2: Event Sourcing Architecture

### Why Event Sourcing?

| Capability | How Events Enable It |
|------------|---------------------|
| **Time-Travel** | Replay events to any seq number |
| **Debugging** | Full history of every state change |
| **Auditing** | Immutable, tamper-evident log |
| **Analytics** | Aggregate events for metrics |
| **Recovery** | Replay from last checkpoint |

### Event Stream Example

```yaml
lctl: "4.0"
chain:
  id: "security-review-001"

events:
  - seq: 1
    type: step_start
    timestamp: "2024-01-15T10:30:00Z"
    agent: "code-analyzer"
    data:
      intent: "analyze"
      input_summary: "UserController.py - 200 lines"

  - seq: 2
    type: fact_added
    timestamp: "2024-01-15T10:30:15Z"
    agent: "code-analyzer"
    data:
      id: "F1"
      text: "SQL query constructed from user input at line 45"
      confidence: 0.85
      source: "static-analysis"

  - seq: 3
    type: tool_call
    timestamp: "2024-01-15T10:30:16Z"
    agent: "code-analyzer"
    data:
      tool: "grep"
      input: {pattern: "execute.*user", file: "UserController.py"}
      output: {matches: 3, lines: [45, 67, 89]}
      duration_ms: 50

  - seq: 4
    type: step_end
    timestamp: "2024-01-15T10:30:30Z"
    agent: "code-analyzer"
    data:
      outcome: "success"
      output_summary: "Found 1 potential SQL injection"
      duration_ms: 30000
      tokens: {input: 500, output: 200}

  - seq: 5
    type: checkpoint
    timestamp: "2024-01-15T10:30:31Z"
    agent: "system"
    data:
      state_hash: "abc123"
      facts_snapshot:
        F1: {text: "SQL query constructed...", confidence: 0.85}

  - seq: 6
    type: step_start
    timestamp: "2024-01-15T10:30:32Z"
    agent: "security-reviewer"
    data:
      intent: "assess"
      input_summary: "Review F1 for exploitability"
```

### Deriving Views from Events

```python
# Replay to get state at any point
def replay_to(events, target_seq):
    state = {"facts": {}, "agents": [], "current_step": None}
    for event in events:
        if event["seq"] > target_seq:
            break
        apply_event(state, event)
    return state

# Derive trace view (for visualization)
def events_to_trace(events):
    return [e for e in events if e["type"] in ("step_start", "step_end")]

# Derive metrics
def events_to_metrics(events):
    return {
        "total_duration": sum(e["data"].get("duration_ms", 0) for e in events),
        "total_tokens": sum(e["data"].get("tokens", {}).get("input", 0) +
                          e["data"].get("tokens", {}).get("output", 0) for e in events),
        "error_count": len([e for e in events if e["type"] == "error"])
    }
```

---

## Part 3: Confidence System (Decay + Consensus)

Based on A/B testing, LCTL v4.0 uses a **tiered confidence system**:

### Tier 1: Decay-Based (All Facts)

Every fact's confidence decays as it propagates:

```yaml
confidence:
  method: decay
  decay_rate: 0.95  # Per hop

# Example: Fact with 0.90 confidence after 5 hops
# 0.90 * 0.95^5 = 0.70
```

### Tier 2: Consensus (High-Stakes)

For critical facts, require multiple agent confirmation:

```yaml
confidence:
  method: consensus
  assessments:
    - agent: "code-analyzer"
      confidence: 0.85
    - agent: "security-reviewer"
      confidence: 0.92
  consensus_score: 0.88  # Weighted average
  conflict_level: 0.0    # Agreement
```

### Confidence Events

```yaml
- seq: 10
  type: fact_modified
  agent: "security-reviewer"
  data:
    id: "F1"
    text: "CONFIRMED: SQL injection vulnerability"
    confidence: 0.95
    reason: "verified_by_consensus"
    consensus:
      assessors: ["code-analyzer", "security-reviewer"]
      agreement: 1.0
```

### Automatic Thresholds

| Confidence | Action |
|------------|--------|
| ≥ 0.80 | Proceed automatically |
| 0.60-0.79 | Warn, but proceed |
| 0.40-0.59 | Request verification |
| < 0.40 | Block, require human review |

---

## Part 4: Handoff System (Query-Based + Contracts)

Based on A/B testing, LCTL v4.0 uses **query-based fact stores** with **contract validation**.

### Fact Store (Query-Based)

Instead of explicitly passing facts, agents query a shared store:

```yaml
# Agent queries what it needs
handoff:
  fact_store: "memory://chain-001/facts"

  # Suggested queries (hints, not requirements)
  suggested_queries:
    - "SELECT * WHERE confidence >= 0.8"
    - "SELECT * WHERE source = 'security-reviewer'"
    - "SELECT * WHERE tag = 'critical'"
```

### Agent Query Example

```python
# Receiving agent queries the fact store
facts = fact_store.query("""
    SELECT * FROM facts
    WHERE confidence >= 0.7
    AND relevance('SQL injection') > 0.5
    ORDER BY confidence DESC
    LIMIT 10
""")
```

### Contracts (Type Safety)

Define expected inputs/outputs per agent type:

```yaml
contracts:
  code-analyzer:
    outputs:
      - type: fact
        required_fields: [id, text, confidence, source]
        confidence_min: 0.5

  security-reviewer:
    inputs:
      - type: fact
        required_fields: [id, text]
    outputs:
      - type: fact
        required_fields: [id, text, confidence, severity]
        severity_values: [low, medium, high, critical]
```

### Contract Validation Events

```yaml
- seq: 15
  type: contract_validation
  agent: "system"
  data:
    contract: "security-reviewer"
    direction: "output"
    status: "passed"
    warnings: []
```

---

## Part 5: Agent Type Definitions

LCTL v4.0 standardizes agent type vocabulary:

### Standard Agent Types

```yaml
agent_types:
  executor:
    capabilities: [code_execution, file_access, tool_use]
    typical_tools: [bash, code_interpreter]

  analyzer:
    capabilities: [pattern_detection, static_analysis]
    typical_tools: [grep, ast_parser]

  reviewer:
    capabilities: [assessment, approval, critique]
    outputs: [confidence_score, severity_rating]

  planner:
    capabilities: [decomposition, prioritization, scheduling]
    outputs: [task_list, dependency_graph]

  researcher:
    capabilities: [search, synthesis, citation]
    typical_tools: [web_search, document_reader]
```

### Agent Profile in Events

```yaml
- seq: 1
  type: step_start
  agent: "security-reviewer"
  data:
    agent_profile:
      type: reviewer
      capabilities: [assessment, approval]
      model: "claude-3-opus"
      reliability_score: 0.92  # Historical accuracy
```

---

## Part 6: Error Taxonomy

Standardized error categories for intelligent recovery:

```yaml
error_taxonomy:
  input_error:
    - invalid_format: "Input doesn't match expected schema"
    - missing_context: "Required context not provided"
    - ambiguous_request: "Cannot determine intent"

  execution_error:
    - tool_failure: "Tool invocation failed"
    - timeout: "Operation exceeded time limit"
    - resource_exhausted: "Token/memory limit reached"
    - rate_limited: "API rate limit hit"

  output_error:
    - generation_failed: "LLM produced invalid output"
    - validation_failed: "Output doesn't meet constraints"
    - unsafe_content: "Output flagged by safety filters"

  orchestration_error:
    - routing_failed: "Could not determine next agent"
    - delegation_rejected: "Target agent refused task"
    - cycle_detected: "Infinite loop detected"
    - chain_depth_exceeded: "Maximum chain depth reached"
```

### Error Events

```yaml
- seq: 20
  type: error
  agent: "code-executor"
  data:
    category: "execution_error"
    type: "timeout"
    message: "Code execution exceeded 30s limit"
    recoverable: true
    suggested_action: "retry_with_timeout_increase"
    context:
      timeout_configured: 30000
      elapsed_ms: 30500
```

---

## Part 7: Tool Integration Patterns

### Auto-Instrumentation

Zero-config integration for major frameworks:

```python
# One line to enable LCTL
import lctl
lctl.auto_instrument()

# That's it. All agent calls are now traced.
```

### Framework-Specific Wrappers

```python
# Claude Code Task Tool
from lctl.claude import traced_task
result = traced_task(subagent_type="explore", prompt="Find auth files")

# LangChain
from lctl.langchain import LCTLCallbackHandler
chain.invoke(input, config={"callbacks": [LCTLCallbackHandler()]})

# CrewAI
from lctl.crewai import LCTLCrew
crew = LCTLCrew(agents=[...], tasks=[...])

# AutoGen
from lctl.autogen import enable_lctl
enable_lctl()  # Patches all agents
```

### Custom Orchestrator

```python
from lctl import LCTLSession

with LCTLSession() as session:
    # All operations within context are traced
    result1 = llm.complete("Analyze requirements")
    session.add_fact("F1", "Requirements analyzed", confidence=0.9)

    result2 = llm.complete(f"Generate code for: {result1}")
    session.add_fact("F2", "Code generated", confidence=0.85)

# Export trace
session.export("workflow.lctl.json")
```

---

## Part 8: CLI Tools (The Real Product)

### Time-Travel Replay (Killer Feature)

```bash
# Replay entire chain
$ lctl replay chain.lctl.json
Replaying 25 events...
[1] code-analyzer: step_start (analyze)
[2] code-analyzer: fact_added F1 (confidence: 0.85)
...
Replay complete. Final state has 5 facts.

# Replay to specific point
$ lctl replay --to-seq 10 chain.lctl.json
Replaying to event 10...
State at seq 10: 2 facts, agent=code-analyzer

# Compare two replays
$ lctl diff chain-v1.lctl.json chain-v2.lctl.json
Events diverged at seq 8:
  v1: fact_added F3 (confidence: 0.70)
  v2: fact_added F3 (confidence: 0.85) [DIFFERENT]
```

### Visual Debugger

```bash
# Launch web-based debugger
$ lctl debug chain.lctl.json
Starting LCTL Debugger at http://localhost:8080
- Chain: security-review-001
- Events: 25
- Duration: 45.2s
- Agents: 3

# Features:
# - Interactive flowchart of agent execution
# - Click any node to see state at that point
# - Confidence timeline visualization
# - Cost breakdown per agent
```

### Analytics

```bash
# Performance summary
$ lctl stats chain.lctl.json
Chain: security-review-001
Duration: 45.2s
Tokens: 2,340 (in: 1,500 / out: 840)
Cost: $0.047
Agents: 3
Errors: 0
Avg confidence: 0.87

# Bottleneck analysis
$ lctl bottleneck chain.lctl.json
Slowest steps:
1. security-reviewer (step 6): 22.5s (50%)
2. code-analyzer (step 1): 15.0s (33%)
3. fix-implementer (step 11): 7.7s (17%)

Recommendation: security-reviewer is bottleneck. Consider parallelization.
```

### Validation

```bash
# Validate against contracts
$ lctl validate chain.lctl.json --contracts ./contracts/
Validating 25 events against 3 contracts...
✓ code-analyzer outputs valid
✓ security-reviewer inputs valid
✓ security-reviewer outputs valid
✓ fix-implementer inputs valid

All contracts satisfied.

# Check confidence degradation
$ lctl confidence chain.lctl.json
Fact confidence timeline:
F1: 0.85 → 0.85 → 0.95 (verified)
F2: 0.90 → 0.86 → 0.81 (decayed)
F3: 0.70 → BLOCKED (below threshold)

Warning: F3 blocked at step 8 due to low confidence.
```

---

## Part 9: Streaming Support

For real-time applications:

```yaml
lctl: "4.0"
streaming:
  enabled: true
  protocol: "websocket"  # or "sse"

events:
  - seq: 1
    type: stream_start
    data:
      expected_chunks: "unknown"

  - seq: 2
    type: stream_chunk
    data:
      index: 0
      content: "Based on my analysis..."
      tokens: 5

  - seq: 3
    type: stream_chunk
    data:
      index: 1
      content: " the vulnerability is..."
      tokens: 4

  - seq: 4
    type: stream_end
    data:
      total_chunks: 47
      total_tokens: 203
      duration_ms: 2340
```

---

## Part 10: Multi-Model Coordination

Track multiple LLMs in single workflow:

```yaml
events:
  - seq: 5
    type: step_start
    agent: "planner"
    data:
      model:
        id: "gpt-4"
        role: "planning"

  - seq: 10
    type: step_start
    agent: "executor"
    data:
      model:
        id: "claude-3-opus"
        role: "execution"

  - seq: 15
    type: model_routing
    agent: "orchestrator"
    data:
      decision: "route_to_opus"
      reason: "code_generation_task"
      alternatives_considered: ["gpt-4", "claude-3-sonnet"]
      confidence: 0.85
```

---

## Migration from v3.0

### What Changed

| v3.0 | v4.0 | Reason |
|------|------|--------|
| `trace` array | `events` stream | Event sourcing enables replay |
| `facts` object | Facts as events | Immutable history |
| `handoff` object | Query-based store | Pull model more flexible |
| Manual instrumentation | Auto-instrumentation | Zero-config adoption |
| Protocol-first | Tool-first | Adoption driven by tooling |

### Conversion

```bash
# Convert v3.0 to v4.0
$ lctl convert --from v3 --to v4 old-trace.lctl.json > new-events.lctl.json
```

---

## Version History

- **4.0**: Tool-first redesign, event sourcing, time-travel replay, auto-instrumentation
- **3.0**: Observability focus, chain tracking, confidence propagation
- **2.0**: Compression + artifacts (abandoned)
- **1.0**: Initial compression format (abandoned)
