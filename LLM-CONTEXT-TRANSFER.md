# LLM Context Transfer Language (LCTL) v3.0

An observability and tracing protocol for multi-agent LLM workflows - enabling debuggable "LLM telepathy."

## Purpose

When multiple LLM agents collaborate in chains or parallel workflows, debugging becomes difficult:
- "Why did agent 3 make that decision?"
- "What context did it actually receive?"
- "Where did information get lost in the chain?"

LCTL provides a **metadata layer** for agent-to-agent communication that enables:
- Chain provenance tracking
- Confidence degradation visibility
- Structured audit trails
- Workflow debugging

**Note**: LCTL v3.0 is NOT about compression. Full natural language context is still passed. LCTL adds structured metadata for observability.

## Changes in v3.0

- **Pivoted from compression to observability**
- **Chain tracing**: Track information flow across agent hops
- **Confidence propagation**: See where certainty degrades
- **Fact registry**: Number and track critical facts through chains
- **Execution logs**: Structured record of agent decisions
- **Debugging-first**: Designed for "why did this happen?" questions

## When to Use LCTL

**Use LCTL when:**
- Multi-agent chains (agent → agent → agent)
- Parallel agent workflows that merge results
- Debugging complex agent failures
- Audit trails required for agent decisions
- Long-running workflows where context may degrade

**Don't use LCTL when:**
- Single agent interactions
- Simple request/response patterns
- Token cost is the primary concern (LCTL adds overhead)

## Format Specification

### Core Structure

```yaml
lctl: "3.0"
purpose: observability

# Chain tracking - WHO is involved
chain:
  id: [unique workflow id]
  step: [current step number]
  source: [sending agent name/type]
  target: [receiving agent name/type]

# Trace - WHAT happened before
trace:
  - step: 1
    agent: [agent name]
    action: [what it did - brief]
    confidence: [0.0-1.0]
    facts_added: [F1, F2]
  - step: 2
    agent: [agent name]
    action: [what it did]
    confidence: [0.0-1.0]
    facts_modified: [F1]

# Fact registry - structured knowledge
facts:
  F1:
    text: [fact content]
    confidence: [0.0-1.0]
    source: [which agent established this]
    step: [when established]
  F2:
    text: [fact content]
    confidence: [0.0-1.0]

# Handoff metadata - WHAT to preserve
handoff:
  critical_facts: [F1, F2]  # MUST be preserved
  context_facts: [F3, F4]   # Nice to have
  warnings: [any concerns]

# The actual instruction (natural language)
instruction: |
  [Full natural language task description]
  [No compression - write clearly]
```

### Minimal Format (Simple Handoffs)

For simpler cases, use abbreviated format:

```yaml
lctl: "3.0"
chain: {id: auth-001, step: 2, from: analyzer, to: fixer}
facts:
  F1: {text: "SQL injection in UserController.py:45", confidence: 0.95}
instruction: Fix the SQL injection vulnerability
```

### Confidence Guidelines

| Score | Meaning | Propagation Rule |
|-------|---------|------------------|
| 0.95-1.0 | Verified/certain | Can propagate as-is |
| 0.80-0.94 | High confidence | Propagate, note source |
| 0.60-0.79 | Moderate | Verify before acting |
| 0.40-0.59 | Low | Flag for human review |
| <0.40 | Uncertain | Do not propagate |

**Propagation rule**: Child confidence ≤ min(parent confidence, own confidence)

## Usage Examples

### Example 1: Sequential Agent Chain

**Scenario**: Code review → Security analysis → Fix implementation

**Step 1: Code Analyzer → Security Reviewer**
```yaml
lctl: "3.0"
purpose: observability

chain:
  id: review-2024-001
  step: 1
  source: code-analyzer
  target: security-reviewer

trace:
  - step: 1
    agent: code-analyzer
    action: "Analyzed UserController.py, found potential issues"
    confidence: 0.85
    facts_added: [F1, F2, F3]

facts:
  F1:
    text: "UserController.py:45-52 constructs SQL from user input"
    confidence: 0.95
    source: code-analyzer
    step: 1
  F2:
    text: "No input validation detected on user_id parameter"
    confidence: 0.90
    source: code-analyzer
    step: 1
  F3:
    text: "Function is called from /api/users endpoint"
    confidence: 1.0
    source: code-analyzer
    step: 1

handoff:
  critical_facts: [F1, F2]
  context_facts: [F3]

instruction: |
  Assess security risk of the SQL construction pattern found in
  UserController.py. Determine if this is exploitable SQL injection
  and severity rating.
```

**Step 2: Security Reviewer → Fix Implementer**
```yaml
lctl: "3.0"
purpose: observability

chain:
  id: review-2024-001
  step: 2
  source: security-reviewer
  target: fix-implementer

trace:
  - step: 1
    agent: code-analyzer
    action: "Found SQL construction from user input"
    confidence: 0.85
    facts_added: [F1, F2, F3]
  - step: 2
    agent: security-reviewer
    action: "Confirmed SQL injection, rated Critical"
    confidence: 0.92
    facts_added: [F4, F5]
    facts_modified: [F1]

facts:
  F1:
    text: "CONFIRMED: SQL injection vulnerability in UserController.py:45-52"
    confidence: 0.95  # upgraded after confirmation
    source: security-reviewer
    step: 2
  F2:
    text: "No input validation on user_id parameter"
    confidence: 0.90
    source: code-analyzer
    step: 1
  F4:
    text: "CVSS Score: 9.8 (Critical) - unauthenticated remote exploitation"
    confidence: 0.88
    source: security-reviewer
    step: 2
  F5:
    text: "Recommended fix: parameterized queries with SQLAlchemy"
    confidence: 0.85
    source: security-reviewer
    step: 2

handoff:
  critical_facts: [F1, F4, F5]
  context_facts: [F2]
  warnings: ["Production system - test thoroughly before deploy"]

instruction: |
  Implement the SQL injection fix for UserController.py.
  Use parameterized queries as recommended. The vulnerability is
  Critical severity so prioritize correctness over optimization.
```

### Example 2: Parallel Agents with Merge

**Scenario**: Frontend, Backend, and Test agents work in parallel, then merge

**Merge Agent receives:**
```yaml
lctl: "3.0"
purpose: observability

chain:
  id: feature-dashboard-001
  step: 4
  source: orchestrator
  target: integration-agent
  parallel_merge: true

trace:
  - step: 1
    agent: orchestrator
    action: "Spawned 3 parallel agents for dashboard feature"
    confidence: 1.0
  - step: 2
    agent: frontend-dev
    parallel: true
    action: "Built DashboardView component with charts"
    confidence: 0.88
    facts_added: [F1, F2]
  - step: 2
    agent: backend-dev
    parallel: true
    action: "Created /api/metrics endpoint"
    confidence: 0.85
    facts_added: [F3, F4]
  - step: 2
    agent: test-writer
    parallel: true
    action: "Wrote 12 unit tests, 3 integration tests"
    confidence: 0.92
    facts_added: [F5]
  - step: 3
    agent: orchestrator
    action: "Collected parallel results, identified conflict"
    confidence: 0.80
    facts_added: [F6]

facts:
  F1:
    text: "DashboardView expects metrics as {labels: [], values: []}"
    confidence: 0.88
    source: frontend-dev
    step: 2
  F3:
    text: "/api/metrics returns {data: [{name, value, timestamp}]}"
    confidence: 0.85
    source: backend-dev
    step: 2
  F6:
    text: "CONFLICT: Frontend expects different format than backend provides"
    confidence: 0.95
    source: orchestrator
    step: 3

handoff:
  critical_facts: [F1, F3, F6]
  warnings: ["Format mismatch must be resolved before integration"]

instruction: |
  Resolve the API format mismatch between frontend and backend.
  Frontend expects {labels: [], values: []} but backend returns
  {data: [{name, value, timestamp}]}.

  Decide: adapt frontend, adapt backend, or add transform layer.
  Consider that 15 tests already written against current implementations.
```

### Example 3: Debugging a Failed Chain

**Scenario**: Agent chain failed, need to understand why

**Debug trace:**
```yaml
lctl: "3.0"
purpose: debugging

chain:
  id: deploy-2024-042
  status: FAILED
  failed_at_step: 4

trace:
  - step: 1
    agent: requirement-analyzer
    action: "Parsed deployment requirements"
    confidence: 0.95
    facts_added: [F1, F2]
    duration_ms: 2340

  - step: 2
    agent: environment-checker
    action: "Verified staging environment"
    confidence: 0.90
    facts_added: [F3]
    duration_ms: 1820

  - step: 3
    agent: build-agent
    action: "Built Docker image successfully"
    confidence: 0.92
    facts_added: [F4, F5]
    duration_ms: 45200

  - step: 4
    agent: deploy-agent
    action: "FAILED: Could not connect to cluster"
    confidence: 0.0
    error: "ConnectionRefused: kubectl cannot reach api-server"
    facts_at_failure: [F1, F2, F3, F4, F5]
    duration_ms: 5000

facts:
  F1:
    text: "Target: production-cluster-us-east"
    confidence: 0.95
    source: requirement-analyzer
  F3:
    text: "Staging environment healthy"
    confidence: 0.90
    source: environment-checker
    note: "ISSUE: Checked staging, not production"
  F5:
    text: "Image pushed to registry: app:v2.3.1"
    confidence: 0.92
    source: build-agent

analysis:
  root_cause: "environment-checker verified wrong environment (staging vs production)"
  confidence_in_analysis: 0.85
  recommendation: "Add explicit environment parameter validation at step 2"
```

## Receiving Protocol

When an agent receives an LCTL block:

1. **Parse** the YAML metadata
2. **Load trace** to understand history
3. **Check confidence** on critical facts (flag <0.7)
4. **Acknowledge** (recommended for chains):
   ```yaml
   lctl_ack:
     chain_id: review-2024-001
     step: 3
     agent: fix-implementer
     status: received
     facts_loaded: [F1, F4, F5]
     confidence_concerns: []  # or list low-confidence facts
   ```
5. **Execute** the instruction with full context
6. **Emit** LCTL block for next agent (if chaining)

## Best Practices

### For Chain Tracking
- Always generate unique chain IDs (uuid or descriptive)
- Increment step numbers sequentially
- Preserve full trace history through chain

### For Fact Management
- Use short IDs (F1, F2) for cross-referencing
- Update confidence when facts are verified/contradicted
- Mark source agent and step for each fact
- Critical facts must propagate through entire chain

### For Debugging
- Include duration_ms for performance analysis
- Log errors with full context at failure point
- Record facts_at_failure for debugging
- Add analysis section for post-mortems

### For Parallel Workflows
- Mark parallel steps explicitly
- Track conflicts between parallel results
- Merge agent should reconcile contradictions

## Integration with Claude Code

For Claude Code's Task tool multi-agent architecture:

```
Parent Agent
    │
    ├─ Task(agent, prompt + LCTL metadata)
    │      └─ Sub-agent parses LCTL, executes, returns result + LCTL
    │
    └─ Parent extracts LCTL, updates trace, continues
```

LCTL enables:
- **Debugging**: "Why did explore-agent miss that file?"
- **Auditing**: "What facts were available at step 3?"
- **Confidence tracking**: "When did certainty drop below threshold?"
- **Chain visualization**: Render trace as flowchart

## Version History

- **3.0**: Pivoted to observability focus, removed compression, added tracing
- **2.0**: Added artifacts, confidence, chain tracking (compression focus)
- **1.0**: Initial specification (compression focus)
