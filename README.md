# LLM Context Transfer Language (LCTL) v3.0

An observability and tracing protocol for multi-agent LLM workflows.

## What This Is

LCTL is a **metadata layer** (not a compression format) that enables:
- Debugging multi-agent chains ("why did agent 3 fail?")
- Tracking confidence degradation across hops
- Auditing what context each agent received
- Visualizing agent workflow execution

## Key Insight

LCTL v1-v2 tried to compress context to save tokens. After analysis, we found:
- Modern LLMs have 200K+ context windows (rarely exhausted)
- Sub-agents get fresh contexts anyway (no accumulation problem)
- Compression overhead often exceeds token savings

**v3.0 pivots to solving the real problem: observability and debugging.**

## Files

- `LLM-CONTEXT-TRANSFER.md` - Full specification with format and examples

## Quick Example

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
    action: "Found SQL injection pattern"
    confidence: 0.85
    facts_added: [F1, F2]
  - step: 2
    agent: security-reviewer
    action: "Confirmed Critical severity"
    confidence: 0.92
    facts_modified: [F1]

facts:
  F1:
    text: "CONFIRMED: SQL injection in UserController.py:45"
    confidence: 0.95
    source: security-reviewer

handoff:
  critical_facts: [F1]
  warnings: ["Production system"]

instruction: |
  Implement parameterized query fix for the SQL injection.
```

## What LCTL Enables

| Capability | How LCTL Helps |
|------------|----------------|
| **Debugging** | Full trace of what each agent saw and did |
| **Auditing** | Numbered facts with sources and timestamps |
| **Confidence** | Track certainty degradation through chain |
| **Conflicts** | Identify contradictions in parallel workflows |
| **Post-mortems** | Structured failure analysis with root cause |

## When to Use

**Use LCTL:**
- Multi-agent chains (3+ agents)
- Parallel agents that merge results
- Workflows requiring audit trails
- Debugging complex agent failures

**Don't use LCTL:**
- Single agent interactions
- Simple request/response
- When token cost is primary concern (LCTL adds overhead)

## Integration

Works with any multi-agent framework:
- Claude Code Task tool
- LangChain/LangGraph
- CrewAI
- Qwen-Agent
- Custom orchestrators

## Origin

Evolved through three iterations:
- v1.0: Compression format (abandoned - didn't solve real problems)
- v2.0: Added artifacts/confidence (still compression-focused)
- v3.0: Pivoted to observability (current - solves validated pain points)
