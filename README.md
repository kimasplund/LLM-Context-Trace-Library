# LLM Context Transfer Language (LCTL) v4.0

**Time-travel debugging for multi-agent LLM workflows.**

> "Don't pitch the protocol. Pitch the tool."

## The Killer Feature

```bash
$ lctl replay --to-seq 10 chain.lctl.json
Replaying to event 10...
State at seq 10: 2 facts, agent=code-analyzer

$ lctl diff chain-v1.lctl.json chain-v2.lctl.json
Events diverged at seq 8:
  v1: fact_added F3 (confidence: 0.70)
  v2: fact_added F3 (confidence: 0.85) [DIFFERENT]
```

Step back through agent execution. See exactly where things diverged. Understand "what if?"

## Evolution Journey

| Version | Focus | Outcome |
|---------|-------|---------|
| v1.0 | Compression | Abandoned - didn't solve real problems |
| v2.0 | Compression + artifacts | Abandoned - still wrong focus |
| v3.0 | Observability | Better - but protocol-first thinking |
| **v4.0** | **Tool-first** | **Event sourcing enables time-travel** |

## Core Protocol (One Page)

```yaml
lctl: "4.0"
chain:
  id: "security-review-001"

events:
  - seq: 1
    type: step_start
    timestamp: "2024-01-15T10:30:00Z"
    agent: "code-analyzer"
    data: {intent: "analyze", input_summary: "UserController.py"}

  - seq: 2
    type: fact_added
    agent: "code-analyzer"
    data: {id: "F1", text: "SQL injection at line 45", confidence: 0.85}

  - seq: 3
    type: step_end
    agent: "code-analyzer"
    data: {outcome: "success", duration_ms: 30000, tokens: {in: 500, out: 200}}
```

That's it. Chain ID + event stream. Everything else derives from this.

## What A/B Testing Revealed

We ran 6 parallel cognitive agents to evolve LCTL:

| Component | A/B Winner | Why |
|-----------|------------|-----|
| **Architecture** | Event Sourcing | Enables time-travel replay |
| **Handoffs** | Query-Based + Contracts | Pull model, type safety |
| **Confidence** | Decay + Consensus | Automatic degradation tracking |
| **Chain Tracking** | DAG projection on events | Handles parallel workflows |
| **Adoption Driver** | CLI tooling | Protocol is implementation detail |

## CLI Tools

```bash
# Time-travel replay
$ lctl replay chain.lctl.json

# Visual debugger (web UI)
$ lctl debug chain.lctl.json

# Performance analytics
$ lctl stats chain.lctl.json
Duration: 45.2s | Tokens: 2,340 | Cost: $0.047

# Bottleneck analysis
$ lctl bottleneck chain.lctl.json
security-reviewer: 50% of time (consider parallelization)

# Contract validation
$ lctl validate chain.lctl.json --contracts ./contracts/
```

## Integration

```python
# Zero-config auto-instrumentation
import lctl
lctl.auto_instrument()

# Or framework-specific
from lctl.langchain import LCTLCallbackHandler
from lctl.crewai import LCTLCrew
from lctl.autogen import enable_lctl
```

## Key Innovations

1. **Event Sourcing**: Immutable event log enables replay to any point
2. **Query-Based Handoffs**: Agents pull facts they need, not push everything
3. **Decay + Consensus Confidence**: Automatic degradation, consensus for critical facts
4. **Contract Validation**: Type safety for agent inputs/outputs
5. **Tool-First Design**: CLI tools drive adoption, protocol is invisible

## Files

- `LLM-CONTEXT-TRANSFER.md` - Full specification (10 parts)

## Research Basis

Developed through integrated reasoning methodology:
- Breadth-of-thought: 10 evolution directions explored
- Self-reflecting chain: Validated real pain points
- A/B testing: 5 approaches tested per component
- Framework analysis: Claude Code, LangChain, CrewAI, AutoGen, custom

## Philosophy

The protocol succeeds when developers never think about it. They use `lctl debug` because it's 10x better than print statements. The fact that it uses LCTL format is irrelevant to them.

**Build the tool. The protocol follows.**
