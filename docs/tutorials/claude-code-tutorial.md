# Claude Code Integration Tutorial

> Time-travel debugging for Claude Code's multi-agent workflows

This tutorial shows how to use LCTL to trace, debug, and analyze Claude Code sessions. Unlike other integrations, Claude Code uses a **hook-based architecture** that automatically captures events across all tool invocations.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Understanding Hooks](#understanding-hooks)
- [CLI Commands](#cli-commands)
- [Programmatic Usage](#programmatic-usage)
- [What Gets Traced](#what-gets-traced)
- [Analyzing Traces](#analyzing-traces)
- [Cost Estimation](#cost-estimation)
- [HTML Reports](#html-reports)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Claude Code integration enables LCTL to trace its own multi-agent workflows. When Claude Code spawns subagents using the `Task` tool, each agent's actions are recorded as LCTL events, enabling:

- **Time-travel debugging** of agent spawning and completions
- **Tool call tracking** across all agent types
- **Cost estimation** based on token usage
- **Performance bottleneck detection**
- **Git commit linking** for version control context

### Architecture

```
Claude Code Session
    │
    ├─→ PreToolUse.sh hook ─→ LCTL records STEP_START
    │
    ├─→ Tool Execution (Task, Bash, Write, etc.)
    │
    └─→ PostToolUse.sh hook ─→ LCTL records STEP_END, TOOL_CALL, etc.
                                   │
                                   └─→ .claude/traces/.lctl-state.json
```

Unlike callback-based integrations (LangChain, CrewAI), Claude Code uses **shell hooks** that execute before/after each tool call, capturing events across the entire session.

---

## Quick Start

### 1. Initialize LCTL Hooks

```bash
# Generate hook scripts in your project
lctl claude init

# Or specify a custom hooks directory
lctl claude init --hooks-dir /path/to/.claude/hooks
```

This creates:
- **Hook scripts:**
  - `PreToolUse.sh` - Captures `Task` tool invocations (agent spawning)
  - `PostToolUse.sh` - Captures tool completions, file changes, git commits
  - `Stop.sh` - Exports final trace when session ends
- **Settings file:** `.claude/settings.json` - Registers hooks with Claude Code

> **Important:** After running `lctl claude init`, restart Claude Code for hooks to activate.

### 2. Verify Installation

```bash
lctl claude validate
```

Expected output:
```
Claude Code LCTL Hooks Status
=============================

Hook Status:
  PreToolUse: OK (installed, executable, LCTL-enabled)
  PostToolUse: OK (installed, executable, LCTL-enabled)
  Stop: OK (installed, executable, LCTL-enabled)

Traces Directory: .claude/traces/ (will be created on first trace)

Status: All hooks installed correctly
```

### 3. Run Claude Code Normally

Once hooks are installed, tracing happens automatically. Just use Claude Code as usual:

```bash
claude
```

When you spawn agents with the `Task` tool, LCTL captures:
- Agent types (implementor, Explore, Plan, etc.)
- Task descriptions and prompts
- Tool calls within agents
- Agent completions with results
- File changes, git commits, and more

### 4. View Your Trace

When the session ends, LCTL automatically exports the trace:

```bash
# List traces
ls .claude/traces/

# View trace summary
lctl stats .claude/traces/claude-code-20260112-143052.lctl.json

# Replay to inspect state at any point
lctl replay --to-seq 20 .claude/traces/claude-code-*.lctl.json
```

---

## Understanding Hooks

Claude Code hooks are shell scripts that execute at specific lifecycle points. LCTL generates hooks that record events without interfering with normal operation.

### Hook Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code Session                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tool Call: Task(agent="implementor", description="...")        │
│       │                                                         │
│       ├──→ PreToolUse.sh executes                               │
│       │         └──→ LCTL: step_start("implementor", ...)       │
│       │                                                         │
│       ├──→ Subagent runs (implementor)                          │
│       │         ├──→ Uses Bash, Write, Edit tools               │
│       │         └──→ Returns result                             │
│       │                                                         │
│       └──→ PostToolUse.sh executes                              │
│                 └──→ LCTL: step_end("implementor", ...)         │
│                                                                 │
│  Session ends                                                   │
│       └──→ Stop.sh executes                                     │
│                 └──→ LCTL: export trace, show summary           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hook Input Format

Hooks receive context as JSON via **stdin** with the following structure:

```json
{
  "session_id": "abc123",
  "hook_event_name": "PostToolUse",
  "tool_name": "Task",
  "tool_input": {
    "subagent_type": "Explore",
    "description": "Find files",
    "prompt": "Search for...",
    "model": "haiku"
  },
  "tool_response": {
    "status": "completed",
    "content": [...],
    "usage": {"input_tokens": 100, "output_tokens": 50}
  }
}
```

| Field | Description |
|-------|-------------|
| `tool_name` | Tool being invoked (Task, Bash, Write, etc.) |
| `tool_input` | Tool input parameters as object |
| `tool_response` | Tool output (in PostToolUse only) |
| `hook_event_name` | Hook type (PreToolUse, PostToolUse, Stop) |

### Hook Scripts Location

Hooks are stored in `.claude/hooks/` (project-level) or `~/.claude/hooks/` (global):

```
.claude/
├── hooks/
│   ├── PreToolUse.sh     # Records Task starts
│   ├── PostToolUse.sh    # Records completions, tool calls
│   └── Stop.sh           # Exports final trace
└── traces/
    ├── .lctl-state.json  # Session state (deleted on export)
    └── claude-code-*.lctl.json  # Exported traces
```

---

## CLI Commands

### `lctl claude init`

Generate LCTL hook scripts.

```bash
# Default location (.claude/hooks/)
lctl claude init

# Custom location
lctl claude init --hooks-dir ~/.claude/hooks

# Overwrite existing hooks
lctl claude init --force
```

### `lctl claude validate`

Check hook installation status.

```bash
lctl claude validate

# Check custom location
lctl claude validate --hooks-dir /path/to/hooks
```

### `lctl claude status`

Show active tracing session status.

```bash
lctl claude status
```

Output:
```
Active LCTL Session
===================

Chain ID: claude-code-20260112-143052
Events: 47
Active Agents: implementor
Tool Calls: 23
Facts: 8
Errors: 0

Trace file: .claude/traces/claude-code-20260112-143052.lctl.json
```

### `lctl claude report`

Generate an HTML report from a trace.

```bash
# Generate report
lctl claude report .claude/traces/claude-code-*.lctl.json

# Custom output path
lctl claude report trace.lctl.json --output report.html

# Open in browser after generating
lctl claude report trace.lctl.json --open
```

### `lctl claude clean`

Clean up old traces.

```bash
# Remove traces older than 7 days (default)
lctl claude clean

# Remove traces older than 30 days
lctl claude clean --older-than 30

# Dry run (show what would be deleted)
lctl claude clean --dry-run
```

---

## Programmatic Usage

For advanced use cases, you can use the tracer directly in Python.

### Basic Usage

```python
from lctl.integrations.claude_code import LCTLClaudeCodeTracer

# Create tracer
tracer = LCTLClaudeCodeTracer(chain_id="my-workflow")

# Record agent spawning
tracer.on_task_start(
    agent_type="implementor",
    description="Implement authentication",
    prompt="Add JWT auth to the API endpoints",
    model="sonnet",
)

# Record tool calls within the agent
tracer.on_tool_call(
    tool_name="Write",
    input_data={"file_path": "/src/auth.py", "content": "..."},
    output_data={"success": True},
    duration_ms=150,
)

# Record file changes
tracer.on_file_change(
    file_path="/src/auth.py",
    change_type="create",
    lines_added=120,
)

# Record agent completion
tracer.on_task_complete(
    agent_type="implementor",
    description="Implement authentication",
    result="Successfully implemented JWT authentication",
    success=True,
    tokens_in=5000,
    tokens_out=3000,
)

# Export trace
path = tracer.export()
print(f"Trace saved to: {path}")
```

### Singleton Pattern for Hooks

Hooks need to share state across invocations. Use `get_or_create()`:

```python
from lctl.integrations.claude_code import LCTLClaudeCodeTracer

# First call creates, subsequent calls return same instance
tracer = LCTLClaudeCodeTracer.get_or_create()

# State is persisted to .claude/traces/.lctl-state.json
# and restored on each hook invocation
```

### Recording Different Event Types

```python
# TodoWrite (task list updates)
tracer.on_todo_write(todos=[
    {"content": "Implement auth", "status": "completed"},
    {"content": "Add tests", "status": "in_progress"},
    {"content": "Deploy", "status": "pending"},
])

# Skill invocations
tracer.on_skill_invoke(
    skill_name="commit",
    args="-m 'Add authentication'",
    result_summary="Created commit abc123",
)

# MCP tool calls
tracer.on_mcp_tool_call(
    server_name="sql",
    tool_name="execute-sql",
    input_data={"query": "SELECT * FROM users"},
    output_data={"rows": 100},
)

# Git commits
tracer.on_git_commit(
    commit_hash="abc123def456",
    message="Add JWT authentication",
    files_changed=5,
    insertions=200,
    deletions=50,
)

# User interactions
tracer.on_user_interaction(
    question="Which database should we use?",
    response="PostgreSQL",
    options=["PostgreSQL", "MySQL", "MongoDB"],
)

# Web fetch
tracer.on_web_fetch(
    url="https://docs.anthropic.com",
    prompt="Get API documentation",
    result_summary="Found Claude API reference",
)

# Web search
tracer.on_web_search(
    query="JWT best practices 2026",
    results_count=15,
    top_result="Use RS256 signing...",
)
```

---

## What Gets Traced

The PostToolUse hook automatically captures these events:

### Agent Events
| Event | Trigger | Data Captured |
|-------|---------|---------------|
| `step_start` | Task tool invoked | agent_type, description, prompt, model |
| `step_end` | Task tool completes | result, success, duration, tokens |

### Tool Events
| Tool | Event Type | Data Captured |
|------|------------|---------------|
| `TodoWrite` | task progress | completed/in_progress/pending counts |
| `Skill` | skill invocation | skill_name, args, result |
| `mcp__*` | MCP tool call | server, tool, input, output |
| `Write` | file create | file_path, lines |
| `Edit` | file edit | file_path |
| `Bash` (git commit) | git commit | hash, message, stats |
| `AskUserQuestion` | user interaction | question, response |
| `WebFetch` | web fetch | url, content summary |
| `WebSearch` | web search | query, results |

### Skipped Tools

High-frequency read-only tools are skipped to reduce noise:
- `Read`, `Glob`, `Grep`, `LS`, `TaskOutput`, `KillShell`

---

## Analyzing Traces

### Summary Statistics

```bash
lctl stats .claude/traces/claude-code-*.lctl.json
```

Output:
```
Chain: claude-code-20260112-143052
========================================
Events: 127
Duration: 5m 23s
Agents: 5 (Plan, implementor, Explore, code-finder, qa-tester)

Token Usage:
  Input: 45,230 tokens
  Output: 12,450 tokens
  Total: 57,680 tokens

Estimated Cost: $0.32 (Claude Sonnet pricing)

Facts: 23
Errors: 2
Tool Calls: 89
File Changes: 12
Git Commits: 3
```

### Execution Flow

```bash
lctl trace .claude/traces/claude-code-*.lctl.json
```

Output:
```
Execution trace for claude-code-20260112-143052:

+- [1] Plan: Plan feature implementation
   +- [15] implementor: Implement authentication
      +- [45] success (2m 15s, 12,340 tokens)
   +- [46] implementor: Add tests
      +- [78] success (1m 45s, 8,230 tokens)
   +- [79] qa-tester: Run test suite
      +- [95] success (45s, 3,120 tokens)
   +- [96] success (4m 50s)
+- [97] success (5m 20s)
```

### Bottleneck Analysis

```bash
lctl bottleneck .claude/traces/claude-code-*.lctl.json
```

Output:
```
Slowest agents:
  1. implementor (seq 15): 2m 15s (42%)
  2. implementor (seq 46): 1m 45s (33%)
  3. qa-tester (seq 79): 45s (14%)

Recommendation: Consider parallelizing independent tasks.
```

### Time-Travel Replay

```bash
# Replay to see state after planning
lctl replay --to-seq 14 trace.lctl.json

# Output shows facts, metrics, current agent at that point
```

---

## Cost Estimation

LCTL includes accurate pricing for all Claude models:

```python
from lctl.integrations.claude_code import estimate_cost, MODEL_PRICING

# Estimate cost
cost = estimate_cost(
    tokens_in=50000,
    tokens_out=15000,
    model="claude-sonnet-4"
)

print(f"Input: ${cost['input_cost']:.4f}")
print(f"Output: ${cost['output_cost']:.4f}")
print(f"Total: ${cost['total_cost']:.4f}")
```

### Supported Models

| Model | Input (per MTok) | Output (per MTok) |
|-------|-----------------|-------------------|
| Claude Opus 4.5 | $5.00 | $25.00 |
| Claude Opus 4/4.1 | $15.00 | $75.00 |
| Claude Sonnet 4/4.5 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |
| Claude Haiku 3.5 | $0.80 | $4.00 |
| Claude Haiku 3 | $0.25 | $1.25 |

---

## HTML Reports

Generate visual reports for sharing or archiving:

```python
from lctl.integrations.claude_code import generate_html_report
from lctl.core.events import Chain

# Load chain
chain = Chain.load("trace.lctl.json")

# Generate report
generate_html_report(chain, "report.html")
```

Or via CLI:

```bash
lctl claude report trace.lctl.json --output report.html --open
```

The HTML report includes:
- Event count, duration, token usage
- Estimated cost
- Agent breakdown with step counts
- Fact registry with confidence scores
- Bottleneck analysis
- Event timeline (first 50 events)

---

## Advanced Usage

### Parallel Agent Tracking

```python
# Start parallel group
group_id = tracer.start_parallel_group()

# Spawn multiple agents
tracer.on_task_start(agent_type="Explore", description="Search backend", parallel_group=group_id)
tracer.on_task_start(agent_type="Explore", description="Search frontend", parallel_group=group_id)

# ... agents complete ...

tracer.end_parallel_group()
```

### Git History Linking

```python
# Get git context for workflow changes
git_info = tracer.link_to_git_history()

print(f"Recent commits: {git_info['commits']}")
print(f"Uncommitted changes: {git_info['uncommitted_changes']}")
```

### Session Metadata

```python
from lctl.integrations.claude_code import get_session_metadata

metadata = get_session_metadata()
print(f"Project: {metadata['project_name']}")
print(f"Git branch: {metadata['git_branch']}")
print(f"Python: {metadata['python_version']}")
```

### Checkpoints

Create checkpoints for fast replay to important points:

```python
# After major milestones
tracer.checkpoint(description="After authentication implemented")

# Checkpoints enable fast replay:
# lctl replay --to-checkpoint "After authentication" trace.lctl.json
```

---

## Troubleshooting

### Hooks Not Executing

1. **Check hook permissions**:
   ```bash
   ls -la .claude/hooks/
   # All scripts should have execute permission
   chmod +x .claude/hooks/*.sh
   ```

2. **Verify Claude Code hooks configuration**:
   ```bash
   claude config get hooks
   ```

3. **Run validation**:
   ```bash
   lctl claude validate
   ```

### No Traces Generated

1. **Check for state file**:
   ```bash
   ls -la .claude/traces/.lctl-state.json
   # If missing, hooks may not be executing
   ```

2. **Check Python path in hooks**:
   ```bash
   head -20 .claude/hooks/PostToolUse.sh
   # Verify python3 is available and lctl is importable
   ```

3. **Test manually**:
   ```bash
   python3 -c "from lctl.integrations.claude_code import LCTLClaudeCodeTracer; print('OK')"
   ```

### Large Trace Files

For very long sessions, traces can grow large. Use cleanup:

```bash
# Remove old traces
lctl claude clean --older-than 7

# Or manually delete
rm .claude/traces/claude-code-2025*.lctl.json
```

### Performance Impact

LCTL hooks add minimal overhead (~10-50ms per tool call). If you notice slowdown:

1. **Reduce traced tools** - Edit `PostToolUse.sh` to skip additional tools
2. **Disable during development** - Rename/remove hooks temporarily
3. **Use sampling** - Modify hooks to trace only every Nth invocation

---

## API Reference

See [Claude Code API Reference](../api.md#claude-code-integration) for complete API documentation.

### Key Classes

- `LCTLClaudeCodeTracer` - Main tracer class
- `generate_hooks()` - Generate hook scripts
- `validate_hooks()` - Validate installation
- `generate_html_report()` - Create visual reports
- `get_session_metadata()` - Get environment context
- `estimate_cost()` - Calculate token costs
- `MODEL_PRICING` - Pricing dict for all models

---

## Next Steps

- [LCTL CLI Reference](../../README.md#cli-reference)
- [Web Dashboard](../../README.md#web-dashboard)
- [Protocol Specification](../../LLM-CONTEXT-TRANSFER.md)
- [Other Framework Integrations](.)
