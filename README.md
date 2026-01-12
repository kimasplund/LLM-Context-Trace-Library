<p align="center">
  <h1 align="center">LCTL v4.0</h1>
  <p align="center">
    <strong>Time-Travel Debugging for Multi-Agent LLM Workflows</strong>
  </p>
  <p align="center">
    Replay agent execution to any point. See exactly where things diverged. Understand "what if?"
  </p>
</p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#framework-integrations">Integrations</a> &bull;
  <a href="#cli-reference">CLI</a> &bull;
  <a href="#web-dashboard">Dashboard</a> &bull;
  <a href="#api-reference">API</a>
</p>

---

## Why LCTL?

Debugging multi-agent LLM systems is hard. When something goes wrong, you need to understand:

- **What happened?** - Which agent did what, and when?
- **Why did it happen?** - What was the state at each decision point?
- **What if?** - How would things change with different inputs?

LCTL solves this with **event sourcing** - an immutable log of every action, enabling you to replay execution to any point in time.

```bash
# Replay to see state at any point
$ lctl replay --to-seq 10 chain.lctl.json
State at seq 10: 2 facts, agent=code-analyzer

# Find where two runs diverged
$ lctl diff run-v1.lctl.json run-v2.lctl.json
Events diverged at seq 8:
  v1: fact_added F3 (confidence: 0.70)
  v2: fact_added F3 (confidence: 0.85) [DIFFERENT]
```

## Key Features

- **Time-Travel Replay** - Step back through agent execution to any point
- **Event Sourcing** - Immutable audit log of every state change
- **Zero-Config Integration** - Auto-instrumentation for popular frameworks (LangChain, CrewAI, AutoGen, OpenAI Agents, PydanticAI, Semantic Kernel)
- **Web Dashboard** - Visual debugger with timeline, swim lanes, and bottleneck analysis
- **Confidence Tracking** - Monitor fact confidence decay and consensus
- **Performance Analysis** - Find bottlenecks and optimize token usage

## Installation

### Basic Installation

```bash
pip install git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git
```

### With Framework Support

```bash
# LangChain integration
pip install "lctl[langchain] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# CrewAI integration
pip install "lctl[crewai] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# AutoGen/AG2 integration
pip install "lctl[autogen] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# OpenAI Agents integration
pip install "lctl[openai-agents] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# PydanticAI integration
pip install "lctl[pydantic-ai] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# Semantic Kernel integration
pip install "lctl[semantic-kernel] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# Web dashboard
pip install "lctl[dashboard] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"

# Everything
pip install "lctl[all] @ git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git"
```

### Development Installation

```bash
git clone https://github.com/kimasplund/LLM-Context-Transfer-Language.git
cd LLM-Context-Transfer-Language
pip install -e '.[dev]'
```

## Quick Start

### Basic Usage

```python
import lctl

# Zero-config auto-instrumentation
lctl.auto_instrument()

# Or use manual session for full control
from lctl import LCTLSession

with LCTLSession(chain_id="my-workflow") as session:
    session.step_start("analyzer", "analyze", "Processing input.py")

    # Your agent logic here
    result = analyze_code("input.py")

    session.add_fact("F1", "Found SQL injection vulnerability", confidence=0.85)
    session.step_end(outcome="success", duration_ms=1500)

# Export for analysis
session.export("workflow.lctl.json")
```

### Using the Decorator

```python
import lctl

@lctl.traced("analyzer", "analyze")
def analyze_code(filepath):
    """Automatically traced function."""
    findings = run_analysis(filepath)

    # Add facts during execution
    session = lctl.get_session()
    for finding in findings:
        session.add_fact(finding.id, finding.text, confidence=finding.score)

    return findings

# Run your code - tracing happens automatically
result = analyze_code("vulnerable.py")
lctl.export("analysis.lctl.json")
```

## Framework Integrations

### LangChain

```python
from lctl.integrations.langchain import LCTLCallbackHandler, trace_chain

# Option 1: Callback handler
handler = LCTLCallbackHandler(chain_id="langchain-demo")
result = chain.invoke(
    {"input": "Analyze this code for security issues"},
    config={"callbacks": [handler]}
)
handler.export("langchain_trace.lctl.json")

# Option 2: Chain wrapper (recommended)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a security analyst."),
    ("human", "{input}")
])
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

traced = trace_chain(chain, chain_id="security-review")
result = traced.invoke({"input": "Review UserController.py"})
traced.export("security_trace.lctl.json")
```

### CrewAI

```python
from lctl.integrations.crewai import LCTLAgent, LCTLTask, LCTLCrew

# Create traced agents
researcher = LCTLAgent(
    role="Security Researcher",
    goal="Find vulnerabilities in code",
    backstory="Expert security researcher with 10 years experience"
)

analyst = LCTLAgent(
    role="Risk Analyst",
    goal="Assess severity of security findings",
    backstory="Specializes in risk assessment and prioritization"
)

# Create tasks
research_task = LCTLTask(
    description="Analyze the codebase for security vulnerabilities",
    expected_output="List of vulnerabilities with severity ratings",
    agent=researcher
)

# Create and run crew
crew = LCTLCrew(
    agents=[researcher, analyst],
    tasks=[research_task],
    chain_id="security-crew"
)

result = crew.kickoff()
crew.export_trace("crew_trace.lctl.json")
```

### AutoGen / AG2

```python
from lctl.integrations.autogen import LCTLAutogenCallback, trace_agent

# Option 1: Callback approach
from autogen import ConversableAgent

callback = LCTLAutogenCallback(chain_id="autogen-demo")

assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"model": "gpt-4"}
)

user = ConversableAgent(
    name="user",
    human_input_mode="NEVER"
)

# Attach tracing
callback.attach(assistant)
callback.attach(user)

# Run conversation
user.initiate_chat(assistant, message="Explain quantum computing")
callback.export("autogen_trace.lctl.json")

# Option 2: Traced wrapper
from lctl.integrations.autogen import LCTLConversableAgent

agent = LCTLConversableAgent(
    name="researcher",
    system_message="You research and summarize topics.",
    chain_id="research-agent"
)

result = agent.initiate_chat(other_agent, message="Research LCTL")
agent.export_trace("research_trace.lctl.json")
```

### OpenAI Agents SDK

```python
from agents import Agent, Runner, function_tool
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer, trace_agent

# Define tools
@function_tool
def search_docs(query: str) -> str:
    """Search documentation for relevant information."""
    return f"Results for: {query}"

# Create agent
agent = Agent(
    name="doc_assistant",
    instructions="Help users find documentation.",
    tools=[search_docs]
)

# Option 1: Tracer with run_config
tracer = LCTLOpenAIAgentTracer(chain_id="openai-agent")
result = await Runner.run(
    agent,
    input="How do I use LCTL?",
    run_config=tracer.run_config
)
tracer.export("openai_trace.lctl.json")

# Option 2: Traced agent wrapper
traced = trace_agent(agent, chain_id="doc-helper")
result = await Runner.run(
    traced.agent,
    input="Explain time-travel debugging",
    run_config=traced.run_config
)
traced.export("helper_trace.lctl.json")
```

### Claude Code

LCTL can trace Claude Code's own multi-agent workflows using hook-based architecture.

```bash
# Initialize LCTL hooks in your project
lctl claude init

# Validate installation
lctl claude validate

# View active session status
lctl claude status

# Generate HTML report
lctl claude report .claude/traces/claude-code-*.lctl.json --open
```

**Automatic Tracing**: Once hooks are installed, LCTL automatically traces:
- Agent spawning via `Task` tool (Plan, implementor, Explore, etc.)
- Tool calls (Bash, Write, Edit, WebFetch, WebSearch)
- TodoWrite updates, Skill invocations, MCP tool calls
- Git commits linked to workflows
- User interactions (AskUserQuestion)

**Programmatic Usage**:

```python
from lctl.integrations.claude_code import LCTLClaudeCodeTracer

tracer = LCTLClaudeCodeTracer(chain_id="my-workflow")

# Record agent activity
tracer.on_task_start(agent_type="implementor", description="Add auth", prompt="...")
tracer.on_tool_call("Bash", {"command": "pytest"}, {"exit_code": 0})
tracer.on_file_change("/src/auth.py", "create", lines_added=100)
tracer.on_task_complete(agent_type="implementor", result="Done", success=True)

# Export trace
tracer.export("workflow.lctl.json")

# Get summary with cost estimation
summary = tracer.get_summary()
print(f"Cost: ${summary['total_tokens_in'] * 3 / 1_000_000:.4f}")
```

See the [Claude Code Tutorial](docs/tutorials/claude-code-tutorial.md) for comprehensive documentation.

## CLI Reference

LCTL provides a powerful CLI for analyzing and debugging agent traces.

### replay - Time-Travel Debugging

```bash
# Replay all events
lctl replay chain.lctl.json

# Replay to specific point
lctl replay --to-seq 10 chain.lctl.json

# Verbose output showing each event
lctl replay --verbose chain.lctl.json
```

### stats - Performance Statistics

```bash
# Human-readable output
lctl stats chain.lctl.json
# Output:
# Chain: security-review-001
# Events: 25
# Agents: 3 (code-analyzer, reviewer, fixer)
# Duration: 45.2s
# Tokens: 2,340 (in: 1,500 / out: 840)
# Est. Cost: $0.0470

# JSON output for scripting
lctl stats --json chain.lctl.json
```

### trace - Execution Flow

```bash
lctl trace chain.lctl.json
# Output:
# Execution trace for security-review-001:
#
# +- [1] code-analyzer: analyze
#    +- [5] security-reviewer: assess
#       +- [10] fix-implementer: fix
#       +- [15] success (7.7s)
#    +- [16] success (22.5s)
# +- [17] success (15.0s)
```

### bottleneck - Performance Analysis

```bash
lctl bottleneck chain.lctl.json
# Output:
# Slowest steps:
#   1. security-reviewer (seq 6): 22.5s (50%)
#   2. code-analyzer (seq 1): 15.0s (33%)
#   3. fix-implementer (seq 11): 7.7s (17%)
#
# Recommendation: security-reviewer is a major bottleneck (50% of time).
# Consider: parallelization, caching, or faster model.

# Show top N bottlenecks
lctl bottleneck --top 3 chain.lctl.json
```

### confidence - Fact Confidence Timeline

```bash
lctl confidence chain.lctl.json
# Output:
# Fact confidence timeline:
#   F1: 0.85 -> 0.85 -> 0.95 +
#   F2: 0.90 -> 0.86 -> 0.81 ~
#   F3: 0.70 -> 0.45 x BLOCKED
#
# Warning: F3 blocked at step 8 due to low confidence.
```

### diff - Compare Chains

```bash
lctl diff chain-v1.lctl.json chain-v2.lctl.json
# Output:
# Found 3 difference(s):
#
# Seq 8: DIVERGED
#   v1: fact_added by code-analyzer
#   v2: fact_modified by security-reviewer
#
# Seq 12: MISSING in first
# Seq 15: MISSING in second
```

### debug - Visual Debugger

```bash
# Launch web-based debugger for a specific chain
lctl debug chain.lctl.json
# Starting LCTL Visual Debugger...
#   Chain: security-review-001
#   Events: 25
#
# Open http://localhost:8080 in your browser

# Custom port
lctl debug --port 3000 chain.lctl.json
```

### dashboard - Web Dashboard

```bash
# Start dashboard in current directory
lctl dashboard

# Custom port and host
lctl dashboard --port 3000 --host 127.0.0.1

# Specify directory with chain files
lctl dashboard --dir ./traces
```

## Web Dashboard

The LCTL dashboard provides a visual interface for exploring and debugging agent traces.

### Features

- **Timeline View** - Chronological display of all events
- **Agent Swim Lanes** - See which agent did what and when
- **Fact Registry** - Track facts with confidence indicators
- **Time-Travel Slider** - Replay to any point interactively
- **Bottleneck Highlighting** - Identify slow operations
- **Error Indicators** - Quick visibility into failures

### Screenshots

<!-- TODO: Add dashboard screenshots -->
```
+---------------------------------------------------+
|  LCTL Dashboard - security-review-001             |
+---------------------------------------------------+
|  Timeline          | Facts        | Analysis     |
|  +--------------+  | F1: [===] 95%| Bottlenecks  |
|  | analyzer [==]|  | F2: [== ] 81%| - reviewer   |
|  | reviewer [===]  | F3: [x] BLOCK|   50% time   |
|  | fixer    [= ]|  |              |              |
|  +--------------+  |              | Tokens: 2340 |
|  [<] Seq: 10 [>]  |              | Cost: $0.047 |
+---------------------------------------------------+
```

### Starting the Dashboard

```bash
# Install dashboard dependencies
pip install 'lctl[dashboard]'

# Launch
lctl dashboard --dir ./my-traces
```

## API Reference

### Core Classes

#### LCTLSession

The main class for recording LCTL events.

```python
from lctl import LCTLSession

session = LCTLSession(chain_id="my-chain")

# Record agent steps
session.step_start(agent, intent, input_summary)
session.step_end(agent, outcome, output_summary, duration_ms, tokens_in, tokens_out)

# Record facts
session.add_fact(fact_id, text, confidence, source)
session.modify_fact(fact_id, text, confidence, reason)

# Record tool calls
session.tool_call(tool, input_data, output_data, duration_ms)

# Record errors
session.error(category, error_type, message, recoverable, suggested_action)

# Create checkpoints
session.checkpoint()

# Export
session.export("trace.lctl.json")
```

#### Chain and Event

```python
from lctl import Chain, Event, EventType

# Load existing chain
chain = Chain.load(Path("trace.lctl.json"))

# Access events
for event in chain.events:
    print(f"[{event.seq}] {event.type.value}: {event.agent}")

# Save chain
chain.save(Path("modified.lctl.json"))
```

#### ReplayEngine

```python
from lctl import Chain, ReplayEngine

chain = Chain.load(Path("trace.lctl.json"))
engine = ReplayEngine(chain)

# Replay to specific point
state = engine.replay_to(target_seq=10)
print(f"Facts at seq 10: {state.facts}")

# Replay all
state = engine.replay_all()
print(f"Total duration: {state.metrics['total_duration_ms']}ms")

# Analysis
trace = engine.get_trace()
bottlenecks = engine.find_bottlenecks()
confidence_timeline = engine.get_confidence_timeline()

# Compare chains
other_chain = Chain.load(Path("other.lctl.json"))
other_engine = ReplayEngine(other_chain)
diffs = engine.diff(other_engine)
```

### Event Types

| Type | Description | Data Payload |
|------|-------------|--------------|
| `step_start` | Agent begins work | `{intent, input_summary}` |
| `step_end` | Agent completes | `{outcome, output_summary, duration_ms, tokens}` |
| `fact_added` | New fact registered | `{id, text, confidence, source}` |
| `fact_modified` | Fact updated | `{id, text, confidence, reason}` |
| `tool_call` | Tool invocation | `{tool, input, output, duration_ms}` |
| `error` | Failure occurred | `{category, type, message, recoverable}` |
| `checkpoint` | State snapshot | `{state_hash, facts_snapshot}` |

### Confidence Thresholds

LCTL uses these default confidence thresholds:

| Confidence | Action |
|------------|--------|
| >= 0.80 | Proceed automatically |
| 0.60-0.79 | Warn, but proceed |
| 0.40-0.59 | Request verification |
| < 0.40 | Block, require human review |

## Protocol Specification

The LCTL v4.0 protocol is based on event sourcing:

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
      text: "SQL injection at line 45"
      confidence: 0.85
      source: "static-analysis"

  - seq: 3
    type: step_end
    timestamp: "2024-01-15T10:30:30Z"
    agent: "code-analyzer"
    data:
      outcome: "success"
      duration_ms: 30000
      tokens: {input: 500, output: 200}
```

For the complete specification, see [LLM-CONTEXT-TRANSFER.md](./LLM-CONTEXT-TRANSFER.md).

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/kimasplund/LLM-Context-Transfer-Language.git
cd LLM-Context-Transfer-Language

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e '.[dev]'

# Run tests
pytest

# Run tests with coverage
pytest --cov=lctl --cov-report=term-missing

# Format code
black lctl tests
ruff check lctl tests --fix
```

### Pull Request Guidelines

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `pytest`
4. Format code: `black lctl tests`
5. Check linting: `ruff check lctl tests`
6. Submit a pull request with a clear description

### Reporting Issues

When reporting bugs, please include:

- Python version
- LCTL version (`lctl --version`)
- Framework versions (if applicable)
- Minimal reproduction steps
- Expected vs actual behavior

## License

GNU Affero General Public License v3.0 (AGPLv3)

Copyright (c) 2024 LCTL Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

---

<p align="center">
  <sub>Built with event sourcing. The protocol is an implementation detail.</sub>
</p>
