# Using LCTL with OpenAI Agents SDK

This tutorial shows you how to integrate LCTL (LLM Context Transfer Language) v4.0 with the OpenAI Agents SDK for time-travel debugging and observability of agent runs, tool calls, and handoffs.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **OpenAI Agents SDK** installed:
   ```bash
   pip install openai-agents
   ```
4. **OpenAI API key** set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

First, verify that both LCTL and OpenAI Agents SDK are properly installed:

```python
from lctl.integrations.openai_agents import is_available

if is_available():
    print("OpenAI Agents SDK integration is ready!")
else:
    print("Please install the SDK: pip install openai-agents")
```

### Step 2: Import Required Modules

```python
from agents import Agent, Runner, function_tool
from lctl.integrations.openai_agents import (
    LCTLOpenAIAgentTracer,
    TracedAgent,
    trace_agent,
)
```

## Complete Working Examples

### Example 1: Basic Tracer with Runner

The most common pattern - use the tracer's `run_config` with `Runner.run()`:

```python
import asyncio
from agents import Agent, Runner
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

async def main():
    # Create the tracer
    tracer = LCTLOpenAIAgentTracer(chain_id="basic-assistant")

    # Create your agent
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant that provides concise answers.",
    )

    # Run with LCTL tracing
    result = await Runner.run(
        agent,
        input="What are three benefits of test-driven development?",
        run_config=tracer.run_config  # This enables tracing
    )

    print(f"Result: {result.final_output}")

    # Export the trace
    tracer.export("basic_assistant.lctl.json")

asyncio.run(main())
```

### Example 2: Using trace_agent Wrapper

Wrap an agent for convenient access to both agent and tracer:

```python
import asyncio
from agents import Agent, Runner
from lctl.integrations.openai_agents import trace_agent

async def main():
    # Create and wrap the agent
    agent = Agent(
        name="researcher",
        instructions="You are a research assistant that finds and summarizes information.",
    )

    traced = trace_agent(agent, chain_id="research-agent", verbose=True)

    # Access agent and tracer through the wrapper
    print(f"Agent name: {traced.agent.name}")
    print(f"Chain ID: {traced.tracer.chain.id}")

    # Run with tracing
    result = await Runner.run(
        traced.agent,
        input="Summarize the key features of Python 3.12",
        run_config=traced.run_config
    )

    print(f"Result: {result.final_output}")

    # Export via the wrapper
    traced.export("research_trace.lctl.json")

asyncio.run(main())
```

### Example 3: Agents with Tools

LCTL automatically captures tool invocations:

```python
import asyncio
from agents import Agent, Runner, function_tool
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

# Define tools using @function_tool decorator
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated response
    return f"Weather in {city}: Sunny, 72F, humidity 45%"

@function_tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@function_tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated response
    return f"Found 5 results for: {query}. Top result: ..."

async def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="tool-agent", verbose=True)

    # Create agent with tools
    agent = Agent(
        name="multi_tool_assistant",
        instructions="""You help users with weather info, calculations, and web searches.
        Use the appropriate tool for each request.""",
        tools=[get_weather, calculate, search_web],
    )

    # Run - tool calls will be traced
    result = await Runner.run(
        agent,
        input="What's the weather in Tokyo? Also, what is 25 * 17?",
        run_config=tracer.run_config
    )

    print(f"Result: {result.final_output}")
    print(f"Events captured: {len(tracer.chain.events)}")

    tracer.export("tool_calls.lctl.json")

asyncio.run(main())
```

### Example 4: Multi-Agent Handoffs

Trace control transfers between agents:

```python
import asyncio
from agents import Agent, Runner
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

async def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="triage-system", verbose=True)

    # Create specialized agents
    triage_agent = Agent(
        name="triage",
        instructions="""You route customer requests to the appropriate specialist.
        - For sales inquiries, hand off to the sales agent
        - For technical issues, hand off to the support agent""",
        handoffs=["sales", "support"],  # Define possible handoffs
    )

    sales_agent = Agent(
        name="sales",
        instructions="You handle sales inquiries. Provide pricing and product info.",
    )

    support_agent = Agent(
        name="support",
        instructions="You handle technical support. Troubleshoot issues and provide solutions.",
    )

    # Run triage agent (handoffs will be traced)
    result = await Runner.run(
        triage_agent,
        input="I'm having trouble logging into my account",
        run_config=tracer.run_config
    )

    print(f"Final result: {result.final_output}")
    tracer.export("triage_trace.lctl.json")

asyncio.run(main())
```

### Example 5: Manual Tracing with Context Manager

For fine-grained control or custom agent implementations:

```python
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="manual-tracing")

    # Use context manager for manual tracing
    with tracer.trace_agent_run("custom_agent", "Process user request"):
        # Simulate agent work
        print("Agent processing...")

        # Record tool calls manually
        tracer.record_tool_call(
            tool="database_query",
            input_data={"query": "SELECT * FROM users WHERE id = 123"},
            output_data={"name": "John", "email": "john@example.com"},
            duration_ms=150,
        )

        tracer.record_tool_call(
            tool="send_email",
            input_data={"to": "john@example.com", "subject": "Welcome"},
            output_data={"status": "sent", "message_id": "abc123"},
            duration_ms=200,
        )

    print(f"Recorded {len(tracer.chain.events)} events")
    tracer.export("manual_trace.lctl.json")

main()
```

### Example 6: Recording Handoffs Manually

When you need explicit control over handoff recording:

```python
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="explicit-handoffs")

    # Simulate triage agent
    with tracer.trace_agent_run("triage", "User request: billing inquiry"):
        print("Triage analyzing request...")
        # Record the handoff decision
        tracer.record_handoff("triage", "billing")

    # Simulate billing agent handling
    with tracer.trace_agent_run("billing", "Handle billing inquiry"):
        print("Billing agent processing...")

        tracer.record_tool_call(
            tool="lookup_invoice",
            input_data={"customer_id": "C123"},
            output_data={"invoice": "INV-001", "amount": 99.99, "status": "paid"},
            duration_ms=75,
        )

    print(f"Total events: {len(tracer.chain.events)}")
    tracer.export("handoffs_trace.lctl.json")

main()
```

### Example 7: Error Handling and Tracing

Errors are automatically captured in traces:

```python
import asyncio
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

async def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="error-handling")

    try:
        with tracer.trace_agent_run("risky_agent", "Attempting risky operation"):
            # Simulate an error
            raise RuntimeError("Database connection failed")

    except RuntimeError as e:
        print(f"Caught error: {e}")
        # Error is already recorded in the trace

    # Trace another agent that succeeds
    with tracer.trace_agent_run("fallback_agent", "Fallback operation"):
        print("Fallback succeeded")

    # Check what was recorded
    data = tracer.to_dict()
    error_events = [e for e in data["events"] if e["type"] == "error"]
    print(f"Errors recorded: {len(error_events)}")

    tracer.export("error_trace.lctl.json")

asyncio.run(main())
```

### Example 8: Async Context Manager

The context manager supports async operations:

```python
import asyncio
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

async def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="async-demo")

    async with tracer.trace_agent_run("async_agent", "Async processing"):
        # Simulate async work
        await asyncio.sleep(0.1)

        tracer.record_tool_call(
            tool="async_api_call",
            input_data={"endpoint": "/api/data"},
            output_data={"data": [1, 2, 3]},
            duration_ms=100,
        )

        await asyncio.sleep(0.05)

    tracer.export("async_trace.lctl.json")

asyncio.run(main())
```

### Example 9: Setting Output and Usage in Context

Track token usage and output summaries:

```python
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

def main():
    tracer = LCTLOpenAIAgentTracer(chain_id="usage-tracking")

    with tracer.trace_agent_run("assistant", "Answer user question") as ctx:
        # Simulate agent work
        answer = "Python is a high-level programming language..."

        # Set output summary
        ctx.set_output(answer[:100])

        # Set token usage
        ctx.set_usage(tokens_in=150, tokens_out=85)

        # Record any tool calls
        ctx.record_tool_call(
            tool="knowledge_base",
            input_data="python definition",
            output_data="Python is...",
            duration_ms=50,
        )

    tracer.export("usage_trace.lctl.json")

main()
```

## How to View Traces in the Dashboard

After exporting your traces, analyze them using LCTL's CLI tools:

### View Execution Flow

```bash
lctl trace tool_calls.lctl.json
```

Output:
```
Execution trace for chain: tool-agent
  [1] fact_added: agent_multi_tool_assistant_config
  [2] step_start: multi_tool_assistant (intent: agent_run)
  [3] tool_call: get_weather (150ms)
  [4] tool_call: calculate (25ms)
  [5] step_end: multi_tool_assistant (outcome: success, 2150ms)
```

### Time-Travel Replay

```bash
lctl replay tool_calls.lctl.json
```

Replay to before a specific tool call:

```bash
lctl replay --to-seq 3 tool_calls.lctl.json
```

### Performance Statistics

```bash
lctl stats triage_trace.lctl.json
```

Output:
```
Duration: 3.2s | Tokens: 420 in, 180 out | Cost: $0.012
Agents: 3 | Handoffs: 1 | Tool calls: 2
```

### Find Bottlenecks

```bash
lctl bottleneck tool_calls.lctl.json
```

Output:
```
get_weather: 45% of time (975ms)
  - Consider caching weather data
search_web: 35% of time (750ms)
calculate: 20% of time (425ms)
```

### Compare Agent Runs

```bash
lctl diff run_v1.lctl.json run_v2.lctl.json
```

### Launch Web UI

```bash
lctl debug triage_trace.lctl.json
```

## Common Patterns and Best Practices

### 1. Use Descriptive Chain IDs

```python
# Good - descriptive and versioned
tracer = LCTLOpenAIAgentTracer(chain_id="customer-support-v2")
tracer = LCTLOpenAIAgentTracer(chain_id="code-review-agent")

# Avoid - generic names
tracer = LCTLOpenAIAgentTracer(chain_id="test")
tracer = LCTLOpenAIAgentTracer()  # Auto-generates UUID
```

### 2. Enable Verbose Mode for Development

```python
# Development - see real-time tracing output
tracer = LCTLOpenAIAgentTracer(chain_id="dev-agent", verbose=True)

# Production - silent tracing
tracer = LCTLOpenAIAgentTracer(chain_id="prod-agent", verbose=False)
```

### 3. Reuse Sessions for Related Operations

```python
from lctl import LCTLSession
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

# Create a shared session
session = LCTLSession(chain_id="multi-agent-pipeline")

# Multiple tracers share the session
tracer1 = LCTLOpenAIAgentTracer(session=session)
tracer2 = LCTLOpenAIAgentTracer(session=session)

# All events go to the same trace
# ... run agents ...

session.export("combined_trace.lctl.json")
```

### 4. Analyze Traces Programmatically

```python
from lctl import Chain, ReplayEngine

# Load the trace
chain = Chain.load("tool_calls.lctl.json")
engine = ReplayEngine(chain)

# Replay all events
state = engine.replay_all()

# Access metrics
print(f"Total events: {state.metrics['event_count']}")
print(f"Duration: {state.metrics['total_duration_ms']}ms")
print(f"Tokens in: {state.metrics['total_tokens_in']}")
print(f"Tokens out: {state.metrics['total_tokens_out']}")
print(f"Errors: {state.metrics['error_count']}")

# Get facts discovered
print("\nFacts:")
for fact_id, fact in state.facts.items():
    print(f"  {fact_id}: {fact['text'][:50]}...")

# Find bottlenecks
bottlenecks = engine.find_bottlenecks()
print("\nBottlenecks:")
for b in bottlenecks[:3]:
    print(f"  {b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")
```

### 5. Export as Dictionary for Integration

```python
trace_data = tracer.to_dict()

# Send to monitoring system
import json
import requests

requests.post(
    "https://monitoring.example.com/traces",
    json=trace_data
)

# Or log for analysis
print(json.dumps(trace_data, indent=2))
```

### 6. Handle Multiple Runs

```python
import asyncio
from agents import Agent, Runner
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

async def process_batch(questions: list):
    tracer = LCTLOpenAIAgentTracer(chain_id="batch-processing")

    agent = Agent(name="batch_agent", instructions="Answer questions concisely")

    for i, question in enumerate(questions):
        # Each question gets traced
        with tracer.trace_agent_run(f"question_{i}", f"Processing: {question[:50]}"):
            result = await Runner.run(
                agent,
                input=question,
                run_config=tracer.run_config
            )
            print(f"Q{i}: {result.final_output[:50]}...")

    tracer.export("batch_trace.lctl.json")

questions = [
    "What is Python?",
    "Explain machine learning",
    "What is REST API?",
]
asyncio.run(process_batch(questions))
```

## Troubleshooting

### Issue: OpenAI Agents SDK Not Found

**Error**: `OpenAIAgentsNotAvailableError: OpenAI Agents SDK is not installed`

**Solution**: Install the SDK:
```bash
pip install openai-agents
```

### Issue: No Events in Trace

**Problem**: Trace file has no events.

**Solutions**:

1. Ensure you're passing `run_config`:
   ```python
   # Correct
   result = await Runner.run(agent, input="...", run_config=tracer.run_config)

   # Wrong - no tracing
   result = await Runner.run(agent, input="...")
   ```

2. If using context managers, ensure code runs inside the `with` block:
   ```python
   with tracer.trace_agent_run("agent", "task"):
       # Work must happen HERE
       do_work()
   # Not here - already outside the trace
   ```

### Issue: Tool Calls Not Captured

**Problem**: Tool invocations don't appear in trace.

**Solutions**:

1. Use `@function_tool` decorator for tools:
   ```python
   @function_tool
   def my_tool(arg: str) -> str:
       """Tool description."""
       return result
   ```

2. Ensure tools are passed to the agent:
   ```python
   agent = Agent(name="...", tools=[my_tool, other_tool])
   ```

3. For manual tracing, use `record_tool_call`:
   ```python
   tracer.record_tool_call(
       tool="my_tool",
       input_data={"arg": "value"},
       output_data="result",
       duration_ms=100,
   )
   ```

### Issue: Handoffs Not Recorded

**Problem**: Agent handoffs don't appear in trace.

**Solution**: The SDK hooks should capture handoffs automatically. If not:
```python
# Record manually
tracer.record_handoff("from_agent", "to_agent")
```

### Issue: Async Context Manager Errors

**Problem**: `async with` raises errors.

**Solution**: Ensure you're in an async function:
```python
async def main():
    async with tracer.trace_agent_run("agent", "task"):
        await do_async_work()

asyncio.run(main())
```

### Issue: Token Counts Show 0

**Problem**: Token usage is 0 in traces.

**Solution**: Token counts depend on the agent returning usage data. For manual tracing:
```python
with tracer.trace_agent_run("agent", "task") as ctx:
    # Set usage explicitly
    ctx.set_usage(tokens_in=100, tokens_out=50)
```

### Issue: Verbose Output Not Showing

**Problem**: No console output despite `verbose=True`.

**Solution**: Verbose output only appears during traced operations:
```python
tracer = LCTLOpenAIAgentTracer(chain_id="debug", verbose=True)

# This will print verbose output
with tracer.trace_agent_run("agent", "task"):
    tracer.record_tool_call(...)  # Prints "[LCTL] Tool ended: ..."
```

## What LCTL Captures

The OpenAI Agents SDK integration automatically captures:

| Event Type | Description |
|------------|-------------|
| `step_start` | Agent run begins |
| `step_end` | Agent run completes with outcome and duration |
| `tool_call` | Tool invocation with input/output/duration |
| `fact_added` | Agent config, handoffs, instructions |
| `error` | Exceptions with context |

## Trace Structure Example

Here's what a typical OpenAI Agents trace looks like:

```json
{
  "lctl": "4.0",
  "chain": {
    "id": "tool-agent"
  },
  "events": [
    {
      "seq": 1,
      "type": "fact_added",
      "agent": "multi_tool_assistant",
      "data": {
        "id": "agent_multi_tool_assistant_instructions",
        "text": "Agent instructions: You help users with weather info..."
      }
    },
    {
      "seq": 2,
      "type": "step_start",
      "agent": "multi_tool_assistant",
      "data": {
        "intent": "agent_run",
        "input_summary": "What's the weather in Tokyo?"
      }
    },
    {
      "seq": 3,
      "type": "tool_call",
      "agent": "system",
      "data": {
        "tool": "get_weather",
        "input": "Tokyo",
        "output": "Weather in Tokyo: Sunny, 72F",
        "duration_ms": 150
      }
    },
    {
      "seq": 4,
      "type": "step_end",
      "agent": "multi_tool_assistant",
      "data": {
        "outcome": "success",
        "output_summary": "The weather in Tokyo is sunny...",
        "duration_ms": 2150,
        "tokens": {"input": 85, "output": 45}
      }
    }
  ]
}
```

## Next Steps

- Learn about [LCTL CLI commands](../cli/README.md)
- Explore [LangChain integration](./langchain-tutorial.md)
- Check out [CrewAI integration](./crewai-tutorial.md)
- See [AutoGen integration](./autogen-tutorial.md)
