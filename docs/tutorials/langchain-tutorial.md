# Using LCTL with LangChain

This tutorial shows you how to integrate LCTL (LLM Context Transfer Language) v4.0 with LangChain chains and agents for time-travel debugging and observability.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **LangChain** installed:
   ```bash
   pip install langchain langchain-openai langchain-core
   ```
4. **OpenAI API key** (or another LLM provider) set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

First, verify that both LCTL and LangChain are properly installed:

```python
from lctl.integrations.langchain import langchain_available

if langchain_available():
    print("LangChain integration is ready!")
else:
    print("Please install LangChain: pip install langchain langchain-openai")
```

### Step 2: Import Required Modules

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import (
    LCTLCallbackHandler,
    LCTLChain,
    trace_chain,
)
```

## Complete Working Examples

### Example 1: Using LCTLCallbackHandler Directly

The callback handler approach gives you full control and works with any LangChain invocation method.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import LCTLCallbackHandler

# Create the callback handler with a custom chain ID
handler = LCTLCallbackHandler(chain_id="qa-assistant")

# Build your LangChain chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions concisely."),
    ("human", "{input}"),
])
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm

# Invoke the chain with LCTL tracing
result = chain.invoke(
    {"input": "What is the capital of France?"},
    config={"callbacks": [handler]},
)

print(f"Answer: {result.content}")
print(f"Recorded {len(handler.chain.events)} events")

# Export the trace for later analysis
handler.export("qa_trace.lctl.json")
```

### Example 2: Using the LCTLChain Wrapper

The wrapper approach automatically adds tracing to all invocations without needing to pass callbacks manually.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import LCTLChain

# Build your chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor. Show your work."),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm

# Wrap it with LCTL tracing
traced = LCTLChain(chain, chain_id="math-tutor")

# Use normally - tracing happens automatically
result = traced.invoke({"question": "What is 15% of 200?"})
print(f"Answer: {result.content}")

# Export when done
traced.export("math_tutor_trace.lctl.json")
```

### Example 3: Using the trace_chain Helper

The most concise way to add LCTL tracing:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import trace_chain

# Build and trace in one step
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm

# One-liner to add tracing
traced = trace_chain(chain, chain_id="quick-trace")

# Use the traced chain
result = traced.invoke({"input": "Hello, world!"})
traced.export("quick_trace.lctl.json")
```

### Example 4: Tracing Chains with Tools

LCTL automatically captures tool invocations and their results:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import LCTLCallbackHandler

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated response
    return f"The weather in {city} is sunny, 72F"

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Bind tools to the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo").bind_tools([get_weather, calculate])

# Create handler and invoke
handler = LCTLCallbackHandler(chain_id="tool-using-agent")

result = llm.invoke(
    "What's the weather in Paris? Also, what is 25 * 4?",
    config={"callbacks": [handler]},
)

print(f"Result: {result}")
print(f"Events captured: {len(handler.chain.events)}")

# Export trace including tool calls
handler.export("tool_calls_trace.lctl.json")
```

### Example 5: Tracing Streaming Responses

LCTL captures events during streaming invocations:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import trace_chain

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller."),
    ("human", "{request}"),
])
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
chain = prompt | llm

traced = trace_chain(chain, chain_id="storyteller")

print("Streaming response:")
for chunk in traced.stream({"request": "Tell me a short joke"}):
    print(chunk.content, end="", flush=True)
print("\n")

traced.export("streaming_trace.lctl.json")
```

### Example 6: Async Operations

LCTL supports async LangChain operations:

```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lctl.integrations.langchain import LCTLChain

async def main():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You translate text to French."),
        ("human", "{text}"),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm

    traced = LCTLChain(chain, chain_id="translator")

    # Async invoke
    result = await traced.ainvoke({"text": "Hello, how are you?"})
    print(f"Translation: {result.content}")

    traced.export("async_trace.lctl.json")

asyncio.run(main())
```

## How to View Traces in the Dashboard

After exporting your traces, you can analyze them using LCTL's CLI tools:

### View Execution Flow

```bash
lctl trace qa_trace.lctl.json
```

Output:
```
Execution trace for chain: qa-assistant
  [1] step_start: llm (intent: chat_model_call)
  [2] step_end: llm (outcome: success, 1250ms)
```

### Time-Travel Replay

Step through your chain execution:

```bash
lctl replay qa_trace.lctl.json
```

Replay to a specific event:

```bash
lctl replay --to-seq 5 qa_trace.lctl.json
```

### Performance Statistics

```bash
lctl stats qa_trace.lctl.json
```

Output:
```
Duration: 1.25s | Tokens: 156 in, 42 out | Cost: $0.002
```

### Find Bottlenecks

```bash
lctl bottleneck tool_calls_trace.lctl.json
```

Output:
```
get_weather: 45% of time (850ms)
calculate: 30% of time (550ms)
llm: 25% of time (450ms)
```

### Compare Traces

```bash
lctl diff trace_v1.lctl.json trace_v2.lctl.json
```

### Launch Web UI (if available)

```bash
lctl debug qa_trace.lctl.json
```

## Common Patterns and Best Practices

### 1. Use Meaningful Chain IDs

Always provide descriptive chain IDs for easier debugging:

```python
# Good - descriptive and unique
handler = LCTLCallbackHandler(chain_id="customer-support-v2")
handler = LCTLCallbackHandler(chain_id="code-review-agent")

# Avoid - generic names
handler = LCTLCallbackHandler(chain_id="test")
handler = LCTLCallbackHandler()  # Auto-generates UUID
```

### 2. Trace Complex Pipelines

When tracing multi-step pipelines, use a single session:

```python
from lctl import LCTLSession
from lctl.integrations.langchain import LCTLCallbackHandler

# Create a shared session
session = LCTLSession(chain_id="multi-step-pipeline")

# Use with multiple chains
handler1 = LCTLCallbackHandler(session=session)
handler2 = LCTLCallbackHandler(session=session)

# Both chains trace to the same session
result1 = chain1.invoke(input1, config={"callbacks": [handler1]})
result2 = chain2.invoke(input2, config={"callbacks": [handler2]})

# Export combined trace
session.export("pipeline_trace.lctl.json")
```

### 3. Analyze Traces Programmatically

```python
from lctl import Chain, ReplayEngine

# Load and analyze a trace
chain = Chain.load("qa_trace.lctl.json")
engine = ReplayEngine(chain)

# Replay all events
state = engine.replay_all()

# Get metrics
print(f"Total events: {state.metrics['event_count']}")
print(f"Duration: {state.metrics['total_duration_ms']}ms")
print(f"Tokens in: {state.metrics['total_tokens_in']}")
print(f"Tokens out: {state.metrics['total_tokens_out']}")

# Get execution trace
trace = engine.get_trace()
for step in trace:
    print(f"[{step['seq']}] {step['type']}: {step['agent']}")

# Find bottlenecks
bottlenecks = engine.find_bottlenecks()
for b in bottlenecks[:3]:
    print(f"{b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")
```

### 4. Export as Dictionary

For integration with other systems:

```python
trace_data = handler.to_dict()

# Send to monitoring system
import json
print(json.dumps(trace_data, indent=2))
```

## Troubleshooting

### Issue: LangChain Not Found

**Error**: `ImportError: LangChain is not installed`

**Solution**: Install LangChain and its dependencies:
```bash
pip install langchain langchain-core langchain-openai
```

### Issue: No Events Recorded

**Problem**: Trace file has 0 events.

**Solutions**:
1. Ensure you pass the callback in the config:
   ```python
   # Correct
   chain.invoke(input, config={"callbacks": [handler]})

   # Wrong - callbacks not passed
   chain.invoke(input)
   ```

2. If using `LCTLChain`, ensure you call methods on the wrapper:
   ```python
   traced = LCTLChain(chain)
   traced.invoke(input)  # Correct - uses wrapper
   chain.invoke(input)   # Wrong - bypasses tracing
   ```

### Issue: Token Counts Show 0

**Problem**: Token usage shows 0 in traces.

**Solution**: This depends on the LLM provider returning usage data. Ensure you're using a provider that reports token usage (like OpenAI). Some local models don't report tokens.

### Issue: Tool Calls Not Captured

**Problem**: Tool invocations aren't showing in traces.

**Solution**: Ensure tools are bound to the LLM and you're using the callback:
```python
llm = ChatOpenAI().bind_tools([my_tool])
result = llm.invoke(prompt, config={"callbacks": [handler]})
```

### Issue: Async Traces Incomplete

**Problem**: Async operations have missing events.

**Solution**: Ensure you're using the async methods on the wrapper:
```python
traced = LCTLChain(chain)
result = await traced.ainvoke(input)  # Correct
result = await chain.ainvoke(input)   # Wrong - bypasses tracing
```

## What LCTL Captures

The LangChain integration automatically captures:

| Event Type | Description |
|------------|-------------|
| `step_start` | LLM/chain/tool execution begins |
| `step_end` | Execution completes with outcome and duration |
| `tool_call` | Tool invocation with input/output |
| `fact_added` | Agent actions and decisions |
| `error` | Exceptions with context |

## Next Steps

- Learn about [LCTL CLI commands](../cli/README.md)
- Explore [CrewAI integration](./crewai-tutorial.md)
- Check out [AutoGen integration](./autogen-tutorial.md)
- See [OpenAI Agents integration](./openai-agents-tutorial.md)
