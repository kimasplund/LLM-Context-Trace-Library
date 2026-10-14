# Using LCTL with PydanticAI

This tutorial shows you how to integrate LCTL (LLM Context Trace Library) v4.0 with PydanticAI for time-travel debugging and observability of agent runs, tool calls, and structured outputs.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **PydanticAI** installed:
   ```bash
   pip install pydantic-ai
   ```
4. **LLM API key** (e.g., OpenAI) set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

Verify that both LCTL and PydanticAI are properly installed:

```python
try:
    import pydantic_ai
    from lctl.integrations.pydantic_ai import trace_agent

    print("PydanticAI integration is ready!")
except ImportError:
    print("Please install required packages: pip install lctl[pydantic-ai]")
```

### Step 2: Import Required Modules

```python
from pydantic_ai import Agent
from lctl.integrations.pydantic_ai import trace_agent, LCTLPydanticAITracer
```

## Complete Working Examples

### Example 1: Basic Agent Tracing

The simplest way to trace an agent is using the `trace_agent` wrapper:

```python
import asyncio
from pydantic_ai import Agent
from lctl.integrations.pydantic_ai import trace_agent


async def main():
    # 1. Create your PydanticAI agent
    agent = Agent("openai:gpt-4o", system_prompt="You are a helpful assistant.")

    # 2. Wrap it with LCTL tracing
    traced_agent = trace_agent(agent, chain_id="basic-demo")

    # 3. Run normally
    result = await traced_agent.run("What is the capital of Finland?")

    print(f"Result: {result.data}")

    # 4. Export the trace
    traced_agent.tracer.export("basic_trace.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Tracing Tool_Calls

LCTL automatically captures tool usage defined in PydanticAI:

```python
import asyncio
from pydantic_ai import Agent, RunContext, tool
from lctl.integrations.pydantic_ai import trace_agent


async def main():
    agent = Agent("openai:gpt-4o", system_prompt="You supply weather information.")

    # Define a tool
    @agent.tool
    async def get_weather(ctx: RunContext[str], city: str) -> str:
        """Get the weather for a city."""
        return f"Weather in {city}: Snowy, -5C"

    # Trace the agent
    traced = trace_agent(agent, chain_id="weather-agent")

    # Run query requiring tools
    result = await traced.run("What is the weather in Helsinki?")

    print(result.data)
    traced.export("tool_trace.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Structured Responses

PydanticAI excels at structured data. LCTL captures the structured output accurately:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from lctl.integrations.pydantic_ai import trace_agent


class CityInfo(BaseModel):
    name: str
    population: int
    country: str


async def main():
    agent = Agent("openai:gpt-4o", result_type=CityInfo, system_prompt="Extract city information.")

    traced = trace_agent(agent, chain_id="structured-agent")

    result = await traced.run("Tell me about Tokyo.")

    print(f"City: {result.data.name}")
    print(f"Population: {result.data.population}")

    # The trace will contain the JSON representation of the CityInfo object
    traced.export("structured_trace.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Streaming Responses

LCTL captures streaming output as it arrives (summarized at the end):

```python
import asyncio
from pydantic_ai import Agent
from lctl.integrations.pydantic_ai import trace_agent


async def main():
    agent = Agent("openai:gpt-4o")
    traced = trace_agent(agent, chain_id="streaming-demo")

    async with traced.run_stream("Tell me a short story") as response:
        async for chunk in response.stream_text():
            print(chunk, end="", flush=True)

    traced.export("streaming_trace.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

## How to View Traces

Use the LCTL CLI to inspect your PydanticAI traces:

### Visual Execution Trace
```bash
lctl trace tool_trace.lctl.json
```
Output:
```
[1] step_start: pydantic_agent (intent: run)
[2] tool_call: get_weather (input: Helsinki)
[3] step_end: pydantic_agent (outcome: success)
```

### Launch Debugger
```bash
lctl debug tool_trace.lctl.json
```

## Troubleshooting

### Issue: "TracedAgent" object has no attribute 'X'
The `trace_agent` function returns a `TracedAgent` wrapper. It proxies most calls to the underlying agent, but if you need direct access, use `.agent`:
```python
traced = trace_agent(agent)
raw_agent = traced.agent
```

### Issue: Missing Token Counts
Ensure your model provider returns usage statistics. PydanticAI passes this through, and LCTL captures it from the `RunResult`.

## What LCTL Captures

| LCTL Event | PydanticAI Concept |
|------------|---------------------|
| `step_start` | `agent.run()` called |
| `step_end` | Run completes (with result data) |
| `tool_call` | `@agent.tool` execution |
| `error` | Exceptions raised during run |
