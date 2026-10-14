"""
PydanticAI Integration Example

This example demonstrates how to use LCTL to trace a PydanticAI agent.
It covers:
1. Basic agent setup
2. Tracing using the trace_agent wrapper
3. Tool usage tracing
4. Structured output
"""

import asyncio
import os

try:
    from pydantic_ai import Agent, RunContext

    from lctl.integrations.pydantic_ai import trace_agent
except ImportError:
    print("Please install pydantic-ai and lctl[pydantic-ai] to run this example")
    exit(1)

# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. This example may fail if run directly.")


async def main():
    print("=== LCTL PydanticAI Example ===")

    # 1. Define an agent with a tool
    agent = Agent("openai:gpt-4o", system_prompt="You are a helpful travel assistant.")

    @agent.tool
    async def get_weather(ctx: RunContext[str], city: str) -> str:
        """Get the current weather for a city."""
        print(f"  [Tool] Getting weather for {city}...")
        # Simulate an API call
        return "Sunny, 25C"

    # 2. Wrap the agent with LCTL tracing
    print("Initializing tracked agent...")
    traced_agent = trace_agent(agent, chain_id="example-pydantic-travel")

    # 3. Run the agent
    print("Running agent query...")
    query = "What is the weather in Barcelona?"

    # We use the traced_agent to run, which captures all events
    result = await traced_agent.run(query)

    print(f"Agent Response: {result.data}")

    # 4. Export the trace
    output_file = "pydantic_travel_trace.lctl.json"
    traced_agent.tracer.export(output_file)
    print(f"Trace exported to {output_file}")

    # Verify trace content
    print(f"Captured {len(traced_agent.tracer.chain.events)} events.")


if __name__ == "__main__":
    asyncio.run(main())
