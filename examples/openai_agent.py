"""
OpenAI Agents SDK Integration Example

This example demonstrates how to use LCTL to trace an OpenAI Agent.
It covers:
1. Agent and Runner setup
2. Tracer initialization
3. Using the run_config to capture events
"""

import asyncio
import os

try:
    from agents import Agent, Runner
    from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer
except ImportError:
    print("Please install openai-agents and lctl[openai-agents] to run this example")
    exit(1)

# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. This example may fail if run directly.")


async def main():
    print("=== LCTL OpenAI Agents Example ===")

    # 1. Create the tracer
    tracer = LCTLOpenAIAgentTracer(chain_id="example-openai-math")

    # 2. Define the agent
    agent = Agent(
        name="CalculateBot",
        instructions="You help with basic math. Be concise.",
    )

    # 3. Run with LCTL tracing via run_config
    query = "What is 10 + 5?"
    print(f"Running agent with query: '{query}'...")

    result = await Runner.run(
        agent,
        input=query,
        run_config=tracer.run_config,  # Critical: Attach the tracer
    )

    print(f"Agent Response: {result.final_output}")

    # 4. Export the trace
    output_file = "openai_agent_trace.lctl.json"
    tracer.export(output_file)
    print(f"Trace exported to {output_file}")
    print(f"Captured {len(tracer.chain.events)} events.")


if __name__ == "__main__":
    asyncio.run(main())
