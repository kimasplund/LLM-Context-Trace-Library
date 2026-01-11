"""Example: Using LCTL with OpenAI Agents SDK.

This example demonstrates how to trace OpenAI Agents SDK agent runs
with LCTL for time-travel debugging and observability.

Prerequisites:
    pip install lctl openai-agents

Usage:
    python openai_agents_example.py
"""

import os

try:
    from agents import Agent, Runner, Tool, function_tool
    OPENAI_AGENTS_INSTALLED = True
except ImportError:
    OPENAI_AGENTS_INSTALLED = False
    print("OpenAI Agents SDK is not installed. Install with: pip install openai-agents")
    print("This example shows the intended usage pattern.\n")

from lctl.integrations.openai_agents import (
    OPENAI_AGENTS_AVAILABLE,
    LCTLOpenAIAgentTracer,
    TracedAgent,
    trace_agent,
    is_available,
)


def example_basic_tracer():
    """Example 1: Using LCTLOpenAIAgentTracer directly.

    This approach gives you full control over the tracer
    and works with the Runner.run() method.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        print("OpenAI Agents SDK not installed. Skipping example.")
        return

    tracer = LCTLOpenAIAgentTracer(chain_id="basic-example")

    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant that provides concise answers.",
    )

    print("To run this agent:")
    print('  result = await Runner.run(agent, input="Hello", run_config=tracer.run_config)')
    print()

    print(f"Tracer configured with chain ID: {tracer.chain.id}")
    print(f"Run config: {tracer.run_config}")
    print()

    tracer.export("basic_trace.lctl.json")
    print("Exported initial trace to: basic_trace.lctl.json")


def example_traced_agent_wrapper():
    """Example 2: Using trace_agent wrapper.

    This approach wraps your agent and provides convenient
    access to both the agent and tracer.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        print("OpenAI Agents SDK not installed. Skipping example.")
        return

    agent = Agent(
        name="researcher",
        instructions="You are a research assistant that finds and summarizes information.",
    )

    traced = trace_agent(agent, chain_id="wrapper-example", verbose=True)

    print(f"Wrapped agent: {traced.agent.name}")
    print(f"Chain ID: {traced.tracer.chain.id}")
    print()

    print("To run the traced agent:")
    print('  result = await Runner.run(traced.agent, input="...", run_config=traced.run_config)')
    print()

    traced.export("wrapper_trace.lctl.json")
    print("Exported trace to: wrapper_trace.lctl.json")


def example_manual_tracing():
    """Example 3: Manual tracing with context manager.

    This approach is useful when you need fine-grained control
    or when using custom agent implementations.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        print("OpenAI Agents SDK not installed. Skipping example.")
        return

    tracer = LCTLOpenAIAgentTracer(chain_id="manual-example")

    with tracer.trace_agent_run("custom_agent", "Process user request"):
        print("Simulating agent execution...")

        tracer.record_tool_call(
            tool="search",
            input_data={"query": "LCTL documentation"},
            output_data={"results": ["result1", "result2"]},
            duration_ms=150,
        )

        tracer.record_tool_call(
            tool="summarize",
            input_data="Long text content...",
            output_data="Summary of the content",
            duration_ms=200,
        )

    print("Agent run completed")
    print(f"Recorded {len(tracer.chain.events)} events")

    tracer.export("manual_trace.lctl.json")
    print("Exported trace to: manual_trace.lctl.json")


def example_with_tools():
    """Example 4: Tracing agents with tools.

    LCTL automatically captures tool invocations and their results
    when using the run hooks.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        print("OpenAI Agents SDK not installed. Skipping example.")
        return

    @function_tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny, 72F"

    @function_tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    agent = Agent(
        name="weather_assistant",
        instructions="You help users with weather information and calculations.",
        tools=[get_weather, calculate],
    )

    tracer = LCTLOpenAIAgentTracer(chain_id="tools-example", verbose=True)

    print(f"Agent configured with tools: {[t.name for t in agent.tools]}")
    print(f"Use tracer.run_config with Runner.run() to capture tool calls")
    print()

    tracer.export("tools_trace.lctl.json")
    print("Exported trace to: tools_trace.lctl.json")


def example_multi_agent_handoffs():
    """Example 5: Tracing multi-agent systems with handoffs.

    LCTL captures handoff events when control transfers between agents.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        print("OpenAI Agents SDK not installed. Skipping example.")
        return

    tracer = LCTLOpenAIAgentTracer(chain_id="handoffs-example")

    triage_agent = Agent(
        name="triage",
        instructions="You route requests to the appropriate specialist.",
    )

    sales_agent = Agent(
        name="sales",
        instructions="You handle sales inquiries.",
    )

    support_agent = Agent(
        name="support",
        instructions="You handle technical support questions.",
    )

    with tracer.trace_agent_run("triage", "User request: 'I need help with billing'"):
        tracer.record_handoff("triage", "support")

    with tracer.trace_agent_run("support", "Handling billing inquiry"):
        tracer.record_tool_call(
            tool="lookup_account",
            input_data={"user_id": "12345"},
            output_data={"status": "active", "balance": 0},
            duration_ms=50,
        )

    print(f"Recorded {len(tracer.chain.events)} events including handoff")
    tracer.export("handoffs_trace.lctl.json")
    print("Exported trace to: handoffs_trace.lctl.json")


def example_error_handling():
    """Example 6: Tracing errors in agent execution.

    LCTL captures errors and provides context for debugging.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        print("OpenAI Agents SDK not installed. Skipping example.")
        return

    tracer = LCTLOpenAIAgentTracer(chain_id="errors-example")

    try:
        with tracer.trace_agent_run("failing_agent", "Attempting risky operation"):
            raise RuntimeError("Simulated agent failure")
    except RuntimeError:
        print("Error was caught and recorded")

    print(f"Recorded {len(tracer.chain.events)} events including error")

    data = tracer.to_dict()
    error_events = [e for e in data["events"] if e["type"] == "error"]
    print(f"Error events: {len(error_events)}")

    tracer.export("errors_trace.lctl.json")
    print("Exported trace to: errors_trace.lctl.json")


def example_analyze_trace():
    """Example 7: Analyzing a trace with LCTL replay engine.

    After recording a trace, you can analyze it with LCTL's
    replay engine and analysis tools.
    """
    from pathlib import Path
    from lctl import Chain, ReplayEngine

    trace_path = Path("manual_trace.lctl.json")
    if not trace_path.exists():
        print("Run example_manual_tracing() first to generate a trace.")
        return

    chain = Chain.load(trace_path)
    engine = ReplayEngine(chain)

    state = engine.replay_all()
    print(f"Total events: {state.metrics['event_count']}")
    print(f"Total duration: {state.metrics['total_duration_ms']}ms")
    print(f"Tokens in: {state.metrics['total_tokens_in']}")
    print(f"Tokens out: {state.metrics['total_tokens_out']}")
    print(f"Errors: {state.metrics['error_count']}")

    print("\nFacts discovered:")
    for fact_id, fact in state.facts.items():
        print(f"  {fact_id}: {fact['text'][:50]}...")

    bottlenecks = engine.find_bottlenecks()
    if bottlenecks:
        print("\nBottlenecks:")
        for b in bottlenecks[:3]:
            print(f"  {b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")


def print_example_code():
    """Print example code when OpenAI Agents SDK is not installed."""
    code = '''
from agents import Agent, Runner, function_tool
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer, trace_agent

# Option 1: Use tracer directly with run_config
tracer = LCTLOpenAIAgentTracer(chain_id="my-agent")

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
)

result = await Runner.run(
    agent,
    input="What is the weather in Paris?",
    run_config=tracer.run_config
)

tracer.export("trace.lctl.json")

# Option 2: Use trace_agent wrapper
@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

agent = Agent(
    name="weather_bot",
    instructions="Help with weather queries.",
    tools=[get_weather],
)

traced = trace_agent(agent, chain_id="weather-bot")
result = await Runner.run(
    traced.agent,
    input="Weather in Tokyo?",
    run_config=traced.run_config
)
traced.export("weather_trace.lctl.json")

# Option 3: Manual tracing for custom scenarios
tracer = LCTLOpenAIAgentTracer(chain_id="custom")

with tracer.trace_agent_run("my_agent", "Processing request"):
    # Your custom logic here
    tracer.record_tool_call(
        tool="search",
        input_data="query",
        output_data="results",
        duration_ms=100
    )

tracer.export("custom_trace.lctl.json")
'''
    print(code)


def main():
    """Run all examples."""
    print("LCTL OpenAI Agents SDK Integration Examples")
    print("=" * 50)

    if not OPENAI_AGENTS_AVAILABLE:
        print("\nOpenAI Agents SDK is not installed.")
        print("Install with: pip install openai-agents")
        print("\nThe integration gracefully degrades when the SDK is not available.")
        print("\nExample code showing intended usage:")
        print("-" * 50)
        print_example_code()
        return

    print("\nOpenAI Agents SDK is available!")
    print("\nRunning examples...")
    print()

    print("-" * 50)
    print("Example 1: Basic Tracer")
    print("-" * 50)
    example_basic_tracer()
    print()

    print("-" * 50)
    print("Example 2: Traced Agent Wrapper")
    print("-" * 50)
    example_traced_agent_wrapper()
    print()

    print("-" * 50)
    print("Example 3: Manual Tracing")
    print("-" * 50)
    example_manual_tracing()
    print()

    print("-" * 50)
    print("Example 4: Agents with Tools")
    print("-" * 50)
    example_with_tools()
    print()

    print("-" * 50)
    print("Example 5: Multi-Agent Handoffs")
    print("-" * 50)
    example_multi_agent_handoffs()
    print()

    print("-" * 50)
    print("Example 6: Error Handling")
    print("-" * 50)
    example_error_handling()
    print()

    print("-" * 50)
    print("Example 7: Analyze Trace")
    print("-" * 50)
    example_analyze_trace()


if __name__ == "__main__":
    main()
