"""Example: Using LCTL with LangChain.

This example demonstrates how to trace LangChain chains with LCTL
for time-travel debugging and observability.

Prerequisites:
    pip install lctl langchain langchain-openai
"""

from lctl.integrations.langchain import (
    LCTLCallbackHandler,
    LCTLChain,
    trace_chain,
    langchain_available,
)


def example_callback_handler():
    """Example 1: Using LCTLCallbackHandler directly.

    This approach gives you full control over the callback handler
    and works with any LangChain invocation method.
    """
    if not langchain_available():
        print("LangChain not installed. Skipping example.")
        return

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    handler = LCTLCallbackHandler(chain_id="callback-example")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm

    result = chain.invoke(
        {"input": "What is the capital of France?"},
        config={"callbacks": [handler]},
    )

    print(f"Result: {result.content}")
    print(f"Recorded {len(handler.chain.events)} events")

    handler.export("callback_trace.lctl.json")
    print("Trace exported to callback_trace.lctl.json")


def example_chain_wrapper():
    """Example 2: Using LCTLChain wrapper.

    This approach wraps your chain and automatically adds tracing
    to all invocations without needing to pass callbacks manually.
    """
    if not langchain_available():
        print("LangChain not installed. Skipping example.")
        return

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm

    traced = LCTLChain(chain, chain_id="wrapper-example")

    result = traced.invoke({"input": "What is 2 + 2?"})

    print(f"Result: {result.content}")
    print(f"Recorded {len(traced.chain.events)} events")

    traced.export("wrapper_trace.lctl.json")
    print("Trace exported to wrapper_trace.lctl.json")


def example_trace_chain_helper():
    """Example 3: Using trace_chain helper function.

    This is the most concise way to add LCTL tracing to a chain.
    """
    if not langchain_available():
        print("LangChain not installed. Skipping example.")
        return

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm

    traced = trace_chain(chain, chain_id="helper-example")

    result = traced.invoke({"input": "Hello, world!"})

    print(f"Result: {result.content}")
    traced.export("helper_trace.lctl.json")


def example_with_tools():
    """Example 4: Tracing chains with tools.

    LCTL automatically captures tool invocations and their results.
    """
    if not langchain_available():
        print("LangChain not installed. Skipping example.")
        return

    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"The weather in {city} is sunny, 72F"

    @tool
    def calculate(expression: str) -> str:
        """Calculate a math expression."""
        return str(eval(expression))

    llm = ChatOpenAI(model="gpt-3.5-turbo").bind_tools([get_weather, calculate])

    handler = LCTLCallbackHandler(chain_id="tools-example")

    result = llm.invoke(
        "What's the weather in Paris?",
        config={"callbacks": [handler]},
    )

    print(f"Result: {result}")
    print(f"Recorded {len(handler.chain.events)} events")

    handler.export("tools_trace.lctl.json")


def example_streaming():
    """Example 5: Tracing streaming responses.

    LCTL captures events during streaming invocations.
    """
    if not langchain_available():
        print("LangChain not installed. Skipping example.")
        return

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    chain = prompt | llm

    traced = trace_chain(chain, chain_id="streaming-example")

    print("Streaming response:")
    for chunk in traced.stream({"input": "Tell me a short joke"}):
        print(chunk.content, end="", flush=True)
    print()

    traced.export("streaming_trace.lctl.json")


def example_analyze_trace():
    """Example 6: Analyzing a trace with LCTL replay engine.

    After recording a trace, you can analyze it with LCTL's
    replay engine and analysis tools.
    """
    from pathlib import Path
    from lctl import Chain, ReplayEngine

    trace_path = Path("callback_trace.lctl.json")
    if not trace_path.exists():
        print("Run example_callback_handler() first to generate a trace.")
        return

    chain = Chain.load(trace_path)
    engine = ReplayEngine(chain)

    state = engine.replay_all()
    print(f"Total events: {state.metrics['event_count']}")
    print(f"Total duration: {state.metrics['total_duration_ms']}ms")
    print(f"Tokens in: {state.metrics['total_tokens_in']}")
    print(f"Tokens out: {state.metrics['total_tokens_out']}")

    trace = engine.get_trace()
    print("\nExecution trace:")
    for step in trace:
        print(f"  [{step['seq']}] {step['type']}: {step['agent']}")

    bottlenecks = engine.find_bottlenecks()
    if bottlenecks:
        print("\nBottlenecks:")
        for b in bottlenecks[:3]:
            print(f"  {b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")


if __name__ == "__main__":
    print("LCTL LangChain Integration Examples")
    print("=" * 40)

    if not langchain_available():
        print("\nLangChain is not installed.")
        print("Install with: pip install langchain langchain-openai")
        print("\nThe integration gracefully degrades when LangChain is not available.")
    else:
        print("\nLangChain is available!")
        print("\nTo run the examples, uncomment the desired example below:")
        print("  example_callback_handler()")
        print("  example_chain_wrapper()")
        print("  example_trace_chain_helper()")
        print("  example_with_tools()")
        print("  example_streaming()")
        print("  example_analyze_trace()")
