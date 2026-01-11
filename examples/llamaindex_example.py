"""Example: Using LCTL with LlamaIndex for traced RAG pipelines.

This example demonstrates how to use LCTL to trace LlamaIndex query engines,
chat engines, and RAG pipelines for time-travel debugging and observability.

Prerequisites:
    pip install lctl llama-index

Usage:
    python llamaindex_example.py
"""

import os

try:
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.callbacks import CallbackManager
    LLAMAINDEX_INSTALLED = True
except ImportError:
    try:
        from llama_index import VectorStoreIndex, Document
        from llama_index.callbacks import CallbackManager
        LLAMAINDEX_INSTALLED = True
    except ImportError:
        LLAMAINDEX_INSTALLED = False
        print("LlamaIndex is not installed. Install with: pip install llama-index")
        print("This example shows the intended usage pattern.\n")

from lctl.integrations.llamaindex import (
    LLAMAINDEX_AVAILABLE,
    LCTLLlamaIndexCallback,
    LCTLQueryEngine,
    LCTLChatEngine,
    trace_query_engine,
    trace_chat_engine,
)


def main() -> None:
    """Run sample LlamaIndex operations with LCTL tracing."""

    if not LLAMAINDEX_AVAILABLE:
        print("=" * 60)
        print("LlamaIndex Integration Example (Mock Mode)")
        print("=" * 60)
        print()
        print("LlamaIndex is not installed. Below is how the code would look:")
        print()
        print_example_code()
        return

    print("=" * 60)
    print("Example 1: Using LCTL Callback with Query Engine")
    print("=" * 60)
    print()

    documents = [
        Document(text="Machine learning is a subset of artificial intelligence."),
        Document(text="Deep learning uses neural networks with many layers."),
        Document(text="Natural language processing enables computers to understand text."),
    ]

    callback = LCTLLlamaIndexCallback(chain_id="rag-pipeline")
    callback_manager = callback.get_callback_manager()

    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager,
    )

    query_engine = index.as_query_engine(callback_manager=callback_manager)

    print("Query engine configured with LCTL tracing:")
    print(f"  - Chain ID: {callback.session.chain.id}")
    print()

    response = query_engine.query("What is machine learning?")
    print(f"Response: {response}")
    print()

    callback.export("rag_pipeline.lctl.json")
    print("Trace exported to: rag_pipeline.lctl.json")
    print()

    print("=" * 60)
    print("Example 2: Using trace_query_engine Helper")
    print("=" * 60)
    print()

    basic_engine = index.as_query_engine()

    traced_engine = trace_query_engine(basic_engine, chain_id="traced-query")

    print("Wrapped existing query engine with tracing:")
    print(f"  - Chain ID: {traced_engine.session.chain.id}")
    print()

    response = traced_engine.query("What is deep learning?")
    print(f"Response: {response}")
    print()

    traced_engine.export("traced_query.lctl.json")
    print("Trace exported to: traced_query.lctl.json")
    print()

    print("=" * 60)
    print("Example 3: Using LCTLQueryEngine Wrapper")
    print("=" * 60)
    print()

    another_engine = index.as_query_engine()

    lctl_engine = LCTLQueryEngine(another_engine, chain_id="lctl-wrapper")

    print("Using LCTLQueryEngine wrapper:")
    print(f"  - Chain ID: {lctl_engine.session.chain.id}")
    print()

    response = lctl_engine.query("What is NLP?")
    print(f"Response: {response}")
    print()

    lctl_engine.export("lctl_wrapper.lctl.json")
    print("Trace exported to: lctl_wrapper.lctl.json")
    print()

    print("=" * 60)
    print("Example 4: Chat Engine Tracing")
    print("=" * 60)
    print()

    chat_engine = index.as_chat_engine()

    traced_chat = trace_chat_engine(chat_engine, chain_id="traced-chat")

    print("Chat engine with LCTL tracing:")
    print(f"  - Chain ID: {traced_chat.session.chain.id}")
    print()

    response = traced_chat.chat("What topics are covered in the documents?")
    print(f"Response: {response}")
    print()

    response = traced_chat.chat("Tell me more about deep learning")
    print(f"Response: {response}")
    print()

    traced_chat.export("chat_trace.lctl.json")
    print("Trace exported to: chat_trace.lctl.json")
    print()

    print("=" * 60)
    print("Example 5: Examining the Trace")
    print("=" * 60)
    print()

    trace_data = traced_chat.to_dict()
    print(f"Total events recorded: {len(trace_data['events'])}")
    print()

    event_types = {}
    for event in trace_data["events"]:
        event_type = event["type"]
        event_types[event_type] = event_types.get(event_type, 0) + 1

    print("Event breakdown:")
    for event_type, count in sorted(event_types.items()):
        print(f"  - {event_type}: {count}")
    print()


def print_example_code() -> None:
    """Print example code when LlamaIndex is not installed."""
    code = '''
from llama_index.core import VectorStoreIndex, Document
from lctl.integrations.llamaindex import (
    LCTLLlamaIndexCallback,
    LCTLQueryEngine,
    LCTLChatEngine,
    trace_query_engine,
    trace_chat_engine,
)

# Create documents
documents = [
    Document(text="Machine learning is a subset of AI."),
    Document(text="Deep learning uses neural networks."),
]

# Option 1: Use LCTL callback directly with index creation
callback = LCTLLlamaIndexCallback(chain_id="my-rag")
callback_manager = callback.get_callback_manager()

index = VectorStoreIndex.from_documents(
    documents,
    callback_manager=callback_manager,
)

query_engine = index.as_query_engine(callback_manager=callback_manager)
response = query_engine.query("What is machine learning?")

callback.export("rag_trace.lctl.json")


# Option 2: Wrap existing query engine with tracing
query_engine = index.as_query_engine()
traced = trace_query_engine(query_engine, chain_id="traced-qa")

response = traced.query("What is deep learning?")
traced.export("query_trace.lctl.json")


# Option 3: Use LCTLQueryEngine wrapper class
lctl_engine = LCTLQueryEngine(query_engine, chain_id="wrapped-qa")
response = lctl_engine.query("Tell me about neural networks")
lctl_engine.export("wrapped_trace.lctl.json")


# Option 4: Trace chat engines for conversational RAG
chat_engine = index.as_chat_engine()
traced_chat = trace_chat_engine(chat_engine, chain_id="chat-session")

response = traced_chat.chat("Hello!")
response = traced_chat.chat("What topics do you know about?")
response = traced_chat.chat("Tell me more about ML")

traced_chat.export("chat_trace.lctl.json")


# Option 5: Use LCTLChatEngine wrapper class
lctl_chat = LCTLChatEngine(chat_engine, chain_id="wrapped-chat")
response = lctl_chat.chat("How are you?")
lctl_chat.reset()  # Reset chat history
lctl_chat.export("wrapped_chat.lctl.json")


# Examining the trace programmatically
trace_data = traced.to_dict()
print(f"Chain ID: {trace_data['chain']['id']}")
print(f"Events recorded: {len(trace_data['events'])}")

for event in trace_data['events']:
    print(f"  [{event['seq']}] {event['type']} by {event['agent']}")
'''
    print(code)
    print()
    print("The trace captures:")
    print("  - LLM calls (prompts, completions, token counts)")
    print("  - Query execution (query string, response)")
    print("  - Retrieval operations (queries, retrieved nodes)")
    print("  - Embedding generation")
    print("  - Response synthesis")
    print("  - Reranking operations")
    print("  - Template processing")
    print("  - Document chunking")
    print()
    print("LlamaIndex event types mapped to LCTL events:")
    print("  - LLM_START/END -> step_start/step_end (agent='llm')")
    print("  - QUERY_START/END -> step_start/step_end (agent='query_engine')")
    print("  - RETRIEVE_START/END -> step_start/step_end + tool_call + fact_added")
    print("  - EMBEDDING_START/END -> step_start/step_end (agent='embedding')")
    print("  - SYNTHESIZE_START/END -> step_start/step_end (agent='synthesizer')")
    print("  - RERANKING_START/END -> step_start/step_end (agent='reranker')")
    print()


if __name__ == "__main__":
    main()
