# Using LCTL with LlamaIndex

This tutorial shows you how to integrate LCTL (LLM Context Trace Library) with LlamaIndex for time-travel debugging and observability of query engines, chat engines, retrievers, and LLM calls.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.11+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **LlamaIndex** installed:
   ```bash
   pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
   ```
4. **OpenAI API key** set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

First, verify that both LCTL and LlamaIndex are properly installed:

```python
from lctl.integrations.llamaindex import llamaindex_available

if llamaindex_available():
    print("LlamaIndex integration is ready!")
else:
    print("Please install LlamaIndex: pip install llama-index")
```

### Step 2: Import Required Modules

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from lctl.integrations.llamaindex import (
    LCTLLlamaIndexCallback,
    trace_query_engine,
    trace_chat_engine,
)
```

## Complete Working Examples

### Example 1: Using LCTLLlamaIndexCallback Directly

The callback approach integrates with LlamaIndex's callback system:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import LCTLLlamaIndexCallback

# Configure LlamaIndex
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# Create callback
callback = LCTLLlamaIndexCallback(chain_id="llamaindex-qa")

# Create documents
documents = [
    Document(text="LCTL is a time-travel debugging library for LLM workflows."),
    Document(text="LCTL uses event sourcing to record agent execution."),
    Document(text="The ReplayEngine allows stepping through events."),
]

# Create index with LCTL callback
index = VectorStoreIndex.from_documents(documents, callback_manager=callback.get_callback_manager())

# Create query engine
query_engine = index.as_query_engine(callback_manager=callback.get_callback_manager())

# Query
response = query_engine.query("What is LCTL?")
print(f"Answer: {response}")
print(f"Recorded {len(callback.chain.events)} events")

# Export trace
callback.export("llamaindex_qa_trace.lctl.json")
```

### Example 2: Using trace_query_engine Helper

The helper wraps an existing query engine with LCTL tracing:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import trace_query_engine

# Configure
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# Create index and query engine
documents = [
    Document(text="Python is a programming language."),
    Document(text="Python was created by Guido van Rossum."),
]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap with tracing
traced = trace_query_engine(query_engine, chain_id="python-qa")

# Use normally
response = traced.query("Who created Python?")
print(f"Answer: {response}")

# Export
traced.export("query_engine_trace.lctl.json")
```

### Example 3: Tracing Chat Engines

LCTL captures conversation history and context in chat engines:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import trace_chat_engine

# Configure
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# Create knowledge base
documents = [
    Document(text="LCTL supports LangChain, CrewAI, AutoGen, and LlamaIndex."),
    Document(text="LCTL provides time-travel replay of agent execution."),
    Document(text="LCTL uses event sourcing for immutable audit logs."),
]
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine()

# Wrap with tracing
traced = trace_chat_engine(chat_engine, chain_id="lctl-chat")

# Have a conversation
response1 = traced.chat("What frameworks does LCTL support?")
print(f"Assistant: {response1}")

response2 = traced.chat("What about debugging features?")
print(f"Assistant: {response2}")

response3 = traced.chat("How does it store data?")
print(f"Assistant: {response3}")

# Export conversation trace
traced.export("chat_trace.lctl.json")
```

### Example 4: Tracing RAG with Custom Retrievers

LCTL captures retrieval operations in detail:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import LCTLLlamaIndexCallback

# Configure
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# Create documents
documents = [
    Document(text="Machine learning is a subset of artificial intelligence."),
    Document(text="Deep learning uses neural networks with many layers."),
    Document(text="Natural language processing handles human language."),
    Document(text="Computer vision processes visual information."),
]

# Create callback
callback = LCTLLlamaIndexCallback(chain_id="custom-rag")

# Build index
index = VectorStoreIndex.from_documents(documents, callback_manager=callback.get_callback_manager())

# Create custom retriever
retriever = VectorIndexRetriever(
    index=index, similarity_top_k=3, callback_manager=callback.get_callback_manager()
)

# Create response synthesizer
synthesizer = get_response_synthesizer(callback_manager=callback.get_callback_manager())

# Build query engine with custom components
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    callback_manager=callback.get_callback_manager(),
)

# Query
response = query_engine.query("What is the relationship between ML and AI?")
print(f"Answer: {response}")

# Export trace with retriever details
callback.export("custom_rag_trace.lctl.json")
```

### Example 5: Tracing Streaming Responses

LCTL captures streaming response events:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import trace_query_engine

# Configure with streaming
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# Create index
documents = [
    Document(text="LCTL provides streaming event capture for real-time monitoring."),
]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)

# Wrap with tracing
traced = trace_query_engine(query_engine, chain_id="streaming-demo")

# Stream response
print("Streaming response:")
response = traced.query("What streaming features does LCTL have?")
for token in response.response_gen:
    print(token, end="", flush=True)
print("\n")

# Export trace
traced.export("streaming_trace.lctl.json")
```

### Example 6: Tracing with Multiple Data Sources

LCTL can trace queries across multiple indices:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import LCTLLlamaIndexCallback

# Configure
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# Create callback
callback = LCTLLlamaIndexCallback(chain_id="multi-source-rag")

# Create separate indices for different topics
python_docs = [
    Document(text="Python is a high-level programming language."),
    Document(text="Python emphasizes readability and simplicity."),
]
python_index = VectorStoreIndex.from_documents(
    python_docs, callback_manager=callback.get_callback_manager()
)

rust_docs = [
    Document(text="Rust is a systems programming language."),
    Document(text="Rust focuses on safety and performance."),
]
rust_index = VectorStoreIndex.from_documents(
    rust_docs, callback_manager=callback.get_callback_manager()
)

# Create query engine tools
python_engine = python_index.as_query_engine(callback_manager=callback.get_callback_manager())
rust_engine = rust_index.as_query_engine(callback_manager=callback.get_callback_manager())

python_tool = QueryEngineTool.from_defaults(
    query_engine=python_engine, description="Answers questions about Python programming."
)
rust_tool = QueryEngineTool.from_defaults(
    query_engine=rust_engine, description="Answers questions about Rust programming."
)

# Create router
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[python_tool, rust_tool],
    callback_manager=callback.get_callback_manager(),
)

# Query different topics
response1 = router.query("What is Python's philosophy?")
print(f"Python: {response1}")

response2 = router.query("What does Rust focus on?")
print(f"Rust: {response2}")

# Export combined trace
callback.export("multi_source_trace.lctl.json")
```

### Example 7: Tracing Async Operations

LCTL supports async LlamaIndex operations:

```python
import asyncio
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lctl.integrations.llamaindex import trace_query_engine


async def main():
    # Configure
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding()

    # Create index
    documents = [
        Document(text="Async operations improve throughput in I/O-bound tasks."),
    ]
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Wrap with tracing
    traced = trace_query_engine(query_engine, chain_id="async-demo")

    # Async query
    response = await traced.aquery("What are async operations good for?")
    print(f"Answer: {response}")

    # Export
    traced.export("async_trace.lctl.json")


asyncio.run(main())
```

## How to View Traces in the Dashboard

After exporting your traces, analyze them using LCTL's CLI tools:

### View Execution Flow

```bash
lctl trace llamaindex_qa_trace.lctl.json
```

Output:
```
Execution trace for chain: llamaindex-qa
  [1] step_start: query_engine (intent: query)
  [2] step_start: retriever (intent: retrieve)
  [3] step_end: retriever (outcome: success, 250ms)
  [4] step_start: synthesizer (intent: synthesize)
  [5] step_end: synthesizer (outcome: success, 1200ms)
  [6] step_end: query_engine (outcome: success, 1500ms)
```

### Time-Travel Replay

Step through query execution:

```bash
lctl replay llamaindex_qa_trace.lctl.json
```

Replay to a specific event:

```bash
lctl replay --to-seq 3 llamaindex_qa_trace.lctl.json
```

### Performance Statistics

```bash
lctl stats chat_trace.lctl.json
```

Output:
```
Duration: 4.2s | Tokens: 512 in, 256 out | Cost: $0.008
```

### Find Bottlenecks

```bash
lctl bottleneck custom_rag_trace.lctl.json
```

Output:
```
synthesizer: 65% of time (1200ms)
retriever: 20% of time (370ms)
embedding: 15% of time (280ms)
```

### Compare Traces

```bash
lctl diff trace_v1.lctl.json trace_v2.lctl.json
```

### Launch Web UI

```bash
lctl debug multi_source_trace.lctl.json
```

## What LCTL Captures

The LlamaIndex integration automatically captures:

| Event Type | Description |
|------------|-------------|
| `step_start` | Query/retrieval/synthesis begins |
| `step_end` | Operation completes with outcome and duration |
| `tool_call` | Retriever operations with document counts |
| `fact_added` | Retrieved documents and responses |
| `error` | Exceptions with context |

### LlamaIndex-Specific Events

| Event | Data Captured |
|-------|---------------|
| Query start | Query text, engine type |
| Retrieval | Number of documents, similarity scores |
| LLM call | Prompt, response, token usage |
| Synthesis | Response text, source nodes |
| Embedding | Text chunks, embedding model |

## Troubleshooting

### Issue: LlamaIndex Not Found

**Error**: `ImportError: LlamaIndex is not installed`

**Solution**: Install LlamaIndex with required components:
```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
```

### Issue: No Events Recorded

**Problem**: Trace file has 0 events.

**Solutions**:

1. Ensure you pass the callback manager:
   ```python
   # Correct
   index = VectorStoreIndex.from_documents(documents, callback_manager=callback.get_callback_manager())

   # Wrong - no callback
   index = VectorStoreIndex.from_documents(documents)
   ```

2. Use the traced wrapper:
   ```python
   # Correct
   traced = trace_query_engine(query_engine, chain_id="demo")
   traced.query("...")

   # Wrong - using original engine
   query_engine.query("...")
   ```

### Issue: Token Counts Show 0

**Problem**: Token usage shows 0 in traces.

**Solution**: Ensure your LLM provider reports token usage. OpenAI models report tokens automatically.

### Issue: Retriever Events Not Captured

**Problem**: RAG pipeline traces don't show retriever operations.

**Solution**: Pass the callback manager to all components:
```python
callback = LCTLLlamaIndexCallback(chain_id="rag")
cb_manager = callback.get_callback_manager()

# Pass to ALL components
index = VectorStoreIndex.from_documents(docs, callback_manager=cb_manager)
retriever = index.as_retriever(callback_manager=cb_manager)
query_engine = index.as_query_engine(callback_manager=cb_manager)
```

### Issue: Streaming Not Captured

**Problem**: Streaming responses don't show in traces.

**Solution**: Streaming events are captured when you consume the generator:
```python
response = traced.query("...")
# Must consume the stream for events to be recorded
for token in response.response_gen:
    print(token, end="")
```

## Next Steps

- Learn about [LCTL CLI commands](../cli/README.md)
- Explore [LangChain integration](./langchain-tutorial.md)
- Check out [DSPy integration](./dspy-tutorial.md)
- See [OpenAI Agents integration](./openai-agents-tutorial.md)
