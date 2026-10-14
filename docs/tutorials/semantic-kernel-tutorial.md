# Using LCTL with Semantic Kernel

This tutorial shows you how to integrate LCTL (LLM Context Trace Library) v4.0 with Semantic Kernel (v1.x) for time-travel debugging and function invocation tracing.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.10+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **Semantic Kernel** installed:
   ```bash
   pip install semantic-kernel
   ```
4. **OpenAI API key** (or Azure OpenAI credentials):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

```python
try:
    import semantic_kernel
    from lctl.integrations.semantic_kernel import trace_kernel

    print("Semantic Kernel integration is ready!")
except ImportError:
    print("Please install required packages: pip install lctl[semantic-kernel]")
```

### Step 2: Import Required Modules

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from lctl.integrations.semantic_kernel import trace_kernel, LCTLSemanticKernelTracer
```

## Complete Working Examples

### Example 1: Basic Kernel Tracing

The `trace_kernel` function attaches a filter to the Kernel that captures all function invocations.

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from lctl.integrations.semantic_kernel import trace_kernel


async def main():
    # 1. Initialize Kernel
    kernel = Kernel()

    # 2. Add AI Service
    service_id = "chat-gpt"
    kernel.add_service(
        OpenAIChatCompletion(service_id=service_id, ai_model_id="gpt-3.5-turbo"),
    )

    # 3. Attach LCTL Tracing
    # This registers the FunctionInvocationFilter
    tracer = trace_kernel(kernel, chain_id="sk-basic-demo", verbose=True)

    # 4. Import a plugin (Native function)
    @kernel.function(name="get_time", description="Get current time")
    def get_time() -> str:
        return "It is currently 12:00 PM"

    kernel.import_plugin_from_object(get_time, plugin_name="TimePlugin")

    # 5. Invoke function
    # LCTL captures the 'step_start', 'step_end', and mapping of inputs/outputs
    result = await kernel.invoke(function_name="get_time", plugin_name="TimePlugin")
    print(f"Result: {result}")

    # 6. Export Trace
    tracer.export("sk_basic.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Prompt Functions

LCTL also traces prompt functions (Semantic Functions).

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from lctl.integrations.semantic_kernel import trace_kernel


async def main():
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-3.5-turbo"),
    )

    # Attach tracer
    tracer = trace_kernel(kernel, chain_id="sk-prompt-demo")

    # Define a prompt function
    prompt = "Summarize this text in one sentence: {{$input}}"
    summarize = kernel.create_function_from_prompt(
        function_name="summarize", plugin_name="Summarizer", prompt=prompt
    )

    # Invoke
    text = """
    LCTL (LLM Context Trace Library) is a library for time-travel debugging.
    It uses event sourcing to capture every agent action as an immutable event.
    """

    result = await kernel.invoke(summarize, input=text)
    print(f"Summary: {result}")

    # Export
    tracer.export("sk_prompt.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Handling Errors

Errors inside functions are captured automatically.

```python
import asyncio
from semantic_kernel import Kernel
from lctl.integrations.semantic_kernel import trace_kernel


async def main():
    kernel = Kernel()
    tracer = trace_kernel(kernel, chain_id="sk-error-demo")

    @kernel.function(name="fail")
    def fail():
        raise ValueError("Something went wrong!")

    kernel.import_plugin_from_object(fail, plugin_name="ErrorPlugin")

    try:
        await kernel.invoke(function_name="fail", plugin_name="ErrorPlugin")
    except Exception as e:
        print(f"Caught error: {e}")

    # Trace will include the ERROR event
    tracer.export("sk_error.lctl.json")


if __name__ == "__main__":
    asyncio.run(main())
```

## How it Works

LCTL uses Semantic Kernel's **Filters API** (`FunctionInvocationFilter`). 

1. **Before Invocation**: Records `step_start` with arguments.
2. **Invocation**: Executes the function (native or prompt).
3. **After Invocation**: Records `step_end` with results and `FunctionResult.metadata` (token usage).
4. **On Exception**: Records `error` event before re-raising.

## Limitations

- **Streaming**: Full streaming support for Semantic Kernel in LCTL depends on SK's streaming filter support (evolving in v1.x). Currently, LCTL primarily traces non-streaming invocations or final aggregated results.
