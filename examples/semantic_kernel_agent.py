"""
Semantic Kernel Integration Example

This example demonstrates how to use LCTL to trace a Semantic Kernel agent.
It covers:
1. Kernel initialization
2. Attaching the LCTL filter
3. Tracing native function invocation
"""

import asyncio
import os

try:
    from semantic_kernel import Kernel

    from lctl.integrations.semantic_kernel import trace_kernel
except ImportError:
    print("Please install semantic-kernel and lctl[semantic-kernel] to run this example")
    exit(1)

# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. This example may fail if run directly.")


async def main():
    print("=== LCTL Semantic Kernel Example ===")

    # 1. Initialize Kernel
    kernel = Kernel()

    # Note: connect AI service if needed
    # kernel.add_service(...)

    # 2. Attach LCTL Tracing
    print("Attaching LCTL tracer...")
    tracer = trace_kernel(kernel, chain_id="example-sk-time")

    # 3. Define and import a native plugin
    @kernel.function(name="get_time", description="Get current server time")
    def get_time() -> str:
        # Simulate time
        return "2023-10-27T10:00:00Z"

    kernel.import_plugin_from_object(get_time, plugin_name="TimePlugin")

    # 4. Invoke the function
    print("Invoking function...")
    result = await kernel.invoke(function_name="get_time", plugin_name="TimePlugin")
    print(f"Function Result: {result}")

    # 5. Export Trace
    output_file = "sk_trace.lctl.json"
    tracer.export(output_file)
    print(f"Trace exported to {output_file}")
    print(f"Captured {len(tracer.chain.events)} events.")


if __name__ == "__main__":
    asyncio.run(main())
