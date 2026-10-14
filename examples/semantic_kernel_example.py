"""Semantic Kernel example: a @kernel_function plugin traced by LCTL (no API key).

Run it in a scratch directory; it writes ``semantic_kernel_trace.lctl.json`` there. View with:

    lctl trace --tree semantic_kernel_trace.lctl.json
"""

import sys

try:
    import semantic_kernel  # noqa: F401
except ImportError:
    print("Semantic Kernel is not installed. Install it with:")
    print(
        '  pip install "lctl[semantic-kernel] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

import asyncio
from pathlib import Path

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

from lctl.integrations.semantic_kernel import LCTLSemanticKernelTracer, is_available


class MathPlugin:
    @kernel_function(name="add", description="Add two integers")
    def add(self, a: int, b: int) -> int:
        return a + b

    @kernel_function(name="square", description="Square an integer")
    def square(self, a: int) -> int:
        return a * a


async def run() -> LCTLSemanticKernelTracer:
    kernel = Kernel()
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    tracer = LCTLSemanticKernelTracer(chain_id="semantic-kernel-example")
    tracer.trace_kernel(kernel)

    total = await kernel.invoke(plugin_name="Math", function_name="add", a=2, b=3)
    squared = await kernel.invoke(plugin_name="Math", function_name="square", a=int(str(total)))
    print(f"2 + 3 = {total}; squared = {squared}")
    return tracer


def main() -> None:
    if not is_available():
        print(
            'LCTL Semantic Kernel integration unavailable: pip install "lctl[semantic-kernel] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
        )
        sys.exit(0)

    tracer = asyncio.run(run())
    path = Path("semantic_kernel_trace.lctl.json")
    tracer.export(str(path))
    print(f"Trace written to {path.resolve()} ({len(tracer.session.chain.events)} events)")


if __name__ == "__main__":
    main()
