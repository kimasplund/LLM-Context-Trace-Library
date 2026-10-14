"""PydanticAI example: Agent(TestModel()) wrapped by trace_agent (no API key).

Run it in a scratch directory; it writes ``pydantic_ai_trace.lctl.json`` there. View with:

    lctl trace --tree pydantic_ai_trace.lctl.json
"""

import sys

try:
    import pydantic_ai  # noqa: F401
except ImportError:
    print("PydanticAI is not installed. Install it with:")
    print(
        '  pip install "lctl[pydantic-ai] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lctl.integrations.pydantic_ai import is_available, trace_agent


async def run():
    agent = Agent(TestModel(custom_output_text="Paris"), system_prompt="Be brief.")
    traced = trace_agent(agent, chain_id="pydantic-ai-example")
    result = await traced.run("Capital of France?")
    print(f"Agent answered: {result.output}")
    return traced


def main() -> None:
    if not is_available():
        print('LCTL PydanticAI integration unavailable: pip install "lctl[pydantic-ai]"')
        sys.exit(0)

    traced = asyncio.run(run())
    path = Path("pydantic_ai_trace.lctl.json")
    traced.export(str(path))
    print(f"Trace written to {path.resolve()} ({len(traced.tracer.session.chain.events)} events)")


if __name__ == "__main__":
    main()
