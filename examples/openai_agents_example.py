"""OpenAI Agents SDK example: a scripted fake model plus a tool (no API key).

Run it in a scratch directory; it writes ``openai_agents_trace.lctl.json`` there. View with:

    lctl trace --tree openai_agents_trace.lctl.json
"""

import sys

try:
    import agents  # noqa: F401
except ImportError:
    print("OpenAI Agents SDK is not installed. Install it with:")
    print(
        '  pip install "lctl[openai-agents] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

import asyncio
import json
from pathlib import Path

from agents import Agent, ModelResponse, RunConfig, Runner, Usage, function_tool
from agents.models.interface import Model
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer, is_available


class ScriptedModel(Model):
    """Replays canned outputs, one per model turn, so no network is needed."""

    def __init__(self, *turns):
        self._turns = list(turns)

    async def get_response(self, *args, **kwargs):
        usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        return ModelResponse(output=self._turns.pop(0), usage=usage, response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def text_turn(text):
    part = ResponseOutputText(type="output_text", text=text, annotations=[])
    return [
        ResponseOutputMessage(
            id="m1", type="message", role="assistant", status="completed", content=[part]
        )
    ]


def tool_turn(name, arguments):
    return [
        ResponseFunctionToolCall(
            id="c1",
            call_id="call-1",
            type="function_call",
            name=name,
            arguments=json.dumps(arguments),
        )
    ]


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


async def run() -> LCTLOpenAIAgentTracer:
    tracer = LCTLOpenAIAgentTracer(chain_id="openai-agents-example")
    model = ScriptedModel(tool_turn("get_weather", {"city": "Tokyo"}), text_turn("It is sunny"))
    agent = Agent(name="weather_bot", instructions="Help.", tools=[get_weather], model=model)
    result = await Runner.run(
        agent,
        "Weather in Tokyo?",
        hooks=tracer.create_hooks(),
        run_config=RunConfig(tracing_disabled=True),  # no export to OpenAI's trace backend
    )
    print(f"Agent answered: {result.final_output}")
    return tracer


def main() -> None:
    if not is_available():
        print('LCTL OpenAI Agents integration unavailable: pip install "lctl[openai-agents]"')
        sys.exit(0)

    tracer = asyncio.run(run())
    path = Path("openai_agents_trace.lctl.json")
    tracer.export(str(path))
    print(f"Trace written to {path.resolve()} ({len(tracer.session.chain.events)} events)")


if __name__ == "__main__":
    main()
