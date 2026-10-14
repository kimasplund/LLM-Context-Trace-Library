"""Live OpenAI Agents SDK tests: a scripted fake model, no network or API key."""

import json

import pytest

pytest.importorskip("agents")

from agents import Agent, ModelResponse, RunConfig, Runner, Usage, function_tool
from agents.models.interface import Model
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from lctl.core.events import EventType
from lctl.integrations.openai_agents import LCTLOpenAIAgentTracer

pytestmark = pytest.mark.integration


class ScriptedModel(Model):
    """Replays canned outputs, one per model turn."""

    def __init__(self, *turns):
        self._turns = list(turns)

    async def get_response(self, *args, **kwargs):
        usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        return ModelResponse(output=self._turns.pop(0), usage=usage, response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def _text(text):
    part = ResponseOutputText(type="output_text", text=text, annotations=[])
    return [
        ResponseOutputMessage(
            id="m1", type="message", role="assistant", status="completed", content=[part]
        )
    ]


def _tool_call(name, arguments):
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


def _agent():
    model = ScriptedModel(_tool_call("get_weather", {"city": "Tokyo"}), _text("It is sunny"))
    return Agent(name="weather_bot", instructions="help", tools=[get_weather], model=model)


_NO_TRACING = RunConfig(tracing_disabled=True)


@pytest.mark.asyncio
async def test_tool_run_records_tool_events():
    tracer = LCTLOpenAIAgentTracer(chain_id="oa-tool")
    result = await Runner.run(_agent(), "go", hooks=tracer.create_hooks(), run_config=_NO_TRACING)

    assert result.final_output == "It is sunny"
    events = tracer.session.chain.events
    starts = [e for e in events if e.type == EventType.STEP_START]
    assert [e.agent for e in starts] == ["weather_bot"]
    (tool,) = [e for e in events if e.type == EventType.TOOL_CALL]
    assert tool.data["tool"] == "get_weather"
    assert "Sunny in Tokyo" in tool.data["output"]
    assert "Tokyo" in tool.data["input"]


def test_run_config_property_warns():
    tracer = LCTLOpenAIAgentTracer(chain_id="oa-rc")
    with pytest.warns(DeprecationWarning, match="hooks=tracer.create_hooks"):
        config = tracer.run_config
    assert "hooks" in config
