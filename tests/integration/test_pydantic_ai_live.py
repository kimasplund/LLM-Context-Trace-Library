"""Live PydanticAI smoke tests: Agent(TestModel()) wrapped by TracedAgent."""

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from lctl.core.events import EventType
from lctl.integrations.pydantic_ai import trace_agent

pytestmark = [
    pytest.mark.integration,
    # pydantic-ai's own Agent.run_sync creates an event loop (and an un-awaited
    # run coroutine) that it never closes; the warning surfaces at GC time in
    # whichever test runs next. Reproduces with a bare, unwrapped Agent.
    pytest.mark.filterwarnings(
        "ignore:Exception ignored in.*:pytest.PytestUnraisableExceptionWarning"
    ),
]


def _events(traced, event_type):
    return [e for e in traced.tracer.session.chain.events if e.type == event_type]


@pytest.mark.asyncio
async def test_run_records_output_and_usage():
    traced = trace_agent(Agent(TestModel(custom_output_text="Hello world")), chain_id="pai-run")
    result = await traced.run("hi")

    assert result.output == "Hello world"
    (end,) = _events(traced, EventType.STEP_END)
    assert end.data["outcome"] == "success"
    assert end.data["output_summary"] == "Hello world"
    assert end.data["tokens"]["input"] > 0
    assert end.data["tokens"]["output"] > 0


def test_run_sync_traced():
    traced = trace_agent(Agent(TestModel(custom_output_text="sync ok")), chain_id="pai-sync")
    result = traced.run_sync("hi")

    assert result.output == "sync ok"
    (start,) = _events(traced, EventType.STEP_START)
    (end,) = _events(traced, EventType.STEP_END)
    assert start.data["intent"] == "run_sync"
    assert end.data["outcome"] == "success"
    assert end.data["output_summary"] == "sync ok"


def test_unknown_attributes_forward():
    agent = Agent(TestModel(), name="proxied")
    traced = trace_agent(agent)

    assert traced.name == "proxied"
    assert traced.override == agent.override
    with pytest.raises(AttributeError):
        traced.no_such_attribute  # noqa: B018


@pytest.mark.asyncio
async def test_stream_records_deltas_once():
    traced = trace_agent(Agent(TestModel(custom_output_text="Hello world")), chain_id="pai-stream")
    seen = []
    async with traced.run_stream("hi") as stream:
        async for text in stream.stream_text():
            seen.append(text)

    assert seen[-1] == "Hello world"  # what the caller sees is unchanged
    (end,) = _events(traced, EventType.STREAM_END)
    assert end.data["total_content"] == "Hello world"
    chunks = "".join(e.data["content"] for e in _events(traced, EventType.STREAM_CHUNK))
    assert chunks == "Hello world"


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [False, True])
async def test_multi_chunk_stream_records_text_once(delta):
    async def pieces(messages, info: AgentInfo):
        yield "Hello"
        yield " wor"
        yield "ld"

    traced = trace_agent(Agent(FunctionModel(stream_function=pieces)), chain_id="pai-chunks")
    seen = []
    async with traced.run_stream("hi") as stream:
        async for text in stream.stream_text(delta=delta, debounce_by=None):
            seen.append(text)

    assert ("".join(seen) if delta else seen[-1]) == "Hello world"
    (end,) = _events(traced, EventType.STREAM_END)
    assert end.data["total_content"] == "Hello world"
    chunks = [e.data["content"] for e in _events(traced, EventType.STREAM_CHUNK)]
    assert "".join(chunks) == "Hello world"
