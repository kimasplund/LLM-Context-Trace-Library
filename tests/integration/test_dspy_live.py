"""Live DSPy smoke test: dspy.Predict with DummyLM wrapped by trace_module."""

import pytest

pytest.importorskip("dspy")

import dspy
from dspy.utils import DummyLM

from lctl.core.events import EventType
from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

pytestmark = pytest.mark.integration


def test_trace_module_records_outer_span_and_returns_answer():
    callback = LCTLDSPyCallback(chain_id="dspy-live")
    with dspy.context(lm=DummyLM([{"answer": "Paris"}])):
        traced = trace_module(dspy.Predict("question -> answer"), callback)
        result = traced(question="capital of France?")

    assert result.answer == "Paris"
    events = callback.session.chain.events
    starts = [e for e in events if e.type == EventType.STEP_START]
    ends = [e for e in events if e.type == EventType.STEP_END]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].data["intent"] == "module_forward"
    assert ends[0].data["outcome"] == "success"
    assert ends[0].span_id == starts[0].span_id
