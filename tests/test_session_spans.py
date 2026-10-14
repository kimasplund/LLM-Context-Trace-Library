"""Tests for span tracking in LCTLSession (trace format 4.2)."""

import asyncio
import contextvars
import uuid

import pytest

from lctl.core.events import EventType
from lctl.core.ids import new_id
from lctl.core.session import LCTLSession


def _is_uuid7(value: str | None) -> bool:
    return value is not None and uuid.UUID(value).version == 7


class TestSpanTracking:
    def test_every_event_gets_uuid7_id(self):
        session = LCTLSession()
        session.step_start("agent", "work")
        session.tool_call("grep", {"q": 1}, {"hits": 0})
        session.add_fact("F1", "a fact")
        session.llm_trace([{"role": "user", "content": "hi"}], "hello", "model-x")
        session.error("category", "Kind", "message")
        session.checkpoint()
        session.step_end()

        ids = [event.id for event in session.chain.events]
        assert len(ids) == 7
        assert all(_is_uuid7(event_id) for event_id in ids)
        assert len(set(ids)) == 7

    def test_nested_steps_link_parents(self):
        session = LCTLSession()
        session.step_start("outer", "plan")
        session.step_start("inner", "do")
        session.step_end()
        session.step_end()

        outer_start, inner_start, inner_end, outer_end = session.chain.events
        assert outer_start.parent_span_id is None
        assert inner_start.parent_span_id == outer_start.span_id
        assert inner_end.span_id == inner_start.span_id
        assert outer_end.span_id == outer_start.span_id
        assert outer_start.span_id != inner_start.span_id

    def test_non_step_events_carry_enclosing_span(self):
        session = LCTLSession()
        session.tool_call("before", {}, {})
        session.step_start("outer", "plan")
        session.step_start("inner", "do")
        session.tool_call("inside_inner", {}, {})
        session.add_fact("F1", "inner fact")
        session.step_end()
        session.tool_call("inside_outer", {}, {})
        session.step_end()

        by_name = {
            e.data.get("tool") or e.data.get("id"): e
            for e in session.chain.events
            if e.type in (EventType.TOOL_CALL, EventType.FACT_ADDED)
        }
        outer_span = session.chain.events[1].span_id
        inner_span = session.chain.events[2].span_id
        assert _is_uuid7(outer_span)
        assert _is_uuid7(inner_span)
        assert outer_span != inner_span
        assert by_name["before"].span_id is None
        assert by_name["inside_inner"].span_id == inner_span
        assert by_name["F1"].span_id == inner_span
        assert by_name["inside_outer"].span_id == outer_span

    def test_explicit_span_and_parent_ids_are_honoured(self):
        session = LCTLSession()
        span, parent = new_id(), new_id()
        session.step_start("agent", "work", span_id=span, parent_span_id=parent)

        start = session.chain.events[0]
        assert start.span_id == span
        assert start.parent_span_id == parent
        assert session.current_span_id == span

    def test_step_end_explicit_span_not_on_top(self):
        session = LCTLSession()
        session.step_start("a", "first")
        span_a = session.current_span_id
        session.step_start("b", "second")
        span_b = session.current_span_id

        session.step_end(span_id=span_a)

        end = session.chain.events[-1]
        assert end.span_id == span_a
        assert end.agent == "a"
        assert session.current_span_id == span_b

    def test_explicit_span_attribution(self):
        session = LCTLSession()
        session.step_start("a", "first")
        span_a = session.current_span_id
        session.step_end()

        session.error("category", "Kind", "late", span_id=span_a)

        error = session.chain.events[-1]
        assert error.span_id == span_a
        assert error.agent == "a"

    def test_redaction_keeps_ids(self):
        session = LCTLSession()
        session.step_start("agent", "work")
        span = session.current_span_id
        session.add_fact("F1", "api_key=sk-abcdefghijklmnopqrstuvwx")

        fact = session.chain.events[-1]
        assert _is_uuid7(fact.id)
        assert fact.span_id == span


class TestSpanConcurrency:
    @pytest.mark.asyncio
    async def test_gather_tasks_get_isolated_spans(self):
        session = LCTLSession()
        session.step_start("orchestrator", "fan out")
        outer = session.current_span_id

        async def work(name: str) -> None:
            session.step_start(name, "work")
            await asyncio.sleep(0)
            session.tool_call(f"tool-{name}", {}, {})
            await asyncio.sleep(0)
            session.step_end()

        await asyncio.gather(work("a"), work("b"))

        events = session.chain.events
        starts = {e.agent: e for e in events if e.type == EventType.STEP_START}
        ends = {e.agent: e for e in events if e.type == EventType.STEP_END}
        tools = {e.data["tool"]: e for e in events if e.type == EventType.TOOL_CALL}

        for name in ("a", "b"):
            assert starts[name].parent_span_id == outer
            assert tools[f"tool-{name}"].span_id == starts[name].span_id
            assert tools[f"tool-{name}"].agent == name
            assert ends[name].span_id == starts[name].span_id
        assert starts["a"].span_id != starts["b"].span_id
        assert session.current_span_id == outer

    def test_fresh_context_starts_without_parent(self):
        session = LCTLSession()
        session.step_start("outer", "plan")
        outer = session.current_span_id

        def in_fresh_context() -> tuple[str | None, str | None]:
            session.step_start("detached", "work")
            implicit_parent = session.chain.events[-1].parent_span_id
            session.step_end()
            session.step_start("attached", "work", parent_span_id=outer)
            explicit_parent = session.chain.events[-1].parent_span_id
            session.step_end()
            return implicit_parent, explicit_parent

        implicit_parent, explicit_parent = contextvars.Context().run(in_fresh_context)
        assert implicit_parent is None
        assert explicit_parent == outer
        assert session.current_span_id == outer


class TestSessionSpanContextManager:
    def test_span_success(self):
        session = LCTLSession()
        with session.span("a", "do") as span_id:
            assert span_id == session.current_span_id

        start, end = session.chain.events
        assert start.type == EventType.STEP_START
        assert start.span_id == span_id
        assert end.type == EventType.STEP_END
        assert end.data["outcome"] == "success"
        assert end.span_id == span_id
        assert end.data["duration_ms"] >= 0
        assert session.current_span_id is None

    def test_span_error_keeps_order_and_attribution(self):
        session = LCTLSession()
        with pytest.raises(RuntimeError, match="boom"), session.span("a", "do") as span_id:
            raise RuntimeError("boom")

        types = [event.type for event in session.chain.events]
        assert types == [EventType.STEP_START, EventType.STEP_END, EventType.ERROR]
        end, error = session.chain.events[1], session.chain.events[2]
        assert end.data["outcome"] == "error"
        assert error.span_id == span_id
        assert error.agent == "a"
        assert error.data["type"] == "RuntimeError"

    def test_span_never_masks_user_exception_when_tracing_breaks(self, monkeypatch):
        session = LCTLSession()
        real_add_event = session._add_event
        calls = {"n": 0}

        def flaky_add_event(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("disk full")
            return real_add_event(*args, **kwargs)

        monkeypatch.setattr(session, "_add_event", flaky_add_event)
        with pytest.raises(ValueError, match="user"), session.span("a", "do"):
            raise ValueError("user")

    def test_span_tracing_failure_on_normal_exit_is_silent(self, monkeypatch):
        session = LCTLSession()
        real_add_event = session._add_event
        calls = {"n": 0}

        def flaky_add_event(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("disk full")
            return real_add_event(*args, **kwargs)

        monkeypatch.setattr(session, "_add_event", flaky_add_event)
        ran = []
        with session.span("a", "do"):
            ran.append(True)
        assert ran == [True]

    def test_traced_step_produces_span(self):
        from lctl.core.session import traced_step

        session = LCTLSession()
        with traced_step(session, "a", "x"):
            pass

        start, end = session.chain.events
        assert start.span_id is not None
        assert end.span_id == start.span_id

    def test_traced_decorator_produces_span_and_error_attribution(self):
        import lctl

        lctl._global_session = None
        try:
            session = lctl.auto_instrument()

            @lctl.traced("a", "x")
            def failing():
                raise ValueError("bad")

            with pytest.raises(ValueError, match="bad"):
                failing()

            types = [event.type for event in session.chain.events]
            assert types == [EventType.STEP_START, EventType.STEP_END, EventType.ERROR]
            start, end, error = session.chain.events
            assert end.span_id == start.span_id
            assert error.span_id == start.span_id
            assert error.agent == "a"
        finally:
            lctl._global_session = None
