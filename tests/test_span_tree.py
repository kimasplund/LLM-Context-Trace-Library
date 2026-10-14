"""Tests for ReplayEngine.get_span_tree() and identity-blind diffing (format 4.2)."""

from datetime import datetime, timedelta
from pathlib import Path

from lctl.core.events import Chain, Event, EventType, ReplayEngine
from lctl.core.session import LCTLSession

FIXTURE = Path(__file__).parent / "fixtures" / "spans_v41_orphans.lctl.json"


def _by_agent(nodes: list[dict]) -> dict[str, dict]:
    return {node["agent"]: node for node in nodes}


def test_exact_tree_from_session():
    session = LCTLSession()
    with session.span("planner", "plan"):
        with session.span("coder", "code"):
            session.tool_call("editor", {}, {})
        with session.span("tester", "test"):
            pass

    tree = ReplayEngine(session.chain).get_span_tree()

    assert [node["agent"] for node in tree] == ["planner"]
    planner = tree[0]
    assert [child["agent"] for child in planner["children"]] == ["coder", "tester"]
    coder, tester = planner["children"]
    assert coder["event_count"] == 1
    assert coder["parent_span_id"] == planner["span_id"]
    starts = {e.agent: e for e in session.chain.events if e.type == EventType.STEP_START}
    for node in (planner, coder, tester):
        assert node["inferred"] is False
        assert node["span_id"] == starts[node["agent"]].span_id
        assert node["outcome"] == "success"


def test_inferred_tree_for_v41_fixture():
    tree = ReplayEngine(Chain.load(FIXTURE)).get_span_tree()

    assert [node["agent"] for node in tree] == ["planner", "reviewer"]
    planner, reviewer = tree
    assert reviewer["span_id"] == "orphan-5"
    assert reviewer["start_seq"] is None
    assert reviewer["end_seq"] == 5
    assert reviewer["inferred"] is True

    children = _by_agent(planner["children"])
    assert [child["agent"] for child in planner["children"]] == ["coder", "tester"]
    coder, tester = children["coder"], children["tester"]
    assert coder["span_id"] == "inferred-2"
    assert coder["parent_span_id"] == "inferred-1"
    assert coder["duration_ms"] == 2000
    assert coder["event_count"] == 1
    assert tester["end_seq"] is None
    assert tester["duration_ms"] is None
    assert tester["outcome"] is None
    assert planner["duration_ms"] == 6000
    assert all(node["inferred"] for node in (planner, coder, tester))


def test_inferred_ids_stable_across_loads():
    first = ReplayEngine(Chain.load(FIXTURE)).get_span_tree()
    second = ReplayEngine(Chain.load(FIXTURE)).get_span_tree()
    assert first == second


def test_interleaved_same_name_agents_v41():
    t0 = datetime(2026, 1, 1)
    chain = Chain(id="interleaved")
    for seq, type_ in enumerate(
        [EventType.STEP_START, EventType.STEP_START, EventType.STEP_END, EventType.STEP_END],
        start=1,
    ):
        chain.add_event(
            Event(seq=seq, type=type_, timestamp=t0 + timedelta(seconds=seq), agent="a", data={})
        )

    tree = ReplayEngine(chain).get_span_tree()

    assert [node["span_id"] for node in tree] == ["inferred-1"]
    outer = tree[0]
    assert outer["end_seq"] == 4
    assert [child["span_id"] for child in outer["children"]] == ["inferred-2"]
    assert outer["children"][0]["end_seq"] == 3


def test_mixed_chain_uses_exact_ids_where_present():
    session = LCTLSession()
    with session.span("modern", "exact"):
        pass
    chain = session.chain
    t = chain.events[-1].timestamp
    for offset, type_ in ((1, EventType.STEP_START), (2, EventType.STEP_END)):
        chain.add_event(
            Event(
                seq=len(chain.events) + 1,
                type=type_,
                timestamp=t + timedelta(seconds=offset),
                agent="legacy",
                data={"intent": "old"},
            )
        )

    nodes = _by_agent(ReplayEngine(chain).get_span_tree())
    assert nodes["modern"]["inferred"] is False
    assert nodes["modern"]["span_id"] == chain.events[0].span_id
    assert nodes["legacy"]["inferred"] is True
    assert nodes["legacy"]["span_id"] == f"inferred-{nodes['legacy']['start_seq']}"


def test_duration_falls_back_to_timestamps():
    t0 = datetime(2026, 1, 1)
    chain = Chain(id="durations")
    chain.add_event(Event(1, EventType.STEP_START, t0, "a", {"intent": "x"}))
    chain.add_event(
        Event(2, EventType.STEP_END, t0 + timedelta(milliseconds=1500), "a", {"duration_ms": 0})
    )

    (node,) = ReplayEngine(chain).get_span_tree()
    assert node["duration_ms"] == 1500


def test_diff_ignores_identity_fields():
    def record() -> Chain:
        session = LCTLSession(chain_id="same")
        with session.span("planner", "plan"):
            session.tool_call("grep", {"q": "x"}, {"hits": 1})
            session.add_fact("F1", "fact", confidence=0.9)
        return session.chain

    first, second = record(), record()
    for a, b in zip(first.events, second.events, strict=True):
        b.timestamp = a.timestamp
    assert first.events[0].id != second.events[0].id
    assert first.events[0].span_id != second.events[0].span_id

    assert ReplayEngine(first).diff(ReplayEngine(second)) == []


def test_get_trace_includes_span_fields_when_present():
    session = LCTLSession()
    with session.span("a", "x"):
        pass
    modern = ReplayEngine(session.chain).get_trace()
    assert all("span_id" in entry for entry in modern)
    assert modern[0]["span_id"] == session.chain.events[0].span_id

    legacy = ReplayEngine(Chain.load(FIXTURE)).get_trace()
    assert all("span_id" not in entry for entry in legacy)


def _event(seq, type_, agent, ts, data=None, **ids) -> Event:
    return Event(seq=seq, type=type_, timestamp=ts, agent=agent, data=data or {}, **ids)


def test_spanless_end_pairs_with_exact_start():
    # Claude Code hooks run as separate processes: the START carries a span id,
    # the END (recorded by a fresh session) does not.
    t0 = datetime(2026, 1, 1)
    chain = Chain(id="cross-process")
    chain.add_event(
        _event(1, EventType.STEP_START, "explorer", t0, {"intent": "find x"}, span_id="S1", id="e1")
    )
    chain.add_event(
        _event(
            2,
            EventType.STEP_END,
            "explorer",
            t0 + timedelta(seconds=3),
            {"outcome": "success"},
            id="e2",
        )
    )

    (node,) = ReplayEngine(chain).get_span_tree()
    assert node["span_id"] == "S1"
    assert node["end_seq"] == 2
    assert node["outcome"] == "success"
    assert node["inferred"] is False


def test_root_level_42_event_not_counted_in_sibling_span():
    t0 = datetime(2026, 1, 1)
    chain = Chain(id="root-fact")
    chain.add_event(
        _event(1, EventType.STEP_START, "worker", t0, {"intent": "w"}, span_id="W", id="e1")
    )
    chain.add_event(_event(2, EventType.FACT_ADDED, "main", t0, {"id": "F"}, id="e2", span_id=None))
    chain.add_event(
        _event(3, EventType.STEP_END, "worker", t0, {"outcome": "success"}, span_id="W", id="e3")
    )

    (node,) = ReplayEngine(chain).get_span_tree()
    assert node["event_count"] == 0


def test_tree_survives_naive_aware_mix_and_string_duration():
    from datetime import UTC

    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC)
    chain = Chain(id="malformed")
    chain.add_event(_event(1, EventType.STEP_START, "a", naive, {"intent": "x"}))
    chain.add_event(_event(2, EventType.STEP_END, "a", aware, {"duration_ms": 0}))
    chain.add_event(_event(3, EventType.STEP_START, "b", naive, {"intent": "y"}))
    chain.add_event(_event(4, EventType.STEP_END, "b", naive, {"duration_ms": "5"}))

    nodes = {n["agent"]: n for n in ReplayEngine(chain).get_span_tree()}
    assert nodes["a"]["duration_ms"] == 2000
    assert nodes["b"]["duration_ms"] == 5
