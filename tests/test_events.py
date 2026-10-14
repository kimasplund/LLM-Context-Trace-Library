"""Tests for LCTL event sourcing core (lctl/core/events.py)."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest
import yaml

from lctl.core.events import Chain, Event, EventType, ReplayEngine, State
from lctl.core.schema import CURRENT_VERSION


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_are_strings(self):
        """EventType values should be strings."""
        assert EventType.STEP_START.value == "step_start"
        assert EventType.STEP_END.value == "step_end"
        assert EventType.FACT_ADDED.value == "fact_added"
        assert EventType.FACT_MODIFIED.value == "fact_modified"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.ERROR.value == "error"
        assert EventType.CHECKPOINT.value == "checkpoint"
        assert EventType.LLM_TRACE.value == "llm_trace"

    def test_all_event_types_defined(self):
        """Verify all expected event types exist."""
        expected_types = {
            "step_start",
            "step_end",
            "fact_added",
            "fact_modified",
            "tool_call",
            "error",
            "checkpoint",
            "stream_start",
            "stream_chunk",
            "stream_end",
            "contract_validation",
            "model_routing",
            "llm_trace",
        }
        actual_types = {e.value for e in EventType}
        assert expected_types == actual_types


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation(self, base_timestamp: datetime):
        """Test basic event creation."""
        event = Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="test_agent",
            data={"key": "value"},
        )
        assert event.seq == 1
        assert event.type == EventType.STEP_START
        assert event.timestamp == base_timestamp
        assert event.agent == "test_agent"
        assert event.data == {"key": "value"}

    def test_event_creation_with_default_data(self, base_timestamp: datetime):
        """Test event creation with default empty data."""
        event = Event(
            seq=1, type=EventType.STEP_START, timestamp=base_timestamp, agent="test_agent"
        )
        assert event.data == {}

    def test_event_to_dict(self, base_timestamp: datetime):
        """Test event serialization to dictionary."""
        event = Event(
            seq=1,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp,
            agent="analyzer",
            data={"id": "F1", "text": "test fact", "confidence": 0.9},
        )
        result = event.to_dict()

        assert result["seq"] == 1
        assert result["type"] == "fact_added"
        assert result["timestamp"] == base_timestamp.isoformat()
        assert result["agent"] == "analyzer"
        assert result["data"]["id"] == "F1"
        assert result["data"]["confidence"] == 0.9

    def test_event_from_dict(self, base_timestamp: datetime):
        """Test event deserialization from dictionary."""
        data = {
            "seq": 5,
            "type": "tool_call",
            "timestamp": base_timestamp.isoformat(),
            "agent": "tool_user",
            "data": {"tool": "search", "input": "query"},
        }
        event = Event.from_dict(data)

        assert event.seq == 5
        assert event.type == EventType.TOOL_CALL
        assert event.timestamp == base_timestamp
        assert event.agent == "tool_user"
        assert event.data["tool"] == "search"

    def test_event_from_dict_without_data(self, base_timestamp: datetime):
        """Test event deserialization with missing data field."""
        data = {
            "seq": 1,
            "type": "step_start",
            "timestamp": base_timestamp.isoformat(),
            "agent": "agent1",
        }
        event = Event.from_dict(data)
        assert event.data == {}

    def test_event_from_dict_with_unknown_type(self, base_timestamp: datetime):
        """Test event deserialization with unknown event type."""
        data = {
            "seq": 1,
            "type": "custom_event",
            "timestamp": base_timestamp.isoformat(),
            "agent": "agent1",
            "data": {},
        }
        event = Event.from_dict(data)
        assert event.type == "custom_event"

    def test_event_roundtrip(self, base_timestamp: datetime):
        """Test event serialization/deserialization roundtrip."""
        original = Event(
            seq=10,
            type=EventType.ERROR,
            timestamp=base_timestamp,
            agent="error_handler",
            data={"category": "validation", "message": "test error"},
        )
        restored = Event.from_dict(original.to_dict())

        assert restored.seq == original.seq
        assert restored.type == original.type
        assert restored.timestamp == original.timestamp
        assert restored.agent == original.agent
        assert restored.data == original.data


class TestChain:
    """Tests for Chain dataclass."""

    def test_chain_creation(self):
        """Test basic chain creation."""
        chain = Chain(id="test-chain-123")
        assert chain.id == "test-chain-123"
        assert chain.events == []
        assert chain.version == CURRENT_VERSION

    def test_chain_creation_with_custom_version(self):
        """Test chain creation with custom version."""
        chain = Chain(id="chain", version="3.0")
        assert chain.version == "3.0"

    def test_add_event(self, base_timestamp: datetime):
        """Test adding events to chain."""
        chain = Chain(id="test")
        event = Event(seq=1, type=EventType.STEP_START, timestamp=base_timestamp, agent="agent1")
        chain.add_event(event)

        assert len(chain.events) == 1
        assert chain.events[0] == event

    def test_chain_to_dict(self, sample_chain: Chain):
        """Test chain serialization to dictionary."""
        result = sample_chain.to_dict()

        assert result["lctl"] == CURRENT_VERSION
        assert result["chain"]["id"] == "test-chain"
        assert len(result["events"]) == 7

    def test_chain_from_dict(self):
        """Test chain deserialization from dictionary."""
        data = {
            "lctl": "4.0",
            "chain": {"id": "restored-chain"},
            "events": [
                {
                    "seq": 1,
                    "type": "step_start",
                    "timestamp": "2025-01-15T10:00:00",
                    "agent": "agent1",
                    "data": {},
                }
            ],
        }
        chain = Chain.from_dict(data)

        assert chain.id == "restored-chain"
        assert chain.version == CURRENT_VERSION
        assert len(chain.events) == 1
        assert chain.events[0].type == EventType.STEP_START

    def test_chain_from_dict_with_defaults(self):
        """Test chain deserialization with missing optional fields."""
        data = {"events": []}
        chain = Chain.from_dict(data)

        assert chain.id == "unknown"
        assert chain.version == CURRENT_VERSION
        assert chain.events == []

    def test_chain_save_json(self, sample_chain: Chain, tmp_path: Path):
        """Test saving chain as JSON."""
        file_path = tmp_path / "chain.lctl.json"
        sample_chain.save(file_path)

        assert file_path.exists()
        content = json.loads(file_path.read_text())
        assert content["lctl"] == CURRENT_VERSION
        assert content["chain"]["id"] == "test-chain"

    def test_chain_save_yaml(self, sample_chain: Chain, tmp_path: Path):
        """Test saving chain as YAML."""
        file_path = tmp_path / "chain.lctl.yaml"
        sample_chain.save(file_path)

        assert file_path.exists()
        content = yaml.safe_load(file_path.read_text())
        assert content["lctl"] == CURRENT_VERSION
        assert content["chain"]["id"] == "test-chain"

    def test_chain_save_yml_extension(self, sample_chain: Chain, tmp_path: Path):
        """Test saving chain with .yml extension."""
        file_path = tmp_path / "chain.lctl.yml"
        sample_chain.save(file_path)

        assert file_path.exists()
        content = yaml.safe_load(file_path.read_text())
        assert content["chain"]["id"] == "test-chain"

    def test_chain_load_json(self, temp_chain_file: Path):
        """Test loading chain from JSON."""
        chain = Chain.load(temp_chain_file)

        assert chain.id == "test-chain"
        assert chain.version == CURRENT_VERSION
        assert len(chain.events) == 7

    def test_chain_load_yaml(self, temp_yaml_chain_file: Path):
        """Test loading chain from YAML."""
        chain = Chain.load(temp_yaml_chain_file)

        assert chain.id == "test-chain"
        assert len(chain.events) == 7

    def test_chain_roundtrip_json(self, sample_chain: Chain, tmp_path: Path):
        """Test JSON save/load roundtrip."""
        file_path = tmp_path / "roundtrip.json"
        sample_chain.save(file_path)
        restored = Chain.load(file_path)

        assert restored.id == sample_chain.id
        assert restored.version == sample_chain.version
        assert len(restored.events) == len(sample_chain.events)

    def test_chain_roundtrip_yaml(self, sample_chain: Chain, tmp_path: Path):
        """Test YAML save/load roundtrip."""
        file_path = tmp_path / "roundtrip.yaml"
        sample_chain.save(file_path)
        restored = Chain.load(file_path)

        assert restored.id == sample_chain.id
        assert len(restored.events) == len(sample_chain.events)

    def test_empty_chain_save_load(self, empty_chain: Chain, tmp_path: Path):
        """Test saving and loading empty chain."""
        file_path = tmp_path / "empty.json"
        empty_chain.save(file_path)
        restored = Chain.load(file_path)

        assert restored.id == "empty-chain"
        assert restored.events == []


class TestState:
    """Tests for State dataclass."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = State()
        assert state.facts == {}
        assert state.current_agent is None
        assert state.current_step is None
        assert state.errors == []
        assert state.metrics["total_duration_ms"] == 0
        assert state.metrics["event_count"] == 0

    def test_apply_step_start(self, base_timestamp: datetime):
        """Test applying STEP_START event."""
        state = State()
        event = Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="planner",
            data={"intent": "plan"},
        )
        state.apply_event(event)

        assert state.current_agent == "planner"
        assert state.current_step == 1
        assert state.metrics["event_count"] == 1

    def test_apply_step_end(self, base_timestamp: datetime):
        """Test applying STEP_END event."""
        state = State()
        event = Event(
            seq=2,
            type=EventType.STEP_END,
            timestamp=base_timestamp,
            agent="planner",
            data={"outcome": "success", "duration_ms": 500, "tokens": {"input": 100, "output": 50}},
        )
        state.apply_event(event)

        assert state.metrics["total_duration_ms"] == 500
        assert state.metrics["total_tokens_in"] == 100
        assert state.metrics["total_tokens_out"] == 50

    def test_apply_step_end_with_alt_token_keys(self, base_timestamp: datetime):
        """Test applying STEP_END with 'in'/'out' token keys."""
        state = State()
        event = Event(
            seq=2,
            type=EventType.STEP_END,
            timestamp=base_timestamp,
            agent="planner",
            data={"outcome": "success", "duration_ms": 300, "tokens": {"in": 80, "out": 40}},
        )
        state.apply_event(event)

        assert state.metrics["total_tokens_in"] == 80
        assert state.metrics["total_tokens_out"] == 40

    def test_apply_fact_added(self, base_timestamp: datetime):
        """Test applying FACT_ADDED event."""
        state = State()
        event = Event(
            seq=3,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp,
            agent="analyzer",
            data={"id": "F1", "text": "Test fact", "confidence": 0.85, "source": "analyzer"},
        )
        state.apply_event(event)

        assert "F1" in state.facts
        assert state.facts["F1"]["text"] == "Test fact"
        assert state.facts["F1"]["confidence"] == 0.85
        assert state.facts["F1"]["source"] == "analyzer"
        assert state.facts["F1"]["seq"] == 3

    def test_apply_fact_added_without_id(self, base_timestamp: datetime):
        """Test applying FACT_ADDED event without id (no-op)."""
        state = State()
        event = Event(
            seq=3,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp,
            agent="analyzer",
            data={"text": "Test fact", "confidence": 0.85},
        )
        state.apply_event(event)

        assert state.facts == {}

    def test_apply_fact_modified(self, base_timestamp: datetime):
        """Test applying FACT_MODIFIED event."""
        state = State()
        state.facts["F1"] = {"text": "Original", "confidence": 0.5, "seq": 1}

        event = Event(
            seq=4,
            type=EventType.FACT_MODIFIED,
            timestamp=base_timestamp,
            agent="verifier",
            data={"id": "F1", "confidence": 0.9, "reason": "verified", "text": "Updated"},
        )
        state.apply_event(event)

        assert state.facts["F1"]["confidence"] == 0.9
        assert state.facts["F1"]["text"] == "Updated"
        assert state.facts["F1"]["reason"] == "verified"
        assert state.facts["F1"]["modified_at"] == 4

    def test_apply_fact_modified_nonexistent(self, base_timestamp: datetime):
        """Test applying FACT_MODIFIED for non-existent fact (no-op)."""
        state = State()
        event = Event(
            seq=4,
            type=EventType.FACT_MODIFIED,
            timestamp=base_timestamp,
            agent="verifier",
            data={"id": "F_NONEXISTENT", "confidence": 0.9},
        )
        state.apply_event(event)

        assert "F_NONEXISTENT" not in state.facts

    def test_apply_error(self, base_timestamp: datetime):
        """Test applying ERROR event."""
        state = State()
        event = Event(
            seq=5,
            type=EventType.ERROR,
            timestamp=base_timestamp,
            agent="agent1",
            data={
                "category": "validation",
                "type": "ValueError",
                "message": "Invalid input",
                "recoverable": True,
            },
        )
        state.apply_event(event)

        assert len(state.errors) == 1
        assert state.errors[0]["category"] == "validation"
        assert state.errors[0]["type"] == "ValueError"
        assert state.errors[0]["message"] == "Invalid input"
        assert state.errors[0]["recoverable"] is True
        assert state.metrics["error_count"] == 1

    def test_apply_tool_call(self, base_timestamp: datetime):
        """Test applying TOOL_CALL event."""
        state = State()
        event = Event(
            seq=6,
            type=EventType.TOOL_CALL,
            timestamp=base_timestamp,
            agent="tool_user",
            data={"tool": "search", "duration_ms": 150},
        )
        state.apply_event(event)

        assert state.metrics["total_duration_ms"] == 150

    def test_apply_multiple_events(self, sample_events: List[Event]):
        """Test applying multiple events in sequence."""
        state = State()
        for event in sample_events:
            state.apply_event(event)

        assert state.metrics["event_count"] == 7
        assert "F1" in state.facts
        assert state.facts["F1"]["confidence"] == 0.95
        assert state.metrics["total_tokens_in"] == 300
        assert state.metrics["total_tokens_out"] == 150


class TestReplayEngine:
    """Tests for ReplayEngine class."""

    def test_replay_engine_creation(self, sample_chain: Chain):
        """Test replay engine creation."""
        engine = ReplayEngine(sample_chain)
        assert engine.chain == sample_chain
        assert len(engine._state_cache) == 0

    def test_replay_to_specific_seq(self, replay_engine: ReplayEngine):
        """Test replaying to specific sequence number."""
        state = replay_engine.replay_to(4)

        assert state.metrics["event_count"] == 4
        assert "F1" in state.facts
        assert state.facts["F1"]["confidence"] == 0.9

    def test_replay_to_first_event(self, replay_engine: ReplayEngine):
        """Test replaying to first event only."""
        state = replay_engine.replay_to(1)

        assert state.metrics["event_count"] == 1
        assert state.current_agent == "planner"

    def test_replay_all(self, replay_engine: ReplayEngine):
        """Test replaying all events."""
        state = replay_engine.replay_all()

        assert state.metrics["event_count"] == 7
        assert "F1" in state.facts
        assert state.facts["F1"]["confidence"] == 0.95

    def test_replay_all_empty_chain(self, empty_chain: Chain):
        """Test replaying empty chain."""
        engine = ReplayEngine(empty_chain)
        state = engine.replay_all()

        assert state.metrics["event_count"] == 0
        assert state.facts == {}

    def test_replay_uses_cache(self, replay_engine: ReplayEngine):
        """Test that replay engine caches intermediate states."""
        replay_engine._state_cache.put(
            3,
            State(
                facts={"CACHED": {"text": "cached", "confidence": 1.0}},
                metrics={
                    "event_count": 3,
                    "total_duration_ms": 0,
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "error_count": 0,
                },
            ),
        )
        state = replay_engine.replay_to(5)

        assert "CACHED" in state.facts

    def test_get_trace(self, replay_engine: ReplayEngine):
        """Test getting execution trace."""
        trace = replay_engine.get_trace()

        step_events = [t for t in trace if t["type"] in ("step_start", "step_end")]
        assert len(step_events) == 4

        assert trace[0]["type"] == "step_start"
        assert trace[0]["agent"] == "planner"

    def test_get_trace_empty_chain(self, empty_chain: Chain):
        """Test getting trace from empty chain."""
        engine = ReplayEngine(empty_chain)
        trace = engine.get_trace()
        assert trace == []

    def test_diff_identical_chains(self, sample_chain: Chain):
        """Test diffing identical chains."""
        engine1 = ReplayEngine(sample_chain)
        engine2 = ReplayEngine(sample_chain)

        diffs = engine1.diff(engine2)
        assert diffs == []

    def test_diff_divergent_chains(self, sample_chain: Chain, divergent_chain: Chain):
        """Test diffing divergent chains."""
        engine1 = ReplayEngine(sample_chain)
        engine2 = ReplayEngine(divergent_chain)

        diffs = engine1.diff(engine2)

        assert len(diffs) > 0
        diverged_diffs = [d for d in diffs if d["type"] == "diverged"]
        assert len(diverged_diffs) >= 1

    def test_diff_missing_events(self, sample_chain: Chain, base_timestamp: datetime):
        """Test diffing chains with different lengths."""
        shorter_chain = Chain(id="shorter")
        for event in sample_chain.events[:3]:
            shorter_chain.add_event(event)

        engine1 = ReplayEngine(sample_chain)
        engine2 = ReplayEngine(shorter_chain)

        diffs = engine1.diff(engine2)

        missing_in_second = [d for d in diffs if d["type"] == "missing_in_second"]
        assert len(missing_in_second) == 4

    def test_diff_extra_events(self, sample_chain: Chain, base_timestamp: datetime):
        """Test diffing chains where second has more events."""
        shorter_chain = Chain(id="shorter")
        for event in sample_chain.events[:3]:
            shorter_chain.add_event(event)

        engine1 = ReplayEngine(shorter_chain)
        engine2 = ReplayEngine(sample_chain)

        diffs = engine1.diff(engine2)

        missing_in_first = [d for d in diffs if d["type"] == "missing_in_first"]
        assert len(missing_in_first) == 4

    def test_find_bottlenecks(self, replay_engine: ReplayEngine):
        """Test finding performance bottlenecks."""
        bottlenecks = replay_engine.find_bottlenecks()

        assert len(bottlenecks) == 2
        assert all("duration_ms" in b for b in bottlenecks)
        assert all("percentage" in b for b in bottlenecks)
        assert bottlenecks[0]["duration_ms"] >= bottlenecks[1]["duration_ms"]

    def test_find_bottlenecks_empty_chain(self, empty_chain: Chain):
        """Test finding bottlenecks in empty chain."""
        engine = ReplayEngine(empty_chain)
        bottlenecks = engine.find_bottlenecks()
        assert bottlenecks == []

    def test_find_bottlenecks_calculates_percentage(self, replay_engine: ReplayEngine):
        """Test that bottleneck percentages are calculated correctly."""
        bottlenecks = replay_engine.find_bottlenecks()

        total_percentage = sum(b["percentage"] for b in bottlenecks)
        assert abs(total_percentage - 100.0) < 0.1

    def test_get_confidence_timeline(self, chain_for_confidence_test: Chain):
        """Test getting confidence timeline for facts."""
        engine = ReplayEngine(chain_for_confidence_test)
        timelines = engine.get_confidence_timeline()

        assert "F1" in timelines
        assert len(timelines["F1"]) == 3
        assert timelines["F1"][0]["confidence"] == 0.5
        assert timelines["F1"][1]["confidence"] == 0.7
        assert timelines["F1"][2]["confidence"] == 0.9

        assert "F2" in timelines
        assert len(timelines["F2"]) == 1
        assert timelines["F2"][0]["confidence"] == 0.3

    def test_get_confidence_timeline_tracks_actions(self, chain_for_confidence_test: Chain):
        """Test that confidence timeline tracks action types."""
        engine = ReplayEngine(chain_for_confidence_test)
        timelines = engine.get_confidence_timeline()

        assert timelines["F1"][0]["action"] == "added"
        assert timelines["F1"][1]["action"] == "modified"
        assert timelines["F1"][2]["action"] == "modified"

    def test_get_confidence_timeline_empty_chain(self, empty_chain: Chain):
        """Test confidence timeline for empty chain."""
        engine = ReplayEngine(empty_chain)
        timelines = engine.get_confidence_timeline()
        assert timelines == {}


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_chain_with_no_step_events(self, base_timestamp: datetime):
        """Test chain containing only fact events."""
        chain = Chain(id="facts-only")
        chain.add_event(
            Event(
                seq=1,
                type=EventType.FACT_ADDED,
                timestamp=base_timestamp,
                agent="agent1",
                data={"id": "F1", "text": "fact1", "confidence": 1.0},
            )
        )

        engine = ReplayEngine(chain)
        state = engine.replay_all()

        assert "F1" in state.facts
        assert engine.find_bottlenecks() == []

    def test_state_with_missing_duration(self, base_timestamp: datetime):
        """Test step end without duration_ms."""
        state = State()
        event = Event(
            seq=2,
            type=EventType.STEP_END,
            timestamp=base_timestamp,
            agent="agent1",
            data={"outcome": "success"},
        )
        state.apply_event(event)

        assert state.metrics["total_duration_ms"] == 0

    def test_state_with_missing_tokens(self, base_timestamp: datetime):
        """Test step end without token data."""
        state = State()
        event = Event(
            seq=2,
            type=EventType.STEP_END,
            timestamp=base_timestamp,
            agent="agent1",
            data={"outcome": "success", "duration_ms": 100},
        )
        state.apply_event(event)

        assert state.metrics["total_tokens_in"] == 0
        assert state.metrics["total_tokens_out"] == 0

    def test_replay_to_beyond_chain_length(self, replay_engine: ReplayEngine):
        """Test replaying to seq beyond chain length."""
        state = replay_engine.replay_to(100)
        assert state.metrics["event_count"] == 7

    def test_error_event_with_minimal_data(self, base_timestamp: datetime):
        """Test error event with minimal data fields."""
        state = State()
        event = Event(
            seq=1, type=EventType.ERROR, timestamp=base_timestamp, agent="agent1", data={}
        )
        state.apply_event(event)

        assert len(state.errors) == 1
        assert state.errors[0]["category"] == "unknown"
        assert state.errors[0]["type"] == "unknown"

    def test_large_chain_performance(self, base_timestamp: datetime):
        """Test performance with larger chain."""
        chain = Chain(id="large-chain")
        for i in range(1000):
            chain.add_event(
                Event(
                    seq=i + 1,
                    type=EventType.FACT_ADDED,
                    timestamp=base_timestamp + timedelta(milliseconds=i),
                    agent=f"agent{i % 5}",
                    data={"id": f"F{i}", "text": f"Fact {i}", "confidence": 0.9},
                )
            )

        engine = ReplayEngine(chain)
        state = engine.replay_all()

        assert state.metrics["event_count"] == 1000
        assert len(state.facts) == 1000
