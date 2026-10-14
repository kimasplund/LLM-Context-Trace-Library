"""Shared fixtures for LCTL tests."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest
import yaml

from lctl.core.events import Chain, Event, EventType, ReplayEngine, State


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed base timestamp for reproducible tests."""
    return datetime(2025, 1, 15, 10, 0, 0)


@pytest.fixture
def sample_events(base_timestamp: datetime) -> List[Event]:
    """Create a list of sample events for testing."""
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="planner",
            data={"intent": "plan", "input_summary": "task description"},
        ),
        Event(
            seq=2,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="planner",
            data={"id": "F1", "text": "Initial requirement identified", "confidence": 0.9},
        ),
        Event(
            seq=3,
            type=EventType.TOOL_CALL,
            timestamp=base_timestamp + timedelta(milliseconds=200),
            agent="planner",
            data={"tool": "search", "input": "query", "output": "results", "duration_ms": 50},
        ),
        Event(
            seq=4,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=500),
            agent="planner",
            data={
                "outcome": "success",
                "output_summary": "plan created",
                "duration_ms": 500,
                "tokens": {"input": 100, "output": 50},
            },
        ),
        Event(
            seq=5,
            type=EventType.STEP_START,
            timestamp=base_timestamp + timedelta(milliseconds=600),
            agent="executor",
            data={"intent": "execute", "input_summary": "plan"},
        ),
        Event(
            seq=6,
            type=EventType.FACT_MODIFIED,
            timestamp=base_timestamp + timedelta(milliseconds=700),
            agent="executor",
            data={"id": "F1", "confidence": 0.95, "reason": "verified"},
        ),
        Event(
            seq=7,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=1100),
            agent="executor",
            data={
                "outcome": "success",
                "output_summary": "executed",
                "duration_ms": 500,
                "tokens": {"input": 200, "output": 100},
            },
        ),
    ]
    return events


@pytest.fixture
def sample_chain(sample_events: List[Event]) -> Chain:
    """Create a sample chain with events."""
    chain = Chain(id="test-chain")
    for event in sample_events:
        chain.add_event(event)
    return chain


@pytest.fixture
def sample_chain_with_errors(base_timestamp: datetime) -> Chain:
    """Create a chain with error events."""
    chain = Chain(id="error-chain")
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="agent1",
            data={"intent": "process", "input_summary": "data"},
        ),
        Event(
            seq=2,
            type=EventType.ERROR,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent1",
            data={
                "category": "validation",
                "type": "ValueError",
                "message": "Invalid input data",
                "recoverable": True,
                "suggested_action": "Retry with corrected input",
            },
        ),
        Event(
            seq=3,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=200),
            agent="agent1",
            data={"outcome": "error", "duration_ms": 200, "tokens": {"input": 50, "output": 10}},
        ),
    ]
    for event in events:
        chain.add_event(event)
    return chain


@pytest.fixture
def empty_chain() -> Chain:
    """Create an empty chain."""
    return Chain(id="empty-chain")


@pytest.fixture
def temp_chain_file(sample_chain: Chain, tmp_path: Path) -> Path:
    """Create a temporary JSON chain file."""
    file_path = tmp_path / "test_chain.lctl.json"
    sample_chain.save(file_path)
    return file_path


@pytest.fixture
def temp_yaml_chain_file(sample_chain: Chain, tmp_path: Path) -> Path:
    """Create a temporary YAML chain file."""
    file_path = tmp_path / "test_chain.lctl.yaml"
    sample_chain.save(file_path)
    return file_path


@pytest.fixture
def temp_empty_chain_file(empty_chain: Chain, tmp_path: Path) -> Path:
    """Create a temporary empty chain file."""
    file_path = tmp_path / "empty_chain.lctl.json"
    empty_chain.save(file_path)
    return file_path


@pytest.fixture
def replay_engine(sample_chain: Chain) -> ReplayEngine:
    """Create a replay engine with sample chain."""
    return ReplayEngine(sample_chain)


@pytest.fixture
def divergent_chain(base_timestamp: datetime) -> Chain:
    """Create a chain that diverges from sample_chain at seq 5."""
    chain = Chain(id="divergent-chain")
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="planner",
            data={"intent": "plan", "input_summary": "task description"},
        ),
        Event(
            seq=2,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="planner",
            data={"id": "F1", "text": "Initial requirement identified", "confidence": 0.9},
        ),
        Event(
            seq=3,
            type=EventType.TOOL_CALL,
            timestamp=base_timestamp + timedelta(milliseconds=200),
            agent="planner",
            data={"tool": "search", "input": "query", "output": "results", "duration_ms": 50},
        ),
        Event(
            seq=4,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=500),
            agent="planner",
            data={
                "outcome": "success",
                "output_summary": "plan created",
                "duration_ms": 500,
                "tokens": {"input": 100, "output": 50},
            },
        ),
        Event(
            seq=5,
            type=EventType.STEP_START,
            timestamp=base_timestamp + timedelta(milliseconds=600),
            agent="different_agent",
            data={"intent": "alternate_action", "input_summary": "alternate plan"},
        ),
    ]
    for event in events:
        chain.add_event(event)
    return chain


@pytest.fixture
def chain_for_confidence_test(base_timestamp: datetime) -> Chain:
    """Create a chain with multiple confidence changes for a fact."""
    chain = Chain(id="confidence-chain")
    events = [
        Event(
            seq=1,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp,
            agent="agent1",
            data={"id": "F1", "text": "Hypothesis A", "confidence": 0.5},
        ),
        Event(
            seq=2,
            type=EventType.FACT_MODIFIED,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent2",
            data={"id": "F1", "confidence": 0.7, "reason": "partial verification"},
        ),
        Event(
            seq=3,
            type=EventType.FACT_MODIFIED,
            timestamp=base_timestamp + timedelta(milliseconds=200),
            agent="agent3",
            data={"id": "F1", "confidence": 0.9, "reason": "full verification"},
        ),
        Event(
            seq=4,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=300),
            agent="agent1",
            data={"id": "F2", "text": "Hypothesis B", "confidence": 0.3},
        ),
    ]
    for event in events:
        chain.add_event(event)
    return chain
