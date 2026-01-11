"""LCTL Event Sourcing Core - The foundation for time-travel debugging."""

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class EventType(str, Enum):
    """Standard LCTL event types."""
    STEP_START = "step_start"
    STEP_END = "step_end"
    FACT_ADDED = "fact_added"
    FACT_MODIFIED = "fact_modified"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    CONTRACT_VALIDATION = "contract_validation"
    MODEL_ROUTING = "model_routing"


@dataclass
class Event:
    """A single LCTL event."""
    seq: int
    type: EventType
    timestamp: datetime
    agent: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Create Event from dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Validate required fields
        required_fields = ["seq", "type", "timestamp", "agent"]
        missing = [f for f in required_fields if f not in d]
        if missing:
            raise ValueError(f"Event missing required fields: {missing}")

        # Parse type
        event_type = d["type"]
        if event_type in [e.value for e in EventType]:
            event_type = EventType(event_type)

        # Parse timestamp
        timestamp = d["timestamp"]
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format '{d['timestamp']}': {e}")

        return cls(
            seq=d["seq"],
            type=event_type,
            timestamp=timestamp,
            agent=d["agent"],
            data=d.get("data", {})
        )


@dataclass
class Chain:
    """An LCTL chain - a collection of events."""
    id: str
    events: List[Event] = field(default_factory=list)
    version: str = "4.0"

    def add_event(self, event: Event) -> None:
        """Add event to the chain."""
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lctl": self.version,
            "chain": {"id": self.id},
            "events": [e.to_dict() for e in self.events]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chain":
        chain_meta = d.get("chain")

        # Validation logic
        if chain_meta is None or not isinstance(chain_meta, dict):
             # If "chain" key is missing or invalid (e.g. string), we fall back ONLY if "events" exists (legacy support)
             if "events" in d:
                 chain_meta = {}
             else:
                 raise ValueError("Invalid chain format: missing 'chain' or 'events' keys")

        return cls(
            id=chain_meta.get("id", "unknown"),
            version=d.get("lctl", "4.0"),
            events=[Event.from_dict(e) for e in d.get("events", [])]
        )

    @classmethod
    def load(cls, path: Path) -> "Chain":
        """Load chain from file (JSON or YAML).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty, has invalid format, or malformed data.
        """
        if not path.exists():
            raise FileNotFoundError(f"Chain file not found: {path}")

        try:
            content = path.read_text()
        except PermissionError:
            raise PermissionError(f"Permission denied reading chain file: {path}")

        if not content.strip():
            raise ValueError(f"Chain file is empty: {path}")

        try:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid file format in {path}: {e}")

        if data is None:
            raise ValueError(f"Chain file contains no data: {path}")

        if not isinstance(data, dict):
            raise ValueError(f"Chain file must contain a dictionary, got {type(data).__name__}: {path}")

        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Save chain to file."""
        data = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(data, indent=2, default=str)
        path.write_text(content)


@dataclass
class State:
    """Materialized state derived from events."""
    facts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_agent: Optional[str] = None
    current_step: Optional[int] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_duration_ms": 0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "event_count": 0,
        "error_count": 0
    })

    def apply_event(self, event: Event) -> None:
        """Apply an event to update state."""
        self.metrics["event_count"] += 1

        if event.type == EventType.STEP_START:
            self.current_agent = event.agent
            self.current_step = event.seq

        elif event.type == EventType.STEP_END:
            duration = event.data.get("duration_ms", 0)
            tokens = event.data.get("tokens", {})
            self.metrics["total_duration_ms"] += duration
            self.metrics["total_tokens_in"] += tokens.get("input", tokens.get("in", 0))
            self.metrics["total_tokens_out"] += tokens.get("output", tokens.get("out", 0))

        elif event.type == EventType.FACT_ADDED:
            fact_id = event.data.get("id")
            if fact_id:
                self.facts[fact_id] = {
                    "text": event.data.get("text", ""),
                    "confidence": event.data.get("confidence", 1.0),
                    "source": event.data.get("source", event.agent),
                    "seq": event.seq
                }

        elif event.type == EventType.FACT_MODIFIED:
            fact_id = event.data.get("id")
            if fact_id and fact_id in self.facts:
                self.facts[fact_id].update({
                    "text": event.data.get("text", self.facts[fact_id]["text"]),
                    "confidence": event.data.get("confidence", self.facts[fact_id]["confidence"]),
                    "modified_at": event.seq,
                    "reason": event.data.get("reason", "")
                })

        elif event.type == EventType.ERROR:
            self.errors.append({
                "seq": event.seq,
                "agent": event.agent,
                "category": event.data.get("category", "unknown"),
                "type": event.data.get("type", "unknown"),
                "message": event.data.get("message", ""),
                "recoverable": event.data.get("recoverable", False)
            })
            self.metrics["error_count"] += 1

        elif event.type == EventType.TOOL_CALL:
            duration = event.data.get("duration_ms", 0)
            self.metrics["total_duration_ms"] += duration


class ReplayEngine:
    """Engine for replaying events and time-travel debugging."""

    def __init__(self, chain: Chain):
        self.chain = chain
        self._state_cache: Dict[int, State] = {}

    def replay_to(self, target_seq: int) -> State:
        """Replay events up to target_seq and return state."""
        # Check cache for nearest checkpoint
        nearest_cached = 0
        for cached_seq in sorted(self._state_cache.keys()):
            if cached_seq <= target_seq:
                nearest_cached = cached_seq
            else:
                break

        # Start from cached state or fresh
        if nearest_cached > 0:
            cached_state = self._state_cache[nearest_cached]
            state = State(
                facts=copy.deepcopy(cached_state.facts),
                metrics=copy.deepcopy(cached_state.metrics),
                errors=copy.deepcopy(cached_state.errors)
            )
            start_seq = nearest_cached + 1
        else:
            state = State()
            start_seq = 1

        # Apply events
        for event in self.chain.events:
            if event.seq < start_seq:
                continue
            if event.seq > target_seq:
                break
            state.apply_event(event)

        return state

    def replay_all(self) -> State:
        """Replay all events."""
        if not self.chain.events:
            return State()
        return self.replay_to(self.chain.events[-1].seq)

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get step-level trace for visualization."""
        trace = []
        for event in self.chain.events:
            if event.type in (EventType.STEP_START, EventType.STEP_END):
                trace.append({
                    "seq": event.seq,
                    "type": event.type.value,
                    "agent": event.agent,
                    "timestamp": event.timestamp.isoformat(),
                    **event.data
                })
        return trace

    def diff(self, other: "ReplayEngine") -> List[Dict[str, Any]]:
        """Compare two chains and find divergence points."""
        diffs = []
        max_len = max(len(self.chain.events), len(other.chain.events))

        for i in range(max_len):
            e1 = self.chain.events[i] if i < len(self.chain.events) else None
            e2 = other.chain.events[i] if i < len(other.chain.events) else None

            if e1 is None:
                diffs.append({"seq": i + 1, "type": "missing_in_first", "event": e2.to_dict()})
            elif e2 is None:
                diffs.append({"seq": i + 1, "type": "missing_in_second", "event": e1.to_dict()})
            elif e1.to_dict() != e2.to_dict():
                diffs.append({
                    "seq": e1.seq,
                    "type": "diverged",
                    "first": e1.to_dict(),
                    "second": e2.to_dict()
                })

        return diffs

    def find_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance and find bottlenecks."""
        step_durations = {}

        current_step = None
        step_start_time = None

        for event in self.chain.events:
            if event.type == EventType.STEP_START:
                current_step = (event.agent, event.seq)
                step_start_time = event.timestamp
            elif event.type == EventType.STEP_END and current_step:
                duration = event.data.get("duration_ms", 0)
                if duration == 0 and step_start_time:
                    duration = (event.timestamp - step_start_time).total_seconds() * 1000
                step_durations[current_step] = {
                    "agent": current_step[0],
                    "seq": current_step[1],
                    "duration_ms": duration,
                    "tokens": event.data.get("tokens", {})
                }
                current_step = None

        # Sort by duration
        sorted_steps = sorted(step_durations.values(), key=lambda x: x["duration_ms"], reverse=True)

        # Calculate percentages
        total_duration = sum(s["duration_ms"] for s in sorted_steps)
        for step in sorted_steps:
            step["percentage"] = (step["duration_ms"] / total_duration * 100) if total_duration > 0 else 0

        return sorted_steps

    def get_confidence_timeline(self) -> Dict[str, List[Dict[str, Any]]]:
        """Track confidence changes for each fact."""
        timelines = {}

        for event in self.chain.events:
            if event.type in (EventType.FACT_ADDED, EventType.FACT_MODIFIED):
                fact_id = event.data.get("id")
                if fact_id:
                    if fact_id not in timelines:
                        timelines[fact_id] = []
                    timelines[fact_id].append({
                        "seq": event.seq,
                        "confidence": event.data.get("confidence", 1.0),
                        "agent": event.agent,
                        "action": "added" if event.type == EventType.FACT_ADDED else "modified"
                    })

        return timelines
