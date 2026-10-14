"""LCTL Event Sourcing Core - The foundation for time-travel debugging."""

import bisect
import copy
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .ids import new_id
from .schema import CURRENT_VERSION, SchemaMigrator, validate_version

T = TypeVar("T")


class LRUCache:
    """LRU cache with configurable max size.

    Uses OrderedDict to maintain insertion order and efficiently
    move accessed items to the end (most recently used position).
    """

    def __init__(self, max_size: int = 100):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store. When exceeded,
                     the least recently used item is evicted.
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._cache: OrderedDict[int, Any] = OrderedDict()

    def get(self, key: int) -> Any | None:
        """Get item by key, moving it to most recently used position.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value if found, None otherwise.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: int, value: Any) -> None:
        """Put item in cache, evicting oldest if at capacity.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest (first) item
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)

    def __contains__(self, key: int) -> bool:
        """Check if key is in cache (does not affect LRU order)."""
        return key in self._cache

    def keys(self) -> list[int]:
        """Return list of keys in LRU order (oldest first)."""
        return list(self._cache.keys())


class EventType(str, Enum):  # noqa: UP042  # StrEnum changes str() output used in serialized chains
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
    LLM_TRACE = "llm_trace"


@dataclass
class Event:
    """A single LCTL event."""

    seq: int
    type: EventType
    timestamp: datetime
    agent: str
    data: dict[str, Any] = field(default_factory=dict)
    # Format 4.2 identity (optional; absent in older traces)
    id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "data": self.data,
        }
        for key in ("id", "span_id", "parent_span_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        """Create Event from dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Validate required fields
        required_fields = ["seq", "type", "timestamp", "agent"]
        missing = [f for f in required_fields if f not in d]
        if missing:
            raise ValueError(f"Event missing required fields: {missing}")

        # Validate seq
        seq = d.get("seq")
        if not isinstance(seq, int):
            raise ValueError(f"seq must be int, got {type(seq).__name__}")
        if seq < 1:
            raise ValueError(f"seq must be positive, got {seq}")

        # Validate timestamp
        timestamp = d.get("timestamp")
        if timestamp is None:
            raise ValueError("timestamp cannot be None")

        # Parse type
        event_type = d["type"]
        if event_type in [e.value for e in EventType]:
            event_type = EventType(event_type)

        # Parse timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format '{d['timestamp']}': {e}") from e

        return cls(
            seq=seq,
            type=event_type,
            timestamp=timestamp,
            agent=d["agent"],
            data=d.get("data", {}),
            id=d.get("id"),
            span_id=d.get("span_id"),
            parent_span_id=d.get("parent_span_id"),
        )


@dataclass
class Chain:
    """An LCTL chain - a collection of events."""

    id: str
    events: list[Event] = field(default_factory=list)
    version: str = CURRENT_VERSION
    trace_id: str | None = field(default_factory=new_id)

    def add_event(self, event: Event) -> None:
        """Add event to the chain."""
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lctl": self.version,
            "chain": {"id": self.id, **({"trace_id": self.trace_id} if self.trace_id else {})},
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chain":
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d).__name__}")

        # Validate and migrate schema version
        validate_version(d)
        d = SchemaMigrator.migrate(d)

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
            version=d.get("lctl", CURRENT_VERSION),
            events=[Event.from_dict(e) for e in d.get("events", [])],
            trace_id=chain_meta.get("trace_id"),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Chain":
        """Load chain from file (JSON or YAML).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty, has invalid format, or malformed data.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Chain file not found: {path}")

        try:
            content = path.read_text()
        except PermissionError as err:
            raise PermissionError(f"Permission denied reading chain file: {path}") from err

        if not content.strip():
            raise ValueError(f"Chain file is empty: {path}")

        try:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid file format in {path}: {e}") from e

        if data is None:
            raise ValueError(f"Chain file contains no data: {path}")

        if not isinstance(data, dict):
            raise ValueError(
                f"Chain file must contain a dictionary, got {type(data).__name__}: {path}"
            )

        return cls.from_dict(data)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save chain to file. Assigns a trace_id first if the chain has none (pre-4.2 files)."""
        # Write to the real file behind a symlink, and keep its permissions.
        path = Path(path).resolve()
        if self.trace_id is None:
            self.trace_id = new_id()
        data = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(data, indent=2, default=str)
        # Write-then-rename so a concurrent reader never sees a half-written chain.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{new_id()}.tmp")
        try:
            tmp.write_text(content)
            if path.exists():
                shutil.copymode(path, tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


@dataclass
class State:
    """Materialized state derived from events."""

    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_agent: str | None = None
    current_step: int | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "total_duration_ms": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "event_count": 0,
            "error_count": 0,
        }
    )

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
                    "seq": event.seq,
                }

        elif event.type == EventType.FACT_MODIFIED:
            fact_id = event.data.get("id")
            if fact_id and fact_id in self.facts:
                # Use copy-on-write to allow shallow copying of State
                new_fact = self.facts[fact_id].copy()
                new_fact.update(
                    {
                        "text": event.data.get("text", self.facts[fact_id]["text"]),
                        "confidence": event.data.get(
                            "confidence", self.facts[fact_id]["confidence"]
                        ),
                        "modified_at": event.seq,
                        "reason": event.data.get("reason", ""),
                    }
                )
                self.facts[fact_id] = new_fact

        elif event.type == EventType.ERROR:
            self.errors.append(
                {
                    "seq": event.seq,
                    "agent": event.agent,
                    "category": event.data.get("category", "unknown"),
                    "type": event.data.get("type", "unknown"),
                    "message": event.data.get("message", ""),
                    "recoverable": event.data.get("recoverable", False),
                }
            )
            self.metrics["error_count"] += 1

        elif event.type == EventType.TOOL_CALL:
            duration = event.data.get("duration_ms", 0)
            self.metrics["total_duration_ms"] += duration


class ReplayEngine:
    """Engine for replaying events and time-travel debugging."""

    def __init__(self, chain: Chain, cache_size: int = 100):
        """Initialize replay engine.

        Args:
            chain: The chain of events to replay.
            cache_size: Maximum number of states to cache. Defaults to 100.
                       Uses LRU eviction when capacity is reached.
        """
        self.chain = chain
        self._state_cache: LRUCache = LRUCache(max_size=cache_size)

    def replay_to(self, target_seq: int) -> State:
        """Replay events up to target_seq and return state."""
        # Check cache for nearest checkpoint
        nearest_cached = 0
        sorted_keys = sorted(self._state_cache.keys())
        # Use bisect to find nearest cached sequence <= target_seq
        idx = bisect.bisect_right(sorted_keys, target_seq)
        if idx > 0:
            nearest_cached = sorted_keys[idx - 1]

        # Start from cached state or fresh
        if nearest_cached > 0:
            cached_state = self._state_cache.get(nearest_cached)
            if cached_state is not None:
                # Use deep copies for nested structures (facts contains dicts, errors is list of dicts)
                state = State(
                    facts=copy.deepcopy(cached_state.facts),
                    metrics=cached_state.metrics.copy(),  # Simple dict, shallow OK
                    errors=copy.deepcopy(cached_state.errors),  # List of dicts
                    current_agent=cached_state.current_agent,
                    current_step=cached_state.current_step,
                )
                start_seq = nearest_cached + 1
            else:
                # Cache miss (shouldn't happen if keys() is consistent, but be safe)
                state = State()
                start_seq = 1
        else:
            state = State()
            start_seq = 1

        # Find start index using bisect if events are sorted by seq
        # (Assuming events are always appended in order, which is the contract)
        events_start_idx = 0
        if start_seq > 1 and self.chain.events:
            # Binary search to find the first event with seq >= start_seq.
            # Note: This relies on chain.events being sorted by seq.
            # Requires Python >= 3.10 for key argument support in bisect.
            events_start_idx = bisect.bisect_left(self.chain.events, start_seq, key=lambda e: e.seq)

        # Apply events
        for i in range(events_start_idx, len(self.chain.events)):
            event = self.chain.events[i]
            if event.seq > target_seq:
                break
            # Double check seq in case list wasn't perfectly sorted or bisect found something else
            if event.seq < start_seq:
                continue

            state.apply_event(event)

        # Cache the result using LRU cache's put method
        # This will evict the oldest entry if cache is at capacity
        self._state_cache.put(target_seq, state)

        return state

    def replay_all(self) -> State:
        """Replay all events."""
        if not self.chain.events:
            return State()
        return self.replay_to(self.chain.events[-1].seq)

    def get_trace(self) -> list[dict[str, Any]]:
        """Get step-level trace for visualization."""
        trace = []
        for event in self.chain.events:
            if event.type in (EventType.STEP_START, EventType.STEP_END):
                entry = {
                    "seq": event.seq,
                    "type": event.type.value,
                    "agent": event.agent,
                    "timestamp": event.timestamp.isoformat(),
                    **event.data,
                }
                if event.span_id is not None:
                    entry["span_id"] = event.span_id
                if event.parent_span_id is not None:
                    entry["parent_span_id"] = event.parent_span_id
                trace.append(entry)
        return trace

    def get_span_tree(self) -> list[dict[str, Any]]:
        """Build the call tree of spans (steps), outermost first.

        Uses recorded span ids (format 4.2) where present. Older events are paired
        by inference: a STEP_END closes the most recent open inferred span of the
        same agent, and a span's parent is the innermost span open when it starts.
        Inferred ids are deterministic ("inferred-<start_seq>"); a STEP_END with no
        matching start becomes an "orphan-<end_seq>" root instead of being dropped.
        """
        nodes: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        open_ids: list[str] = []
        start_times: dict[str, datetime] = {}

        for event in self.chain.events:
            if event.type == EventType.STEP_START:
                span_id = event.span_id or f"inferred-{event.seq}"
                if event.span_id is not None:
                    parent_id = event.parent_span_id
                else:
                    parent_id = open_ids[-1] if open_ids else None
                node: dict[str, Any] = {
                    "span_id": span_id,
                    "parent_span_id": parent_id,
                    "agent": event.agent,
                    "intent": event.data.get("intent"),
                    "start_seq": event.seq,
                    "end_seq": None,
                    "start_time": event.timestamp.isoformat(),
                    "end_time": None,
                    "duration_ms": None,
                    "outcome": None,
                    "event_count": 0,
                    "children": [],
                    "inferred": event.span_id is None,
                }
                if parent_id in nodes:
                    nodes[parent_id]["children"].append(node)
                else:
                    roots.append(node)
                nodes[span_id] = node
                start_times[span_id] = event.timestamp
                open_ids.append(span_id)
            elif event.type == EventType.STEP_END:
                closing = self._closing_span(event, nodes, open_ids)
                if closing is None:
                    node = {
                        "span_id": f"orphan-{event.seq}",
                        "parent_span_id": None,
                        "agent": event.agent,
                        "intent": None,
                        "start_seq": None,
                        "end_seq": None,
                        "start_time": None,
                        "end_time": None,
                        "duration_ms": None,
                        "outcome": None,
                        "event_count": 0,
                        "children": [],
                        "inferred": True,
                    }
                    roots.append(node)
                else:
                    node = nodes[closing]
                    open_ids.remove(closing)
                node["end_seq"] = event.seq
                node["end_time"] = event.timestamp.isoformat()
                node["outcome"] = event.data.get("outcome")
                node["duration_ms"] = self._span_duration_ms(
                    start_times.get(node["span_id"]), event
                )
            else:
                if event.span_id in nodes:
                    owner: str | None = event.span_id
                elif event.id is None:
                    # Pre-4.2 event: attribute to the innermost open span by inference.
                    owner = open_ids[-1] if open_ids else None
                else:
                    owner = None  # 4.2 event recorded outside any span
                if owner is not None:
                    nodes[owner]["event_count"] += 1

        return roots

    @staticmethod
    def _closing_span(
        event: Event, nodes: dict[str, dict[str, Any]], open_ids: list[str]
    ) -> str | None:
        """Which open span a STEP_END closes.

        Its recorded span when known; otherwise (no span id, e.g. an end recorded by a
        separate process or session) the most recent open span of the same agent,
        whether that span's id was recorded or inferred.
        """
        if event.span_id is not None and event.span_id in nodes:
            return event.span_id if event.span_id in open_ids else None
        for span_id in reversed(open_ids):
            if nodes[span_id]["agent"] == event.agent:
                return span_id
        return None

    @staticmethod
    def _span_duration_ms(start: datetime | None, end_event: Event) -> int | None:
        """Recorded duration if positive, else end minus start (naive times taken as UTC)."""
        try:
            recorded = float(end_event.data.get("duration_ms") or 0)
        except (TypeError, ValueError):
            recorded = 0
        if recorded > 0:
            return int(recorded)
        if start is None:
            return None

        def as_utc(moment: datetime) -> datetime:
            return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)

        return int((as_utc(end_event.timestamp) - as_utc(start)).total_seconds() * 1000)

    def diff(self, other: "ReplayEngine", *, ignore_volatile: bool = True) -> list[dict[str, Any]]:
        """Compare two chains and find divergence points.

        By default only content is compared: ids, span ids, timestamps and
        ``data["duration_ms"]`` differ between any two runs and are ignored.
        Pass ``ignore_volatile=False`` to compare every field.
        """
        diffs = []
        max_len = max(len(self.chain.events), len(other.chain.events))

        def comparable(event: Event) -> dict[str, Any]:
            d = event.to_dict()
            if not ignore_volatile:
                return d
            d = {
                k: v
                for k, v in d.items()
                if k not in ("id", "span_id", "parent_span_id", "timestamp")
            }
            if isinstance(d.get("data"), dict):
                d["data"] = {k: v for k, v in d["data"].items() if k != "duration_ms"}
            return d

        for i in range(max_len):
            e1 = self.chain.events[i] if i < len(self.chain.events) else None
            e2 = other.chain.events[i] if i < len(other.chain.events) else None

            if e1 is None and e2 is not None:
                diffs.append({"seq": i + 1, "type": "missing_in_first", "event": e2.to_dict()})
            elif e2 is None and e1 is not None:
                diffs.append({"seq": i + 1, "type": "missing_in_second", "event": e1.to_dict()})
            elif e1 is not None and e2 is not None and comparable(e1) != comparable(e2):
                diffs.append(
                    {
                        "seq": e1.seq,
                        "type": "diverged",
                        "first": e1.to_dict(),
                        "second": e2.to_dict(),
                    }
                )

        return diffs

    def find_bottlenecks(self, threshold_ms: float = 0) -> list[dict[str, Any]]:
        """Find steps that took longer than threshold.

        Uses a stack-based approach to correctly handle nested steps (LIFO matching).

        Args:
            threshold_ms: Minimum duration in milliseconds to be considered a bottleneck.
                         Default 0 returns all steps.

        Returns:
            List of bottleneck steps sorted by duration (descending), with percentage
            of total time for each step.
        """
        step_durations = []
        step_stack: list[
            dict[str, Any]
        ] = []  # Stack of {'agent': str, 'seq': int, 'start_time': datetime}

        for event in self.chain.events:
            if event.type == EventType.STEP_START:
                step_stack.append(
                    {"agent": event.agent, "seq": event.seq, "start_time": event.timestamp}
                )
            elif event.type == EventType.STEP_END and step_stack:
                # Find matching start (same agent, LIFO)
                for i in range(len(step_stack) - 1, -1, -1):
                    if step_stack[i]["agent"] == event.agent:
                        start_info = step_stack.pop(i)
                        duration_ms = event.data.get("duration_ms", 0)
                        if duration_ms == 0 and start_info["start_time"]:
                            duration_ms = (
                                event.timestamp - start_info["start_time"]
                            ).total_seconds() * 1000
                        if duration_ms >= threshold_ms:
                            step_durations.append(
                                {
                                    "agent": event.agent,
                                    "seq": start_info["seq"],
                                    "duration_ms": duration_ms,
                                    "tokens": event.data.get("tokens", {}),
                                }
                            )
                        break

        # Sort by duration
        sorted_steps = sorted(step_durations, key=lambda x: x["duration_ms"], reverse=True)

        # Calculate percentages
        total_duration = sum(s["duration_ms"] for s in sorted_steps)
        for step in sorted_steps:
            step["percentage"] = (
                (step["duration_ms"] / total_duration * 100) if total_duration > 0 else 0
            )

        return sorted_steps

    def get_confidence_timeline(self) -> dict[str, list[dict[str, Any]]]:
        """Track confidence changes for each fact."""
        timelines: dict[str, list[dict[str, Any]]] = {}

        for event in self.chain.events:
            if event.type in (EventType.FACT_ADDED, EventType.FACT_MODIFIED):
                fact_id = event.data.get("id")
                if fact_id:
                    if fact_id not in timelines:
                        timelines[fact_id] = []
                    timelines[fact_id].append(
                        {
                            "seq": event.seq,
                            "confidence": event.data.get("confidence", 1.0),
                            "agent": event.agent,
                            "action": "added" if event.type == EventType.FACT_ADDED else "modified",
                        }
                    )

        return timelines
