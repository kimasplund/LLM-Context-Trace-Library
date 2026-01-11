"""Benchmark fixtures for generating test chains of various sizes.

Provides factory functions to create chains with realistic event distributions
for performance benchmarking.
"""

import random
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple

from lctl.core.events import Chain, Event, EventType


class ChainSize(IntEnum):
    """Standard chain sizes for benchmarking."""

    SMALL = 100
    MEDIUM = 1000
    LARGE = 10000
    XLARGE = 50000


def generate_chain(
    size: int,
    chain_id: str = "benchmark-chain",
    seed: int = 42,
) -> Chain:
    """Generate a chain with the specified number of events.

    Creates a realistic distribution of events including:
    - step_start/step_end pairs
    - fact_added events
    - tool_call events
    - occasional errors

    Args:
        size: Number of events to generate.
        chain_id: Identifier for the chain.
        seed: Random seed for reproducibility.

    Returns:
        A Chain populated with the specified number of events.
    """
    random.seed(seed)
    chain = Chain(id=chain_id)
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    agents = ["planner", "executor", "analyzer", "validator", "synthesizer"]
    current_step_agent = None
    fact_counter = 0

    for seq in range(1, size + 1):
        timestamp = base_time + timedelta(milliseconds=seq * 10)
        agent = random.choice(agents)

        if current_step_agent is None and random.random() < 0.3:
            event_type = EventType.STEP_START
            current_step_agent = agent
            data = {
                "intent": random.choice(["analyze", "plan", "execute", "validate"]),
                "input_summary": f"input_{seq}",
            }
        elif current_step_agent is not None and random.random() < 0.25:
            event_type = EventType.STEP_END
            agent = current_step_agent
            current_step_agent = None
            data = {
                "outcome": random.choice(["success", "success", "success", "partial"]),
                "output_summary": f"output_{seq}",
                "duration_ms": random.randint(50, 2000),
                "tokens": {
                    "input": random.randint(50, 500),
                    "output": random.randint(20, 200),
                },
            }
        elif random.random() < 0.4:
            event_type = EventType.FACT_ADDED
            fact_counter += 1
            data = {
                "id": f"F{fact_counter}",
                "text": f"Benchmark fact {fact_counter}",
                "confidence": round(random.uniform(0.5, 1.0), 2),
                "source": agent,
            }
        elif random.random() < 0.3:
            event_type = EventType.TOOL_CALL
            data = {
                "tool": random.choice(["search", "read_file", "write_file", "execute"]),
                "input": f"tool_input_{seq}",
                "output": f"tool_output_{seq}",
                "duration_ms": random.randint(10, 500),
            }
        elif random.random() < 0.05:
            event_type = EventType.ERROR
            data = {
                "category": random.choice(["validation", "execution", "timeout"]),
                "type": random.choice(["ValueError", "RuntimeError", "TimeoutError"]),
                "message": f"Error at sequence {seq}",
                "recoverable": random.choice([True, True, False]),
            }
        else:
            event_type = EventType.FACT_ADDED
            fact_counter += 1
            data = {
                "id": f"F{fact_counter}",
                "text": f"Benchmark fact {fact_counter}",
                "confidence": round(random.uniform(0.5, 1.0), 2),
                "source": agent,
            }

        event = Event(
            seq=seq,
            type=event_type,
            timestamp=timestamp,
            agent=agent,
            data=data,
        )
        chain.add_event(event)

    return chain


def generate_chain_with_errors(
    size: int,
    error_rate: float = 0.1,
    chain_id: str = "error-chain",
    seed: int = 42,
) -> Chain:
    """Generate a chain with a specified error rate.

    Args:
        size: Number of events to generate.
        error_rate: Proportion of events that should be errors (0.0 to 1.0).
        chain_id: Identifier for the chain.
        seed: Random seed for reproducibility.

    Returns:
        A Chain with the specified error rate.
    """
    random.seed(seed)
    chain = Chain(id=chain_id)
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    agents = ["agent1", "agent2", "agent3"]

    for seq in range(1, size + 1):
        timestamp = base_time + timedelta(milliseconds=seq * 10)
        agent = random.choice(agents)

        if random.random() < error_rate:
            event_type = EventType.ERROR
            data = {
                "category": random.choice(["validation", "execution", "timeout", "network"]),
                "type": random.choice(["ValueError", "RuntimeError", "TimeoutError", "IOError"]),
                "message": f"Error at sequence {seq}",
                "recoverable": random.choice([True, False]),
                "suggested_action": "Retry or escalate",
            }
        else:
            event_type = EventType.STEP_START
            data = {
                "intent": "process",
                "input_summary": f"input_{seq}",
            }

        event = Event(
            seq=seq,
            type=event_type,
            timestamp=timestamp,
            agent=agent,
            data=data,
        )
        chain.add_event(event)

    return chain


def generate_chain_with_confidence_changes(
    num_facts: int,
    modifications_per_fact: int = 3,
    chain_id: str = "confidence-chain",
    seed: int = 42,
) -> Chain:
    """Generate a chain focused on fact confidence changes.

    Creates facts and then modifies their confidence values over time,
    useful for benchmarking confidence timeline generation.

    Args:
        num_facts: Number of facts to create.
        modifications_per_fact: Number of times each fact is modified.
        chain_id: Identifier for the chain.
        seed: Random seed for reproducibility.

    Returns:
        A Chain with confidence timeline data.
    """
    random.seed(seed)
    chain = Chain(id=chain_id)
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    agents = ["analyst1", "analyst2", "verifier", "reviewer"]
    seq = 0

    for fact_num in range(1, num_facts + 1):
        seq += 1
        fact_id = f"F{fact_num}"
        initial_confidence = round(random.uniform(0.3, 0.6), 2)

        event = Event(
            seq=seq,
            type=EventType.FACT_ADDED,
            timestamp=base_time + timedelta(milliseconds=seq * 10),
            agent=random.choice(agents),
            data={
                "id": fact_id,
                "text": f"Hypothesis {fact_num}",
                "confidence": initial_confidence,
            },
        )
        chain.add_event(event)

        current_confidence = initial_confidence
        for mod_num in range(modifications_per_fact):
            seq += 1
            delta = random.uniform(-0.1, 0.2)
            new_confidence = max(0.1, min(1.0, current_confidence + delta))
            current_confidence = round(new_confidence, 2)

            event = Event(
                seq=seq,
                type=EventType.FACT_MODIFIED,
                timestamp=base_time + timedelta(milliseconds=seq * 10),
                agent=random.choice(agents),
                data={
                    "id": fact_id,
                    "confidence": current_confidence,
                    "reason": f"Modification {mod_num + 1}",
                },
            )
            chain.add_event(event)

    return chain


def generate_divergent_chains(
    size: int,
    divergence_point: int,
    chain_id_prefix: str = "divergent",
    seed: int = 42,
) -> Tuple[Chain, Chain]:
    """Generate two chains that diverge at a specified point.

    Useful for benchmarking chain diff operations.

    Args:
        size: Total number of events in each chain.
        divergence_point: Sequence number where chains start to differ.
        chain_id_prefix: Prefix for chain identifiers.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of two chains (original, divergent).
    """
    random.seed(seed)

    chain1 = generate_chain(size, chain_id=f"{chain_id_prefix}-original", seed=seed)

    chain2 = Chain(id=f"{chain_id_prefix}-alternate")
    for event in chain1.events:
        if event.seq < divergence_point:
            chain2.add_event(event)
        else:
            random.seed(seed + event.seq)
            modified_event = Event(
                seq=event.seq,
                type=event.type,
                timestamp=event.timestamp + timedelta(milliseconds=random.randint(1, 100)),
                agent=f"alt_{event.agent}",
                data={**event.data, "modified": True},
            )
            chain2.add_event(modified_event)

    return chain1, chain2


def generate_step_pairs(
    num_pairs: int,
    avg_duration_ms: int = 500,
    chain_id: str = "step-pairs-chain",
    seed: int = 42,
) -> Chain:
    """Generate a chain with matched step_start/step_end pairs.

    Useful for benchmarking bottleneck analysis which relies on step pairs.

    Args:
        num_pairs: Number of step start/end pairs to generate.
        avg_duration_ms: Average duration for each step.
        chain_id: Identifier for the chain.
        seed: Random seed for reproducibility.

    Returns:
        A Chain with step pairs.
    """
    random.seed(seed)
    chain = Chain(id=chain_id)
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    agents = ["planner", "executor", "analyzer", "validator"]
    seq = 0
    current_time = base_time

    for pair_num in range(num_pairs):
        seq += 1
        agent = agents[pair_num % len(agents)]
        duration = max(10, int(random.gauss(avg_duration_ms, avg_duration_ms / 3)))

        start_event = Event(
            seq=seq,
            type=EventType.STEP_START,
            timestamp=current_time,
            agent=agent,
            data={
                "intent": f"task_{pair_num}",
                "input_summary": f"input_{pair_num}",
            },
        )
        chain.add_event(start_event)

        current_time += timedelta(milliseconds=duration)
        seq += 1

        end_event = Event(
            seq=seq,
            type=EventType.STEP_END,
            timestamp=current_time,
            agent=agent,
            data={
                "outcome": "success",
                "output_summary": f"output_{pair_num}",
                "duration_ms": duration,
                "tokens": {
                    "input": random.randint(50, 500),
                    "output": random.randint(20, 200),
                },
            },
        )
        chain.add_event(end_event)

        current_time += timedelta(milliseconds=random.randint(10, 100))

    return chain


def save_chain_to_temp_file(chain: Chain, format: str = "json") -> Path:
    """Save a chain to a temporary file for I/O benchmarks.

    Args:
        chain: The chain to save.
        format: File format, either "json" or "yaml".

    Returns:
        Path to the temporary file.
    """
    suffix = f".lctl.{format}"
    with NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        temp_path = Path(f.name)

    chain.save(temp_path)
    return temp_path


def create_benchmark_chains() -> List[Tuple[str, Chain]]:
    """Create a standard set of benchmark chains.

    Returns:
        List of (name, chain) tuples for benchmarking.
    """
    return [
        ("small_100", generate_chain(ChainSize.SMALL)),
        ("medium_1000", generate_chain(ChainSize.MEDIUM)),
        ("large_10000", generate_chain(ChainSize.LARGE)),
        ("step_pairs_500", generate_step_pairs(500)),
        ("confidence_100x5", generate_chain_with_confidence_changes(100, 5)),
        ("error_heavy_1000", generate_chain_with_errors(1000, error_rate=0.2)),
    ]
