"""Benchmarks for LCTL event processing.

Measures:
- Chain loading from JSON/YAML
- Event serialization/deserialization
- Chain save operations
- Event creation and addition
"""

import gc
import json
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from lctl.core.events import Chain, Event, EventType

from .bench_replay import BenchmarkResult, benchmark
from .fixtures import (
    ChainSize,
    generate_chain,
    save_chain_to_temp_file,
)


class EventBenchmarks:
    """Event processing benchmarks."""

    def __init__(self, baseline: Optional[Dict[str, float]] = None):
        """Initialize with optional baseline for comparison.

        Args:
            baseline: Dictionary mapping benchmark names to baseline ops/sec.
        """
        self.baseline = baseline or {}
        self.results: List[BenchmarkResult] = []
        self._temp_files: List[Path] = []

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self._temp_files = []

    def bench_chain_load_json_small(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark loading 100 events from JSON."""
        chain = generate_chain(ChainSize.SMALL)
        temp_path = save_chain_to_temp_file(chain, format="json")
        self._temp_files.append(temp_path)

        result = benchmark(
            name="load_json_100_events",
            func=lambda: Chain.load(temp_path),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.SMALL
        result.metadata["format"] = "json"
        self.results.append(result)
        return result

    def bench_chain_load_json_medium(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark loading 1000 events from JSON."""
        chain = generate_chain(ChainSize.MEDIUM)
        temp_path = save_chain_to_temp_file(chain, format="json")
        self._temp_files.append(temp_path)

        result = benchmark(
            name="load_json_1000_events",
            func=lambda: Chain.load(temp_path),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["format"] = "json"
        self.results.append(result)
        return result

    def bench_chain_load_json_large(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark loading 10000 events from JSON."""
        chain = generate_chain(ChainSize.LARGE)
        temp_path = save_chain_to_temp_file(chain, format="json")
        self._temp_files.append(temp_path)

        result = benchmark(
            name="load_json_10000_events",
            func=lambda: Chain.load(temp_path),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.LARGE
        result.metadata["format"] = "json"
        self.results.append(result)
        return result

    def bench_chain_load_yaml_small(self, iterations: int = 30) -> BenchmarkResult:
        """Benchmark loading 100 events from YAML."""
        chain = generate_chain(ChainSize.SMALL)
        temp_path = save_chain_to_temp_file(chain, format="yaml")
        self._temp_files.append(temp_path)

        result = benchmark(
            name="load_yaml_100_events",
            func=lambda: Chain.load(temp_path),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.SMALL
        result.metadata["format"] = "yaml"
        self.results.append(result)
        return result

    def bench_chain_load_yaml_medium(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark loading 1000 events from YAML."""
        chain = generate_chain(ChainSize.MEDIUM)
        temp_path = save_chain_to_temp_file(chain, format="yaml")
        self._temp_files.append(temp_path)

        result = benchmark(
            name="load_yaml_1000_events",
            func=lambda: Chain.load(temp_path),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["format"] = "yaml"
        self.results.append(result)
        return result

    def bench_chain_save_json(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark saving 1000 events to JSON."""
        chain = generate_chain(ChainSize.MEDIUM)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "bench_save.json"

            result = benchmark(
                name="save_json_1000_events",
                func=lambda: chain.save(temp_path),
                iterations=iterations,
            )

        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["format"] = "json"
        self.results.append(result)
        return result

    def bench_chain_save_yaml(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark saving 1000 events to YAML."""
        chain = generate_chain(ChainSize.MEDIUM)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "bench_save.yaml"

            result = benchmark(
                name="save_yaml_1000_events",
                func=lambda: chain.save(temp_path),
                iterations=iterations,
            )

        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["format"] = "yaml"
        self.results.append(result)
        return result

    def bench_event_to_dict(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark event serialization to dict."""
        from datetime import datetime, timezone

        events = [
            Event(
                seq=i,
                type=EventType.FACT_ADDED,
                timestamp=datetime.now(timezone.utc),
                agent="agent1",
                data={"id": f"F{i}", "text": f"Fact {i}", "confidence": 0.9},
            )
            for i in range(100)
        ]

        def serialize_all():
            return [e.to_dict() for e in events]

        result = benchmark(
            name="event_to_dict_100",
            func=serialize_all,
            iterations=iterations,
        )
        result.metadata["event_count"] = 100
        self.results.append(result)
        return result

    def bench_event_from_dict(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark event deserialization from dict."""
        from datetime import datetime, timezone

        dicts = [
            {
                "seq": i,
                "type": "fact_added",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "agent1",
                "data": {"id": f"F{i}", "text": f"Fact {i}", "confidence": 0.9},
            }
            for i in range(100)
        ]

        def deserialize_all():
            return [Event.from_dict(d) for d in dicts]

        result = benchmark(
            name="event_from_dict_100",
            func=deserialize_all,
            iterations=iterations,
        )
        result.metadata["event_count"] = 100
        self.results.append(result)
        return result

    def bench_chain_to_dict(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark chain serialization to dict."""
        chain = generate_chain(ChainSize.MEDIUM)

        result = benchmark(
            name="chain_to_dict_1000",
            func=lambda: chain.to_dict(),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        self.results.append(result)
        return result

    def bench_chain_from_dict(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark chain deserialization from dict."""
        chain = generate_chain(ChainSize.MEDIUM)
        chain_dict = chain.to_dict()

        result = benchmark(
            name="chain_from_dict_1000",
            func=lambda: Chain.from_dict(chain_dict),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        self.results.append(result)
        return result

    def bench_add_events(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark adding events to chain."""
        from datetime import datetime, timezone

        def add_1000_events():
            chain = Chain(id="bench-add")
            for i in range(1000):
                event = Event(
                    seq=i + 1,
                    type=EventType.FACT_ADDED,
                    timestamp=datetime.now(timezone.utc),
                    agent="agent1",
                    data={"id": f"F{i}", "text": f"Fact {i}", "confidence": 0.9},
                )
                chain.add_event(event)
            return chain

        result = benchmark(
            name="add_events_1000",
            func=add_1000_events,
            iterations=iterations,
        )
        result.metadata["event_count"] = 1000
        self.results.append(result)
        return result

    def bench_json_vs_yaml_comparison(self) -> List[BenchmarkResult]:
        """Run comparison benchmarks between JSON and YAML."""
        results = []

        chain = generate_chain(ChainSize.MEDIUM)
        json_path = save_chain_to_temp_file(chain, format="json")
        yaml_path = save_chain_to_temp_file(chain, format="yaml")
        self._temp_files.extend([json_path, yaml_path])

        json_result = benchmark(
            name="load_json_1000_comparison",
            func=lambda: Chain.load(json_path),
            iterations=20,
        )
        json_result.metadata["format"] = "json"
        json_result.metadata["file_size_kb"] = json_path.stat().st_size / 1024
        results.append(json_result)

        yaml_result = benchmark(
            name="load_yaml_1000_comparison",
            func=lambda: Chain.load(yaml_path),
            iterations=20,
        )
        yaml_result.metadata["format"] = "yaml"
        yaml_result.metadata["file_size_kb"] = yaml_path.stat().st_size / 1024
        results.append(yaml_result)

        self.results.extend(results)
        return results

    def run_all(self) -> List[BenchmarkResult]:
        """Run all event benchmarks."""
        try:
            self.results = []
            self.bench_chain_load_json_small()
            self.bench_chain_load_json_medium()
            self.bench_chain_load_json_large()
            self.bench_chain_load_yaml_small()
            self.bench_chain_load_yaml_medium()
            self.bench_chain_save_json()
            self.bench_chain_save_yaml()
            self.bench_event_to_dict()
            self.bench_event_from_dict()
            self.bench_chain_to_dict()
            self.bench_chain_from_dict()
            self.bench_add_events()
            return self.results
        finally:
            self.cleanup()

    def run_quick(self) -> List[BenchmarkResult]:
        """Run a quick subset of benchmarks."""
        try:
            self.results = []
            self.bench_chain_load_json_small(iterations=10)
            self.bench_chain_load_yaml_small(iterations=10)
            self.bench_event_to_dict(iterations=20)
            return self.results
        finally:
            self.cleanup()

    def get_comparison(self) -> List[Dict[str, Any]]:
        """Compare results against baseline."""
        comparisons = []
        for result in self.results:
            comparison = result.to_dict()
            if result.name in self.baseline:
                baseline_ops = self.baseline[result.name]
                comparison["baseline_ops_per_second"] = baseline_ops
                if baseline_ops > 0:
                    comparison["speedup"] = round(result.ops_per_second / baseline_ops, 2)
                    comparison["percentage_change"] = round(
                        ((result.ops_per_second - baseline_ops) / baseline_ops) * 100, 1
                    )
            comparisons.append(comparison)
        return comparisons


def run_event_benchmarks(
    quick: bool = False,
    baseline: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Run event benchmarks and return results.

    Args:
        quick: If True, run a quick subset of benchmarks.
        baseline: Optional baseline for comparison.

    Returns:
        List of benchmark results as dictionaries.
    """
    benchmarks = EventBenchmarks(baseline=baseline)

    if quick:
        benchmarks.run_quick()
    else:
        benchmarks.run_all()

    if baseline:
        return benchmarks.get_comparison()
    return [r.to_dict() for r in benchmarks.results]


if __name__ == "__main__":
    print("Running event benchmarks...")
    print("=" * 60)

    results = run_event_benchmarks()

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Avg time: {result['avg_time_ms']:.3f} ms")
        print(f"  Ops/sec: {result['ops_per_second']:.2f}")
        print(f"  Memory peak: {result['memory_peak_mb']:.3f} MB")
        if "file_size_kb" in result.get("metadata", {}):
            print(f"  File size: {result['metadata']['file_size_kb']:.1f} KB")

    print("\n" + "=" * 60)
    print("Event benchmarks complete.")
