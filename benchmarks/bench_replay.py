"""Benchmarks for LCTL replay performance.

Measures:
- Replay 100, 1000, 10000 events
- State computation at various points
- Cache effectiveness
- Memory usage during replay
"""

import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from lctl.core.events import Chain, ReplayEngine, State

from .fixtures import ChainSize, generate_chain, generate_step_pairs


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    ops_per_second: float
    memory_peak_mb: float
    memory_current_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "ops_per_second": round(self.ops_per_second, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 3),
            "memory_current_mb": round(self.memory_current_mb, 3),
            "metadata": self.metadata,
        }


def benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Run a benchmark with timing and memory tracking.

    Args:
        name: Name of the benchmark.
        func: Function to benchmark.
        iterations: Number of iterations to run.
        warmup: Number of warmup iterations (not counted).

    Returns:
        BenchmarkResult with timing and memory data.
    """
    for _ in range(warmup):
        func()

    gc.collect()
    tracemalloc.start()
    memory_before = tracemalloc.get_traced_memory()[0]

    start_time = time.perf_counter()
    for _ in range(iterations):
        func()
    end_time = time.perf_counter()

    memory_current, memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / iterations
    ops_per_second = iterations / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time_ms,
        avg_time_ms=avg_time_ms,
        ops_per_second=ops_per_second,
        memory_peak_mb=(memory_peak - memory_before) / (1024 * 1024),
        memory_current_mb=(memory_current - memory_before) / (1024 * 1024),
    )


class ReplayBenchmarks:
    """Replay performance benchmarks."""

    def __init__(self, baseline: Optional[Dict[str, float]] = None):
        """Initialize with optional baseline for comparison.

        Args:
            baseline: Dictionary mapping benchmark names to baseline ops/sec.
        """
        self.baseline = baseline or {}
        self.results: List[BenchmarkResult] = []

    def bench_replay_small(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark replaying 100 events."""
        chain = generate_chain(ChainSize.SMALL)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="replay_100_events",
            func=lambda: engine.replay_all(),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.SMALL
        self.results.append(result)
        return result

    def bench_replay_medium(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark replaying 1000 events."""
        chain = generate_chain(ChainSize.MEDIUM)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="replay_1000_events",
            func=lambda: engine.replay_all(),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        self.results.append(result)
        return result

    def bench_replay_large(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark replaying 10000 events."""
        chain = generate_chain(ChainSize.LARGE)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="replay_10000_events",
            func=lambda: engine.replay_all(),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.LARGE
        self.results.append(result)
        return result

    def bench_replay_to_midpoint(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark replaying to midpoint of chain."""
        chain = generate_chain(ChainSize.MEDIUM)
        engine = ReplayEngine(chain)
        midpoint = ChainSize.MEDIUM // 2

        result = benchmark(
            name="replay_to_midpoint_1000",
            func=lambda: engine.replay_to(midpoint),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["target_seq"] = midpoint
        self.results.append(result)
        return result

    def bench_replay_incremental(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark incremental replay (stepping through events)."""
        chain = generate_chain(500)
        engine = ReplayEngine(chain)

        def incremental_replay():
            for seq in range(1, 501, 50):
                engine.replay_to(seq)

        result = benchmark(
            name="replay_incremental_500",
            func=incremental_replay,
            iterations=iterations,
        )
        result.metadata["event_count"] = 500
        result.metadata["steps"] = 10
        self.results.append(result)
        return result

    def bench_replay_with_cache(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark replay with pre-populated cache."""
        chain = generate_chain(ChainSize.MEDIUM)
        engine = ReplayEngine(chain)

        state_at_500 = engine.replay_to(500)
        engine._state_cache[500] = state_at_500

        result = benchmark(
            name="replay_cached_1000",
            func=lambda: engine.replay_to(ChainSize.MEDIUM),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["cache_point"] = 500
        self.results.append(result)
        return result

    def bench_state_apply_events(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark raw event application to state."""
        chain = generate_chain(ChainSize.MEDIUM)

        def apply_all_events():
            state = State()
            for event in chain.events:
                state.apply_event(event)
            return state

        result = benchmark(
            name="state_apply_1000_events",
            func=apply_all_events,
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        self.results.append(result)
        return result

    def bench_replay_xlarge(self, iterations: int = 2) -> BenchmarkResult:
        """Benchmark replaying 50000 events (stress test)."""
        chain = generate_chain(ChainSize.XLARGE)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="replay_50000_events",
            func=lambda: engine.replay_all(),
            iterations=iterations,
            warmup=1,
        )
        result.metadata["event_count"] = ChainSize.XLARGE
        self.results.append(result)
        return result

    def run_all(self) -> List[BenchmarkResult]:
        """Run all replay benchmarks."""
        self.results = []
        self.bench_replay_small()
        self.bench_replay_medium()
        self.bench_replay_large()
        self.bench_replay_to_midpoint()
        self.bench_replay_incremental()
        self.bench_replay_with_cache()
        self.bench_state_apply_events()
        return self.results

    def run_quick(self) -> List[BenchmarkResult]:
        """Run a quick subset of benchmarks."""
        self.results = []
        self.bench_replay_small(iterations=10)
        self.bench_replay_medium(iterations=5)
        return self.results

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


def run_replay_benchmarks(
    quick: bool = False,
    baseline: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Run replay benchmarks and return results.

    Args:
        quick: If True, run a quick subset of benchmarks.
        baseline: Optional baseline for comparison.

    Returns:
        List of benchmark results as dictionaries.
    """
    benchmarks = ReplayBenchmarks(baseline=baseline)

    if quick:
        benchmarks.run_quick()
    else:
        benchmarks.run_all()

    if baseline:
        return benchmarks.get_comparison()
    return [r.to_dict() for r in benchmarks.results]


if __name__ == "__main__":
    print("Running replay benchmarks...")
    print("=" * 60)

    results = run_replay_benchmarks()

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Avg time: {result['avg_time_ms']:.3f} ms")
        print(f"  Ops/sec: {result['ops_per_second']:.2f}")
        print(f"  Memory peak: {result['memory_peak_mb']:.3f} MB")

    print("\n" + "=" * 60)
    print("Replay benchmarks complete.")
