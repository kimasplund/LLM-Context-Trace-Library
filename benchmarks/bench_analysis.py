"""Benchmarks for LCTL analysis functions.

Measures:
- Bottleneck analysis
- Confidence timeline generation
- Chain diff comparison
- Execution trace retrieval
"""

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from lctl.core.events import Chain, ReplayEngine

from .bench_replay import BenchmarkResult, benchmark
from .fixtures import (
    ChainSize,
    generate_chain,
    generate_chain_with_confidence_changes,
    generate_divergent_chains,
    generate_step_pairs,
)


class AnalysisBenchmarks:
    """Analysis function benchmarks."""

    def __init__(self, baseline: Optional[Dict[str, float]] = None):
        """Initialize with optional baseline for comparison.

        Args:
            baseline: Dictionary mapping benchmark names to baseline ops/sec.
        """
        self.baseline = baseline or {}
        self.results: List[BenchmarkResult] = []

    def bench_find_bottlenecks_small(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark bottleneck analysis on 100 step pairs."""
        chain = generate_step_pairs(100)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="find_bottlenecks_100_pairs",
            func=lambda: engine.find_bottlenecks(),
            iterations=iterations,
        )
        result.metadata["step_pairs"] = 100
        self.results.append(result)
        return result

    def bench_find_bottlenecks_medium(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark bottleneck analysis on 500 step pairs."""
        chain = generate_step_pairs(500)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="find_bottlenecks_500_pairs",
            func=lambda: engine.find_bottlenecks(),
            iterations=iterations,
        )
        result.metadata["step_pairs"] = 500
        self.results.append(result)
        return result

    def bench_find_bottlenecks_large(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark bottleneck analysis on 1000 step pairs."""
        chain = generate_step_pairs(1000)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="find_bottlenecks_1000_pairs",
            func=lambda: engine.find_bottlenecks(),
            iterations=iterations,
        )
        result.metadata["step_pairs"] = 1000
        self.results.append(result)
        return result

    def bench_confidence_timeline_small(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark confidence timeline for 50 facts with 3 modifications each."""
        chain = generate_chain_with_confidence_changes(50, modifications_per_fact=3)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="confidence_timeline_50x3",
            func=lambda: engine.get_confidence_timeline(),
            iterations=iterations,
        )
        result.metadata["num_facts"] = 50
        result.metadata["modifications_per_fact"] = 3
        self.results.append(result)
        return result

    def bench_confidence_timeline_medium(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark confidence timeline for 200 facts with 5 modifications each."""
        chain = generate_chain_with_confidence_changes(200, modifications_per_fact=5)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="confidence_timeline_200x5",
            func=lambda: engine.get_confidence_timeline(),
            iterations=iterations,
        )
        result.metadata["num_facts"] = 200
        result.metadata["modifications_per_fact"] = 5
        self.results.append(result)
        return result

    def bench_confidence_timeline_large(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark confidence timeline for 500 facts with 10 modifications each."""
        chain = generate_chain_with_confidence_changes(500, modifications_per_fact=10)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="confidence_timeline_500x10",
            func=lambda: engine.get_confidence_timeline(),
            iterations=iterations,
        )
        result.metadata["num_facts"] = 500
        result.metadata["modifications_per_fact"] = 10
        self.results.append(result)
        return result

    def bench_chain_diff_small(self, iterations: int = 30) -> BenchmarkResult:
        """Benchmark chain diff for 100 events diverging at 50."""
        chain1, chain2 = generate_divergent_chains(100, divergence_point=50)
        engine1 = ReplayEngine(chain1)
        engine2 = ReplayEngine(chain2)

        result = benchmark(
            name="chain_diff_100_events",
            func=lambda: engine1.diff(engine2),
            iterations=iterations,
        )
        result.metadata["event_count"] = 100
        result.metadata["divergence_point"] = 50
        self.results.append(result)
        return result

    def bench_chain_diff_medium(self, iterations: int = 15) -> BenchmarkResult:
        """Benchmark chain diff for 1000 events diverging at 500."""
        chain1, chain2 = generate_divergent_chains(1000, divergence_point=500)
        engine1 = ReplayEngine(chain1)
        engine2 = ReplayEngine(chain2)

        result = benchmark(
            name="chain_diff_1000_events",
            func=lambda: engine1.diff(engine2),
            iterations=iterations,
        )
        result.metadata["event_count"] = 1000
        result.metadata["divergence_point"] = 500
        self.results.append(result)
        return result

    def bench_chain_diff_large(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark chain diff for 5000 events diverging at 2500."""
        chain1, chain2 = generate_divergent_chains(5000, divergence_point=2500)
        engine1 = ReplayEngine(chain1)
        engine2 = ReplayEngine(chain2)

        result = benchmark(
            name="chain_diff_5000_events",
            func=lambda: engine1.diff(engine2),
            iterations=iterations,
        )
        result.metadata["event_count"] = 5000
        result.metadata["divergence_point"] = 2500
        self.results.append(result)
        return result

    def bench_chain_diff_identical(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark chain diff for identical chains (best case)."""
        chain = generate_chain(ChainSize.MEDIUM)
        engine1 = ReplayEngine(chain)
        engine2 = ReplayEngine(chain)

        result = benchmark(
            name="chain_diff_identical_1000",
            func=lambda: engine1.diff(engine2),
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        result.metadata["identical"] = True
        self.results.append(result)
        return result

    def bench_get_trace_small(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark execution trace retrieval for 200 events."""
        chain = generate_step_pairs(100)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="get_trace_200_events",
            func=lambda: engine.get_trace(),
            iterations=iterations,
        )
        result.metadata["event_count"] = 200
        self.results.append(result)
        return result

    def bench_get_trace_medium(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark execution trace retrieval for 1000 events."""
        chain = generate_step_pairs(500)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="get_trace_1000_events",
            func=lambda: engine.get_trace(),
            iterations=iterations,
        )
        result.metadata["event_count"] = 1000
        self.results.append(result)
        return result

    def bench_get_trace_large(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark execution trace retrieval for 5000 events."""
        chain = generate_step_pairs(2500)
        engine = ReplayEngine(chain)

        result = benchmark(
            name="get_trace_5000_events",
            func=lambda: engine.get_trace(),
            iterations=iterations,
        )
        result.metadata["event_count"] = 5000
        self.results.append(result)
        return result

    def bench_combined_analysis(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark running all analysis functions on a chain."""
        chain = generate_chain(ChainSize.MEDIUM)
        engine = ReplayEngine(chain)

        def run_all_analysis():
            engine.replay_all()
            engine.find_bottlenecks()
            engine.get_confidence_timeline()
            engine.get_trace()

        result = benchmark(
            name="combined_analysis_1000",
            func=run_all_analysis,
            iterations=iterations,
        )
        result.metadata["event_count"] = ChainSize.MEDIUM
        self.results.append(result)
        return result

    def run_all(self) -> List[BenchmarkResult]:
        """Run all analysis benchmarks."""
        self.results = []
        self.bench_find_bottlenecks_small()
        self.bench_find_bottlenecks_medium()
        self.bench_find_bottlenecks_large()
        self.bench_confidence_timeline_small()
        self.bench_confidence_timeline_medium()
        self.bench_confidence_timeline_large()
        self.bench_chain_diff_small()
        self.bench_chain_diff_medium()
        self.bench_chain_diff_large()
        self.bench_chain_diff_identical()
        self.bench_get_trace_small()
        self.bench_get_trace_medium()
        self.bench_get_trace_large()
        self.bench_combined_analysis()
        return self.results

    def run_quick(self) -> List[BenchmarkResult]:
        """Run a quick subset of benchmarks."""
        self.results = []
        self.bench_find_bottlenecks_small(iterations=10)
        self.bench_confidence_timeline_small(iterations=10)
        self.bench_chain_diff_small(iterations=10)
        self.bench_get_trace_small(iterations=10)
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


def run_analysis_benchmarks(
    quick: bool = False,
    baseline: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Run analysis benchmarks and return results.

    Args:
        quick: If True, run a quick subset of benchmarks.
        baseline: Optional baseline for comparison.

    Returns:
        List of benchmark results as dictionaries.
    """
    benchmarks = AnalysisBenchmarks(baseline=baseline)

    if quick:
        benchmarks.run_quick()
    else:
        benchmarks.run_all()

    if baseline:
        return benchmarks.get_comparison()
    return [r.to_dict() for r in benchmarks.results]


if __name__ == "__main__":
    print("Running analysis benchmarks...")
    print("=" * 60)

    results = run_analysis_benchmarks()

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Avg time: {result['avg_time_ms']:.3f} ms")
        print(f"  Ops/sec: {result['ops_per_second']:.2f}")
        print(f"  Memory peak: {result['memory_peak_mb']:.3f} MB")

    print("\n" + "=" * 60)
    print("Analysis benchmarks complete.")
