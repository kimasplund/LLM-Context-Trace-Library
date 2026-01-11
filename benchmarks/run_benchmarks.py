#!/usr/bin/env python
"""CLI to run all LCTL benchmarks.

Usage:
    python -m benchmarks.run_benchmarks
    python -m benchmarks.run_benchmarks --suite replay
    python -m benchmarks.run_benchmarks --suite events --quick
    python -m benchmarks.run_benchmarks --output report.json
    python -m benchmarks.run_benchmarks --baseline baseline.json
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bench_analysis import AnalysisBenchmarks, run_analysis_benchmarks
from .bench_events import EventBenchmarks, run_event_benchmarks
from .bench_replay import ReplayBenchmarks, run_replay_benchmarks


BENCHMARK_SUITES = {
    "replay": ("Replay Performance", run_replay_benchmarks),
    "events": ("Event Processing", run_event_benchmarks),
    "analysis": ("Analysis Functions", run_analysis_benchmarks),
}


def get_system_info() -> Dict[str, Any]:
    """Gather system information for the report."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def load_baseline(path: Path) -> Optional[Dict[str, float]]:
    """Load baseline results from a JSON file.

    Args:
        path: Path to the baseline JSON file.

    Returns:
        Dictionary mapping benchmark names to ops/sec, or None if not found.
    """
    if not path.exists():
        print(f"Warning: Baseline file not found: {path}")
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        baseline = {}
        for suite in data.get("suites", {}).values():
            for result in suite.get("results", []):
                baseline[result["name"]] = result.get("ops_per_second", 0)
        return baseline
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse baseline file: {e}")
        return None


def generate_report(
    results: Dict[str, List[Dict[str, Any]]],
    baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive benchmark report.

    Args:
        results: Dictionary mapping suite names to result lists.
        baseline: Optional baseline for comparison.

    Returns:
        Complete benchmark report as a dictionary.
    """
    report = {
        "lctl_version": "4.0.0",
        "benchmark_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": get_system_info(),
        "suites": {},
        "summary": {
            "total_benchmarks": 0,
            "total_time_ms": 0,
            "avg_ops_per_second": 0,
            "memory_peak_mb": 0,
        },
    }

    all_ops = []
    max_memory = 0

    for suite_name, suite_results in results.items():
        suite_title = BENCHMARK_SUITES.get(suite_name, (suite_name, None))[0]

        suite_total_time = sum(r["total_time_ms"] for r in suite_results)
        suite_avg_ops = sum(r["ops_per_second"] for r in suite_results) / len(suite_results) if suite_results else 0

        report["suites"][suite_name] = {
            "title": suite_title,
            "benchmark_count": len(suite_results),
            "total_time_ms": round(suite_total_time, 3),
            "avg_ops_per_second": round(suite_avg_ops, 2),
            "results": suite_results,
        }

        report["summary"]["total_benchmarks"] += len(suite_results)
        report["summary"]["total_time_ms"] += suite_total_time

        for r in suite_results:
            all_ops.append(r["ops_per_second"])
            max_memory = max(max_memory, r.get("memory_peak_mb", 0))

    if all_ops:
        report["summary"]["avg_ops_per_second"] = round(sum(all_ops) / len(all_ops), 2)
    report["summary"]["memory_peak_mb"] = round(max_memory, 3)
    report["summary"]["total_time_ms"] = round(report["summary"]["total_time_ms"], 3)

    if baseline:
        report["comparison"] = generate_comparison_summary(results, baseline)

    return report


def generate_comparison_summary(
    results: Dict[str, List[Dict[str, Any]]],
    baseline: Dict[str, float],
) -> Dict[str, Any]:
    """Generate comparison summary against baseline.

    Args:
        results: Current benchmark results.
        baseline: Baseline ops/sec values.

    Returns:
        Comparison summary dictionary.
    """
    improvements = []
    regressions = []
    unchanged = []

    for suite_results in results.values():
        for result in suite_results:
            name = result["name"]
            if name in baseline:
                baseline_ops = baseline[name]
                current_ops = result["ops_per_second"]
                if baseline_ops > 0:
                    change_pct = ((current_ops - baseline_ops) / baseline_ops) * 100

                    item = {
                        "name": name,
                        "baseline_ops": round(baseline_ops, 2),
                        "current_ops": round(current_ops, 2),
                        "change_percent": round(change_pct, 1),
                    }

                    if change_pct > 5:
                        improvements.append(item)
                    elif change_pct < -5:
                        regressions.append(item)
                    else:
                        unchanged.append(item)

    improvements.sort(key=lambda x: x["change_percent"], reverse=True)
    regressions.sort(key=lambda x: x["change_percent"])

    return {
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "improvement_count": len(improvements),
        "regression_count": len(regressions),
        "unchanged_count": len(unchanged),
    }


def print_results(report: Dict[str, Any]) -> None:
    """Print benchmark results to console.

    Args:
        report: The benchmark report dictionary.
    """
    print("\n" + "=" * 70)
    print("LCTL Benchmark Report")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Python: {report['system']['python_version']}")
    print(f"Platform: {report['system']['platform']}")
    print("=" * 70)

    for suite_name, suite_data in report["suites"].items():
        print(f"\n{suite_data['title']}")
        print("-" * 50)

        for result in suite_data["results"]:
            name = result["name"]
            avg_ms = result["avg_time_ms"]
            ops_sec = result["ops_per_second"]
            mem_mb = result["memory_peak_mb"]

            line = f"  {name:<40} {avg_ms:>8.3f} ms  {ops_sec:>10.2f} ops/s  {mem_mb:>6.3f} MB"

            if "speedup" in result:
                speedup = result["speedup"]
                if speedup >= 1.05:
                    line += f"  [+{(speedup - 1) * 100:.1f}%]"
                elif speedup <= 0.95:
                    line += f"  [{(speedup - 1) * 100:.1f}%]"

            print(line)

        print(f"\n  Suite total: {suite_data['total_time_ms']:.3f} ms")
        print(f"  Suite avg: {suite_data['avg_ops_per_second']:.2f} ops/s")

    print("\n" + "=" * 70)
    print("Summary")
    print("-" * 50)
    print(f"  Total benchmarks: {report['summary']['total_benchmarks']}")
    print(f"  Total time: {report['summary']['total_time_ms']:.3f} ms")
    print(f"  Average ops/s: {report['summary']['avg_ops_per_second']:.2f}")
    print(f"  Peak memory: {report['summary']['memory_peak_mb']:.3f} MB")

    if "comparison" in report:
        comp = report["comparison"]
        print("\n" + "-" * 50)
        print("Comparison to Baseline")
        print(f"  Improvements: {comp['improvement_count']}")
        print(f"  Regressions: {comp['regression_count']}")
        print(f"  Unchanged: {comp['unchanged_count']}")

        if comp["improvements"]:
            print("\n  Top improvements:")
            for item in comp["improvements"][:3]:
                print(f"    {item['name']}: +{item['change_percent']:.1f}%")

        if comp["regressions"]:
            print("\n  Regressions:")
            for item in comp["regressions"][:3]:
                print(f"    {item['name']}: {item['change_percent']:.1f}%")

    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point for the benchmark CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Run LCTL benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.run_benchmarks                    # Run all benchmarks
  python -m benchmarks.run_benchmarks --suite replay     # Run only replay benchmarks
  python -m benchmarks.run_benchmarks --quick            # Quick run with fewer iterations
  python -m benchmarks.run_benchmarks --output report.json
  python -m benchmarks.run_benchmarks --baseline baseline.json
        """,
    )

    parser.add_argument(
        "--suite",
        choices=list(BENCHMARK_SUITES.keys()) + ["all"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with fewer iterations",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for JSON report",
    )

    parser.add_argument(
        "--baseline",
        "-b",
        type=Path,
        help="Baseline file for comparison",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (only write to file)",
    )

    args = parser.parse_args()

    baseline = None
    if args.baseline:
        baseline = load_baseline(args.baseline)

    suites_to_run = (
        list(BENCHMARK_SUITES.keys())
        if args.suite == "all"
        else [args.suite]
    )

    results: Dict[str, List[Dict[str, Any]]] = {}

    if not args.quiet:
        print("LCTL Benchmark Suite v4.0")
        print(f"Running: {', '.join(suites_to_run)}")
        if args.quick:
            print("Mode: Quick")
        print()

    for suite_name in suites_to_run:
        suite_title, run_func = BENCHMARK_SUITES[suite_name]
        if not args.quiet:
            print(f"Running {suite_title}...")

        try:
            suite_results = run_func(quick=args.quick, baseline=baseline)
            results[suite_name] = suite_results
        except Exception as e:
            print(f"Error running {suite_name}: {e}", file=sys.stderr)
            return 1

    report = generate_report(results, baseline)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        if not args.quiet:
            print(f"\nReport saved to: {args.output}")

    if not args.quiet:
        print_results(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
