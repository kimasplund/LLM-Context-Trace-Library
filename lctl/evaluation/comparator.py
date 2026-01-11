"""LCTL Chain Comparator - A/B testing for chain executions."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.events import Chain, ReplayEngine
from .metrics import (
    ChainMetrics,
    MetricComparison,
    compute_all_metrics,
)
from .statistical import (
    StatisticalResult,
    is_significant,
    run_all_tests,
)


@dataclass
class ComparisonResult:
    """Result of comparing two chains or chain sets."""

    chain_a_id: str
    chain_b_id: str
    metrics_a: ChainMetrics
    metrics_b: ChainMetrics
    metric_comparisons: Dict[str, MetricComparison]
    statistical_results: Dict[str, Dict[str, StatisticalResult]]
    divergences: List[Dict[str, Any]]
    winner: Optional[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now())

    def summary(self) -> str:
        """Generate a human-readable summary of the comparison.

        Returns:
            Multi-line string summarizing the comparison results.
        """
        lines = [
            "=" * 60,
            "LCTL A/B Test Comparison Report",
            "=" * 60,
            "",
            f"Chain A: {self.chain_a_id}",
            f"Chain B: {self.chain_b_id}",
            f"Generated: {self.timestamp.isoformat()}",
            "",
            "-" * 60,
            "METRICS COMPARISON",
            "-" * 60,
        ]

        for name, comparison in self.metric_comparisons.items():
            improvement_str = ""
            if comparison.improvement is True:
                improvement_str = " [IMPROVED]"
            elif comparison.improvement is False:
                improvement_str = " [REGRESSED]"

            pct_str = (
                f"{comparison.percent_change:+.1f}%"
                if not (comparison.percent_change == float("inf") or
                        comparison.percent_change == float("-inf"))
                else "N/A"
            )

            lines.append(
                f"  {name}:"
            )
            lines.append(
                f"    A: {comparison.value_a:.4f} -> B: {comparison.value_b:.4f}"
            )
            lines.append(
                f"    Change: {comparison.difference:+.4f} ({pct_str}){improvement_str}"
            )

        lines.extend([
            "",
            "-" * 60,
            "STATISTICAL SIGNIFICANCE",
            "-" * 60,
        ])

        for metric_name, tests in self.statistical_results.items():
            lines.append(f"  {metric_name}:")
            for test_name, result in tests.items():
                sig_str = "SIGNIFICANT" if result.significant else "not significant"
                lines.append(
                    f"    {test_name}: p={result.p_value:.4f} ({sig_str})"
                )
                if result.confidence_interval:
                    ci = result.confidence_interval
                    lines.append(
                        f"      95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
                    )

        lines.extend([
            "",
            "-" * 60,
            "DIVERGENCES",
            "-" * 60,
        ])

        if self.divergences:
            lines.append(f"  Found {len(self.divergences)} divergence(s)")
            for d in self.divergences[:5]:
                lines.append(f"    - Seq {d.get('seq', '?')}: {d.get('type', 'unknown')}")
        else:
            lines.append("  No divergences found")

        lines.extend([
            "",
            "-" * 60,
            "CONCLUSION",
            "-" * 60,
        ])

        if self.winner:
            lines.append(f"  Winner: {self.winner}")
        else:
            lines.append("  No clear winner (results inconclusive)")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def is_significant(self, metric: Optional[str] = None) -> bool:
        """Check if the comparison shows statistical significance.

        Args:
            metric: Specific metric to check, or None for any metric.

        Returns:
            True if there is statistical significance.
        """
        if metric:
            if metric in self.statistical_results:
                return is_significant(self.statistical_results[metric])
            return False

        for tests in self.statistical_results.values():
            if is_significant(tests):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison result to dictionary."""
        return {
            "chain_a_id": self.chain_a_id,
            "chain_b_id": self.chain_b_id,
            "metrics_a": self.metrics_a.to_dict(),
            "metrics_b": self.metrics_b.to_dict(),
            "metric_comparisons": {
                k: v.to_dict() for k, v in self.metric_comparisons.items()
            },
            "statistical_results": {
                metric: {test: result.to_dict() for test, result in tests.items()}
                for metric, tests in self.statistical_results.items()
            },
            "divergences": self.divergences,
            "winner": self.winner,
            "timestamp": self.timestamp.isoformat(),
        }


class ChainComparator:
    """Compare two chain executions for A/B testing.

    This class provides comprehensive comparison of chain executions,
    including metric calculations, statistical significance testing,
    and divergence analysis.

    Example:
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()
        print(result.summary())
        print(f"Significant: {result.is_significant()}")
    """

    def __init__(
        self,
        chain_a: Chain,
        chain_b: Chain,
        alpha: float = 0.05,
    ):
        """Initialize the comparator.

        Args:
            chain_a: First chain (control/baseline).
            chain_b: Second chain (treatment/variant).
            alpha: Significance level for statistical tests.

        Raises:
            ValueError: If either chain is None.
        """
        if chain_a is None:
            raise ValueError("chain_a must not be None")
        if chain_b is None:
            raise ValueError("chain_b must not be None")

        self.chain_a = chain_a
        self.chain_b = chain_b
        self.alpha = alpha

        self._metrics_a: Optional[ChainMetrics] = None
        self._metrics_b: Optional[ChainMetrics] = None

    @property
    def metrics_a(self) -> ChainMetrics:
        """Compute or return cached metrics for chain A."""
        if self._metrics_a is None:
            self._metrics_a = ChainMetrics.from_chain(self.chain_a)
        return self._metrics_a

    @property
    def metrics_b(self) -> ChainMetrics:
        """Compute or return cached metrics for chain B."""
        if self._metrics_b is None:
            self._metrics_b = ChainMetrics.from_chain(self.chain_b)
        return self._metrics_b

    def compare(
        self,
        include_divergences: bool = True,
        bootstrap_seed: Optional[int] = None,
    ) -> ComparisonResult:
        """Perform full comparison of the two chains.

        Args:
            include_divergences: Whether to compute event divergences.
            bootstrap_seed: Random seed for bootstrap confidence intervals.

        Returns:
            ComparisonResult with all comparison data.
        """
        metric_comparisons = compute_all_metrics(self.metrics_a, self.metrics_b)

        statistical_results = self._compute_statistical_tests(bootstrap_seed)

        divergences: List[Dict[str, Any]] = []
        if include_divergences:
            engine_a = ReplayEngine(self.chain_a)
            engine_b = ReplayEngine(self.chain_b)
            divergences = engine_a.diff(engine_b)

        winner = self._determine_winner(metric_comparisons, statistical_results)

        return ComparisonResult(
            chain_a_id=self.chain_a.id,
            chain_b_id=self.chain_b.id,
            metrics_a=self.metrics_a,
            metrics_b=self.metrics_b,
            metric_comparisons=metric_comparisons,
            statistical_results=statistical_results,
            divergences=divergences,
            winner=winner,
        )

    def _compute_statistical_tests(
        self,
        bootstrap_seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, StatisticalResult]]:
        """Compute statistical tests for each metric.

        For single chain comparisons, we use step-level data as samples.
        """
        results: Dict[str, Dict[str, StatisticalResult]] = {}

        durations_a = self._extract_step_durations(self.chain_a)
        durations_b = self._extract_step_durations(self.chain_b)

        if len(durations_a) >= 2 and len(durations_b) >= 2:
            results["latency"] = run_all_tests(
                durations_a, durations_b, self.alpha, bootstrap_seed
            )

        confidences_a = list(self.metrics_a.facts.get(fid, {}).get("confidence", 1.0)
                            for fid in self.metrics_a.facts)
        confidences_b = list(self.metrics_b.facts.get(fid, {}).get("confidence", 1.0)
                            for fid in self.metrics_b.facts)

        if len(confidences_a) >= 2 and len(confidences_b) >= 2:
            results["confidence"] = run_all_tests(
                list(confidences_a), list(confidences_b), self.alpha, bootstrap_seed
            )

        return results

    def _extract_step_durations(self, chain: Chain) -> List[float]:
        """Extract step durations from a chain as sample data."""
        engine = ReplayEngine(chain)
        bottlenecks = engine.find_bottlenecks()
        return [float(b["duration_ms"]) for b in bottlenecks]

    def _determine_winner(
        self,
        metric_comparisons: Dict[str, MetricComparison],
        statistical_results: Dict[str, Dict[str, StatisticalResult]],
    ) -> Optional[str]:
        """Determine the winning chain based on metrics and significance.

        A chain wins if it has more improvements and at least one is significant.
        """
        improvements_a = 0
        improvements_b = 0

        for comparison in metric_comparisons.values():
            if comparison.improvement is True:
                improvements_b += 1
            elif comparison.improvement is False:
                improvements_a += 1

        has_significance = any(
            is_significant(tests)
            for tests in statistical_results.values()
        )

        if not has_significance:
            return None

        if improvements_b > improvements_a:
            return self.chain_b.id
        elif improvements_a > improvements_b:
            return self.chain_a.id

        return None


def compare_chains(
    chain_a: Union[str, Path, Chain],
    chain_b: Union[str, Path, Chain],
    alpha: float = 0.05,
) -> ComparisonResult:
    """Compare two chains by path or Chain object.

    This is the main entry point for comparing chains.

    Args:
        chain_a: Path to chain file or Chain object.
        chain_b: Path to chain file or Chain object.
        alpha: Significance level for statistical tests.

    Returns:
        ComparisonResult with comparison data.

    Raises:
        FileNotFoundError: If a chain file does not exist.
        ValueError: If a chain file is invalid.

    Example:
        result = compare_chains("chain_a.lctl.json", "chain_b.lctl.json")
        print(result.summary())
        print(result.is_significant())
    """
    if isinstance(chain_a, (str, Path)):
        chain_a = Chain.load(Path(chain_a))

    if isinstance(chain_b, (str, Path)):
        chain_b = Chain.load(Path(chain_b))

    comparator = ChainComparator(chain_a, chain_b, alpha=alpha)
    return comparator.compare()


def compare_chain_sets(
    chains_a: List[Union[str, Path, Chain]],
    chains_b: List[Union[str, Path, Chain]],
    alpha: float = 0.05,
    bootstrap_seed: Optional[int] = None,
) -> ComparisonResult:
    """Compare two sets of chains for aggregate A/B testing.

    This allows comparing multiple runs of each variant for more
    robust statistical analysis.

    Args:
        chains_a: List of chain files or Chain objects for variant A.
        chains_b: List of chain files or Chain objects for variant B.
        alpha: Significance level for statistical tests.
        bootstrap_seed: Random seed for bootstrap confidence intervals.

    Returns:
        ComparisonResult with aggregate comparison data.

    Raises:
        ValueError: If either chain set is empty.
    """
    if not chains_a:
        raise ValueError("chains_a must not be empty")
    if not chains_b:
        raise ValueError("chains_b must not be empty")

    loaded_a: List[Chain] = []
    for c in chains_a:
        if isinstance(c, (str, Path)):
            loaded_a.append(Chain.load(Path(c)))
        else:
            loaded_a.append(c)

    loaded_b: List[Chain] = []
    for c in chains_b:
        if isinstance(c, (str, Path)):
            loaded_b.append(Chain.load(Path(c)))
        else:
            loaded_b.append(c)

    metrics_list_a = [ChainMetrics.from_chain(c) for c in loaded_a]
    metrics_list_b = [ChainMetrics.from_chain(c) for c in loaded_b]

    latencies_a = [float(m.total_duration_ms) for m in metrics_list_a]
    latencies_b = [float(m.total_duration_ms) for m in metrics_list_b]

    confidences_a = [m.avg_fact_confidence for m in metrics_list_a]
    confidences_b = [m.avg_fact_confidence for m in metrics_list_b]

    error_rates_a = [m.error_rate for m in metrics_list_a]
    error_rates_b = [m.error_rate for m in metrics_list_b]

    token_eff_a = [m.token_efficiency for m in metrics_list_a]
    token_eff_b = [m.token_efficiency for m in metrics_list_b]

    aggregate_a = ChainMetrics(
        chain_id=f"aggregate_a_{len(loaded_a)}",
        total_events=sum(m.total_events for m in metrics_list_a),
        total_duration_ms=int(sum(latencies_a)),
        total_tokens_in=sum(m.total_tokens_in for m in metrics_list_a),
        total_tokens_out=sum(m.total_tokens_out for m in metrics_list_a),
        total_tokens=sum(m.total_tokens for m in metrics_list_a),
        error_count=sum(m.error_count for m in metrics_list_a),
        fact_count=sum(m.fact_count for m in metrics_list_a),
        step_count=sum(m.step_count for m in metrics_list_a),
        agent_count=len(set(a for m in metrics_list_a for a in m.agents)),
        agents=list(set(a for m in metrics_list_a for a in m.agents)),
        avg_step_duration_ms=sum(m.avg_step_duration_ms for m in metrics_list_a) / len(metrics_list_a),
        avg_fact_confidence=sum(confidences_a) / len(confidences_a) if confidences_a else 0.0,
        token_efficiency=sum(token_eff_a) / len(token_eff_a) if token_eff_a else 0.0,
        error_rate=sum(error_rates_a) / len(error_rates_a) if error_rates_a else 0.0,
    )

    aggregate_b = ChainMetrics(
        chain_id=f"aggregate_b_{len(loaded_b)}",
        total_events=sum(m.total_events for m in metrics_list_b),
        total_duration_ms=int(sum(latencies_b)),
        total_tokens_in=sum(m.total_tokens_in for m in metrics_list_b),
        total_tokens_out=sum(m.total_tokens_out for m in metrics_list_b),
        total_tokens=sum(m.total_tokens for m in metrics_list_b),
        error_count=sum(m.error_count for m in metrics_list_b),
        fact_count=sum(m.fact_count for m in metrics_list_b),
        step_count=sum(m.step_count for m in metrics_list_b),
        agent_count=len(set(a for m in metrics_list_b for a in m.agents)),
        agents=list(set(a for m in metrics_list_b for a in m.agents)),
        avg_step_duration_ms=sum(m.avg_step_duration_ms for m in metrics_list_b) / len(metrics_list_b),
        avg_fact_confidence=sum(confidences_b) / len(confidences_b) if confidences_b else 0.0,
        token_efficiency=sum(token_eff_b) / len(token_eff_b) if token_eff_b else 0.0,
        error_rate=sum(error_rates_b) / len(error_rates_b) if error_rates_b else 0.0,
    )

    metric_comparisons = compute_all_metrics(aggregate_a, aggregate_b)

    statistical_results: Dict[str, Dict[str, StatisticalResult]] = {}

    if len(latencies_a) >= 2 and len(latencies_b) >= 2:
        statistical_results["latency"] = run_all_tests(
            latencies_a, latencies_b, alpha, bootstrap_seed
        )

    if len(confidences_a) >= 2 and len(confidences_b) >= 2:
        statistical_results["confidence"] = run_all_tests(
            confidences_a, confidences_b, alpha, bootstrap_seed
        )

    if len(error_rates_a) >= 2 and len(error_rates_b) >= 2:
        statistical_results["error_rate"] = run_all_tests(
            error_rates_a, error_rates_b, alpha, bootstrap_seed
        )

    if len(token_eff_a) >= 2 and len(token_eff_b) >= 2:
        statistical_results["token_efficiency"] = run_all_tests(
            token_eff_a, token_eff_b, alpha, bootstrap_seed
        )

    improvements_a = 0
    improvements_b = 0
    for comparison in metric_comparisons.values():
        if comparison.improvement is True:
            improvements_b += 1
        elif comparison.improvement is False:
            improvements_a += 1

    has_significance = any(
        any(r.significant for r in tests.values())
        for tests in statistical_results.values()
    )

    winner = None
    if has_significance:
        if improvements_b > improvements_a:
            winner = aggregate_b.chain_id
        elif improvements_a > improvements_b:
            winner = aggregate_a.chain_id

    return ComparisonResult(
        chain_a_id=aggregate_a.chain_id,
        chain_b_id=aggregate_b.chain_id,
        metrics_a=aggregate_a,
        metrics_b=aggregate_b,
        metric_comparisons=metric_comparisons,
        statistical_results=statistical_results,
        divergences=[],
        winner=winner,
    )
