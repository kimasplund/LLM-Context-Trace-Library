"""LCTL Evaluation Module - A/B testing for chain comparisons.

This module provides tools for comparing chain executions, computing
metrics, and determining statistical significance of differences.

Example:
    from lctl.evaluation import ChainComparator, compare_chains

    # Compare two chains
    result = compare_chains("chain_a.lctl.json", "chain_b.lctl.json")
    print(result.summary())
    print(result.is_significant())

    # Compare chain objects
    comparator = ChainComparator(chain_a, chain_b)
    result = comparator.compare()

    # Compare multiple runs for statistical power
    result = compare_chain_sets(
        ["run1_a.json", "run2_a.json"],
        ["run1_b.json", "run2_b.json"],
    )
"""

from .comparator import (
    ChainComparator,
    ComparisonResult,
    compare_chain_sets,
    compare_chains,
)
from .metrics import (
    ChainMetrics,
    MetricComparison,
    compute_all_metrics,
    compute_error_rate,
    compute_fact_confidence_avg,
    compute_latency_diff,
    compute_token_efficiency,
)
from .statistical import (
    StatisticalResult,
    bootstrap_confidence_interval,
    consensus_significant,
    is_significant,
    mann_whitney_u,
    run_all_tests,
    welch_t_test,
)

__all__ = [
    # Main comparison API
    "ChainComparator",
    "ComparisonResult",
    "compare_chains",
    "compare_chain_sets",
    # Metrics
    "ChainMetrics",
    "MetricComparison",
    "compute_all_metrics",
    "compute_latency_diff",
    "compute_token_efficiency",
    "compute_error_rate",
    "compute_fact_confidence_avg",
    # Statistical tests
    "StatisticalResult",
    "welch_t_test",
    "mann_whitney_u",
    "bootstrap_confidence_interval",
    "run_all_tests",
    "is_significant",
    "consensus_significant",
]
