"""Tests for LCTL evaluation module (A/B testing)."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from lctl.core.events import Chain, Event, EventType
from lctl.evaluation import (
    ChainComparator,
    ChainMetrics,
    ComparisonResult,
    MetricComparison,
    StatisticalResult,
    bootstrap_confidence_interval,
    compare_chain_sets,
    compare_chains,
    compute_all_metrics,
    compute_error_rate,
    compute_fact_confidence_avg,
    compute_latency_diff,
    compute_token_efficiency,
    consensus_significant,
    is_significant,
    mann_whitney_u,
    run_all_tests,
    welch_t_test,
)


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed base timestamp for reproducible tests."""
    return datetime(2025, 1, 15, 10, 0, 0)


@pytest.fixture
def chain_a(base_timestamp: datetime) -> Chain:
    """Create first chain for comparison (baseline)."""
    chain = Chain(id="chain-a")
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="planner",
            data={"intent": "plan", "input_summary": "task"}
        ),
        Event(
            seq=2,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="planner",
            data={"id": "F1", "text": "Fact 1", "confidence": 0.8}
        ),
        Event(
            seq=3,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=500),
            agent="planner",
            data={
                "outcome": "success",
                "duration_ms": 500,
                "tokens": {"input": 100, "output": 50}
            }
        ),
        Event(
            seq=4,
            type=EventType.STEP_START,
            timestamp=base_timestamp + timedelta(milliseconds=600),
            agent="executor",
            data={"intent": "execute", "input_summary": "plan"}
        ),
        Event(
            seq=5,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=700),
            agent="executor",
            data={"id": "F2", "text": "Fact 2", "confidence": 0.9}
        ),
        Event(
            seq=6,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=1100),
            agent="executor",
            data={
                "outcome": "success",
                "duration_ms": 500,
                "tokens": {"input": 150, "output": 75}
            }
        ),
    ]
    for event in events:
        chain.add_event(event)
    return chain


@pytest.fixture
def chain_b(base_timestamp: datetime) -> Chain:
    """Create second chain for comparison (variant with improvements)."""
    chain = Chain(id="chain-b")
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="planner",
            data={"intent": "plan", "input_summary": "task"}
        ),
        Event(
            seq=2,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=50),
            agent="planner",
            data={"id": "F1", "text": "Fact 1", "confidence": 0.95}
        ),
        Event(
            seq=3,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=300),
            agent="planner",
            data={
                "outcome": "success",
                "duration_ms": 300,
                "tokens": {"input": 80, "output": 40}
            }
        ),
        Event(
            seq=4,
            type=EventType.STEP_START,
            timestamp=base_timestamp + timedelta(milliseconds=400),
            agent="executor",
            data={"intent": "execute", "input_summary": "plan"}
        ),
        Event(
            seq=5,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=500),
            agent="executor",
            data={"id": "F2", "text": "Fact 2", "confidence": 0.98}
        ),
        Event(
            seq=6,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=700),
            agent="executor",
            data={
                "outcome": "success",
                "duration_ms": 300,
                "tokens": {"input": 120, "output": 60}
            }
        ),
    ]
    for event in events:
        chain.add_event(event)
    return chain


@pytest.fixture
def chain_with_errors(base_timestamp: datetime) -> Chain:
    """Create a chain with error events."""
    chain = Chain(id="chain-errors")
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="agent1",
            data={"intent": "process"}
        ),
        Event(
            seq=2,
            type=EventType.ERROR,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent1",
            data={
                "category": "validation",
                "type": "ValueError",
                "message": "Invalid input",
                "recoverable": True
            }
        ),
        Event(
            seq=3,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=200),
            agent="agent1",
            data={"outcome": "error", "duration_ms": 200, "tokens": {"input": 50, "output": 10}}
        ),
    ]
    for event in events:
        chain.add_event(event)
    return chain


class TestChainMetrics:
    """Tests for ChainMetrics."""

    def test_from_chain_basic(self, chain_a: Chain):
        """Test computing metrics from a chain."""
        metrics = ChainMetrics.from_chain(chain_a)

        assert metrics.chain_id == "chain-a"
        assert metrics.total_events == 6
        assert metrics.total_duration_ms == 1000
        assert metrics.total_tokens_in == 250
        assert metrics.total_tokens_out == 125
        assert metrics.total_tokens == 375
        assert metrics.error_count == 0
        assert metrics.fact_count == 2
        assert metrics.step_count == 2
        assert metrics.agent_count == 2
        assert set(metrics.agents) == {"planner", "executor"}

    def test_from_chain_with_errors(self, chain_with_errors: Chain):
        """Test computing metrics from chain with errors."""
        metrics = ChainMetrics.from_chain(chain_with_errors)

        assert metrics.error_count == 1
        assert metrics.error_rate > 0

    def test_avg_step_duration(self, chain_a: Chain):
        """Test average step duration calculation."""
        metrics = ChainMetrics.from_chain(chain_a)
        assert metrics.avg_step_duration_ms == 500.0

    def test_avg_fact_confidence(self, chain_a: Chain):
        """Test average fact confidence calculation."""
        metrics = ChainMetrics.from_chain(chain_a)
        assert abs(metrics.avg_fact_confidence - 0.85) < 1e-10

    def test_token_efficiency(self, chain_a: Chain):
        """Test token efficiency calculation."""
        metrics = ChainMetrics.from_chain(chain_a)
        assert metrics.token_efficiency == 375.0

    def test_to_dict(self, chain_a: Chain):
        """Test metrics serialization."""
        metrics = ChainMetrics.from_chain(chain_a)
        data = metrics.to_dict()

        assert data["chain_id"] == "chain-a"
        assert data["total_events"] == 6
        assert data["total_duration_ms"] == 1000
        assert "agents" in data


class TestMetricComparison:
    """Tests for MetricComparison."""

    def test_create_improvement(self):
        """Test creating comparison showing improvement."""
        comparison = MetricComparison.create(
            name="latency",
            value_a=1000.0,
            value_b=800.0,
            higher_is_better=False,
        )

        assert comparison.metric_name == "latency"
        assert comparison.value_a == 1000.0
        assert comparison.value_b == 800.0
        assert comparison.difference == -200.0
        assert comparison.percent_change == -20.0
        assert comparison.improvement is True

    def test_create_regression(self):
        """Test creating comparison showing regression."""
        comparison = MetricComparison.create(
            name="confidence",
            value_a=0.9,
            value_b=0.8,
            higher_is_better=True,
        )

        assert comparison.improvement is False

    def test_create_no_change(self):
        """Test creating comparison with no change."""
        comparison = MetricComparison.create(
            name="metric",
            value_a=100.0,
            value_b=100.0,
            higher_is_better=True,
        )

        assert comparison.difference == 0.0
        assert comparison.improvement is None

    def test_create_from_zero(self):
        """Test creating comparison from zero baseline."""
        comparison = MetricComparison.create(
            name="metric",
            value_a=0.0,
            value_b=100.0,
            higher_is_better=True,
        )

        assert comparison.percent_change == float("inf")
        assert comparison.improvement is True

    def test_to_dict(self):
        """Test comparison serialization."""
        comparison = MetricComparison.create("test", 100.0, 110.0, True)
        data = comparison.to_dict()

        assert data["metric_name"] == "test"
        assert data["value_a"] == 100.0
        assert data["value_b"] == 110.0


class TestMetricComputations:
    """Tests for metric computation functions."""

    def test_compute_latency_diff(self, chain_a: Chain, chain_b: Chain):
        """Test latency difference computation."""
        metrics_a = ChainMetrics.from_chain(chain_a)
        metrics_b = ChainMetrics.from_chain(chain_b)

        comparison = compute_latency_diff(metrics_a, metrics_b)

        assert comparison.metric_name == "latency_ms"
        assert comparison.value_a == 1000.0
        assert comparison.value_b == 600.0
        assert comparison.improvement is True

    def test_compute_token_efficiency(self, chain_a: Chain, chain_b: Chain):
        """Test token efficiency comparison."""
        metrics_a = ChainMetrics.from_chain(chain_a)
        metrics_b = ChainMetrics.from_chain(chain_b)

        comparison = compute_token_efficiency(metrics_a, metrics_b)

        assert comparison.metric_name == "token_efficiency"

    def test_compute_error_rate(self, chain_a: Chain, chain_with_errors: Chain):
        """Test error rate comparison."""
        metrics_a = ChainMetrics.from_chain(chain_a)
        metrics_b = ChainMetrics.from_chain(chain_with_errors)

        comparison = compute_error_rate(metrics_a, metrics_b)

        assert comparison.metric_name == "error_rate"
        assert comparison.value_a == 0.0
        assert comparison.value_b > 0.0
        assert comparison.improvement is False

    def test_compute_fact_confidence_avg(self, chain_a: Chain, chain_b: Chain):
        """Test fact confidence average comparison."""
        metrics_a = ChainMetrics.from_chain(chain_a)
        metrics_b = ChainMetrics.from_chain(chain_b)

        comparison = compute_fact_confidence_avg(metrics_a, metrics_b)

        assert comparison.metric_name == "fact_confidence_avg"
        assert comparison.value_b > comparison.value_a
        assert comparison.improvement is True

    def test_compute_all_metrics(self, chain_a: Chain, chain_b: Chain):
        """Test computing all metrics at once."""
        metrics_a = ChainMetrics.from_chain(chain_a)
        metrics_b = ChainMetrics.from_chain(chain_b)

        comparisons = compute_all_metrics(metrics_a, metrics_b)

        assert "latency_diff" in comparisons
        assert "token_efficiency" in comparisons
        assert "error_rate" in comparisons
        assert "fact_confidence_avg" in comparisons


class TestStatisticalFunctions:
    """Tests for statistical test functions."""

    def test_welch_t_test_significant(self):
        """Test Welch's t-test with significant difference."""
        sample_a = [10.0, 11.0, 9.0, 10.5, 10.2]
        sample_b = [15.0, 16.0, 14.0, 15.5, 15.2]

        result = welch_t_test(sample_a, sample_b, alpha=0.05)

        assert result.test_name == "welch_t_test"
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.sample_size_a == 5
        assert result.sample_size_b == 5

    def test_welch_t_test_not_significant(self):
        """Test Welch's t-test with no significant difference."""
        sample_a = [10.0, 11.0, 9.0, 10.5, 10.2]
        sample_b = [10.1, 10.9, 9.1, 10.4, 10.3]

        result = welch_t_test(sample_a, sample_b, alpha=0.05)

        assert result.significant is False

    def test_welch_t_test_effect_size(self):
        """Test that effect size is computed."""
        sample_a = [10.0, 11.0, 9.0]
        sample_b = [15.0, 16.0, 14.0]

        result = welch_t_test(sample_a, sample_b)

        assert result.effect_size is not None
        assert abs(result.effect_size) > 1.0

    def test_welch_t_test_insufficient_data(self):
        """Test Welch's t-test with insufficient data."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            welch_t_test([1.0], [2.0, 3.0])

    def test_mann_whitney_u_significant(self):
        """Test Mann-Whitney U test with significant difference."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [6.0, 7.0, 8.0, 9.0, 10.0]

        result = mann_whitney_u(sample_a, sample_b, alpha=0.05)

        assert result.test_name == "mann_whitney_u"
        assert result.p_value < 0.05
        assert result.significant is True

    def test_mann_whitney_u_not_significant(self):
        """Test Mann-Whitney U test with overlapping distributions."""
        sample_a = [1.0, 3.0, 5.0, 7.0, 9.0]
        sample_b = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = mann_whitney_u(sample_a, sample_b, alpha=0.05)

        assert result.significant is False

    def test_mann_whitney_u_empty_sample(self):
        """Test Mann-Whitney U with empty sample."""
        with pytest.raises(ValueError, match="must not be empty"):
            mann_whitney_u([], [1.0, 2.0])

    def test_bootstrap_ci_significant(self):
        """Test bootstrap confidence interval with significant difference."""
        sample_a = [10.0, 11.0, 9.0, 10.5, 10.2, 9.8, 10.1, 10.3]
        sample_b = [15.0, 16.0, 14.0, 15.5, 15.2, 14.8, 15.1, 15.3]

        result = bootstrap_confidence_interval(
            sample_a, sample_b, n_bootstrap=1000, seed=42
        )

        assert result.test_name == "bootstrap_ci"
        assert result.confidence_interval is not None
        ci_lower, ci_upper = result.confidence_interval
        assert not (ci_lower <= 0 <= ci_upper)
        assert result.significant is True

    def test_bootstrap_ci_not_significant(self):
        """Test bootstrap CI with overlapping distributions."""
        sample_a = [10.0, 11.0, 9.0, 10.5, 12.0, 8.0]
        sample_b = [10.1, 10.9, 9.1, 10.4, 11.5, 8.5]

        result = bootstrap_confidence_interval(
            sample_a, sample_b, n_bootstrap=1000, seed=42
        )

        assert result.confidence_interval is not None
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower <= 0 <= ci_upper or result.significant is False

    def test_bootstrap_ci_reproducibility(self):
        """Test bootstrap CI reproducibility with seed."""
        sample_a = [10.0, 11.0, 9.0]
        sample_b = [12.0, 13.0, 11.0]

        result1 = bootstrap_confidence_interval(sample_a, sample_b, seed=123)
        result2 = bootstrap_confidence_interval(sample_a, sample_b, seed=123)

        assert result1.confidence_interval == result2.confidence_interval

    def test_bootstrap_ci_empty_sample(self):
        """Test bootstrap CI with empty sample."""
        with pytest.raises(ValueError, match="must not be empty"):
            bootstrap_confidence_interval([], [1.0, 2.0])

    def test_run_all_tests(self):
        """Test running all statistical tests."""
        sample_a = [10.0, 11.0, 9.0, 10.5, 10.2]
        sample_b = [15.0, 16.0, 14.0, 15.5, 15.2]

        results = run_all_tests(sample_a, sample_b, alpha=0.05)

        assert "t_test" in results
        assert "mann_whitney_u" in results
        assert "bootstrap_ci" in results

    def test_is_significant(self):
        """Test is_significant helper."""
        results = {
            "test1": StatisticalResult("t1", 2.0, 0.03, True, 0.05),
            "test2": StatisticalResult("t2", 1.0, 0.10, False, 0.05),
        }

        assert is_significant(results) is True

    def test_is_significant_none(self):
        """Test is_significant with no significant results."""
        results = {
            "test1": StatisticalResult("t1", 1.0, 0.10, False, 0.05),
            "test2": StatisticalResult("t2", 0.5, 0.20, False, 0.05),
        }

        assert is_significant(results) is False

    def test_consensus_significant(self):
        """Test consensus_significant helper."""
        results = {
            "test1": StatisticalResult("t1", 2.0, 0.03, True, 0.05),
            "test2": StatisticalResult("t2", 2.5, 0.02, True, 0.05),
            "test3": StatisticalResult("t3", 1.0, 0.10, False, 0.05),
        }

        assert consensus_significant(results, threshold=0.5) is True
        assert consensus_significant(results, threshold=0.8) is False


class TestChainComparator:
    """Tests for ChainComparator class."""

    def test_comparator_creation(self, chain_a: Chain, chain_b: Chain):
        """Test creating a comparator."""
        comparator = ChainComparator(chain_a, chain_b)

        assert comparator.chain_a == chain_a
        assert comparator.chain_b == chain_b
        assert comparator.alpha == 0.05

    def test_comparator_creation_with_alpha(self, chain_a: Chain, chain_b: Chain):
        """Test creating comparator with custom alpha."""
        comparator = ChainComparator(chain_a, chain_b, alpha=0.01)

        assert comparator.alpha == 0.01

    def test_comparator_none_chain(self, chain_a: Chain):
        """Test comparator rejects None chains."""
        with pytest.raises(ValueError, match="must not be None"):
            ChainComparator(None, chain_a)

        with pytest.raises(ValueError, match="must not be None"):
            ChainComparator(chain_a, None)

    def test_metrics_caching(self, chain_a: Chain, chain_b: Chain):
        """Test that metrics are cached."""
        comparator = ChainComparator(chain_a, chain_b)

        metrics1 = comparator.metrics_a
        metrics2 = comparator.metrics_a

        assert metrics1 is metrics2

    def test_compare_basic(self, chain_a: Chain, chain_b: Chain):
        """Test basic comparison."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()

        assert isinstance(result, ComparisonResult)
        assert result.chain_a_id == "chain-a"
        assert result.chain_b_id == "chain-b"
        assert result.metrics_a is not None
        assert result.metrics_b is not None
        assert len(result.metric_comparisons) > 0

    def test_compare_without_divergences(self, chain_a: Chain, chain_b: Chain):
        """Test comparison without divergence analysis."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare(include_divergences=False)

        assert result.divergences == []

    def test_compare_with_divergences(self, chain_a: Chain, chain_b: Chain):
        """Test comparison with divergence analysis."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare(include_divergences=True)

        assert len(result.divergences) > 0


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_summary(self, chain_a: Chain, chain_b: Chain):
        """Test summary generation."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()

        summary = result.summary()

        assert "LCTL A/B Test Comparison Report" in summary
        assert "Chain A: chain-a" in summary
        assert "Chain B: chain-b" in summary
        assert "METRICS COMPARISON" in summary
        assert "STATISTICAL SIGNIFICANCE" in summary
        assert "CONCLUSION" in summary

    def test_is_significant_method(self, chain_a: Chain, chain_b: Chain):
        """Test is_significant method."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()

        sig = result.is_significant()
        assert isinstance(sig, bool)

    def test_is_significant_specific_metric(self, chain_a: Chain, chain_b: Chain):
        """Test is_significant for specific metric."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()

        sig_latency = result.is_significant("latency")
        assert isinstance(sig_latency, bool)

    def test_is_significant_nonexistent_metric(self, chain_a: Chain, chain_b: Chain):
        """Test is_significant for nonexistent metric."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()

        assert result.is_significant("nonexistent") is False

    def test_to_dict(self, chain_a: Chain, chain_b: Chain):
        """Test serialization to dictionary."""
        comparator = ChainComparator(chain_a, chain_b)
        result = comparator.compare()

        data = result.to_dict()

        assert data["chain_a_id"] == "chain-a"
        assert data["chain_b_id"] == "chain-b"
        assert "metrics_a" in data
        assert "metrics_b" in data
        assert "metric_comparisons" in data
        assert "statistical_results" in data
        assert "divergences" in data
        assert "winner" in data
        assert "timestamp" in data


class TestCompareChains:
    """Tests for compare_chains function."""

    def test_compare_chain_objects(self, chain_a: Chain, chain_b: Chain):
        """Test comparing Chain objects directly."""
        result = compare_chains(chain_a, chain_b)

        assert isinstance(result, ComparisonResult)
        assert result.chain_a_id == "chain-a"
        assert result.chain_b_id == "chain-b"

    def test_compare_chain_files(self, chain_a: Chain, chain_b: Chain, tmp_path: Path):
        """Test comparing chains from files."""
        path_a = tmp_path / "chain_a.lctl.json"
        path_b = tmp_path / "chain_b.lctl.json"

        chain_a.save(path_a)
        chain_b.save(path_b)

        result = compare_chains(str(path_a), str(path_b))

        assert isinstance(result, ComparisonResult)

    def test_compare_chain_path_objects(
        self, chain_a: Chain, chain_b: Chain, tmp_path: Path
    ):
        """Test comparing chains using Path objects."""
        path_a = tmp_path / "chain_a.lctl.json"
        path_b = tmp_path / "chain_b.lctl.json"

        chain_a.save(path_a)
        chain_b.save(path_b)

        result = compare_chains(path_a, path_b)

        assert isinstance(result, ComparisonResult)

    def test_compare_chain_nonexistent_file(self, chain_a: Chain, tmp_path: Path):
        """Test comparing with nonexistent file."""
        path_a = tmp_path / "chain_a.lctl.json"
        chain_a.save(path_a)

        with pytest.raises(FileNotFoundError):
            compare_chains(str(path_a), str(tmp_path / "nonexistent.json"))


class TestCompareChainSets:
    """Tests for compare_chain_sets function."""

    def test_compare_chain_sets_basic(self, chain_a: Chain, chain_b: Chain):
        """Test comparing sets of chains."""
        result = compare_chain_sets(
            [chain_a, chain_a],
            [chain_b, chain_b],
        )

        assert isinstance(result, ComparisonResult)
        assert "aggregate_a" in result.chain_a_id
        assert "aggregate_b" in result.chain_b_id

    def test_compare_chain_sets_files(
        self, chain_a: Chain, chain_b: Chain, tmp_path: Path
    ):
        """Test comparing chain sets from files."""
        paths_a = []
        paths_b = []

        for i in range(3):
            path_a = tmp_path / f"chain_a_{i}.lctl.json"
            path_b = tmp_path / f"chain_b_{i}.lctl.json"
            chain_a.save(path_a)
            chain_b.save(path_b)
            paths_a.append(str(path_a))
            paths_b.append(str(path_b))

        result = compare_chain_sets(paths_a, paths_b)

        assert isinstance(result, ComparisonResult)

    def test_compare_chain_sets_empty(self, chain_a: Chain):
        """Test comparing empty chain sets."""
        with pytest.raises(ValueError, match="must not be empty"):
            compare_chain_sets([], [chain_a])

        with pytest.raises(ValueError, match="must not be empty"):
            compare_chain_sets([chain_a], [])

    def test_compare_chain_sets_statistical_tests(
        self, chain_a: Chain, chain_b: Chain
    ):
        """Test that chain sets run proper statistical tests."""
        result = compare_chain_sets(
            [chain_a, chain_a, chain_a],
            [chain_b, chain_b, chain_b],
            bootstrap_seed=42,
        )

        assert len(result.statistical_results) > 0


class TestStatisticalResult:
    """Tests for StatisticalResult dataclass."""

    def test_to_dict_basic(self):
        """Test basic serialization."""
        result = StatisticalResult(
            test_name="t_test",
            statistic=2.5,
            p_value=0.03,
            significant=True,
            alpha=0.05,
            sample_size_a=10,
            sample_size_b=10,
        )

        data = result.to_dict()

        assert data["test_name"] == "t_test"
        assert data["statistic"] == 2.5
        assert data["p_value"] == 0.03
        assert data["significant"] is True
        assert data["alpha"] == 0.05

    def test_to_dict_with_effect_size(self):
        """Test serialization with effect size."""
        result = StatisticalResult(
            test_name="t_test",
            statistic=2.5,
            p_value=0.03,
            significant=True,
            alpha=0.05,
            effect_size=0.8,
        )

        data = result.to_dict()

        assert data["effect_size"] == 0.8

    def test_to_dict_with_confidence_interval(self):
        """Test serialization with confidence interval."""
        result = StatisticalResult(
            test_name="bootstrap",
            statistic=-5.0,
            p_value=0.001,
            significant=True,
            alpha=0.05,
            confidence_interval=(-7.0, -3.0),
        )

        data = result.to_dict()

        assert data["confidence_interval"]["lower"] == -7.0
        assert data["confidence_interval"]["upper"] == -3.0


class TestIntegration:
    """Integration tests for the evaluation module."""

    def test_full_workflow(self, chain_a: Chain, chain_b: Chain, tmp_path: Path):
        """Test complete A/B testing workflow."""
        path_a = tmp_path / "chain_a.lctl.json"
        path_b = tmp_path / "chain_b.lctl.json"

        chain_a.save(path_a)
        chain_b.save(path_b)

        result = compare_chains(str(path_a), str(path_b))

        assert result.metrics_a.total_duration_ms == 1000
        assert result.metrics_b.total_duration_ms == 600

        assert result.metric_comparisons["latency_diff"].improvement is True

        summary = result.summary()
        assert len(summary) > 0

        data = result.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_usage_example(self, chain_a: Chain, chain_b: Chain, tmp_path: Path):
        """Test the documented usage example works."""
        path_a = tmp_path / "chain_a.lctl.json"
        path_b = tmp_path / "chain_b.lctl.json"
        chain_a.save(path_a)
        chain_b.save(path_b)

        result = compare_chains(str(path_a), str(path_b))
        print(result.summary())
        print(result.is_significant())

        assert isinstance(result.summary(), str)
        assert isinstance(result.is_significant(), bool)
