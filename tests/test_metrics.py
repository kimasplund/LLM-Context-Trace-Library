"""Tests for LCTL metrics module (lctl/metrics/)."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from lctl.core.events import Chain, Event, EventType
from lctl.metrics import (
    AgentCollector,
    AgentMetrics,
    AggregatedMetrics,
    CollectedMetrics,
    ErrorCollector,
    ErrorMetrics,
    FactCollector,
    FactMetrics,
    LatencyCollector,
    LatencyMetrics,
    MetricsAggregator,
    MetricsCollector,
    TokenCollector,
    TokenMetrics,
    export_csv,
    export_json,
    export_prometheus,
    to_dict,
    to_json,
    to_prometheus,
)


class TestLatencyMetrics:
    """Tests for LatencyMetrics and LatencyCollector."""

    def test_latency_metrics_default(self):
        """Test default latency metrics values."""
        metrics = LatencyMetrics()
        assert metrics.step_durations == {}
        assert metrics.tool_call_durations == {}
        assert metrics.total_time_ms == 0
        assert metrics.avg_step_duration_ms == 0.0

    def test_latency_collector_empty_chain(self, empty_chain: Chain):
        """Test latency collection from empty chain."""
        collector = LatencyCollector()
        metrics = collector.collect(empty_chain)

        assert metrics.total_time_ms == 0
        assert metrics.avg_step_duration_ms == 0.0
        assert metrics.step_durations == {}

    def test_latency_collector_with_steps(self, sample_chain: Chain):
        """Test latency collection with step events."""
        collector = LatencyCollector()
        metrics = collector.collect(sample_chain)

        assert metrics.total_time_ms == 1000
        assert metrics.avg_step_duration_ms == 500.0
        assert "planner" in metrics.step_durations
        assert "executor" in metrics.step_durations
        assert metrics.min_step_duration_ms == 500
        assert metrics.max_step_duration_ms == 500

    def test_latency_collector_with_tool_calls(self, sample_chain: Chain):
        """Test latency collection with tool call events."""
        collector = LatencyCollector()
        metrics = collector.collect(sample_chain)

        assert "search" in metrics.tool_call_durations
        assert metrics.tool_call_durations["search"] == [50]
        assert metrics.avg_tool_call_duration_ms == 50.0

    def test_latency_metrics_to_dict(self, sample_chain: Chain):
        """Test latency metrics serialization."""
        collector = LatencyCollector()
        metrics = collector.collect(sample_chain)
        result = metrics.to_dict()

        assert "step_durations" in result
        assert "tool_call_durations" in result
        assert result["total_time_ms"] == 1000


class TestTokenMetrics:
    """Tests for TokenMetrics and TokenCollector."""

    def test_token_metrics_default(self):
        """Test default token metrics values."""
        metrics = TokenMetrics()
        assert metrics.tokens_by_agent == {}
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0
        assert metrics.efficiency_ratio == 0.0

    def test_token_collector_empty_chain(self, empty_chain: Chain):
        """Test token collection from empty chain."""
        collector = TokenCollector()
        metrics = collector.collect(empty_chain)

        assert metrics.total_tokens == 0
        assert metrics.efficiency_ratio == 0.0

    def test_token_collector_with_steps(self, sample_chain: Chain):
        """Test token collection with step events."""
        collector = TokenCollector()
        metrics = collector.collect(sample_chain)

        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 150
        assert metrics.total_tokens == 450
        assert metrics.efficiency_ratio == 0.5

    def test_token_collector_per_agent(self, sample_chain: Chain):
        """Test token collection per agent."""
        collector = TokenCollector()
        metrics = collector.collect(sample_chain)

        assert "planner" in metrics.tokens_by_agent
        assert "executor" in metrics.tokens_by_agent
        assert metrics.tokens_by_agent["planner"]["input"] == 100
        assert metrics.tokens_by_agent["planner"]["output"] == 50
        assert metrics.tokens_by_agent["executor"]["input"] == 200
        assert metrics.tokens_by_agent["executor"]["output"] == 100

    def test_token_metrics_to_dict(self, sample_chain: Chain):
        """Test token metrics serialization."""
        collector = TokenCollector()
        metrics = collector.collect(sample_chain)
        result = metrics.to_dict()

        assert "tokens_by_agent" in result
        assert result["total_tokens"] == 450


class TestErrorMetrics:
    """Tests for ErrorMetrics and ErrorCollector."""

    def test_error_metrics_default(self):
        """Test default error metrics values."""
        metrics = ErrorMetrics()
        assert metrics.error_count == 0
        assert metrics.error_rate == 0.0
        assert metrics.error_types == {}

    def test_error_collector_empty_chain(self, empty_chain: Chain):
        """Test error collection from empty chain."""
        collector = ErrorCollector()
        metrics = collector.collect(empty_chain)

        assert metrics.error_count == 0
        assert metrics.error_rate == 0.0

    def test_error_collector_with_errors(self, sample_chain_with_errors: Chain):
        """Test error collection with error events."""
        collector = ErrorCollector()
        metrics = collector.collect(sample_chain_with_errors)

        assert metrics.error_count == 1
        assert metrics.error_rate == 1 / 3
        assert "ValueError" in metrics.error_types
        assert metrics.recoverable_count == 1
        assert metrics.non_recoverable_count == 0

    def test_error_collector_per_agent(self, sample_chain_with_errors: Chain):
        """Test error collection per agent."""
        collector = ErrorCollector()
        metrics = collector.collect(sample_chain_with_errors)

        assert "agent1" in metrics.errors_by_agent
        assert metrics.errors_by_agent["agent1"] == 1

    def test_error_metrics_to_dict(self, sample_chain_with_errors: Chain):
        """Test error metrics serialization."""
        collector = ErrorCollector()
        metrics = collector.collect(sample_chain_with_errors)
        result = metrics.to_dict()

        assert "error_count" in result
        assert "error_types" in result
        assert result["recoverable_count"] == 1


class TestFactMetrics:
    """Tests for FactMetrics and FactCollector."""

    def test_fact_metrics_default(self):
        """Test default fact metrics values."""
        metrics = FactMetrics()
        assert metrics.fact_count == 0
        assert metrics.avg_confidence == 0.0
        assert metrics.confidence_changes == {}

    def test_fact_collector_empty_chain(self, empty_chain: Chain):
        """Test fact collection from empty chain."""
        collector = FactCollector()
        metrics = collector.collect(empty_chain)

        assert metrics.fact_count == 0
        assert metrics.avg_confidence == 0.0

    def test_fact_collector_with_facts(self, sample_chain: Chain):
        """Test fact collection with fact events."""
        collector = FactCollector()
        metrics = collector.collect(sample_chain)

        assert metrics.fact_count == 1
        assert "F1" in metrics.confidence_changes

    def test_fact_collector_confidence_changes(self, chain_for_confidence_test: Chain):
        """Test fact collection with confidence changes."""
        collector = FactCollector()
        metrics = collector.collect(chain_for_confidence_test)

        assert metrics.fact_count == 2
        assert len(metrics.confidence_changes["F1"]) == 3
        assert metrics.confidence_changes["F1"] == [0.5, 0.7, 0.9]
        assert metrics.min_confidence == 0.3
        assert metrics.max_confidence == 0.9

    def test_fact_collector_per_agent(self, sample_chain: Chain):
        """Test fact collection per agent."""
        collector = FactCollector()
        metrics = collector.collect(sample_chain)

        assert "planner" in metrics.facts_by_agent
        assert metrics.facts_by_agent["planner"] == 1

    def test_fact_metrics_to_dict(self, sample_chain: Chain):
        """Test fact metrics serialization."""
        collector = FactCollector()
        metrics = collector.collect(sample_chain)
        result = metrics.to_dict()

        assert "fact_count" in result
        assert "confidence_changes" in result


class TestAgentMetrics:
    """Tests for AgentMetrics and AgentCollector."""

    def test_agent_metrics_default(self):
        """Test default agent metrics values."""
        metrics = AgentMetrics()
        assert metrics.steps_per_agent == {}
        assert metrics.handoff_count == 0
        assert metrics.agent_count == 0

    def test_agent_collector_empty_chain(self, empty_chain: Chain):
        """Test agent collection from empty chain."""
        collector = AgentCollector()
        metrics = collector.collect(empty_chain)

        assert metrics.agent_count == 0
        assert metrics.handoff_count == 0

    def test_agent_collector_with_steps(self, sample_chain: Chain):
        """Test agent collection with step events."""
        collector = AgentCollector()
        metrics = collector.collect(sample_chain)

        assert metrics.agent_count == 2
        assert "planner" in metrics.agents
        assert "executor" in metrics.agents
        assert metrics.steps_per_agent["planner"] == 1
        assert metrics.steps_per_agent["executor"] == 1

    def test_agent_collector_handoffs(self, sample_chain: Chain):
        """Test agent collection handoff counting."""
        collector = AgentCollector()
        metrics = collector.collect(sample_chain)

        assert metrics.handoff_count == 1
        assert ("planner", "executor") in metrics.handoff_pairs
        assert metrics.handoff_pairs[("planner", "executor")] == 1

    def test_agent_collector_collaboration_graph(self, sample_chain: Chain):
        """Test agent collection collaboration graph."""
        collector = AgentCollector()
        metrics = collector.collect(sample_chain)

        assert "planner" in metrics.collaboration_graph
        assert "executor" in metrics.collaboration_graph["planner"]

    def test_agent_metrics_to_dict(self, sample_chain: Chain):
        """Test agent metrics serialization."""
        collector = AgentCollector()
        metrics = collector.collect(sample_chain)
        result = metrics.to_dict()

        assert "steps_per_agent" in result
        assert "handoff_pairs" in result
        assert "planner->executor" in result["handoff_pairs"]


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_metrics_collector_empty_chain(self, empty_chain: Chain):
        """Test collecting all metrics from empty chain."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(empty_chain)

        assert metrics.chain_id == "empty-chain"
        assert metrics.total_events == 0
        assert metrics.latency.total_time_ms == 0
        assert metrics.tokens.total_tokens == 0
        assert metrics.errors.error_count == 0
        assert metrics.facts.fact_count == 0
        assert metrics.agents.agent_count == 0

    def test_metrics_collector_full_chain(self, sample_chain: Chain):
        """Test collecting all metrics from sample chain."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        assert metrics.chain_id == "test-chain"
        assert metrics.total_events == 7
        assert metrics.latency.total_time_ms == 1000
        assert metrics.tokens.total_tokens == 450
        assert metrics.errors.error_count == 0
        assert metrics.facts.fact_count == 1
        assert metrics.agents.agent_count == 2

    def test_metrics_collector_has_timestamp(self, sample_chain: Chain):
        """Test that collection includes timestamp."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        assert metrics.collection_timestamp is not None
        assert isinstance(metrics.collection_timestamp, datetime)

    def test_collected_metrics_summary(self, sample_chain: Chain):
        """Test collected metrics summary generation."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)
        summary = metrics.summary()

        assert "test-chain" in summary
        assert "1000ms" in summary
        assert "450" in summary
        assert "planner" in summary

    def test_collected_metrics_to_dict(self, sample_chain: Chain):
        """Test collected metrics serialization."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)
        result = metrics.to_dict()

        assert result["chain_id"] == "test-chain"
        assert "latency" in result
        assert "tokens" in result
        assert "errors" in result
        assert "facts" in result
        assert "agents" in result

    def test_individual_collectors(self, sample_chain: Chain):
        """Test individual collector methods."""
        collector = MetricsCollector()

        latency = collector.collect_latency(sample_chain)
        assert latency.total_time_ms == 1000

        tokens = collector.collect_tokens(sample_chain)
        assert tokens.total_tokens == 450

        errors = collector.collect_errors(sample_chain)
        assert errors.error_count == 0

        facts = collector.collect_facts(sample_chain)
        assert facts.fact_count == 1

        agents = collector.collect_agents(sample_chain)
        assert agents.agent_count == 2


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    @pytest.fixture
    def multiple_metrics(self, sample_chain: Chain, sample_chain_with_errors: Chain) -> List[CollectedMetrics]:
        """Create metrics from multiple chains."""
        collector = MetricsCollector()
        return [
            collector.collect_from_chain(sample_chain),
            collector.collect_from_chain(sample_chain_with_errors),
        ]

    def test_aggregator_empty_list(self):
        """Test aggregator with empty list."""
        aggregator = MetricsAggregator()

        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            aggregator.aggregate([])

    def test_aggregator_single_chain(self, sample_chain: Chain):
        """Test aggregator with single chain."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate([metrics])

        assert aggregated.chain_count == 1
        assert aggregated.latency.total_time_ms == 1000

    def test_aggregator_multiple_chains(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregator with multiple chains."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)

        assert aggregated.chain_count == 2
        assert "test-chain" in aggregated.chain_ids
        assert "error-chain" in aggregated.chain_ids

    def test_aggregated_latency(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated latency metrics."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)

        assert aggregated.latency.total_time_ms == 1200
        assert aggregated.latency.avg_total_time_ms == 600.0
        assert aggregated.latency.p50_step_duration_ms > 0

    def test_aggregated_tokens(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated token metrics."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)

        assert aggregated.tokens.total_tokens == 510
        assert aggregated.tokens.avg_tokens_per_chain == 255.0

    def test_aggregated_errors(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated error metrics."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)

        assert aggregated.errors.total_error_count == 1
        assert aggregated.errors.chains_with_errors == 1
        assert aggregated.errors.error_free_rate == 0.5

    def test_aggregated_facts(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated fact metrics."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)

        assert aggregated.facts.total_fact_count == 1
        assert aggregated.facts.avg_facts_per_chain == 0.5

    def test_aggregated_agents(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated agent metrics."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)

        assert len(aggregated.agents.unique_agents) == 3
        assert aggregated.agents.total_handoffs == 1

    def test_aggregated_summary(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated metrics summary."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)
        summary = aggregated.summary()

        assert "2 chains" in summary
        assert "1200ms" in summary

    def test_aggregated_to_dict(self, multiple_metrics: List[CollectedMetrics]):
        """Test aggregated metrics serialization."""
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(multiple_metrics)
        result = aggregated.to_dict()

        assert result["chain_count"] == 2
        assert "latency" in result
        assert "tokens" in result


class TestExporters:
    """Tests for metric exporters."""

    def test_export_json(self, sample_chain: Chain, tmp_path: Path):
        """Test JSON export."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        output_path = tmp_path / "metrics.json"
        export_json(metrics, output_path)

        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert content["chain_id"] == "test-chain"

    def test_export_json_directory_not_found(self, sample_chain: Chain, tmp_path: Path):
        """Test JSON export with nonexistent directory."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        output_path = tmp_path / "nonexistent" / "metrics.json"
        with pytest.raises(FileNotFoundError):
            export_json(metrics, output_path)

    def test_export_prometheus(self, sample_chain: Chain, tmp_path: Path):
        """Test Prometheus export."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        output_path = tmp_path / "metrics.prom"
        export_prometheus(metrics, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "lctl_total_events" in content
        assert "test-chain" in content

    def test_export_prometheus_aggregated(self, sample_chain: Chain, tmp_path: Path):
        """Test Prometheus export with aggregated metrics."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate([metrics])

        output_path = tmp_path / "aggregated.prom"
        export_prometheus(aggregated, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "lctl_chain_count" in content

    def test_export_prometheus_custom_prefix(self, sample_chain: Chain, tmp_path: Path):
        """Test Prometheus export with custom prefix."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        output_path = tmp_path / "metrics.prom"
        export_prometheus(metrics, output_path, prefix="custom")

        content = output_path.read_text()
        assert "custom_total_events" in content

    def test_export_csv(self, sample_chain: Chain, tmp_path: Path):
        """Test CSV export."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        output_path = tmp_path / "metrics.csv"
        export_csv(metrics, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "metric,value" in content
        assert "total_events,7" in content

    def test_export_csv_aggregated(self, sample_chain: Chain, tmp_path: Path):
        """Test CSV export with aggregated metrics."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate([metrics])

        output_path = tmp_path / "aggregated.csv"
        export_csv(aggregated, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "chain_count,1" in content

    def test_to_dict(self, sample_chain: Chain):
        """Test to_dict helper."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)
        result = to_dict(metrics)

        assert result["chain_id"] == "test-chain"

    def test_to_json(self, sample_chain: Chain):
        """Test to_json helper."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)
        result = to_json(metrics)

        parsed = json.loads(result)
        assert parsed["chain_id"] == "test-chain"

    def test_to_prometheus(self, sample_chain: Chain):
        """Test to_prometheus helper."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)
        result = to_prometheus(metrics)

        assert "lctl_total_events" in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_chain_with_only_errors(self, base_timestamp: datetime):
        """Test chain containing only error events."""
        chain = Chain(id="errors-only")
        chain.add_event(Event(
            seq=1,
            type=EventType.ERROR,
            timestamp=base_timestamp,
            agent="agent1",
            data={"type": "Error1", "recoverable": False}
        ))
        chain.add_event(Event(
            seq=2,
            type=EventType.ERROR,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent2",
            data={"type": "Error2", "recoverable": True}
        ))

        collector = MetricsCollector()
        metrics = collector.collect_from_chain(chain)

        assert metrics.errors.error_count == 2
        assert metrics.errors.recoverable_count == 1
        assert metrics.errors.non_recoverable_count == 1
        assert metrics.latency.total_time_ms == 0

    def test_chain_with_only_facts(self, base_timestamp: datetime):
        """Test chain containing only fact events."""
        chain = Chain(id="facts-only")
        chain.add_event(Event(
            seq=1,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp,
            agent="agent1",
            data={"id": "F1", "text": "Fact 1", "confidence": 0.8}
        ))
        chain.add_event(Event(
            seq=2,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent2",
            data={"id": "F2", "text": "Fact 2", "confidence": 0.9}
        ))

        collector = MetricsCollector()
        metrics = collector.collect_from_chain(chain)

        assert metrics.facts.fact_count == 2
        assert abs(metrics.facts.avg_confidence - 0.85) < 0.0001
        assert metrics.latency.total_time_ms == 0

    def test_chain_with_system_agent(self, base_timestamp: datetime):
        """Test that system agent is handled correctly."""
        chain = Chain(id="system-chain")
        chain.add_event(Event(
            seq=1,
            type=EventType.CHECKPOINT,
            timestamp=base_timestamp,
            agent="system",
            data={"state_hash": "abc123"}
        ))
        chain.add_event(Event(
            seq=2,
            type=EventType.STEP_START,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent1",
            data={"intent": "process"}
        ))

        collector = MetricsCollector()
        metrics = collector.collect_from_chain(chain)

        assert metrics.agents.agent_count == 1
        assert "system" not in metrics.agents.agents
        assert "agent1" in metrics.agents.agents

    def test_step_without_tokens(self, base_timestamp: datetime):
        """Test step end without token information."""
        chain = Chain(id="no-tokens")
        chain.add_event(Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="agent1",
            data={"intent": "process"}
        ))
        chain.add_event(Event(
            seq=2,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="agent1",
            data={"outcome": "success", "duration_ms": 100}
        ))

        collector = MetricsCollector()
        metrics = collector.collect_from_chain(chain)

        assert metrics.tokens.total_tokens == 0
        assert metrics.latency.total_time_ms == 100

    def test_fact_without_id(self, base_timestamp: datetime):
        """Test fact event without id field."""
        chain = Chain(id="no-fact-id")
        chain.add_event(Event(
            seq=1,
            type=EventType.FACT_ADDED,
            timestamp=base_timestamp,
            agent="agent1",
            data={"text": "Orphan fact", "confidence": 0.5}
        ))

        collector = MetricsCollector()
        metrics = collector.collect_from_chain(chain)

        assert metrics.facts.fact_count == 0

    def test_multiple_handoffs_same_pair(self, base_timestamp: datetime):
        """Test multiple handoffs between same agent pair."""
        chain = Chain(id="multi-handoff")
        events = [
            Event(seq=1, type=EventType.STEP_START, timestamp=base_timestamp,
                  agent="agent1", data={"intent": "step1"}),
            Event(seq=2, type=EventType.STEP_START, timestamp=base_timestamp + timedelta(milliseconds=100),
                  agent="agent2", data={"intent": "step2"}),
            Event(seq=3, type=EventType.STEP_START, timestamp=base_timestamp + timedelta(milliseconds=200),
                  agent="agent1", data={"intent": "step3"}),
            Event(seq=4, type=EventType.STEP_START, timestamp=base_timestamp + timedelta(milliseconds=300),
                  agent="agent2", data={"intent": "step4"}),
        ]
        for e in events:
            chain.add_event(e)

        collector = MetricsCollector()
        metrics = collector.collect_from_chain(chain)

        assert metrics.agents.handoff_count == 3
        assert metrics.agents.handoff_pairs[("agent1", "agent2")] == 2
        assert metrics.agents.handoff_pairs[("agent2", "agent1")] == 1

    def test_percentile_calculation(self, base_timestamp: datetime):
        """Test percentile calculation in aggregation."""
        chains: List[Chain] = []

        for i in range(10):
            chain = Chain(id=f"chain-{i}")
            chain.add_event(Event(
                seq=1, type=EventType.STEP_START, timestamp=base_timestamp,
                agent="agent1", data={"intent": "step"}
            ))
            chain.add_event(Event(
                seq=2, type=EventType.STEP_END, timestamp=base_timestamp + timedelta(milliseconds=100),
                agent="agent1", data={"outcome": "success", "duration_ms": (i + 1) * 100}
            ))
            chains.append(chain)

        collector = MetricsCollector()
        metrics_list = [collector.collect_from_chain(c) for c in chains]

        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(metrics_list)

        assert aggregated.latency.p50_step_duration_ms > 0
        assert aggregated.latency.p90_step_duration_ms > aggregated.latency.p50_step_duration_ms
        assert aggregated.latency.p99_step_duration_ms >= aggregated.latency.p90_step_duration_ms


class TestIntegration:
    """Integration tests matching the example usage."""

    def test_example_usage(self, sample_chain: Chain, tmp_path: Path):
        """Test the documented example usage."""
        collector = MetricsCollector()
        metrics = collector.collect_from_chain(sample_chain)

        summary = metrics.summary()
        assert "test-chain" in summary

        prom_path = tmp_path / "metrics.prom"
        export_prometheus(metrics, prom_path)
        assert prom_path.exists()

    def test_full_workflow(self, sample_chain: Chain, sample_chain_with_errors: Chain, tmp_path: Path):
        """Test full metrics workflow: collect, aggregate, export."""
        collector = MetricsCollector()

        metrics1 = collector.collect_from_chain(sample_chain)
        metrics2 = collector.collect_from_chain(sample_chain_with_errors)

        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate([metrics1, metrics2])

        json_path = tmp_path / "aggregated.json"
        export_json(aggregated, json_path)
        assert json_path.exists()

        prom_path = tmp_path / "aggregated.prom"
        export_prometheus(aggregated, prom_path)
        assert prom_path.exists()

        csv_path = tmp_path / "aggregated.csv"
        export_csv(aggregated, csv_path)
        assert csv_path.exists()

        json_content = json.loads(json_path.read_text())
        assert json_content["chain_count"] == 2

        prom_content = prom_path.read_text()
        assert "lctl_chain_count 2" in prom_content

        csv_content = csv_path.read_text()
        assert "chain_count,2" in csv_content
