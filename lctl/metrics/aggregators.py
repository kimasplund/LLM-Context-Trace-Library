"""LCTL Metric Aggregators - Aggregate metrics across multiple chains."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .collectors import (
    AgentMetrics,
    CollectedMetrics,
    ErrorMetrics,
    FactMetrics,
    LatencyMetrics,
    TokenMetrics,
)


@dataclass
class AggregatedLatencyMetrics:
    """Aggregated latency metrics across multiple chains."""

    chain_count: int = 0
    total_time_ms: int = 0
    avg_total_time_ms: float = 0.0
    avg_step_duration_ms: float = 0.0
    min_step_duration_ms: int = 0
    max_step_duration_ms: int = 0
    p50_step_duration_ms: float = 0.0
    p90_step_duration_ms: float = 0.0
    p99_step_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_count": self.chain_count,
            "total_time_ms": self.total_time_ms,
            "avg_total_time_ms": self.avg_total_time_ms,
            "avg_step_duration_ms": self.avg_step_duration_ms,
            "min_step_duration_ms": self.min_step_duration_ms,
            "max_step_duration_ms": self.max_step_duration_ms,
            "p50_step_duration_ms": self.p50_step_duration_ms,
            "p90_step_duration_ms": self.p90_step_duration_ms,
            "p99_step_duration_ms": self.p99_step_duration_ms,
        }


@dataclass
class AggregatedTokenMetrics:
    """Aggregated token metrics across multiple chains."""

    chain_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_chain: float = 0.0
    avg_efficiency_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_count": self.chain_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_chain": self.avg_tokens_per_chain,
            "avg_efficiency_ratio": self.avg_efficiency_ratio,
        }


@dataclass
class AggregatedErrorMetrics:
    """Aggregated error metrics across multiple chains."""

    chain_count: int = 0
    total_error_count: int = 0
    avg_error_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    chains_with_errors: int = 0
    error_free_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_count": self.chain_count,
            "total_error_count": self.total_error_count,
            "avg_error_rate": self.avg_error_rate,
            "error_types": self.error_types,
            "chains_with_errors": self.chains_with_errors,
            "error_free_rate": self.error_free_rate,
        }


@dataclass
class AggregatedFactMetrics:
    """Aggregated fact metrics across multiple chains."""

    chain_count: int = 0
    total_fact_count: int = 0
    avg_facts_per_chain: float = 0.0
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_count": self.chain_count,
            "total_fact_count": self.total_fact_count,
            "avg_facts_per_chain": self.avg_facts_per_chain,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
        }


@dataclass
class AggregatedAgentMetrics:
    """Aggregated agent metrics across multiple chains."""

    chain_count: int = 0
    unique_agents: List[str] = field(default_factory=list)
    total_handoffs: int = 0
    avg_handoffs_per_chain: float = 0.0
    avg_agents_per_chain: float = 0.0
    agent_participation: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_count": self.chain_count,
            "unique_agents": self.unique_agents,
            "total_handoffs": self.total_handoffs,
            "avg_handoffs_per_chain": self.avg_handoffs_per_chain,
            "avg_agents_per_chain": self.avg_agents_per_chain,
            "agent_participation": self.agent_participation,
        }


@dataclass
class AggregatedMetrics:
    """All aggregated metrics across multiple chains."""

    chain_count: int
    chain_ids: List[str]
    latency: AggregatedLatencyMetrics
    tokens: AggregatedTokenMetrics
    errors: AggregatedErrorMetrics
    facts: AggregatedFactMetrics
    agents: AggregatedAgentMetrics

    def summary(self) -> str:
        """Generate a human-readable summary of aggregated metrics."""
        lines = [
            f"LCTL Aggregated Metrics Summary - {self.chain_count} chains",
            "=" * 50,
            "",
            "Latency:",
            f"  Total time: {self.latency.total_time_ms}ms",
            f"  Avg per chain: {self.latency.avg_total_time_ms:.2f}ms",
            f"  Step duration: avg={self.latency.avg_step_duration_ms:.2f}ms, "
            f"p50={self.latency.p50_step_duration_ms:.2f}ms, "
            f"p90={self.latency.p90_step_duration_ms:.2f}ms",
            "",
            "Tokens:",
            f"  Total: {self.tokens.total_tokens}",
            f"  Avg per chain: {self.tokens.avg_tokens_per_chain:.2f}",
            f"  Avg efficiency: {self.tokens.avg_efficiency_ratio:.2f}",
            "",
            "Errors:",
            f"  Total: {self.errors.total_error_count}",
            f"  Avg rate: {self.errors.avg_error_rate:.2%}",
            f"  Error-free chains: {self.errors.error_free_rate:.2%}",
            "",
            "Facts:",
            f"  Total: {self.facts.total_fact_count}",
            f"  Avg per chain: {self.facts.avg_facts_per_chain:.2f}",
            f"  Avg confidence: {self.facts.avg_confidence:.2f}",
            "",
            "Agents:",
            f"  Unique agents: {len(self.agents.unique_agents)}",
            f"  Total handoffs: {self.agents.total_handoffs}",
            f"  Avg handoffs per chain: {self.agents.avg_handoffs_per_chain:.2f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all aggregated metrics to dictionary."""
        return {
            "chain_count": self.chain_count,
            "chain_ids": self.chain_ids,
            "latency": self.latency.to_dict(),
            "tokens": self.tokens.to_dict(),
            "errors": self.errors.to_dict(),
            "facts": self.facts.to_dict(),
            "agents": self.agents.to_dict(),
        }


def _percentile(sorted_values: List[float], p: float) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


class LatencyAggregator:
    """Aggregates latency metrics across multiple chains."""

    def aggregate(self, metrics_list: List[LatencyMetrics]) -> AggregatedLatencyMetrics:
        """Aggregate latency metrics from multiple chains."""
        if not metrics_list:
            return AggregatedLatencyMetrics()

        result = AggregatedLatencyMetrics(chain_count=len(metrics_list))

        all_step_durations: List[int] = []
        total_times: List[int] = []

        for metrics in metrics_list:
            result.total_time_ms += metrics.total_time_ms
            total_times.append(metrics.total_time_ms)

            for durations in metrics.step_durations.values():
                all_step_durations.extend(durations)

        if total_times:
            result.avg_total_time_ms = sum(total_times) / len(total_times)

        if all_step_durations:
            result.avg_step_duration_ms = sum(all_step_durations) / len(all_step_durations)
            result.min_step_duration_ms = min(all_step_durations)
            result.max_step_duration_ms = max(all_step_durations)

            sorted_durations = sorted(all_step_durations)
            result.p50_step_duration_ms = _percentile(sorted_durations, 0.5)
            result.p90_step_duration_ms = _percentile(sorted_durations, 0.9)
            result.p99_step_duration_ms = _percentile(sorted_durations, 0.99)

        return result


class TokenAggregator:
    """Aggregates token metrics across multiple chains."""

    def aggregate(self, metrics_list: List[TokenMetrics]) -> AggregatedTokenMetrics:
        """Aggregate token metrics from multiple chains."""
        if not metrics_list:
            return AggregatedTokenMetrics()

        result = AggregatedTokenMetrics(chain_count=len(metrics_list))

        efficiency_ratios: List[float] = []

        for metrics in metrics_list:
            result.total_input_tokens += metrics.total_input_tokens
            result.total_output_tokens += metrics.total_output_tokens
            result.total_tokens += metrics.total_tokens
            if metrics.efficiency_ratio > 0:
                efficiency_ratios.append(metrics.efficiency_ratio)

        result.avg_tokens_per_chain = result.total_tokens / len(metrics_list)

        if efficiency_ratios:
            result.avg_efficiency_ratio = sum(efficiency_ratios) / len(efficiency_ratios)

        return result


class ErrorAggregator:
    """Aggregates error metrics across multiple chains."""

    def aggregate(self, metrics_list: List[ErrorMetrics]) -> AggregatedErrorMetrics:
        """Aggregate error metrics from multiple chains."""
        if not metrics_list:
            return AggregatedErrorMetrics()

        result = AggregatedErrorMetrics(chain_count=len(metrics_list))

        error_rates: List[float] = []

        for metrics in metrics_list:
            result.total_error_count += metrics.error_count
            error_rates.append(metrics.error_rate)

            if metrics.error_count > 0:
                result.chains_with_errors += 1

            for error_type, count in metrics.error_types.items():
                result.error_types[error_type] = result.error_types.get(error_type, 0) + count

        result.avg_error_rate = sum(error_rates) / len(error_rates)
        result.error_free_rate = 1 - (result.chains_with_errors / len(metrics_list))

        return result


class FactAggregator:
    """Aggregates fact metrics across multiple chains."""

    def aggregate(self, metrics_list: List[FactMetrics]) -> AggregatedFactMetrics:
        """Aggregate fact metrics from multiple chains."""
        if not metrics_list:
            return AggregatedFactMetrics()

        result = AggregatedFactMetrics(chain_count=len(metrics_list))

        confidences: List[float] = []
        min_conf = 1.0
        max_conf = 0.0

        for metrics in metrics_list:
            result.total_fact_count += metrics.fact_count
            if metrics.avg_confidence > 0:
                confidences.append(metrics.avg_confidence)
            if metrics.fact_count > 0:
                min_conf = min(min_conf, metrics.min_confidence)
                max_conf = max(max_conf, metrics.max_confidence)

        result.avg_facts_per_chain = result.total_fact_count / len(metrics_list)

        if confidences:
            result.avg_confidence = sum(confidences) / len(confidences)

        if result.total_fact_count > 0:
            result.min_confidence = min_conf
            result.max_confidence = max_conf

        return result


class AgentAggregator:
    """Aggregates agent metrics across multiple chains."""

    def aggregate(self, metrics_list: List[AgentMetrics]) -> AggregatedAgentMetrics:
        """Aggregate agent metrics from multiple chains."""
        if not metrics_list:
            return AggregatedAgentMetrics()

        result = AggregatedAgentMetrics(chain_count=len(metrics_list))

        all_agents: set = set()
        agent_counts: List[int] = []

        for metrics in metrics_list:
            all_agents.update(metrics.agents)
            result.total_handoffs += metrics.handoff_count
            agent_counts.append(metrics.agent_count)

            for agent in metrics.agents:
                result.agent_participation[agent] = result.agent_participation.get(agent, 0) + 1

        result.unique_agents = sorted(all_agents)
        result.avg_handoffs_per_chain = result.total_handoffs / len(metrics_list)
        result.avg_agents_per_chain = sum(agent_counts) / len(agent_counts)

        return result


class MetricsAggregator:
    """Main aggregator that combines all metric types."""

    def __init__(self):
        self._latency_aggregator = LatencyAggregator()
        self._token_aggregator = TokenAggregator()
        self._error_aggregator = ErrorAggregator()
        self._fact_aggregator = FactAggregator()
        self._agent_aggregator = AgentAggregator()

    def aggregate(self, metrics_list: List[CollectedMetrics]) -> AggregatedMetrics:
        """Aggregate metrics from multiple chains.

        Args:
            metrics_list: List of CollectedMetrics from individual chains.

        Returns:
            AggregatedMetrics containing combined metrics.

        Raises:
            ValueError: If metrics_list is empty.
        """
        if not metrics_list:
            raise ValueError("Cannot aggregate empty metrics list")

        return AggregatedMetrics(
            chain_count=len(metrics_list),
            chain_ids=[m.chain_id for m in metrics_list],
            latency=self._latency_aggregator.aggregate([m.latency for m in metrics_list]),
            tokens=self._token_aggregator.aggregate([m.tokens for m in metrics_list]),
            errors=self._error_aggregator.aggregate([m.errors for m in metrics_list]),
            facts=self._fact_aggregator.aggregate([m.facts for m in metrics_list]),
            agents=self._agent_aggregator.aggregate([m.agents for m in metrics_list]),
        )

    def aggregate_latency(self, metrics_list: List[LatencyMetrics]) -> AggregatedLatencyMetrics:
        """Aggregate only latency metrics."""
        return self._latency_aggregator.aggregate(metrics_list)

    def aggregate_tokens(self, metrics_list: List[TokenMetrics]) -> AggregatedTokenMetrics:
        """Aggregate only token metrics."""
        return self._token_aggregator.aggregate(metrics_list)

    def aggregate_errors(self, metrics_list: List[ErrorMetrics]) -> AggregatedErrorMetrics:
        """Aggregate only error metrics."""
        return self._error_aggregator.aggregate(metrics_list)

    def aggregate_facts(self, metrics_list: List[FactMetrics]) -> AggregatedFactMetrics:
        """Aggregate only fact metrics."""
        return self._fact_aggregator.aggregate(metrics_list)

    def aggregate_agents(self, metrics_list: List[AgentMetrics]) -> AggregatedAgentMetrics:
        """Aggregate only agent metrics."""
        return self._agent_aggregator.aggregate(metrics_list)
