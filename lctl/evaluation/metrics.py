"""LCTL Evaluation Metrics - Metrics for comparing chain executions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.events import Chain, EventType, ReplayEngine, State


@dataclass
class ChainMetrics:
    """Computed metrics for a single chain execution."""

    chain_id: str
    total_events: int
    total_duration_ms: int
    total_tokens_in: int
    total_tokens_out: int
    total_tokens: int
    error_count: int
    fact_count: int
    step_count: int
    agent_count: int
    agents: List[str]
    avg_step_duration_ms: float
    avg_fact_confidence: float
    token_efficiency: float
    error_rate: float
    agent_stats: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_chain(cls, chain: Chain, state: Optional[State] = None) -> "ChainMetrics":
        """Compute metrics from a chain.

        Args:
            chain: The chain to analyze.
            state: Optional pre-computed state. If None, chain will be replayed.
        """
        if state is None:
            engine = ReplayEngine(chain)
            state = engine.replay_all()

        agents = list(set(e.agent for e in chain.events if e.agent != "system"))
        step_starts = [e for e in chain.events if e.type == EventType.STEP_START]
        [e for e in chain.events if e.type == EventType.STEP_END]

        # Calculate per-agent metrics
        agent_stats: Dict[str, Any] = {}
        for event in chain.events:
            agent = event.agent
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "event_count": 0,
                    "duration_ms": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error_count": 0,
                    "fact_count": 0,
                    "tool_calls": 0
                }

            agent_stats[agent]["event_count"] += 1

            # Normalize event type
            event_type = event.type.value if hasattr(event.type, 'value') else event.type

            if event_type == "step_end":
                agent_stats[agent]["duration_ms"] += event.data.get("duration_ms", 0)
                tokens = event.data.get("tokens", {})
                agent_stats[agent]["tokens_in"] += tokens.get("input", tokens.get("in", 0))
                agent_stats[agent]["tokens_out"] += tokens.get("output", tokens.get("out", 0))
            elif event_type == "error":
                agent_stats[agent]["error_count"] += 1
            elif event_type in ("fact_added", "fact_modified"):
                agent_stats[agent]["fact_count"] += 1
            elif event_type == "tool_call":
                agent_stats[agent]["tool_calls"] += 1
                agent_stats[agent]["duration_ms"] += event.data.get("duration_ms", 0)

        total_duration = state.metrics["total_duration_ms"]
        total_tokens_in = state.metrics["total_tokens_in"]
        total_tokens_out = state.metrics["total_tokens_out"]
        total_tokens = total_tokens_in + total_tokens_out

        step_count = len(step_starts)
        avg_step_duration = total_duration / step_count if step_count > 0 else 0.0

        confidences = [f.get("confidence", 1.0) for f in state.facts.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        token_efficiency = (
            total_tokens / (total_duration / 1000) if total_duration > 0 else 0.0
        )

        error_rate = (
            state.metrics["error_count"] / len(chain.events)
            if chain.events else 0.0
        )

        return cls(
            chain_id=chain.id,
            total_events=len(chain.events),
            total_duration_ms=total_duration,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_tokens=total_tokens,
            error_count=state.metrics["error_count"],
            fact_count=len(state.facts),
            step_count=step_count,
            agent_count=len(agents),
            agents=agents,
            avg_step_duration_ms=avg_step_duration,
            avg_fact_confidence=avg_confidence,
            token_efficiency=token_efficiency,
            error_rate=error_rate,
            agent_stats=agent_stats,
            facts=state.facts,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "chain_id": self.chain_id,
            "total_events": self.total_events,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "fact_count": self.fact_count,
            "step_count": self.step_count,
            "agent_count": self.agent_count,
            "agents": self.agents,
            "avg_step_duration_ms": self.avg_step_duration_ms,
            "avg_fact_confidence": self.avg_fact_confidence,
            "token_efficiency": self.token_efficiency,
            "error_rate": self.error_rate,
            "agent_stats": self.agent_stats,
        }


@dataclass
class MetricComparison:
    """Comparison of a single metric between two chains."""

    metric_name: str
    value_a: float
    value_b: float
    difference: float
    percent_change: float
    improvement: Optional[bool]

    @classmethod
    def create(
        cls,
        name: str,
        value_a: float,
        value_b: float,
        higher_is_better: bool = True,
    ) -> "MetricComparison":
        """Create a metric comparison.

        Args:
            name: Name of the metric.
            value_a: Value from chain A.
            value_b: Value from chain B.
            higher_is_better: Whether higher values indicate improvement.

        Returns:
            MetricComparison instance.
        """
        difference = value_b - value_a
        percent_change = (
            (difference / value_a * 100) if value_a != 0 else
            (float("inf") if difference > 0 else (float("-inf") if difference < 0 else 0.0))
        )

        if difference == 0:
            improvement = None
        elif higher_is_better:
            improvement = difference > 0
        else:
            improvement = difference < 0

        return cls(
            metric_name=name,
            value_a=value_a,
            value_b=value_b,
            difference=difference,
            percent_change=percent_change,
            improvement=improvement,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "difference": self.difference,
            "percent_change": self.percent_change,
            "improvement": self.improvement,
        }


def compute_latency_diff(metrics_a: ChainMetrics, metrics_b: ChainMetrics) -> MetricComparison:
    """Compute latency difference between two chains.

    Lower latency is better.
    """
    return MetricComparison.create(
        name="latency_ms",
        value_a=float(metrics_a.total_duration_ms),
        value_b=float(metrics_b.total_duration_ms),
        higher_is_better=False,
    )


def compute_token_efficiency(
    metrics_a: ChainMetrics, metrics_b: ChainMetrics
) -> MetricComparison:
    """Compute token efficiency difference between two chains.

    Higher efficiency (tokens per second) is better.
    """
    return MetricComparison.create(
        name="token_efficiency",
        value_a=metrics_a.token_efficiency,
        value_b=metrics_b.token_efficiency,
        higher_is_better=True,
    )


def compute_error_rate(metrics_a: ChainMetrics, metrics_b: ChainMetrics) -> MetricComparison:
    """Compute error rate difference between two chains.

    Lower error rate is better.
    """
    return MetricComparison.create(
        name="error_rate",
        value_a=metrics_a.error_rate,
        value_b=metrics_b.error_rate,
        higher_is_better=False,
    )


def compute_fact_confidence_avg(
    metrics_a: ChainMetrics, metrics_b: ChainMetrics
) -> MetricComparison:
    """Compute average fact confidence difference between two chains.

    Higher confidence is better.
    """
    return MetricComparison.create(
        name="fact_confidence_avg",
        value_a=metrics_a.avg_fact_confidence,
        value_b=metrics_b.avg_fact_confidence,
        higher_is_better=True,
    )


def compute_all_metrics(
    metrics_a: ChainMetrics, metrics_b: ChainMetrics
) -> Dict[str, MetricComparison]:
    """Compute all metric comparisons between two chains.

    Returns:
        Dictionary mapping metric names to MetricComparison instances.
    """
    return {
        "latency_diff": compute_latency_diff(metrics_a, metrics_b),
        "token_efficiency": compute_token_efficiency(metrics_a, metrics_b),
        "error_rate": compute_error_rate(metrics_a, metrics_b),
        "fact_confidence_avg": compute_fact_confidence_avg(metrics_a, metrics_b),
    }
