"""LCTL Metric Collectors - Collect metrics from LCTL chains."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.events import Chain, EventType


@dataclass
class LatencyMetrics:
    """Latency-related metrics from chain execution."""

    step_durations: Dict[str, List[int]] = field(default_factory=dict)
    tool_call_durations: Dict[str, List[int]] = field(default_factory=dict)
    total_time_ms: int = 0
    avg_step_duration_ms: float = 0.0
    avg_tool_call_duration_ms: float = 0.0
    min_step_duration_ms: int = 0
    max_step_duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_durations": self.step_durations,
            "tool_call_durations": self.tool_call_durations,
            "total_time_ms": self.total_time_ms,
            "avg_step_duration_ms": self.avg_step_duration_ms,
            "avg_tool_call_duration_ms": self.avg_tool_call_duration_ms,
            "min_step_duration_ms": self.min_step_duration_ms,
            "max_step_duration_ms": self.max_step_duration_ms,
        }


@dataclass
class TokenMetrics:
    """Token usage metrics per agent."""

    tokens_by_agent: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    efficiency_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_by_agent": self.tokens_by_agent,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "efficiency_ratio": self.efficiency_ratio,
        }


@dataclass
class ErrorMetrics:
    """Error-related metrics from chain execution."""

    error_count: int = 0
    error_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    errors_by_agent: Dict[str, int] = field(default_factory=dict)
    recoverable_count: int = 0
    non_recoverable_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "error_types": self.error_types,
            "errors_by_agent": self.errors_by_agent,
            "recoverable_count": self.recoverable_count,
            "non_recoverable_count": self.non_recoverable_count,
        }


@dataclass
class FactMetrics:
    """Fact-related metrics from chain execution."""

    fact_count: int = 0
    avg_confidence: float = 0.0
    confidence_changes: Dict[str, List[float]] = field(default_factory=dict)
    facts_by_agent: Dict[str, int] = field(default_factory=dict)
    min_confidence: float = 1.0
    max_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fact_count": self.fact_count,
            "avg_confidence": self.avg_confidence,
            "confidence_changes": self.confidence_changes,
            "facts_by_agent": self.facts_by_agent,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
        }


@dataclass
class AgentMetrics:
    """Agent-related metrics from chain execution."""

    steps_per_agent: Dict[str, int] = field(default_factory=dict)
    handoff_count: int = 0
    handoff_pairs: Dict[Tuple[str, str], int] = field(default_factory=dict)
    collaboration_graph: Dict[str, Set[str]] = field(default_factory=dict)
    agent_count: int = 0
    agents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        handoff_pairs_str = {
            f"{k[0]}->{k[1]}": v for k, v in self.handoff_pairs.items()
        }
        collaboration_graph_list = {
            k: list(v) for k, v in self.collaboration_graph.items()
        }
        return {
            "steps_per_agent": self.steps_per_agent,
            "handoff_count": self.handoff_count,
            "handoff_pairs": handoff_pairs_str,
            "collaboration_graph": collaboration_graph_list,
            "agent_count": self.agent_count,
            "agents": self.agents,
        }


@dataclass
class CollectedMetrics:
    """All collected metrics from a chain."""

    chain_id: str
    latency: LatencyMetrics
    tokens: TokenMetrics
    errors: ErrorMetrics
    facts: FactMetrics
    agents: AgentMetrics
    total_events: int = 0
    collection_timestamp: Optional[datetime] = None

    def summary(self) -> str:
        """Generate a human-readable summary of the metrics."""
        lines = [
            f"LCTL Metrics Summary - Chain: {self.chain_id}",
            "=" * 50,
            "",
            "Latency:",
            f"  Total time: {self.latency.total_time_ms}ms",
            f"  Avg step duration: {self.latency.avg_step_duration_ms:.2f}ms",
            f"  Avg tool call duration: {self.latency.avg_tool_call_duration_ms:.2f}ms",
            "",
            "Tokens:",
            f"  Total: {self.tokens.total_tokens} (in: {self.tokens.total_input_tokens}, out: {self.tokens.total_output_tokens})",
            f"  Efficiency ratio: {self.tokens.efficiency_ratio:.2f}",
            "",
            "Errors:",
            f"  Count: {self.errors.error_count} (rate: {self.errors.error_rate:.2%})",
            f"  Recoverable: {self.errors.recoverable_count}, Non-recoverable: {self.errors.non_recoverable_count}",
            "",
            "Facts:",
            f"  Count: {self.facts.fact_count}",
            f"  Avg confidence: {self.facts.avg_confidence:.2f}",
            f"  Confidence range: [{self.facts.min_confidence:.2f}, {self.facts.max_confidence:.2f}]",
            "",
            "Agents:",
            f"  Count: {self.agents.agent_count}",
            f"  Handoffs: {self.agents.handoff_count}",
            f"  Agents: {', '.join(self.agents.agents)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "chain_id": self.chain_id,
            "total_events": self.total_events,
            "collection_timestamp": (
                self.collection_timestamp.isoformat()
                if self.collection_timestamp
                else None
            ),
            "latency": self.latency.to_dict(),
            "tokens": self.tokens.to_dict(),
            "errors": self.errors.to_dict(),
            "facts": self.facts.to_dict(),
            "agents": self.agents.to_dict(),
        }


class LatencyCollector:
    """Collects latency metrics from a chain."""

    def collect(self, chain: Chain) -> LatencyMetrics:
        """Collect latency metrics from chain events."""
        metrics = LatencyMetrics()
        all_step_durations: List[int] = []
        all_tool_durations: List[int] = []

        for event in chain.events:
            if event.type == EventType.STEP_END:
                duration = event.data.get("duration_ms", 0)
                agent = event.agent
                if agent not in metrics.step_durations:
                    metrics.step_durations[agent] = []
                metrics.step_durations[agent].append(duration)
                all_step_durations.append(duration)
                metrics.total_time_ms += duration

            elif event.type == EventType.TOOL_CALL:
                duration = event.data.get("duration_ms", 0)
                tool = event.data.get("tool", "unknown")
                if tool not in metrics.tool_call_durations:
                    metrics.tool_call_durations[tool] = []
                metrics.tool_call_durations[tool].append(duration)
                all_tool_durations.append(duration)

        if all_step_durations:
            metrics.avg_step_duration_ms = sum(all_step_durations) / len(all_step_durations)
            metrics.min_step_duration_ms = min(all_step_durations)
            metrics.max_step_duration_ms = max(all_step_durations)

        if all_tool_durations:
            metrics.avg_tool_call_duration_ms = sum(all_tool_durations) / len(all_tool_durations)

        return metrics


class TokenCollector:
    """Collects token usage metrics from a chain."""

    def collect(self, chain: Chain) -> TokenMetrics:
        """Collect token metrics from chain events."""
        metrics = TokenMetrics()

        for event in chain.events:
            if event.type == EventType.STEP_END:
                tokens = event.data.get("tokens", {})
                input_tokens = tokens.get("input", tokens.get("in", 0))
                output_tokens = tokens.get("output", tokens.get("out", 0))

                agent = event.agent
                if agent not in metrics.tokens_by_agent:
                    metrics.tokens_by_agent[agent] = {"input": 0, "output": 0}

                metrics.tokens_by_agent[agent]["input"] += input_tokens
                metrics.tokens_by_agent[agent]["output"] += output_tokens
                metrics.total_input_tokens += input_tokens
                metrics.total_output_tokens += output_tokens

        metrics.total_tokens = metrics.total_input_tokens + metrics.total_output_tokens

        if metrics.total_input_tokens > 0:
            metrics.efficiency_ratio = metrics.total_output_tokens / metrics.total_input_tokens

        return metrics


class ErrorCollector:
    """Collects error metrics from a chain."""

    def collect(self, chain: Chain) -> ErrorMetrics:
        """Collect error metrics from chain events."""
        metrics = ErrorMetrics()

        for event in chain.events:
            if event.type == EventType.ERROR:
                metrics.error_count += 1

                error_type = event.data.get("type", "unknown")
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

                agent = event.agent
                metrics.errors_by_agent[agent] = metrics.errors_by_agent.get(agent, 0) + 1

                if event.data.get("recoverable", False):
                    metrics.recoverable_count += 1
                else:
                    metrics.non_recoverable_count += 1

        total_events = len(chain.events)
        if total_events > 0:
            metrics.error_rate = metrics.error_count / total_events

        return metrics


class FactCollector:
    """Collects fact-related metrics from a chain."""

    def collect(self, chain: Chain) -> FactMetrics:
        """Collect fact metrics from chain events."""
        metrics = FactMetrics()
        confidences: List[float] = []
        seen_facts: Set[str] = set()

        for event in chain.events:
            if event.type == EventType.FACT_ADDED:
                fact_id = event.data.get("id")
                if fact_id:
                    seen_facts.add(fact_id)
                    confidence = event.data.get("confidence", 1.0)
                    confidences.append(confidence)

                    if fact_id not in metrics.confidence_changes:
                        metrics.confidence_changes[fact_id] = []
                    metrics.confidence_changes[fact_id].append(confidence)

                    agent = event.agent
                    metrics.facts_by_agent[agent] = metrics.facts_by_agent.get(agent, 0) + 1

            elif event.type == EventType.FACT_MODIFIED:
                fact_id = event.data.get("id")
                if fact_id and "confidence" in event.data:
                    confidence = event.data["confidence"]
                    confidences.append(confidence)

                    if fact_id not in metrics.confidence_changes:
                        metrics.confidence_changes[fact_id] = []
                    metrics.confidence_changes[fact_id].append(confidence)

        metrics.fact_count = len(seen_facts)

        if confidences:
            metrics.avg_confidence = sum(confidences) / len(confidences)
            metrics.min_confidence = min(confidences)
            metrics.max_confidence = max(confidences)

        return metrics


class AgentCollector:
    """Collects agent-related metrics from a chain."""

    def collect(self, chain: Chain) -> AgentMetrics:
        """Collect agent metrics from chain events."""
        metrics = AgentMetrics()
        agents_seen: Set[str] = set()
        previous_agent: Optional[str] = None

        for event in chain.events:
            if event.agent == "system":
                continue

            agents_seen.add(event.agent)

            if event.type == EventType.STEP_START:
                agent = event.agent
                metrics.steps_per_agent[agent] = metrics.steps_per_agent.get(agent, 0) + 1

                if previous_agent and previous_agent != agent:
                    metrics.handoff_count += 1
                    pair = (previous_agent, agent)
                    metrics.handoff_pairs[pair] = metrics.handoff_pairs.get(pair, 0) + 1

                    if previous_agent not in metrics.collaboration_graph:
                        metrics.collaboration_graph[previous_agent] = set()
                    metrics.collaboration_graph[previous_agent].add(agent)

                previous_agent = agent

        metrics.agents = sorted(agents_seen)
        metrics.agent_count = len(agents_seen)

        return metrics


class MetricsCollector:
    """Main metrics collector that aggregates all metric types."""

    def __init__(self):
        self._latency_collector = LatencyCollector()
        self._token_collector = TokenCollector()
        self._error_collector = ErrorCollector()
        self._fact_collector = FactCollector()
        self._agent_collector = AgentCollector()

    def collect_from_chain(self, chain: Chain) -> CollectedMetrics:
        """Collect all metrics from a chain.

        Args:
            chain: The LCTL chain to collect metrics from.

        Returns:
            CollectedMetrics containing all metric types.
        """
        return CollectedMetrics(
            chain_id=chain.id,
            latency=self._latency_collector.collect(chain),
            tokens=self._token_collector.collect(chain),
            errors=self._error_collector.collect(chain),
            facts=self._fact_collector.collect(chain),
            agents=self._agent_collector.collect(chain),
            total_events=len(chain.events),
            collection_timestamp=datetime.now(timezone.utc),
        )

    def collect_latency(self, chain: Chain) -> LatencyMetrics:
        """Collect only latency metrics."""
        return self._latency_collector.collect(chain)

    def collect_tokens(self, chain: Chain) -> TokenMetrics:
        """Collect only token metrics."""
        return self._token_collector.collect(chain)

    def collect_errors(self, chain: Chain) -> ErrorMetrics:
        """Collect only error metrics."""
        return self._error_collector.collect(chain)

    def collect_facts(self, chain: Chain) -> FactMetrics:
        """Collect only fact metrics."""
        return self._fact_collector.collect(chain)

    def collect_agents(self, chain: Chain) -> AgentMetrics:
        """Collect only agent metrics."""
        return self._agent_collector.collect(chain)
