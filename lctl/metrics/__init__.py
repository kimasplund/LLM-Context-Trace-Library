"""LCTL Metrics Module - Collect, aggregate, and export metrics from LCTL chains.

Usage:
    from lctl.metrics import MetricsCollector, export_prometheus

    collector = MetricsCollector()
    metrics = collector.collect_from_chain(chain)
    print(metrics.summary())
    export_prometheus(metrics, "metrics.prom")
"""

from .aggregators import (
    AgentAggregator,
    AggregatedAgentMetrics,
    AggregatedErrorMetrics,
    AggregatedFactMetrics,
    AggregatedLatencyMetrics,
    AggregatedMetrics,
    AggregatedTokenMetrics,
    ErrorAggregator,
    FactAggregator,
    LatencyAggregator,
    MetricsAggregator,
    TokenAggregator,
)
from .collectors import (
    AgentCollector,
    AgentMetrics,
    CollectedMetrics,
    ErrorCollector,
    ErrorMetrics,
    FactCollector,
    FactMetrics,
    LatencyCollector,
    LatencyMetrics,
    MetricsCollector,
    TokenCollector,
    TokenMetrics,
)
from .exporters import (
    export_csv,
    export_json,
    export_prometheus,
    to_dict,
    to_json,
    to_prometheus,
)

__all__ = [
    # Metric dataclasses
    "LatencyMetrics",
    "TokenMetrics",
    "ErrorMetrics",
    "FactMetrics",
    "AgentMetrics",
    "CollectedMetrics",
    # Collectors
    "LatencyCollector",
    "TokenCollector",
    "ErrorCollector",
    "FactCollector",
    "AgentCollector",
    "MetricsCollector",
    # Aggregated metric dataclasses
    "AggregatedLatencyMetrics",
    "AggregatedTokenMetrics",
    "AggregatedErrorMetrics",
    "AggregatedFactMetrics",
    "AggregatedAgentMetrics",
    "AggregatedMetrics",
    # Aggregators
    "LatencyAggregator",
    "TokenAggregator",
    "ErrorAggregator",
    "FactAggregator",
    "AgentAggregator",
    "MetricsAggregator",
    # Export functions
    "export_json",
    "export_prometheus",
    "export_csv",
    "to_dict",
    "to_json",
    "to_prometheus",
]
