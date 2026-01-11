"""LCTL Metric Exporters - Export metrics to various formats."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .aggregators import AggregatedMetrics
from .collectors import CollectedMetrics


def export_json(
    metrics: Union[CollectedMetrics, AggregatedMetrics],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """Export metrics to JSON file.

    Args:
        metrics: CollectedMetrics or AggregatedMetrics to export.
        path: Output file path.
        indent: JSON indentation level.

    Raises:
        FileNotFoundError: If parent directory does not exist.
        PermissionError: If write permission is denied.
    """
    export_path = Path(path)

    if not export_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {export_path.parent}")

    try:
        content = json.dumps(metrics.to_dict(), indent=indent, default=str)
        export_path.write_text(content)
    except PermissionError:
        raise PermissionError(f"Permission denied writing to: {path}")


def export_prometheus(
    metrics: Union[CollectedMetrics, AggregatedMetrics],
    path: Union[str, Path],
    prefix: str = "lctl",
) -> None:
    """Export metrics in Prometheus exposition format.

    Args:
        metrics: CollectedMetrics or AggregatedMetrics to export.
        path: Output file path.
        prefix: Metric name prefix.

    Raises:
        FileNotFoundError: If parent directory does not exist.
        PermissionError: If write permission is denied.
    """
    export_path = Path(path)

    if not export_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {export_path.parent}")

    lines: List[str] = []

    if isinstance(metrics, CollectedMetrics):
        lines.extend(_format_collected_prometheus(metrics, prefix))
    else:
        lines.extend(_format_aggregated_prometheus(metrics, prefix))

    try:
        export_path.write_text("\n".join(lines) + "\n")
    except PermissionError:
        raise PermissionError(f"Permission denied writing to: {path}")


def _format_collected_prometheus(metrics: CollectedMetrics, prefix: str) -> List[str]:
    """Format CollectedMetrics as Prometheus exposition format."""
    lines: List[str] = []
    chain_id = metrics.chain_id

    lines.append(f"# HELP {prefix}_total_events Total number of events in chain")
    lines.append(f"# TYPE {prefix}_total_events gauge")
    lines.append(f'{prefix}_total_events{{chain_id="{chain_id}"}} {metrics.total_events}')
    lines.append("")

    lines.append(f"# HELP {prefix}_total_time_ms Total execution time in milliseconds")
    lines.append(f"# TYPE {prefix}_total_time_ms gauge")
    lines.append(f'{prefix}_total_time_ms{{chain_id="{chain_id}"}} {metrics.latency.total_time_ms}')
    lines.append("")

    lines.append(f"# HELP {prefix}_avg_step_duration_ms Average step duration in milliseconds")
    lines.append(f"# TYPE {prefix}_avg_step_duration_ms gauge")
    lines.append(f'{prefix}_avg_step_duration_ms{{chain_id="{chain_id}"}} {metrics.latency.avg_step_duration_ms:.2f}')
    lines.append("")

    lines.append(f"# HELP {prefix}_total_tokens Total tokens used")
    lines.append(f"# TYPE {prefix}_total_tokens gauge")
    lines.append(f'{prefix}_total_tokens{{chain_id="{chain_id}"}} {metrics.tokens.total_tokens}')
    lines.append("")

    lines.append(f"# HELP {prefix}_input_tokens Total input tokens")
    lines.append(f"# TYPE {prefix}_input_tokens gauge")
    lines.append(f'{prefix}_input_tokens{{chain_id="{chain_id}"}} {metrics.tokens.total_input_tokens}')
    lines.append("")

    lines.append(f"# HELP {prefix}_output_tokens Total output tokens")
    lines.append(f"# TYPE {prefix}_output_tokens gauge")
    lines.append(f'{prefix}_output_tokens{{chain_id="{chain_id}"}} {metrics.tokens.total_output_tokens}')
    lines.append("")

    lines.append(f"# HELP {prefix}_token_efficiency Token efficiency ratio")
    lines.append(f"# TYPE {prefix}_token_efficiency gauge")
    lines.append(f'{prefix}_token_efficiency{{chain_id="{chain_id}"}} {metrics.tokens.efficiency_ratio:.4f}')
    lines.append("")

    lines.append(f"# HELP {prefix}_error_count Total error count")
    lines.append(f"# TYPE {prefix}_error_count gauge")
    lines.append(f'{prefix}_error_count{{chain_id="{chain_id}"}} {metrics.errors.error_count}')
    lines.append("")

    lines.append(f"# HELP {prefix}_error_rate Error rate")
    lines.append(f"# TYPE {prefix}_error_rate gauge")
    lines.append(f'{prefix}_error_rate{{chain_id="{chain_id}"}} {metrics.errors.error_rate:.4f}')
    lines.append("")

    if metrics.errors.error_types:
        lines.append(f"# HELP {prefix}_errors_by_type Errors by type")
        lines.append(f"# TYPE {prefix}_errors_by_type gauge")
        for error_type, count in metrics.errors.error_types.items():
            lines.append(f'{prefix}_errors_by_type{{chain_id="{chain_id}",type="{error_type}"}} {count}')
        lines.append("")

    lines.append(f"# HELP {prefix}_fact_count Total fact count")
    lines.append(f"# TYPE {prefix}_fact_count gauge")
    lines.append(f'{prefix}_fact_count{{chain_id="{chain_id}"}} {metrics.facts.fact_count}')
    lines.append("")

    lines.append(f"# HELP {prefix}_avg_confidence Average fact confidence")
    lines.append(f"# TYPE {prefix}_avg_confidence gauge")
    lines.append(f'{prefix}_avg_confidence{{chain_id="{chain_id}"}} {metrics.facts.avg_confidence:.4f}')
    lines.append("")

    lines.append(f"# HELP {prefix}_agent_count Number of unique agents")
    lines.append(f"# TYPE {prefix}_agent_count gauge")
    lines.append(f'{prefix}_agent_count{{chain_id="{chain_id}"}} {metrics.agents.agent_count}')
    lines.append("")

    lines.append(f"# HELP {prefix}_handoff_count Number of agent handoffs")
    lines.append(f"# TYPE {prefix}_handoff_count gauge")
    lines.append(f'{prefix}_handoff_count{{chain_id="{chain_id}"}} {metrics.agents.handoff_count}')
    lines.append("")

    if metrics.agents.steps_per_agent:
        lines.append(f"# HELP {prefix}_steps_per_agent Steps per agent")
        lines.append(f"# TYPE {prefix}_steps_per_agent gauge")
        for agent, steps in metrics.agents.steps_per_agent.items():
            lines.append(f'{prefix}_steps_per_agent{{chain_id="{chain_id}",agent="{agent}"}} {steps}')
        lines.append("")

    if metrics.tokens.tokens_by_agent:
        lines.append(f"# HELP {prefix}_tokens_by_agent Tokens by agent")
        lines.append(f"# TYPE {prefix}_tokens_by_agent gauge")
        for agent, tokens in metrics.tokens.tokens_by_agent.items():
            lines.append(
                f'{prefix}_tokens_by_agent{{chain_id="{chain_id}",agent="{agent}",type="input"}} {tokens["input"]}'
            )
            lines.append(
                f'{prefix}_tokens_by_agent{{chain_id="{chain_id}",agent="{agent}",type="output"}} {tokens["output"]}'
            )
        lines.append("")

    return lines


def _format_aggregated_prometheus(metrics: AggregatedMetrics, prefix: str) -> List[str]:
    """Format AggregatedMetrics as Prometheus exposition format."""
    lines: List[str] = []

    lines.append(f"# HELP {prefix}_chain_count Number of aggregated chains")
    lines.append(f"# TYPE {prefix}_chain_count gauge")
    lines.append(f"{prefix}_chain_count {metrics.chain_count}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_total_time_ms Total aggregated execution time")
    lines.append(f"# TYPE {prefix}_agg_total_time_ms gauge")
    lines.append(f"{prefix}_agg_total_time_ms {metrics.latency.total_time_ms}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_avg_total_time_ms Average total time per chain")
    lines.append(f"# TYPE {prefix}_agg_avg_total_time_ms gauge")
    lines.append(f"{prefix}_agg_avg_total_time_ms {metrics.latency.avg_total_time_ms:.2f}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_step_duration_ms Step duration percentiles")
    lines.append(f"# TYPE {prefix}_agg_step_duration_ms gauge")
    lines.append(f'{prefix}_agg_step_duration_ms{{quantile="0.5"}} {metrics.latency.p50_step_duration_ms:.2f}')
    lines.append(f'{prefix}_agg_step_duration_ms{{quantile="0.9"}} {metrics.latency.p90_step_duration_ms:.2f}')
    lines.append(f'{prefix}_agg_step_duration_ms{{quantile="0.99"}} {metrics.latency.p99_step_duration_ms:.2f}')
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_total_tokens Total tokens across all chains")
    lines.append(f"# TYPE {prefix}_agg_total_tokens gauge")
    lines.append(f"{prefix}_agg_total_tokens {metrics.tokens.total_tokens}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_avg_tokens_per_chain Average tokens per chain")
    lines.append(f"# TYPE {prefix}_agg_avg_tokens_per_chain gauge")
    lines.append(f"{prefix}_agg_avg_tokens_per_chain {metrics.tokens.avg_tokens_per_chain:.2f}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_total_errors Total errors across all chains")
    lines.append(f"# TYPE {prefix}_agg_total_errors gauge")
    lines.append(f"{prefix}_agg_total_errors {metrics.errors.total_error_count}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_error_free_rate Rate of error-free chains")
    lines.append(f"# TYPE {prefix}_agg_error_free_rate gauge")
    lines.append(f"{prefix}_agg_error_free_rate {metrics.errors.error_free_rate:.4f}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_total_facts Total facts across all chains")
    lines.append(f"# TYPE {prefix}_agg_total_facts gauge")
    lines.append(f"{prefix}_agg_total_facts {metrics.facts.total_fact_count}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_avg_confidence Average confidence across all chains")
    lines.append(f"# TYPE {prefix}_agg_avg_confidence gauge")
    lines.append(f"{prefix}_agg_avg_confidence {metrics.facts.avg_confidence:.4f}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_unique_agents Number of unique agents")
    lines.append(f"# TYPE {prefix}_agg_unique_agents gauge")
    lines.append(f"{prefix}_agg_unique_agents {len(metrics.agents.unique_agents)}")
    lines.append("")

    lines.append(f"# HELP {prefix}_agg_total_handoffs Total handoffs across all chains")
    lines.append(f"# TYPE {prefix}_agg_total_handoffs gauge")
    lines.append(f"{prefix}_agg_total_handoffs {metrics.agents.total_handoffs}")
    lines.append("")

    if metrics.agents.agent_participation:
        lines.append(f"# HELP {prefix}_agg_agent_participation Agent participation across chains")
        lines.append(f"# TYPE {prefix}_agg_agent_participation gauge")
        for agent, count in metrics.agents.agent_participation.items():
            lines.append(f'{prefix}_agg_agent_participation{{agent="{agent}"}} {count}')
        lines.append("")

    return lines


def export_csv(
    metrics: Union[CollectedMetrics, AggregatedMetrics],
    path: Union[str, Path],
) -> None:
    """Export metrics to CSV file.

    Args:
        metrics: CollectedMetrics or AggregatedMetrics to export.
        path: Output file path.

    Raises:
        FileNotFoundError: If parent directory does not exist.
        PermissionError: If write permission is denied.
    """
    export_path = Path(path)

    if not export_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {export_path.parent}")

    lines: List[str] = []

    if isinstance(metrics, CollectedMetrics):
        lines.extend(_format_collected_csv(metrics))
    else:
        lines.extend(_format_aggregated_csv(metrics))

    try:
        export_path.write_text("\n".join(lines) + "\n")
    except PermissionError:
        raise PermissionError(f"Permission denied writing to: {path}")


def _format_collected_csv(metrics: CollectedMetrics) -> List[str]:
    """Format CollectedMetrics as CSV."""
    lines: List[str] = []

    lines.append("metric,value")
    lines.append(f"chain_id,{metrics.chain_id}")
    lines.append(f"total_events,{metrics.total_events}")
    lines.append(f"total_time_ms,{metrics.latency.total_time_ms}")
    lines.append(f"avg_step_duration_ms,{metrics.latency.avg_step_duration_ms:.2f}")
    lines.append(f"min_step_duration_ms,{metrics.latency.min_step_duration_ms}")
    lines.append(f"max_step_duration_ms,{metrics.latency.max_step_duration_ms}")
    lines.append(f"total_input_tokens,{metrics.tokens.total_input_tokens}")
    lines.append(f"total_output_tokens,{metrics.tokens.total_output_tokens}")
    lines.append(f"total_tokens,{metrics.tokens.total_tokens}")
    lines.append(f"token_efficiency,{metrics.tokens.efficiency_ratio:.4f}")
    lines.append(f"error_count,{metrics.errors.error_count}")
    lines.append(f"error_rate,{metrics.errors.error_rate:.4f}")
    lines.append(f"recoverable_errors,{metrics.errors.recoverable_count}")
    lines.append(f"non_recoverable_errors,{metrics.errors.non_recoverable_count}")
    lines.append(f"fact_count,{metrics.facts.fact_count}")
    lines.append(f"avg_confidence,{metrics.facts.avg_confidence:.4f}")
    lines.append(f"min_confidence,{metrics.facts.min_confidence:.4f}")
    lines.append(f"max_confidence,{metrics.facts.max_confidence:.4f}")
    lines.append(f"agent_count,{metrics.agents.agent_count}")
    lines.append(f"handoff_count,{metrics.agents.handoff_count}")

    return lines


def _format_aggregated_csv(metrics: AggregatedMetrics) -> List[str]:
    """Format AggregatedMetrics as CSV."""
    lines: List[str] = []

    lines.append("metric,value")
    lines.append(f"chain_count,{metrics.chain_count}")
    lines.append(f"total_time_ms,{metrics.latency.total_time_ms}")
    lines.append(f"avg_total_time_ms,{metrics.latency.avg_total_time_ms:.2f}")
    lines.append(f"avg_step_duration_ms,{metrics.latency.avg_step_duration_ms:.2f}")
    lines.append(f"p50_step_duration_ms,{metrics.latency.p50_step_duration_ms:.2f}")
    lines.append(f"p90_step_duration_ms,{metrics.latency.p90_step_duration_ms:.2f}")
    lines.append(f"p99_step_duration_ms,{metrics.latency.p99_step_duration_ms:.2f}")
    lines.append(f"total_input_tokens,{metrics.tokens.total_input_tokens}")
    lines.append(f"total_output_tokens,{metrics.tokens.total_output_tokens}")
    lines.append(f"total_tokens,{metrics.tokens.total_tokens}")
    lines.append(f"avg_tokens_per_chain,{metrics.tokens.avg_tokens_per_chain:.2f}")
    lines.append(f"avg_efficiency_ratio,{metrics.tokens.avg_efficiency_ratio:.4f}")
    lines.append(f"total_error_count,{metrics.errors.total_error_count}")
    lines.append(f"avg_error_rate,{metrics.errors.avg_error_rate:.4f}")
    lines.append(f"chains_with_errors,{metrics.errors.chains_with_errors}")
    lines.append(f"error_free_rate,{metrics.errors.error_free_rate:.4f}")
    lines.append(f"total_fact_count,{metrics.facts.total_fact_count}")
    lines.append(f"avg_facts_per_chain,{metrics.facts.avg_facts_per_chain:.2f}")
    lines.append(f"avg_confidence,{metrics.facts.avg_confidence:.4f}")
    lines.append(f"unique_agent_count,{len(metrics.agents.unique_agents)}")
    lines.append(f"total_handoffs,{metrics.agents.total_handoffs}")
    lines.append(f"avg_handoffs_per_chain,{metrics.agents.avg_handoffs_per_chain:.2f}")

    return lines


def to_dict(metrics: Union[CollectedMetrics, AggregatedMetrics]) -> Dict[str, Any]:
    """Convert metrics to dictionary.

    Args:
        metrics: CollectedMetrics or AggregatedMetrics to convert.

    Returns:
        Dictionary representation of the metrics.
    """
    return metrics.to_dict()


def to_json(
    metrics: Union[CollectedMetrics, AggregatedMetrics],
    indent: int = 2,
) -> str:
    """Convert metrics to JSON string.

    Args:
        metrics: CollectedMetrics or AggregatedMetrics to convert.
        indent: JSON indentation level.

    Returns:
        JSON string representation of the metrics.
    """
    return json.dumps(metrics.to_dict(), indent=indent, default=str)


def to_prometheus(
    metrics: Union[CollectedMetrics, AggregatedMetrics],
    prefix: str = "lctl",
) -> str:
    """Convert metrics to Prometheus exposition format string.

    Args:
        metrics: CollectedMetrics or AggregatedMetrics to convert.
        prefix: Metric name prefix.

    Returns:
        Prometheus exposition format string.
    """
    if isinstance(metrics, CollectedMetrics):
        lines = _format_collected_prometheus(metrics, prefix)
    else:
        lines = _format_aggregated_prometheus(metrics, prefix)
    return "\n".join(lines) + "\n"
