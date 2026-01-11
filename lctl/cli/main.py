"""LCTL CLI - Time-travel debugging for multi-agent LLM workflows."""

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import click

from ..core.events import Chain, Event, ReplayEngine
from ..evaluation import ChainMetrics, compute_all_metrics


def _load_chain_safely(chain_file: str) -> Optional[Chain]:
    """Load a chain file with proper error handling.

    Returns:
        Chain if successful, None if failed (error message already printed).
    """
    try:
        return Chain.load(Path(chain_file))
    except FileNotFoundError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        return None
    except PermissionError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        return None
    except ValueError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        return None
    except Exception as e:
        click.echo(click.style(f"Unexpected error loading chain: {e}", fg="red"), err=True)
        return None


@click.group()
@click.version_option(version="4.0.0", prog_name="lctl")
def cli():
    """LCTL - Time-travel debugging for multi-agent workflows.

    The killer feature: replay agent execution to any point in time.
    """
    pass


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--to-seq", "-s", type=int, help="Replay to specific sequence number")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed event output")
def replay(chain_file: str, to_seq: Optional[int], verbose: bool):
    """Replay agent execution (time-travel debugging).

    Examples:
        lctl replay chain.lctl.json
        lctl replay --to-seq 10 chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    engine = ReplayEngine(chain)

    if not chain.events:
        click.echo("No events in chain.")
        return

    target = to_seq if to_seq else chain.events[-1].seq
    click.echo(f"Replaying {len(chain.events)} events to seq {target}...")
    click.echo()

    if verbose:
        for event in chain.events:
            if event.seq > target:
                break
            _print_event(event)
        click.echo()

    state = engine.replay_to(target)

    click.echo(f"State at seq {target}:")
    click.echo(f"  Facts: {len(state.facts)}")
    click.echo(f"  Current agent: {state.current_agent or 'none'}")
    click.echo(f"  Errors: {state.metrics['error_count']}")

    if state.facts:
        click.echo()
        click.echo("Facts:")
        for fid, fact in state.facts.items():
            conf = fact.get('confidence', 1.0)
            text = fact.get('text', '') or ''
            display_text = text[:60] + "..." if len(text) > 60 else text
            click.echo(f"  {fid}: {display_text} (conf: {conf:.2f})")


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(chain_file: str, as_json: bool):
    """Show performance statistics for a chain.

    Examples:
        lctl stats chain.lctl.json
        lctl stats --json chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    engine = ReplayEngine(chain)
    state = engine.replay_all()

    # Get unique agents
    agents = set(e.agent for e in chain.events)

    stats_data = {
        "chain_id": chain.id,
        "version": chain.version,
        "events": len(chain.events),
        "agents": len(agents),
        "facts": len(state.facts),
        "errors": state.metrics["error_count"],
        "duration_ms": state.metrics["total_duration_ms"],
        "tokens": {
            "input": state.metrics["total_tokens_in"],
            "output": state.metrics["total_tokens_out"],
            "total": state.metrics["total_tokens_in"] + state.metrics["total_tokens_out"]
        },
        "estimated_cost_usd": _estimate_cost(
            state.metrics["total_tokens_in"],
            state.metrics["total_tokens_out"]
        )
    }

    if as_json:
        click.echo(json.dumps(stats_data, indent=2))
    else:
        click.echo(f"Chain: {stats_data['chain_id']}")
        click.echo(f"Version: {stats_data['version']}")
        click.echo(f"Events: {stats_data['events']}")
        click.echo(f"Agents: {stats_data['agents']} ({', '.join(sorted(agents))})")
        click.echo(f"Facts: {stats_data['facts']}")
        click.echo(f"Errors: {stats_data['errors']}")
        click.echo(f"Duration: {stats_data['duration_ms'] / 1000:.1f}s")
        click.echo(f"Tokens: {stats_data['tokens']['total']:,} (in: {stats_data['tokens']['input']:,} / out: {stats_data['tokens']['output']:,})")
        click.echo(f"Est. Cost: ${stats_data['estimated_cost_usd']:.4f}")


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--top", "-n", default=5, help="Show top N bottlenecks")
def bottleneck(chain_file: str, top: int):
    """Analyze performance bottlenecks.

    Examples:
        lctl bottleneck chain.lctl.json
        lctl bottleneck --top 3 chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    engine = ReplayEngine(chain)

    all_bottlenecks = engine.find_bottlenecks()
    bottlenecks = all_bottlenecks[:top] if top > 0 else []

    if not bottlenecks:
        click.echo("No step timing data found.")
        return

    click.echo("Slowest steps:")
    for i, step in enumerate(bottlenecks, 1):
        duration_s = step["duration_ms"] / 1000
        click.echo(f"  {i}. {step['agent']} (seq {step['seq']}): {duration_s:.1f}s ({step['percentage']:.0f}%)")

    # Recommendations - check list is not empty before accessing index
    click.echo()
    if bottlenecks and bottlenecks[0]["percentage"] > 50:
        click.echo(f"Recommendation: {bottlenecks[0]['agent']} is a major bottleneck ({bottlenecks[0]['percentage']:.0f}% of time).")
        click.echo("Consider: parallelization, caching, or faster model.")


@cli.command()
@click.argument("chain_file", type=click.Path())
def confidence(chain_file: str):
    """Show confidence timeline for facts.

    Examples:
        lctl confidence chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    engine = ReplayEngine(chain)

    timelines = engine.get_confidence_timeline()

    if not timelines:
        click.echo("No facts found in chain.")
        return

    click.echo("Fact confidence timeline:")
    for fact_id, history in timelines.items():
        values = [f"{h['confidence']:.2f}" for h in history]
        trend = " → ".join(values)

        # Status indicator
        final_conf = history[-1]["confidence"]
        if final_conf >= 0.8:
            status = click.style("✓", fg="green")
        elif final_conf >= 0.6:
            status = click.style("~", fg="yellow")
        elif final_conf >= 0.4:
            status = click.style("?", fg="yellow")
        else:
            status = click.style("✗ BLOCKED", fg="red")

        click.echo(f"  {fact_id}: {trend} {status}")


@cli.command("diff")
@click.argument("chain1", type=click.Path())
@click.argument("chain2", type=click.Path())
def diff_chains(chain1: str, chain2: str):
    """Compare two chains and find divergence.

    Examples:
        lctl diff chain-v1.lctl.json chain-v2.lctl.json
    """
    c1 = _load_chain_safely(chain1)
    if c1 is None:
        sys.exit(1)

    c2 = _load_chain_safely(chain2)
    if c2 is None:
        sys.exit(1)

    engine1 = ReplayEngine(c1)
    engine2 = ReplayEngine(c2)

    diffs = engine1.diff(engine2)

    if not diffs:
        click.echo(click.style("Chains are identical.", fg="green"))
        return

    click.echo(f"Found {len(diffs)} difference(s):")
    click.echo()

    for d in diffs[:10]:  # Show first 10
        if d["type"] == "diverged":
            click.echo(f"Seq {d['seq']}: " + click.style("DIVERGED", fg="yellow"))
            click.echo(f"  v1: {d['first']['type']} by {d['first']['agent']}")
            click.echo(f"  v2: {d['second']['type']} by {d['second']['agent']}")
        elif d["type"] == "missing_in_first":
            click.echo(f"Seq {d['seq']}: " + click.style("MISSING in first", fg="red"))
        elif d["type"] == "missing_in_second":
            click.echo(f"Seq {d['seq']}: " + click.style("MISSING in second", fg="red"))
        click.echo()


@cli.command()
@click.argument("chain_file", type=click.Path())
def trace(chain_file: str):
    """Show step-level execution trace.

    Examples:
        lctl trace chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    engine = ReplayEngine(chain)

    trace_data = engine.get_trace()

    if not trace_data:
        click.echo("No step events found.")
        return

    click.echo(f"Execution trace for {chain.id}:")
    click.echo()

    indent = 0
    for t in trace_data:
        if t["type"] == "step_start":
            prefix = "  " * indent + "┌─"
            click.echo(f"{prefix} [{t['seq']}] {t['agent']}: {t.get('intent', 'unknown')}")
            indent += 1
        elif t["type"] == "step_end":
            indent = max(0, indent - 1)
            prefix = "  " * indent + "└─"
            outcome = t.get("outcome", "?")
            duration = t.get("duration_ms", 0) / 1000

            if outcome == "success":
                outcome_str = click.style(outcome, fg="green")
            elif outcome == "error":
                outcome_str = click.style(outcome, fg="red")
            else:
                outcome_str = outcome

            click.echo(f"{prefix} [{t['seq']}] {outcome_str} ({duration:.1f}s)")


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--port", "-p", default=8080, help="Port for web UI")
def debug(chain_file: str, port: int):
    """Launch visual debugger (web UI) for a specific chain.

    Examples:
        lctl debug chain.lctl.json
        lctl debug --port 3000 chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    click.echo("Starting LCTL Visual Debugger...")
    click.echo(f"  Chain: {chain.id}")
    click.echo(f"  Events: {len(chain.events)}")
    click.echo()
    click.echo(f"Open http://localhost:{port} in your browser")
    click.echo()

    # Start the dashboard
    try:
        from ..dashboard import run_dashboard
        run_dashboard(port=port, working_dir=Path(chain_file).parent.absolute())
    except ImportError as e:
        click.echo(click.style(f"Error: Dashboard dependencies not installed: {e}", fg="red"))
        click.echo("Install with: pip install 'lctl[dashboard]'")
        sys.exit(1)


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def evaluate(chain_file: str, as_json: bool):
    """Show comprehensive evaluation metrics for a chain.

    Provides detailed analysis including:
    - Performance metrics (duration, tokens, efficiency)
    - Quality metrics (fact confidence, error rates)
    - Agent-level breakdown
    - Bottleneck analysis

    Examples:
        lctl evaluate chain.lctl.json
        lctl evaluate --json chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    metrics = ChainMetrics.from_chain(chain)
    engine = ReplayEngine(chain)
    bottlenecks = engine.find_bottlenecks()

    if as_json:
        output = metrics.to_dict()
        output["bottlenecks"] = bottlenecks[:5]
        output["confidence_timeline"] = engine.get_confidence_timeline()
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        click.echo(click.style("Chain Evaluation Report", fg="cyan", bold=True))
        click.echo(click.style("=" * 50, fg="cyan"))
        click.echo()

        click.echo(click.style("Overview", fg="white", bold=True))
        click.echo(f"  Chain ID: {metrics.chain_id}")
        click.echo(f"  Total Events: {metrics.total_events}")
        click.echo(f"  Steps: {metrics.step_count}")
        click.echo(f"  Agents: {metrics.agent_count} ({', '.join(sorted(metrics.agents))})")
        click.echo()

        click.echo(click.style("Performance", fg="white", bold=True))
        duration_s = metrics.total_duration_ms / 1000
        click.echo(f"  Total Duration: {duration_s:.2f}s")
        click.echo(f"  Avg Step Duration: {metrics.avg_step_duration_ms:.0f}ms")
        click.echo(f"  Token Efficiency: {metrics.token_efficiency:.1f} tokens/sec")
        click.echo()

        click.echo(click.style("Token Usage", fg="white", bold=True))
        click.echo(f"  Input Tokens: {metrics.total_tokens_in:,}")
        click.echo(f"  Output Tokens: {metrics.total_tokens_out:,}")
        click.echo(f"  Total Tokens: {metrics.total_tokens:,}")
        cost = _estimate_cost(metrics.total_tokens_in, metrics.total_tokens_out)
        click.echo(f"  Est. Cost: ${cost:.4f}")
        click.echo()

        click.echo(click.style("Quality", fg="white", bold=True))
        click.echo(f"  Facts Generated: {metrics.fact_count}")
        click.echo(f"  Avg Fact Confidence: {metrics.avg_fact_confidence:.2f}")
        error_color = "red" if metrics.error_count > 0 else "green"
        click.echo("  Errors: " + click.style(str(metrics.error_count), fg=error_color))
        click.echo(f"  Error Rate: {metrics.error_rate:.2%}")
        click.echo()

        if bottlenecks:
            click.echo(click.style("Top Bottlenecks", fg="white", bold=True))
            for i, b in enumerate(bottlenecks[:3], 1):
                pct = b["percentage"]
                color = "red" if pct > 50 else ("yellow" if pct > 30 else "white")
                click.echo(
                    f"  {i}. {b['agent']} (seq {b['seq']}): "
                    + click.style(f"{b['duration_ms']/1000:.1f}s ({pct:.0f}%)", fg=color)
                )


@cli.command()
@click.argument("chain1", type=click.Path())
@click.argument("chain2", type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def compare(chain1: str, chain2: str, as_json: bool):
    """Compare two chains with statistical analysis.

    Compares performance, quality, and efficiency metrics between
    two chain executions. Shows improvement/regression indicators.

    Examples:
        lctl compare baseline.lctl.json optimized.lctl.json
        lctl compare --json v1.lctl.json v2.lctl.json
    """
    c1 = _load_chain_safely(chain1)
    if c1 is None:
        sys.exit(1)

    c2 = _load_chain_safely(chain2)
    if c2 is None:
        sys.exit(1)

    metrics_a = ChainMetrics.from_chain(c1)
    metrics_b = ChainMetrics.from_chain(c2)
    comparisons = compute_all_metrics(metrics_a, metrics_b)

    if as_json:
        output = {
            "chain_a": {"id": metrics_a.chain_id, "metrics": metrics_a.to_dict()},
            "chain_b": {"id": metrics_b.chain_id, "metrics": metrics_b.to_dict()},
            "comparisons": {k: v.to_dict() for k, v in comparisons.items()},
            "summary": {
                "improvements": sum(1 for c in comparisons.values() if c.improvement is True),
                "regressions": sum(1 for c in comparisons.values() if c.improvement is False),
                "unchanged": sum(1 for c in comparisons.values() if c.improvement is None),
            }
        }
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        click.echo(click.style("Chain Comparison Report", fg="cyan", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()

        click.echo(click.style("Chains", fg="white", bold=True))
        click.echo(f"  A: {metrics_a.chain_id} ({metrics_a.total_events} events)")
        click.echo(f"  B: {metrics_b.chain_id} ({metrics_b.total_events} events)")
        click.echo()

        click.echo(click.style("Metric Comparisons", fg="white", bold=True))
        click.echo(f"  {'Metric':<25} {'Chain A':>12} {'Chain B':>12} {'Change':>12} {'Status':>10}")
        click.echo(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

        metric_display = [
            ("Latency (ms)", metrics_a.total_duration_ms, metrics_b.total_duration_ms, comparisons["latency_diff"]),
            ("Token Efficiency", metrics_a.token_efficiency, metrics_b.token_efficiency, comparisons["token_efficiency"]),
            ("Error Rate", metrics_a.error_rate, metrics_b.error_rate, comparisons["error_rate"]),
            ("Fact Confidence", metrics_a.avg_fact_confidence, metrics_b.avg_fact_confidence, comparisons["fact_confidence_avg"]),
        ]

        for name, val_a, val_b, comp in metric_display:
            if comp.improvement is True:
                status = click.style("BETTER", fg="green")
            elif comp.improvement is False:
                status = click.style("WORSE", fg="red")
            else:
                status = click.style("SAME", fg="yellow")

            pct_str = f"{comp.percent_change:+.1f}%" if abs(comp.percent_change) < float("inf") else "N/A"
            click.echo(f"  {name:<25} {val_a:>12.2f} {val_b:>12.2f} {pct_str:>12} {status:>10}")

        click.echo()

        improvements = sum(1 for c in comparisons.values() if c.improvement is True)
        regressions = sum(1 for c in comparisons.values() if c.improvement is False)

        click.echo(click.style("Summary", fg="white", bold=True))
        click.echo("  Improvements: " + click.style(str(improvements), fg="green"))
        click.echo("  Regressions: " + click.style(str(regressions), fg="red"))

        if improvements > regressions:
            click.echo()
            click.echo(click.style("  Overall: Chain B is better", fg="green", bold=True))
        elif regressions > improvements:
            click.echo()
            click.echo(click.style("  Overall: Chain A is better", fg="red", bold=True))
        else:
            click.echo()
            click.echo(click.style("  Overall: Chains are comparable", fg="yellow", bold=True))


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--iterations", "-n", default=10, help="Number of benchmark iterations")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def benchmark(chain_file: str, iterations: int, as_json: bool):
    """Run performance benchmarks on a chain.

    Measures replay performance across multiple iterations to
    provide statistical analysis of execution characteristics.

    Examples:
        lctl benchmark chain.lctl.json
        lctl benchmark --iterations 50 chain.lctl.json
        lctl benchmark --json chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    if iterations < 1:
        click.echo(click.style("Error: iterations must be at least 1", fg="red"), err=True)
        sys.exit(1)

    replay_times: list[float] = []
    state_sizes: list[int] = []

    if not as_json:
        click.echo(f"Running {iterations} benchmark iterations...")

    for i in range(iterations):
        engine = ReplayEngine(chain)

        start_time = time.perf_counter()
        state = engine.replay_all()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        replay_times.append(elapsed_ms)
        state_sizes.append(len(state.facts))

        if not as_json and iterations > 1:
            progress = (i + 1) / iterations * 100
            if (i + 1) % max(1, iterations // 10) == 0:
                click.echo(f"  Progress: {progress:.0f}%")

    avg_time = statistics.mean(replay_times)
    min_time = min(replay_times)
    max_time = max(replay_times)
    std_dev = statistics.stdev(replay_times) if len(replay_times) > 1 else 0.0
    median_time = statistics.median(replay_times)

    events_per_ms = len(chain.events) / avg_time if avg_time > 0 else 0
    throughput = events_per_ms * 1000

    results = {
        "chain_id": chain.id,
        "events": len(chain.events),
        "iterations": iterations,
        "timing": {
            "avg_ms": round(avg_time, 3),
            "min_ms": round(min_time, 3),
            "max_ms": round(max_time, 3),
            "std_dev_ms": round(std_dev, 3),
            "median_ms": round(median_time, 3),
        },
        "throughput": {
            "events_per_second": round(throughput, 1),
        },
        "memory": {
            "avg_facts": round(statistics.mean(state_sizes), 1),
        }
    }

    if as_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo()
        click.echo(click.style("Benchmark Results", fg="cyan", bold=True))
        click.echo(click.style("=" * 40, fg="cyan"))
        click.echo()

        click.echo(click.style("Chain Info", fg="white", bold=True))
        click.echo(f"  Chain ID: {chain.id}")
        click.echo(f"  Events: {len(chain.events)}")
        click.echo(f"  Iterations: {iterations}")
        click.echo()

        click.echo(click.style("Replay Timing", fg="white", bold=True))
        click.echo(f"  Average: {avg_time:.3f}ms")
        click.echo(f"  Median: {median_time:.3f}ms")
        click.echo(f"  Min: {min_time:.3f}ms")
        click.echo(f"  Max: {max_time:.3f}ms")
        click.echo(f"  Std Dev: {std_dev:.3f}ms")
        click.echo()

        click.echo(click.style("Throughput", fg="white", bold=True))
        click.echo(f"  Events/second: {throughput:,.1f}")
        click.echo()

        if throughput > 10000:
            perf_color = "green"
            perf_label = "Excellent"
        elif throughput > 1000:
            perf_color = "green"
            perf_label = "Good"
        elif throughput > 100:
            perf_color = "yellow"
            perf_label = "Moderate"
        else:
            perf_color = "red"
            perf_label = "Needs Optimization"

        click.echo(click.style(f"Performance Rating: {perf_label}", fg=perf_color, bold=True))


@cli.command()
@click.argument("chain_file", type=click.Path())
@click.option("--format", "output_format", type=click.Choice(["json", "prometheus"]), default="json", help="Output format")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (same as --format json)")
def metrics(chain_file: str, output_format: str, as_json: bool):
    """Export metrics in various formats.

    Supports JSON and Prometheus formats for integration with
    monitoring and observability systems.

    Examples:
        lctl metrics chain.lctl.json
        lctl metrics --format prometheus chain.lctl.json
        lctl metrics --json chain.lctl.json
    """
    chain = _load_chain_safely(chain_file)
    if chain is None:
        sys.exit(1)

    chain_metrics = ChainMetrics.from_chain(chain)

    if as_json:
        output_format = "json"

    if output_format == "json":
        output = chain_metrics.to_dict()
        output["cost_estimate_usd"] = _estimate_cost(
            chain_metrics.total_tokens_in,
            chain_metrics.total_tokens_out
        )
        click.echo(json.dumps(output, indent=2, default=str))

    elif output_format == "prometheus":
        chain_id = chain_metrics.chain_id.replace("-", "_").replace(".", "_")

        prometheus_lines = [
            '# HELP lctl_chain_events_total Total number of events in chain',
            '# TYPE lctl_chain_events_total gauge',
            f'lctl_chain_events_total{{chain_id="{chain_id}"}} {chain_metrics.total_events}',
            '',
            '# HELP lctl_chain_duration_ms Total duration in milliseconds',
            '# TYPE lctl_chain_duration_ms gauge',
            f'lctl_chain_duration_ms{{chain_id="{chain_id}"}} {chain_metrics.total_duration_ms}',
            '',
            '# HELP lctl_chain_tokens_total Total tokens used',
            '# TYPE lctl_chain_tokens_total gauge',
            f'lctl_chain_tokens_total{{chain_id="{chain_id}",type="input"}} {chain_metrics.total_tokens_in}',
            f'lctl_chain_tokens_total{{chain_id="{chain_id}",type="output"}} {chain_metrics.total_tokens_out}',
            '',
            '# HELP lctl_chain_errors_total Total number of errors',
            '# TYPE lctl_chain_errors_total gauge',
            f'lctl_chain_errors_total{{chain_id="{chain_id}"}} {chain_metrics.error_count}',
            '',
            '# HELP lctl_chain_facts_total Total number of facts',
            '# TYPE lctl_chain_facts_total gauge',
            f'lctl_chain_facts_total{{chain_id="{chain_id}"}} {chain_metrics.fact_count}',
            '',
            '# HELP lctl_chain_steps_total Total number of steps',
            '# TYPE lctl_chain_steps_total gauge',
            f'lctl_chain_steps_total{{chain_id="{chain_id}"}} {chain_metrics.step_count}',
            '',
            '# HELP lctl_chain_agents_total Total number of agents',
            '# TYPE lctl_chain_agents_total gauge',
            f'lctl_chain_agents_total{{chain_id="{chain_id}"}} {chain_metrics.agent_count}',
            '',
            '# HELP lctl_chain_avg_step_duration_ms Average step duration in milliseconds',
            '# TYPE lctl_chain_avg_step_duration_ms gauge',
            f'lctl_chain_avg_step_duration_ms{{chain_id="{chain_id}"}} {chain_metrics.avg_step_duration_ms:.2f}',
            '',
            '# HELP lctl_chain_avg_fact_confidence Average fact confidence',
            '# TYPE lctl_chain_avg_fact_confidence gauge',
            f'lctl_chain_avg_fact_confidence{{chain_id="{chain_id}"}} {chain_metrics.avg_fact_confidence:.4f}',
            '',
            '# HELP lctl_chain_token_efficiency Tokens per second',
            '# TYPE lctl_chain_token_efficiency gauge',
            f'lctl_chain_token_efficiency{{chain_id="{chain_id}"}} {chain_metrics.token_efficiency:.2f}',
            '',
            '# HELP lctl_chain_error_rate Error rate (errors/events)',
            '# TYPE lctl_chain_error_rate gauge',
            f'lctl_chain_error_rate{{chain_id="{chain_id}"}} {chain_metrics.error_rate:.6f}',
        ]

        click.echo('\n'.join(prometheus_lines))


@cli.command()
@click.option("--port", "-p", default=8080, help="Port to run dashboard on")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--dir", "-d", "working_dir", type=click.Path(exists=True), help="Directory containing .lctl.json files")
def dashboard(port: int, host: str, working_dir: Optional[str]):
    """Launch the web dashboard for visualizing LCTL chains.

    The dashboard provides:
    - Timeline view of events
    - Agent swim lanes
    - Fact registry with confidence indicators
    - Time-travel slider to replay
    - Bottleneck highlighting
    - Error indicators

    Examples:
        lctl dashboard
        lctl dashboard --port 3000
        lctl dashboard --dir ./traces
    """
    click.echo(click.style("LCTL Dashboard", fg="cyan", bold=True))
    click.echo()

    work_path = Path(working_dir) if working_dir else Path.cwd()

    # Count available chains
    chain_count = len(list(work_path.glob("*.lctl.json"))) + len(list(work_path.glob("*.lctl.yaml")))

    click.echo(f"  Working directory: {work_path}")
    click.echo(f"  Available chains: {chain_count}")
    click.echo()
    click.echo("  Dashboard URL: " + click.style(f"http://{host}:{port}", fg="green", bold=True))
    click.echo()
    click.echo("Press Ctrl+C to stop the server")
    click.echo()

    try:
        from ..dashboard import run_dashboard
        run_dashboard(host=host, port=port, working_dir=work_path)
    except ImportError as e:
        click.echo(click.style(f"Error: Dashboard dependencies not installed: {e}", fg="red"))
        click.echo()
        click.echo("Install dashboard dependencies with:")
        click.echo("  pip install 'lctl[dashboard]'")
        click.echo()
        click.echo("Or install manually:")
        click.echo("  pip install fastapi uvicorn")
        sys.exit(1)


def _print_event(event: Event) -> None:
    """Print a single event."""
    type_colors = {
        "step_start": "cyan",
        "step_end": "cyan",
        "fact_added": "green",
        "fact_modified": "yellow",
        "tool_call": "blue",
        "error": "red",
        "checkpoint": "magenta"
    }

    etype = event.type.value if hasattr(event.type, 'value') else str(event.type)
    color = type_colors.get(etype, "white")

    click.echo(f"[{event.seq}] " + click.style(etype, fg=color) + f" ({event.agent})")

    # Show key data - handle None values safely
    if etype == "fact_added":
        fact_id = event.data.get('id', 'unknown')
        text = event.data.get('text', '') or ''
        display_text = text[:50] + "..." if len(text) > 50 else text
        click.echo(f"     {fact_id}: {display_text}")
    elif etype == "error":
        category = event.data.get('category', 'unknown')
        error_type = event.data.get('type', 'unknown')
        message = event.data.get('message', '') or ''
        display_msg = message[:50] + "..." if len(message) > 50 else message
        click.echo(f"     {category}/{error_type}: {display_msg}")
    elif etype == "tool_call":
        tool = event.data.get('tool', 'unknown')
        duration = event.data.get('duration_ms', 0) or 0
        click.echo(f"     {tool}: {duration}ms")


def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    """Estimate cost based on typical Claude pricing.

    Note: This is a rough estimate using Claude 3 Opus pricing.
    Actual costs vary by model and pricing tier.
    """
    # Rough estimate: $3/1M input, $15/1M output (Claude 3 Opus pricing)
    # Handle None or negative values gracefully
    tokens_in = max(0, tokens_in or 0)
    tokens_out = max(0, tokens_out or 0)

    cost_in = (tokens_in / 1_000_000) * 3
    cost_out = (tokens_out / 1_000_000) * 15
    return cost_in + cost_out


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
