"""LCTL CLI - Time-travel debugging for multi-agent LLM workflows."""

import click
import json
from pathlib import Path
from typing import Optional

from ..core.events import Chain, ReplayEngine, State


@click.group()
@click.version_option(version="4.0.0", prog_name="lctl")
def cli():
    """LCTL - Time-travel debugging for multi-agent workflows.

    The killer feature: replay agent execution to any point in time.
    """
    pass


@cli.command()
@click.argument("chain_file", type=click.Path(exists=True))
@click.option("--to-seq", "-s", type=int, help="Replay to specific sequence number")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed event output")
def replay(chain_file: str, to_seq: Optional[int], verbose: bool):
    """Replay agent execution (time-travel debugging).

    Examples:
        lctl replay chain.lctl.json
        lctl replay --to-seq 10 chain.lctl.json
    """
    chain = Chain.load(Path(chain_file))
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
            click.echo(f"  {fid}: {fact['text'][:60]}... (conf: {conf:.2f})")


@cli.command()
@click.argument("chain_file", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(chain_file: str, as_json: bool):
    """Show performance statistics for a chain.

    Examples:
        lctl stats chain.lctl.json
        lctl stats --json chain.lctl.json
    """
    chain = Chain.load(Path(chain_file))
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
@click.argument("chain_file", type=click.Path(exists=True))
@click.option("--top", "-n", default=5, help="Show top N bottlenecks")
def bottleneck(chain_file: str, top: int):
    """Analyze performance bottlenecks.

    Examples:
        lctl bottleneck chain.lctl.json
        lctl bottleneck --top 3 chain.lctl.json
    """
    chain = Chain.load(Path(chain_file))
    engine = ReplayEngine(chain)

    bottlenecks = engine.find_bottlenecks()[:top]

    if not bottlenecks:
        click.echo("No step timing data found.")
        return

    click.echo("Slowest steps:")
    for i, step in enumerate(bottlenecks, 1):
        duration_s = step["duration_ms"] / 1000
        click.echo(f"  {i}. {step['agent']} (seq {step['seq']}): {duration_s:.1f}s ({step['percentage']:.0f}%)")

    # Recommendations
    click.echo()
    if bottlenecks[0]["percentage"] > 50:
        click.echo(f"Recommendation: {bottlenecks[0]['agent']} is a major bottleneck ({bottlenecks[0]['percentage']:.0f}% of time).")
        click.echo("Consider: parallelization, caching, or faster model.")


@cli.command()
@click.argument("chain_file", type=click.Path(exists=True))
def confidence(chain_file: str):
    """Show confidence timeline for facts.

    Examples:
        lctl confidence chain.lctl.json
    """
    chain = Chain.load(Path(chain_file))
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
@click.argument("chain1", type=click.Path(exists=True))
@click.argument("chain2", type=click.Path(exists=True))
def diff_chains(chain1: str, chain2: str):
    """Compare two chains and find divergence.

    Examples:
        lctl diff chain-v1.lctl.json chain-v2.lctl.json
    """
    c1 = Chain.load(Path(chain1))
    c2 = Chain.load(Path(chain2))

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
@click.argument("chain_file", type=click.Path(exists=True))
def trace(chain_file: str):
    """Show step-level execution trace.

    Examples:
        lctl trace chain.lctl.json
    """
    chain = Chain.load(Path(chain_file))
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
@click.argument("chain_file", type=click.Path(exists=True))
@click.option("--port", "-p", default=8080, help="Port for web UI")
def debug(chain_file: str, port: int):
    """Launch visual debugger (web UI).

    Examples:
        lctl debug chain.lctl.json
        lctl debug --port 3000 chain.lctl.json
    """
    chain = Chain.load(Path(chain_file))

    click.echo(f"Starting LCTL Visual Debugger...")
    click.echo(f"  Chain: {chain.id}")
    click.echo(f"  Events: {len(chain.events)}")
    click.echo()
    click.echo(f"Open http://localhost:{port} in your browser")
    click.echo()
    click.echo(click.style("Note: Web UI not yet implemented. Use CLI commands for now:", fg="yellow"))
    click.echo("  lctl replay - Time-travel to any point")
    click.echo("  lctl trace  - View execution flow")
    click.echo("  lctl stats  - Performance metrics")


def _print_event(event):
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

    etype = event.type.value if hasattr(event.type, 'value') else event.type
    color = type_colors.get(etype, "white")

    click.echo(f"[{event.seq}] " + click.style(etype, fg=color) + f" ({event.agent})")

    # Show key data
    if etype == "fact_added":
        click.echo(f"     {event.data.get('id')}: {event.data.get('text', '')[:50]}...")
    elif etype == "error":
        click.echo(f"     {event.data.get('category')}/{event.data.get('type')}: {event.data.get('message', '')[:50]}")
    elif etype == "tool_call":
        click.echo(f"     {event.data.get('tool')}: {event.data.get('duration_ms', 0)}ms")


def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    """Estimate cost based on typical Claude pricing."""
    # Rough estimate: $3/1M input, $15/1M output (Claude 3 Opus pricing)
    cost_in = (tokens_in / 1_000_000) * 3
    cost_out = (tokens_out / 1_000_000) * 15
    return cost_in + cost_out


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
