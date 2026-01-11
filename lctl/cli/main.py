"""LCTL CLI - Time-travel debugging for multi-agent LLM workflows."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from ..core.events import Chain, Event, ReplayEngine, State


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

    click.echo(f"Starting LCTL Visual Debugger...")
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
    click.echo(f"  Dashboard URL: " + click.style(f"http://{host}:{port}", fg="green", bold=True))
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
