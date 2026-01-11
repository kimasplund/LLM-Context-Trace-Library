"""Tests for LCTL CLI commands (lctl/cli/main.py)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from lctl.cli.main import cli, _estimate_cost
from lctl.core.events import Chain, Event, EventType
from lctl.evaluation.metrics import ChainMetrics


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli_chain_file(tmp_path: Path, base_timestamp: datetime) -> Path:
    """Create a chain file for CLI testing."""
    chain = Chain(id="cli-test-chain")
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
            data={"id": "F1", "text": "Important finding with sufficient text for display", "confidence": 0.9}
        ),
        Event(
            seq=3,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(seconds=2),
            agent="planner",
            data={
                "outcome": "success",
                "output_summary": "plan complete",
                "duration_ms": 2000,
                "tokens": {"input": 500, "output": 200}
            }
        ),
        Event(
            seq=4,
            type=EventType.STEP_START,
            timestamp=base_timestamp + timedelta(seconds=2, milliseconds=100),
            agent="executor",
            data={"intent": "execute", "input_summary": "plan"}
        ),
        Event(
            seq=5,
            type=EventType.TOOL_CALL,
            timestamp=base_timestamp + timedelta(seconds=2, milliseconds=500),
            agent="executor",
            data={"tool": "code_gen", "input": "spec", "output": "code", "duration_ms": 300}
        ),
        Event(
            seq=6,
            type=EventType.FACT_MODIFIED,
            timestamp=base_timestamp + timedelta(seconds=3),
            agent="executor",
            data={"id": "F1", "confidence": 0.95, "reason": "verified"}
        ),
        Event(
            seq=7,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(seconds=5),
            agent="executor",
            data={
                "outcome": "success",
                "output_summary": "executed",
                "duration_ms": 3000,
                "tokens": {"input": 1000, "output": 500}
            }
        ),
    ]
    for event in events:
        chain.add_event(event)

    file_path = tmp_path / "cli_test.lctl.json"
    chain.save(file_path)
    return file_path


@pytest.fixture
def empty_cli_chain_file(tmp_path: Path) -> Path:
    """Create an empty chain file for CLI testing."""
    chain = Chain(id="empty-cli-chain")
    file_path = tmp_path / "empty_cli.lctl.json"
    chain.save(file_path)
    return file_path


@pytest.fixture
def error_chain_file(tmp_path: Path, base_timestamp: datetime) -> Path:
    """Create a chain file with errors for CLI testing."""
    chain = Chain(id="error-chain")
    events = [
        Event(
            seq=1,
            type=EventType.STEP_START,
            timestamp=base_timestamp,
            agent="validator",
            data={"intent": "validate"}
        ),
        Event(
            seq=2,
            type=EventType.ERROR,
            timestamp=base_timestamp + timedelta(milliseconds=100),
            agent="validator",
            data={
                "category": "validation",
                "type": "ValueError",
                "message": "Invalid input data format"
            }
        ),
        Event(
            seq=3,
            type=EventType.STEP_END,
            timestamp=base_timestamp + timedelta(milliseconds=200),
            agent="validator",
            data={"outcome": "error", "duration_ms": 200, "tokens": {"input": 50, "output": 10}}
        ),
    ]
    for event in events:
        chain.add_event(event)

    file_path = tmp_path / "error_chain.lctl.json"
    chain.save(file_path)
    return file_path


class TestCliVersion:
    """Tests for CLI version command."""

    def test_version_option(self, runner: CliRunner):
        """Test --version option."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "4.0.0" in result.output


class TestReplayCommand:
    """Tests for replay command."""

    def test_replay_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic replay command."""
        result = runner.invoke(cli, ["replay", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Replaying 7 events" in result.output
        assert "State at seq 7:" in result.output
        assert "Facts: 1" in result.output

    def test_replay_to_specific_seq(self, runner: CliRunner, cli_chain_file: Path):
        """Test replay to specific sequence number."""
        result = runner.invoke(cli, ["replay", "--to-seq", "3", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "to seq 3" in result.output
        assert "State at seq 3:" in result.output

    def test_replay_verbose(self, runner: CliRunner, cli_chain_file: Path):
        """Test replay with verbose output."""
        result = runner.invoke(cli, ["replay", "-v", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "step_start" in result.output
        assert "fact_added" in result.output
        assert "planner" in result.output

    def test_replay_shows_facts(self, runner: CliRunner, cli_chain_file: Path):
        """Test replay shows facts in output."""
        result = runner.invoke(cli, ["replay", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Facts:" in result.output
        assert "F1:" in result.output
        assert "conf:" in result.output

    def test_replay_empty_chain(self, runner: CliRunner, empty_cli_chain_file: Path):
        """Test replay on empty chain."""
        result = runner.invoke(cli, ["replay", str(empty_cli_chain_file)])

        assert result.exit_code == 0
        assert "No events in chain" in result.output

    def test_replay_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test replay with nonexistent file."""
        result = runner.invoke(cli, ["replay", str(tmp_path / "nonexistent.json")])

        assert result.exit_code != 0


class TestStatsCommand:
    """Tests for stats command."""

    def test_stats_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic stats command."""
        result = runner.invoke(cli, ["stats", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Chain: cli-test-chain" in result.output
        assert "Version: 4.0" in result.output
        assert "Events: 7" in result.output
        assert "Agents:" in result.output
        assert "Facts: 1" in result.output
        assert "Duration:" in result.output
        assert "Tokens:" in result.output

    def test_stats_json_output(self, runner: CliRunner, cli_chain_file: Path):
        """Test stats with JSON output."""
        result = runner.invoke(cli, ["stats", "--json", str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["chain_id"] == "cli-test-chain"
        assert data["version"] == "4.0"
        assert data["events"] == 7
        assert "tokens" in data
        assert "estimated_cost_usd" in data

    def test_stats_shows_agents(self, runner: CliRunner, cli_chain_file: Path):
        """Test stats shows agent list."""
        result = runner.invoke(cli, ["stats", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "executor" in result.output
        assert "planner" in result.output

    def test_stats_with_errors(self, runner: CliRunner, error_chain_file: Path):
        """Test stats shows error count."""
        result = runner.invoke(cli, ["stats", str(error_chain_file)])

        assert result.exit_code == 0
        assert "Errors: 1" in result.output


class TestBottleneckCommand:
    """Tests for bottleneck command."""

    def test_bottleneck_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic bottleneck command."""
        result = runner.invoke(cli, ["bottleneck", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Slowest steps:" in result.output

    def test_bottleneck_shows_duration(self, runner: CliRunner, cli_chain_file: Path):
        """Test bottleneck shows duration."""
        result = runner.invoke(cli, ["bottleneck", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "s (" in result.output
        assert "%)" in result.output

    def test_bottleneck_top_n(self, runner: CliRunner, cli_chain_file: Path):
        """Test bottleneck with --top option."""
        result = runner.invoke(cli, ["bottleneck", "--top", "1", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "1." in result.output
        assert "2." not in result.output

    def test_bottleneck_recommendation(self, runner: CliRunner, cli_chain_file: Path):
        """Test bottleneck shows recommendations for major bottlenecks."""
        result = runner.invoke(cli, ["bottleneck", str(cli_chain_file)])

        assert result.exit_code == 0
        if "major bottleneck" in result.output.lower() or "Recommendation" in result.output:
            assert "Consider:" in result.output or "parallelization" in result.output

    def test_bottleneck_empty_chain(self, runner: CliRunner, empty_cli_chain_file: Path):
        """Test bottleneck on empty chain."""
        result = runner.invoke(cli, ["bottleneck", str(empty_cli_chain_file)])

        assert result.exit_code == 0
        assert "No step timing data found" in result.output


class TestConfidenceCommand:
    """Tests for confidence command."""

    def test_confidence_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic confidence command."""
        result = runner.invoke(cli, ["confidence", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Fact confidence timeline:" in result.output
        assert "F1:" in result.output

    def test_confidence_shows_timeline(self, runner: CliRunner, cli_chain_file: Path):
        """Test confidence shows change timeline."""
        result = runner.invoke(cli, ["confidence", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "0.90" in result.output or "0.9" in result.output
        assert "0.95" in result.output

    def test_confidence_empty_chain(self, runner: CliRunner, empty_cli_chain_file: Path):
        """Test confidence on empty chain."""
        result = runner.invoke(cli, ["confidence", str(empty_cli_chain_file)])

        assert result.exit_code == 0
        assert "No facts found" in result.output


class TestDiffCommand:
    """Tests for diff command."""

    def test_diff_identical_chains(self, runner: CliRunner, cli_chain_file: Path):
        """Test diff of identical chains."""
        result = runner.invoke(cli, ["diff", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        assert "identical" in result.output.lower()

    def test_diff_different_chains(
        self, runner: CliRunner, cli_chain_file: Path, error_chain_file: Path
    ):
        """Test diff of different chains."""
        result = runner.invoke(cli, ["diff", str(cli_chain_file), str(error_chain_file)])

        assert result.exit_code == 0
        assert "difference" in result.output.lower() or "DIVERGED" in result.output

    def test_diff_shows_divergence_type(
        self, runner: CliRunner, cli_chain_file: Path, empty_cli_chain_file: Path
    ):
        """Test diff shows divergence types."""
        result = runner.invoke(cli, ["diff", str(cli_chain_file), str(empty_cli_chain_file)])

        assert result.exit_code == 0
        assert "MISSING" in result.output or "difference" in result.output


class TestTraceCommand:
    """Tests for trace command."""

    def test_trace_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic trace command."""
        result = runner.invoke(cli, ["trace", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Execution trace" in result.output
        assert "cli-test-chain" in result.output

    def test_trace_shows_steps(self, runner: CliRunner, cli_chain_file: Path):
        """Test trace shows step events."""
        result = runner.invoke(cli, ["trace", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "planner" in result.output
        assert "executor" in result.output

    def test_trace_shows_intents(self, runner: CliRunner, cli_chain_file: Path):
        """Test trace shows step intents."""
        result = runner.invoke(cli, ["trace", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "plan" in result.output
        assert "execute" in result.output

    def test_trace_shows_outcomes(self, runner: CliRunner, cli_chain_file: Path):
        """Test trace shows step outcomes."""
        result = runner.invoke(cli, ["trace", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "success" in result.output.lower()

    def test_trace_shows_error_outcome(self, runner: CliRunner, error_chain_file: Path):
        """Test trace shows error outcomes."""
        result = runner.invoke(cli, ["trace", str(error_chain_file)])

        assert result.exit_code == 0
        assert "error" in result.output.lower()

    def test_trace_empty_chain(self, runner: CliRunner, empty_cli_chain_file: Path):
        """Test trace on empty chain."""
        result = runner.invoke(cli, ["trace", str(empty_cli_chain_file)])

        assert result.exit_code == 0
        assert "No step events found" in result.output


class TestDebugCommand:
    """Tests for debug command."""

    def test_debug_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic debug command shows chain info before starting dashboard."""
        # Use catch_exceptions=False to see actual errors if any
        from unittest.mock import patch, MagicMock

        # Mock the dashboard to prevent actual server startup
        mock_run_dashboard = MagicMock()
        with patch.dict('sys.modules', {'lctl.dashboard': MagicMock(run_dashboard=mock_run_dashboard)}):
            with patch('lctl.cli.main.run_dashboard', mock_run_dashboard, create=True):
                result = runner.invoke(cli, ["debug", str(cli_chain_file)])

        # Either succeeds with mocked dashboard or exits with error message about dashboard
        assert "Visual Debugger" in result.output
        assert "cli-test-chain" in result.output

    def test_debug_custom_port(self, runner: CliRunner, cli_chain_file: Path):
        """Test debug with custom port."""
        from unittest.mock import patch, MagicMock

        mock_run_dashboard = MagicMock()
        with patch.dict('sys.modules', {'lctl.dashboard': MagicMock(run_dashboard=mock_run_dashboard)}):
            with patch('lctl.cli.main.run_dashboard', mock_run_dashboard, create=True):
                result = runner.invoke(cli, ["debug", "--port", "3000", str(cli_chain_file)])

        assert "3000" in result.output

    def test_debug_shows_event_count(self, runner: CliRunner, cli_chain_file: Path):
        """Test debug shows event count."""
        from unittest.mock import patch, MagicMock

        mock_run_dashboard = MagicMock()
        with patch.dict('sys.modules', {'lctl.dashboard': MagicMock(run_dashboard=mock_run_dashboard)}):
            with patch('lctl.cli.main.run_dashboard', mock_run_dashboard, create=True):
                result = runner.invoke(cli, ["debug", str(cli_chain_file)])

        assert "Events: 7" in result.output

    def test_debug_with_missing_dashboard(self, runner: CliRunner, cli_chain_file: Path):
        """Test debug shows helpful error when dashboard dependencies are missing."""
        from unittest.mock import patch

        # Simulate ImportError when dashboard is not available
        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'fastapi'")

        with patch.dict('sys.modules', {'lctl.dashboard': None}):
            result = runner.invoke(cli, ["debug", str(cli_chain_file)])

        # Should show info about the chain even if dashboard fails
        assert "Visual Debugger" in result.output or "Dashboard" in result.output


class TestEstimateCost:
    """Tests for cost estimation helper function."""

    def test_estimate_cost_zero(self):
        """Test cost estimation with zero tokens."""
        cost = _estimate_cost(0, 0)
        assert cost == 0.0

    def test_estimate_cost_input_only(self):
        """Test cost estimation with input tokens only."""
        cost = _estimate_cost(1_000_000, 0)
        assert cost == pytest.approx(3.0, rel=0.01)

    def test_estimate_cost_output_only(self):
        """Test cost estimation with output tokens only."""
        cost = _estimate_cost(0, 1_000_000)
        assert cost == pytest.approx(15.0, rel=0.01)

    def test_estimate_cost_mixed(self):
        """Test cost estimation with both input and output tokens."""
        cost = _estimate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0, rel=0.01)

    def test_estimate_cost_small_values(self):
        """Test cost estimation with small token counts."""
        cost = _estimate_cost(1000, 500)
        assert cost > 0
        assert cost < 0.02


class TestCliEdgeCases:
    """Edge case tests for CLI."""

    def test_help_output(self, runner: CliRunner):
        """Test help output is displayed."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "LCTL" in result.output
        assert "replay" in result.output
        assert "stats" in result.output
        assert "bottleneck" in result.output

    def test_command_help(self, runner: CliRunner):
        """Test individual command help."""
        result = runner.invoke(cli, ["replay", "--help"])

        assert result.exit_code == 0
        assert "Replay" in result.output
        assert "--to-seq" in result.output

    def test_yaml_file_input(self, runner: CliRunner, sample_chain: Chain, tmp_path: Path):
        """Test CLI commands work with YAML files."""
        yaml_path = tmp_path / "test.lctl.yaml"
        sample_chain.save(yaml_path)

        result = runner.invoke(cli, ["stats", str(yaml_path)])

        assert result.exit_code == 0
        assert "test-chain" in result.output


class TestCliIntegration:
    """Integration tests for CLI commands."""

    def test_full_analysis_workflow(self, runner: CliRunner, cli_chain_file: Path):
        """Test full analysis workflow using multiple commands."""
        stats_result = runner.invoke(cli, ["stats", str(cli_chain_file)])
        assert stats_result.exit_code == 0
        assert "cli-test-chain" in stats_result.output

        trace_result = runner.invoke(cli, ["trace", str(cli_chain_file)])
        assert trace_result.exit_code == 0
        assert "planner" in trace_result.output

        bottleneck_result = runner.invoke(cli, ["bottleneck", str(cli_chain_file)])
        assert bottleneck_result.exit_code == 0
        assert "Slowest steps" in bottleneck_result.output

        confidence_result = runner.invoke(cli, ["confidence", str(cli_chain_file)])
        assert confidence_result.exit_code == 0
        assert "F1" in confidence_result.output

    def test_replay_workflow(self, runner: CliRunner, cli_chain_file: Path):
        """Test replay workflow with different sequences."""
        early_result = runner.invoke(cli, ["replay", "--to-seq", "2", str(cli_chain_file)])
        assert early_result.exit_code == 0
        assert "seq 2" in early_result.output

        mid_result = runner.invoke(cli, ["replay", "--to-seq", "4", str(cli_chain_file)])
        assert mid_result.exit_code == 0
        assert "seq 4" in mid_result.output

        full_result = runner.invoke(cli, ["replay", str(cli_chain_file)])
        assert full_result.exit_code == 0
        assert "seq 7" in full_result.output


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic evaluate command."""
        result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Chain Evaluation Report" in result.output
        assert "Overview" in result.output
        assert "Performance" in result.output
        assert "Quality" in result.output

    def test_evaluate_shows_chain_id(self, runner: CliRunner, cli_chain_file: Path):
        """Test evaluate shows chain ID."""
        result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "cli-test-chain" in result.output

    def test_evaluate_shows_metrics(self, runner: CliRunner, cli_chain_file: Path):
        """Test evaluate shows comprehensive metrics."""
        result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Total Events:" in result.output
        assert "Steps:" in result.output
        assert "Agents:" in result.output
        assert "Total Duration:" in result.output
        assert "Token Efficiency:" in result.output

    def test_evaluate_shows_token_usage(self, runner: CliRunner, cli_chain_file: Path):
        """Test evaluate shows token usage."""
        result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Input Tokens:" in result.output
        assert "Output Tokens:" in result.output
        assert "Total Tokens:" in result.output
        assert "Est. Cost:" in result.output

    def test_evaluate_shows_quality_metrics(self, runner: CliRunner, cli_chain_file: Path):
        """Test evaluate shows quality metrics."""
        result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Facts Generated:" in result.output
        assert "Avg Fact Confidence:" in result.output
        assert "Error Rate:" in result.output

    def test_evaluate_shows_bottlenecks(self, runner: CliRunner, cli_chain_file: Path):
        """Test evaluate shows bottleneck analysis."""
        result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Bottleneck" in result.output

    def test_evaluate_json_output(self, runner: CliRunner, cli_chain_file: Path):
        """Test evaluate with JSON output."""
        result = runner.invoke(cli, ["evaluate", "--json", str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["chain_id"] == "cli-test-chain"
        assert "total_events" in data
        assert "total_duration_ms" in data
        assert "total_tokens" in data
        assert "error_rate" in data
        assert "bottlenecks" in data
        assert "confidence_timeline" in data

    def test_evaluate_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test evaluate with nonexistent file."""
        result = runner.invoke(cli, ["evaluate", str(tmp_path / "nonexistent.json")])

        assert result.exit_code != 0

    def test_evaluate_empty_chain(self, runner: CliRunner, empty_cli_chain_file: Path):
        """Test evaluate on empty chain."""
        result = runner.invoke(cli, ["evaluate", str(empty_cli_chain_file)])

        assert result.exit_code == 0
        assert "Total Events: 0" in result.output


class TestCompareCommand:
    """Tests for compare command."""

    def test_compare_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic compare command."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Chain Comparison Report" in result.output
        assert "Chains" in result.output
        assert "Metric Comparisons" in result.output

    def test_compare_shows_both_chains(self, runner: CliRunner, cli_chain_file: Path):
        """Test compare shows both chain IDs."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        assert "cli-test-chain" in result.output

    def test_compare_shows_metrics_table(self, runner: CliRunner, cli_chain_file: Path):
        """Test compare shows metrics comparison table."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Latency" in result.output
        assert "Token Efficiency" in result.output
        assert "Error Rate" in result.output
        assert "Fact Confidence" in result.output

    def test_compare_shows_status_indicators(self, runner: CliRunner, cli_chain_file: Path):
        """Test compare shows status indicators."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        assert "SAME" in result.output or "BETTER" in result.output or "WORSE" in result.output

    def test_compare_shows_summary(self, runner: CliRunner, cli_chain_file: Path):
        """Test compare shows summary."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Summary" in result.output
        assert "Improvements:" in result.output
        assert "Regressions:" in result.output

    def test_compare_different_chains(
        self, runner: CliRunner, cli_chain_file: Path, error_chain_file: Path
    ):
        """Test compare with different chains."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(error_chain_file)])

        assert result.exit_code == 0
        assert "Chain Comparison Report" in result.output

    def test_compare_json_output(self, runner: CliRunner, cli_chain_file: Path):
        """Test compare with JSON output."""
        result = runner.invoke(cli, ["compare", "--json", str(cli_chain_file), str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "chain_a" in data
        assert "chain_b" in data
        assert "comparisons" in data
        assert "summary" in data
        assert "improvements" in data["summary"]
        assert "regressions" in data["summary"]

    def test_compare_nonexistent_first_file(self, runner: CliRunner, cli_chain_file: Path, tmp_path: Path):
        """Test compare with nonexistent first file."""
        result = runner.invoke(cli, ["compare", str(tmp_path / "nonexistent.json"), str(cli_chain_file)])

        assert result.exit_code != 0

    def test_compare_nonexistent_second_file(self, runner: CliRunner, cli_chain_file: Path, tmp_path: Path):
        """Test compare with nonexistent second file."""
        result = runner.invoke(cli, ["compare", str(cli_chain_file), str(tmp_path / "nonexistent.json")])

        assert result.exit_code != 0


class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic benchmark command."""
        result = runner.invoke(cli, ["benchmark", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Benchmark Results" in result.output

    def test_benchmark_shows_chain_info(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark shows chain info."""
        result = runner.invoke(cli, ["benchmark", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Chain ID:" in result.output
        assert "Events:" in result.output
        assert "Iterations:" in result.output

    def test_benchmark_shows_timing(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark shows timing stats."""
        result = runner.invoke(cli, ["benchmark", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Replay Timing" in result.output
        assert "Average:" in result.output
        assert "Median:" in result.output
        assert "Min:" in result.output
        assert "Max:" in result.output

    def test_benchmark_shows_throughput(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark shows throughput."""
        result = runner.invoke(cli, ["benchmark", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Throughput" in result.output
        assert "Events/second:" in result.output

    def test_benchmark_shows_rating(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark shows performance rating."""
        result = runner.invoke(cli, ["benchmark", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Performance Rating:" in result.output

    def test_benchmark_custom_iterations(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark with custom iteration count."""
        result = runner.invoke(cli, ["benchmark", "--iterations", "5", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Iterations: 5" in result.output

    def test_benchmark_json_output(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark with JSON output."""
        result = runner.invoke(cli, ["benchmark", "--json", str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["chain_id"] == "cli-test-chain"
        assert "timing" in data
        assert "avg_ms" in data["timing"]
        assert "min_ms" in data["timing"]
        assert "max_ms" in data["timing"]
        assert "throughput" in data
        assert "events_per_second" in data["throughput"]

    def test_benchmark_single_iteration(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark with single iteration."""
        result = runner.invoke(cli, ["benchmark", "-n", "1", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "Iterations: 1" in result.output

    def test_benchmark_invalid_iterations(self, runner: CliRunner, cli_chain_file: Path):
        """Test benchmark with invalid iteration count."""
        result = runner.invoke(cli, ["benchmark", "--iterations", "0", str(cli_chain_file)])

        assert result.exit_code != 0

    def test_benchmark_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test benchmark with nonexistent file."""
        result = runner.invoke(cli, ["benchmark", str(tmp_path / "nonexistent.json")])

        assert result.exit_code != 0


class TestMetricsCommand:
    """Tests for metrics command."""

    def test_metrics_basic(self, runner: CliRunner, cli_chain_file: Path):
        """Test basic metrics command (JSON by default)."""
        result = runner.invoke(cli, ["metrics", str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "chain_id" in data
        assert "total_events" in data

    def test_metrics_json_format(self, runner: CliRunner, cli_chain_file: Path):
        """Test metrics with explicit JSON format."""
        result = runner.invoke(cli, ["metrics", "--format", "json", str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["chain_id"] == "cli-test-chain"
        assert data["total_events"] == 7
        assert "total_duration_ms" in data
        assert "total_tokens_in" in data
        assert "total_tokens_out" in data
        assert "cost_estimate_usd" in data

    def test_metrics_json_flag(self, runner: CliRunner, cli_chain_file: Path):
        """Test metrics with --json flag."""
        result = runner.invoke(cli, ["metrics", "--json", str(cli_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "chain_id" in data

    def test_metrics_prometheus_format(self, runner: CliRunner, cli_chain_file: Path):
        """Test metrics with Prometheus format."""
        result = runner.invoke(cli, ["metrics", "--format", "prometheus", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "# HELP lctl_chain_events_total" in result.output
        assert "# TYPE lctl_chain_events_total gauge" in result.output
        assert "lctl_chain_events_total" in result.output

    def test_metrics_prometheus_all_metrics(self, runner: CliRunner, cli_chain_file: Path):
        """Test Prometheus format includes all metrics."""
        result = runner.invoke(cli, ["metrics", "--format", "prometheus", str(cli_chain_file)])

        assert result.exit_code == 0
        assert "lctl_chain_duration_ms" in result.output
        assert "lctl_chain_tokens_total" in result.output
        assert "lctl_chain_errors_total" in result.output
        assert "lctl_chain_facts_total" in result.output
        assert "lctl_chain_steps_total" in result.output
        assert "lctl_chain_agents_total" in result.output
        assert "lctl_chain_avg_step_duration_ms" in result.output
        assert "lctl_chain_avg_fact_confidence" in result.output
        assert "lctl_chain_token_efficiency" in result.output
        assert "lctl_chain_error_rate" in result.output

    def test_metrics_prometheus_labels(self, runner: CliRunner, cli_chain_file: Path):
        """Test Prometheus format uses proper labels."""
        result = runner.invoke(cli, ["metrics", "--format", "prometheus", str(cli_chain_file)])

        assert result.exit_code == 0
        assert 'chain_id="cli_test_chain"' in result.output

    def test_metrics_prometheus_token_types(self, runner: CliRunner, cli_chain_file: Path):
        """Test Prometheus format distinguishes token types."""
        result = runner.invoke(cli, ["metrics", "--format", "prometheus", str(cli_chain_file)])

        assert result.exit_code == 0
        assert 'type="input"' in result.output
        assert 'type="output"' in result.output

    def test_metrics_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test metrics with nonexistent file."""
        result = runner.invoke(cli, ["metrics", str(tmp_path / "nonexistent.json")])

        assert result.exit_code != 0

    def test_metrics_with_errors(self, runner: CliRunner, error_chain_file: Path):
        """Test metrics shows error count correctly."""
        result = runner.invoke(cli, ["metrics", str(error_chain_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["error_count"] == 1


class TestEvaluationCommandsHelp:
    """Tests for evaluation commands help output."""

    def test_evaluate_help(self, runner: CliRunner):
        """Test evaluate command help."""
        result = runner.invoke(cli, ["evaluate", "--help"])

        assert result.exit_code == 0
        assert "comprehensive evaluation metrics" in result.output.lower()
        assert "--json" in result.output

    def test_compare_help(self, runner: CliRunner):
        """Test compare command help."""
        result = runner.invoke(cli, ["compare", "--help"])

        assert result.exit_code == 0
        assert "compare" in result.output.lower()
        assert "statistical" in result.output.lower()
        assert "--json" in result.output

    def test_benchmark_help(self, runner: CliRunner):
        """Test benchmark command help."""
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()
        assert "--iterations" in result.output
        assert "--json" in result.output

    def test_metrics_help(self, runner: CliRunner):
        """Test metrics command help."""
        result = runner.invoke(cli, ["metrics", "--help"])

        assert result.exit_code == 0
        assert "export metrics" in result.output.lower()
        assert "--format" in result.output
        assert "prometheus" in result.output


class TestEvaluationCommandsIntegration:
    """Integration tests for evaluation commands."""

    def test_full_evaluation_workflow(self, runner: CliRunner, cli_chain_file: Path):
        """Test complete evaluation workflow."""
        evaluate_result = runner.invoke(cli, ["evaluate", str(cli_chain_file)])
        assert evaluate_result.exit_code == 0
        assert "Chain Evaluation Report" in evaluate_result.output

        benchmark_result = runner.invoke(cli, ["benchmark", "-n", "3", str(cli_chain_file)])
        assert benchmark_result.exit_code == 0
        assert "Benchmark Results" in benchmark_result.output

        metrics_result = runner.invoke(cli, ["metrics", str(cli_chain_file)])
        assert metrics_result.exit_code == 0
        metrics_data = json.loads(metrics_result.output)
        assert metrics_data["chain_id"] == "cli-test-chain"

    def test_compare_workflow(
        self, runner: CliRunner, cli_chain_file: Path, error_chain_file: Path
    ):
        """Test comparison workflow with different chains."""
        compare_result = runner.invoke(
            cli, ["compare", str(cli_chain_file), str(error_chain_file)]
        )
        assert compare_result.exit_code == 0
        assert "Chain Comparison Report" in compare_result.output

        compare_json = runner.invoke(
            cli, ["compare", "--json", str(cli_chain_file), str(error_chain_file)]
        )
        assert compare_json.exit_code == 0
        data = json.loads(compare_json.output)
        assert data["chain_a"]["id"] == "cli-test-chain"
        assert data["chain_b"]["id"] == "error-chain"

    def test_metrics_export_workflow(self, runner: CliRunner, cli_chain_file: Path):
        """Test metrics export in both formats."""
        json_result = runner.invoke(cli, ["metrics", "--format", "json", str(cli_chain_file)])
        assert json_result.exit_code == 0
        json_data = json.loads(json_result.output)
        assert "total_events" in json_data

        prom_result = runner.invoke(cli, ["metrics", "--format", "prometheus", str(cli_chain_file)])
        assert prom_result.exit_code == 0
        assert "# HELP" in prom_result.output
        assert "# TYPE" in prom_result.output
