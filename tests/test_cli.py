"""Tests for LCTL CLI commands (lctl/cli/main.py)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from lctl.cli.main import cli, _estimate_cost
from lctl.core.events import Chain, Event, EventType


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
