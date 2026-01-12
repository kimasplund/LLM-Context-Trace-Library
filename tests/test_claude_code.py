"""Comprehensive test suite for Claude Code integration."""

from pathlib import Path

from lctl.core.events import Chain, EventType
from lctl.integrations.claude_code import (
    MODEL_PRICING,
    LCTLClaudeCodeTracer,
    estimate_cost,
    generate_hooks,
    generate_html_report,
    get_session_metadata,
    is_available,
    validate_hooks,
)


class TestLCTLClaudeCodeTracer:
    """Tests for the LCTLClaudeCodeTracer class."""

    def test_init_default(self):
        """Test default initialization."""
        tracer = LCTLClaudeCodeTracer()
        assert tracer.session is not None
        assert tracer.session.chain.id.startswith("claude-code-")
        assert tracer.agent_stack == []
        assert tracer.tool_counts == {}

    def test_init_with_chain_id(self):
        """Test initialization with custom chain ID."""
        tracer = LCTLClaudeCodeTracer(chain_id="my-custom-chain")
        assert tracer.session.chain.id == "my-custom-chain"

    def test_init_with_output_dir(self, tmp_path):
        """Test initialization with custom output directory."""
        tracer = LCTLClaudeCodeTracer(output_dir=str(tmp_path))
        assert tracer.output_dir == tmp_path

    def test_on_task_start(self):
        """Test recording task start."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")
        tracer.on_task_start(
            agent_type="implementor",
            description="Implement feature",
            prompt="Add authentication to API",
            model="sonnet",
        )

        assert len(tracer.agent_stack) == 1
        assert tracer.agent_stack[0]["agent_type"] == "implementor"
        assert "implementor" in tracer._start_times

        # Check events were recorded
        events = tracer.session.chain.events
        assert len(events) >= 2  # step_start + fact_added

        step_start = next(e for e in events if e.type == EventType.STEP_START)
        assert step_start.agent == "implementor"

    def test_on_task_start_background(self):
        """Test recording background task start."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")
        tracer.on_task_start(
            agent_type="Explore",
            description="Search codebase",
            prompt="Find authentication code",
            run_in_background=True,
        )

        assert tracer.agent_stack[-1]["background"] is True

    def test_on_task_start_resume(self):
        """Test recording resumed task."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")
        tracer.on_task_start(
            agent_type="implementor",
            description="Continue work",
            prompt="Continue implementing",
            resume_agent_id="agent-123",
        )

        assert tracer.agent_stack[-1]["resume_from"] == "agent-123"

    def test_on_task_complete(self):
        """Test recording task completion."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        # Start first
        tracer.on_task_start(
            agent_type="implementor",
            description="Implement feature",
            prompt="Add auth",
        )

        # Complete
        tracer.on_task_complete(
            agent_type="implementor",
            description="Implement feature",
            result="Successfully implemented authentication",
            success=True,
            tokens_in=1000,
            tokens_out=500,
        )

        assert len(tracer.agent_stack) == 0
        assert "implementor" not in tracer._start_times

        # Check events
        events = tracer.session.chain.events
        step_end = next(e for e in events if e.type == EventType.STEP_END)
        assert step_end.agent == "implementor"
        assert step_end.data.get("outcome") == "success"

    def test_on_task_complete_failure(self):
        """Test recording task failure."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_task_start(agent_type="builder", description="Build", prompt="Build project")
        tracer.on_task_complete(
            agent_type="builder",
            description="Build",
            result="Build failed with errors",
            success=False,
            error_message="Compilation error",
        )

        events = tracer.session.chain.events
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) >= 1

    def test_on_task_complete_with_agent_id(self):
        """Test recording completion with agent ID for resume."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_task_start(agent_type="Explore", description="Search", prompt="Find code")
        tracer.on_task_complete(
            agent_type="Explore",
            result="Found relevant files",
            success=True,
            agent_id="agent-456",
        )

        assert tracer._agent_ids.get("Explore") == "agent-456"
        assert tracer.get_agent_ids()["Explore"] == "agent-456"

    def test_on_tool_call(self):
        """Test recording tool calls."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_tool_call(
            tool_name="Bash",
            input_data={"command": "pytest"},
            output_data={"exit_code": 0, "stdout": "All tests passed"},
            duration_ms=5000,
            agent="implementor",
        )

        assert tracer.tool_counts.get("implementor:Bash") == 1

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "Bash"

    def test_on_tool_call_infers_agent(self):
        """Test tool call infers agent from stack."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_task_start(agent_type="developer", description="Dev", prompt="Develop")
        tracer.on_tool_call(
            tool_name="Write",
            input_data={"file_path": "/src/main.py"},
        )

        assert "developer:Write" in tracer.tool_counts

    def test_on_fact_discovered(self):
        """Test recording discovered facts."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_fact_discovered(
            fact_id="auth-method",
            text="Application uses JWT for authentication",
            confidence=0.85,
            agent="analyzer",
        )

        events = tracer.session.chain.events
        fact_event = next(e for e in events if e.type == EventType.FACT_ADDED)
        assert fact_event.data["id"] == "auth-method"  # id not fact_id
        assert fact_event.data["confidence"] == 0.85

    def test_on_fact_updated(self):
        """Test updating facts."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        # Add first
        tracer.on_fact_discovered(
            fact_id="requirement-1",
            text="Needs authentication",
            confidence=0.7,
        )

        # Update
        tracer.on_fact_updated(
            fact_id="requirement-1",
            confidence=0.95,
            reason="Confirmed by stakeholder",
        )

        events = tracer.session.chain.events
        mod_events = [e for e in events if e.type == EventType.FACT_MODIFIED]
        assert len(mod_events) >= 1

    def test_on_user_interaction(self):
        """Test recording user interactions."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_user_interaction(
            question="Which database should we use?",
            response="PostgreSQL",
            options=["PostgreSQL", "MySQL", "MongoDB"],
        )

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "AskUserQuestion"

        fact = next(e for e in events if e.type == EventType.FACT_ADDED)
        assert fact.data["confidence"] == 1.0  # User decisions are ground truth

    def test_on_file_change(self):
        """Test recording file changes."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_file_change(
            file_path="/src/main.py",
            change_type="edit",
            lines_added=50,
            lines_removed=10,
        )

        assert len(tracer._file_changes) == 1
        assert tracer._file_changes[0]["file_path"] == "/src/main.py"
        assert tracer.get_file_changes()[0]["lines_added"] == 50

    def test_on_web_fetch(self):
        """Test recording web fetch."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_web_fetch(
            url="https://docs.python.org",
            prompt="Get Python documentation",
            result_summary="Python 3.12 documentation",
            duration_ms=1500,
        )

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "WebFetch"

    def test_on_web_search(self):
        """Test recording web search."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_web_search(
            query="Python async best practices",
            results_count=10,
            top_result="Use asyncio for concurrent operations",
        )

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "WebSearch"

    def test_on_todo_write(self):
        """Test recording todo list updates."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        todos = [
            {"content": "Implement auth", "status": "completed"},
            {"content": "Add tests", "status": "in_progress"},
            {"content": "Deploy", "status": "pending"},
        ]
        tracer.on_todo_write(todos=todos)

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "TodoWrite"
        assert tool_call.data["input"]["completed"] == 1
        assert tool_call.data["input"]["in_progress"] == 1
        assert tool_call.data["input"]["pending"] == 1

    def test_on_skill_invoke(self):
        """Test recording skill invocations."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_skill_invoke(
            skill_name="commit",
            args="-m 'Fix bug'",
            result_summary="Created commit abc123",
            duration_ms=2000,
        )

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "Skill"
        assert tool_call.data["input"]["skill"] == "commit"

    def test_on_mcp_tool_call(self):
        """Test recording MCP tool calls."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_mcp_tool_call(
            server_name="sql",
            tool_name="execute-sql",
            input_data={"query": "SELECT * FROM users"},
            output_data={"rows": 100},
            duration_ms=150,
        )

        assert "claude-code:mcp__sql__execute-sql" in tracer.tool_counts

        events = tracer.session.chain.events
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "mcp__sql__execute-sql"

    def test_on_git_commit(self):
        """Test recording git commits."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.on_git_commit(
            commit_hash="abc123def456",
            message="Fix authentication bug",
            files_changed=3,
            insertions=50,
            deletions=10,
        )

        events = tracer.session.chain.events

        # Check tool call
        tool_call = next(e for e in events if e.type == EventType.TOOL_CALL)
        assert tool_call.data["tool"] == "GitCommit"
        assert tool_call.data["output"]["commit"] == "abc123def456"

        # Check fact
        fact = next(e for e in events if e.type == EventType.FACT_ADDED)
        assert "abc123de" in fact.data["text"]

        # Check checkpoint was created
        checkpoints = [e for e in events if e.type == EventType.CHECKPOINT]
        assert len(checkpoints) >= 1

    def test_link_to_git_history(self, tmp_path):
        """Test git history linking."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        # Track some file changes
        tracer._file_changes = [{"file_path": "test.py"}]

        # Run against actual git repo (this project)
        result = tracer.link_to_git_history(".")

        assert "commits" in result
        assert "uncommitted_changes" in result
        # Should find recent commits in this repo
        assert len(result["commits"]) > 0

    def test_parallel_groups(self):
        """Test parallel execution group tracking."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        group_id = tracer.start_parallel_group("parallel-1")
        assert group_id == "parallel-1"
        assert tracer._parallel_group == "parallel-1"

        tracer.end_parallel_group()
        assert tracer._parallel_group is None

    def test_checkpoint(self):
        """Test checkpoint creation."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        tracer.checkpoint(description="Before major change")

        events = tracer.session.chain.events

        # Description is stored as a fact before the checkpoint
        fact = next(e for e in events if e.type == EventType.FACT_ADDED)
        assert "Checkpoint: Before major change" in fact.data["text"]

        # Checkpoint event exists
        cp = next(e for e in events if e.type == EventType.CHECKPOINT)
        assert cp is not None

    def test_export(self, tmp_path):
        """Test trace export."""
        tracer = LCTLClaudeCodeTracer(chain_id="test-export", output_dir=str(tmp_path))

        tracer.on_task_start(agent_type="test", description="Test", prompt="Test prompt")
        tracer.on_task_complete(agent_type="test", result="Done", success=True)

        path = tracer.export()
        assert Path(path).exists()

        # Verify can be loaded
        chain = Chain.load(Path(path))
        assert chain.id == "test-export"

    def test_export_custom_path(self, tmp_path):
        """Test export to custom path."""
        tracer = LCTLClaudeCodeTracer(chain_id="test")

        custom_path = tmp_path / "custom.lctl.json"
        path = tracer.export(str(custom_path))

        assert Path(path).exists()
        assert path == str(custom_path)

    def test_get_summary(self):
        """Test summary generation."""
        tracer = LCTLClaudeCodeTracer(chain_id="test-summary")

        # Add some activity
        tracer.on_task_start(agent_type="analyzer", description="Analyze", prompt="Analyze code")
        tracer.on_fact_discovered("f1", "Found issue", confidence=0.8)
        tracer.on_tool_call("Bash", {"command": "ls"}, {"files": ["a.py"]})
        tracer.on_task_complete(agent_type="analyzer", result="Done", success=True, tokens_in=100, tokens_out=50)
        tracer.on_file_change("/test.py", "edit")

        summary = tracer.get_summary()

        assert summary["chain_id"] == "test-summary"
        assert summary["event_count"] > 0
        assert "analyzer" in summary["agent_stats"]
        assert summary["fact_count"] >= 1
        assert summary["file_changes"] == 1
        assert len(summary["tool_counts"]) > 0

    def test_reset(self):
        """Test tracer reset."""
        tracer = LCTLClaudeCodeTracer(chain_id="test-reset")

        # Add activity
        tracer.on_task_start(agent_type="test", description="Test", prompt="Test")
        tracer.on_file_change("/test.py", "create")

        # Reset
        tracer.reset()

        assert len(tracer.agent_stack) == 0
        assert len(tracer.tool_counts) == 0
        assert len(tracer._file_changes) == 0
        assert tracer.session.chain.id != "test-reset"

    def test_cleanup_stale_start_times(self):
        """Test cleanup of stale start times."""
        import time

        tracer = LCTLClaudeCodeTracer(chain_id="test-cleanup")

        # Add some start times
        tracer._start_times["agent1"] = time.time()  # Fresh
        tracer._start_times["agent2"] = time.time() - 7200  # 2 hours old (stale)
        tracer._start_times["agent3"] = time.time() - 100  # 100 seconds old

        # Cleanup with 1 hour max age
        removed = tracer.cleanup_stale_start_times(max_age_seconds=3600.0)

        assert removed == 1  # Only agent2 should be removed
        assert "agent1" in tracer._start_times
        assert "agent2" not in tracer._start_times
        assert "agent3" in tracer._start_times

    def test_cleanup_stale_start_times_none_stale(self):
        """Test cleanup when no start times are stale."""
        import time

        tracer = LCTLClaudeCodeTracer(chain_id="test-cleanup-none")

        # Add only fresh start times
        tracer._start_times["agent1"] = time.time()
        tracer._start_times["agent2"] = time.time() - 60

        removed = tracer.cleanup_stale_start_times(max_age_seconds=3600.0)

        assert removed == 0
        assert len(tracer._start_times) == 2

    def test_agent_stack_out_of_order_completion(self):
        """Test that agents can complete out of order."""
        tracer = LCTLClaudeCodeTracer(chain_id="test-out-of-order")

        # Start multiple agents
        tracer.on_task_start(agent_type="agent1", description="Task 1", prompt="Prompt 1")
        tracer.on_task_start(agent_type="agent2", description="Task 2", prompt="Prompt 2")
        tracer.on_task_start(agent_type="agent3", description="Task 3", prompt="Prompt 3")

        assert len(tracer.agent_stack) == 3

        # Complete out of order (agent2 completes before agent3)
        tracer.on_task_complete(agent_type="agent2", result="Done 2", success=True)

        # agent1 and agent3 should still be in stack
        assert len(tracer.agent_stack) == 2
        agent_types = [a["agent_type"] for a in tracer.agent_stack]
        assert "agent1" in agent_types
        assert "agent3" in agent_types
        assert "agent2" not in agent_types


class TestGetOrCreate:
    """Tests for singleton pattern and state persistence."""

    def test_get_or_create_new(self, tmp_path):
        """Test creating new tracer instance."""
        # Clear global state
        import lctl.integrations.claude_code as cc
        cc._tracer_instance = None
        cc._tracer_file = None

        state_file = tmp_path / ".lctl-state.json"
        tracer = LCTLClaudeCodeTracer.get_or_create(
            chain_id="singleton-test",
            state_file=str(state_file),
        )

        assert tracer.session.chain.id == "singleton-test"

        # Clean up
        cc._tracer_instance = None
        cc._tracer_file = None

    def test_get_or_create_returns_same(self, tmp_path):
        """Test that get_or_create returns same instance."""
        import lctl.integrations.claude_code as cc
        cc._tracer_instance = None
        cc._tracer_file = None

        state_file = tmp_path / ".lctl-state.json"

        tracer1 = LCTLClaudeCodeTracer.get_or_create(
            chain_id="singleton-test",
            state_file=str(state_file),
        )
        tracer2 = LCTLClaudeCodeTracer.get_or_create()

        assert tracer1 is tracer2

        # Clean up
        cc._tracer_instance = None
        cc._tracer_file = None


class TestGenerateHooks:
    """Tests for hook script generation."""

    def test_generate_hooks(self, tmp_path):
        """Test hook generation."""
        hooks_dir = tmp_path / "hooks"
        hooks = generate_hooks(str(hooks_dir))

        assert "PreToolUse" in hooks
        assert "PostToolUse" in hooks
        assert "Stop" in hooks

        # Check files exist and are executable
        for name, path in hooks.items():
            hook_path = Path(path)
            assert hook_path.exists()
            assert hook_path.stat().st_mode & 0o755

    def test_generate_hooks_content(self, tmp_path):
        """Test hook script content."""
        hooks_dir = tmp_path / "hooks"
        hooks = generate_hooks(str(hooks_dir))

        # Check PreToolUse hook
        pre_content = Path(hooks["PreToolUse"]).read_text()
        assert "LCTL" in pre_content
        assert "Task" in pre_content
        assert "on_task_start" in pre_content

        # Check PostToolUse hook
        post_content = Path(hooks["PostToolUse"]).read_text()
        assert "on_task_complete" in post_content
        assert "TodoWrite" in post_content
        assert "Skill" in post_content
        assert "mcp__" in post_content

        # Check Stop hook
        stop_content = Path(hooks["Stop"]).read_text()
        assert "export" in stop_content


class TestValidateHooks:
    """Tests for hook validation."""

    def test_validate_hooks_missing(self, tmp_path):
        """Test validation with missing hooks."""
        result = validate_hooks(str(tmp_path))

        assert result["valid"] is False
        assert not result["hooks"]["PreToolUse"]["exists"]

    def test_validate_hooks_present(self, tmp_path):
        """Test validation with present hooks."""
        hooks_dir = tmp_path / "hooks"
        generate_hooks(str(hooks_dir))

        result = validate_hooks(str(hooks_dir))

        assert result["valid"] is True
        assert result["hooks"]["PreToolUse"]["exists"]
        assert result["hooks"]["PreToolUse"]["executable"]
        assert result["hooks"]["PreToolUse"]["contains_lctl"]


class TestGenerateHtmlReport:
    """Tests for HTML report generation."""

    def test_generate_html_report(self, tmp_path):
        """Test HTML report generation."""
        # Create a chain with some data
        tracer = LCTLClaudeCodeTracer(chain_id="report-test")
        tracer.on_task_start(agent_type="analyzer", description="Analyze", prompt="Test")
        tracer.on_fact_discovered("f1", "Found something", confidence=0.9)
        tracer.on_task_complete(agent_type="analyzer", result="Done", success=True, tokens_in=1000, tokens_out=500)

        output_path = tmp_path / "report.html"
        result = generate_html_report(tracer.session.chain, str(output_path))

        assert Path(result).exists()

        content = Path(result).read_text()
        assert "LCTL Workflow Report" in content
        assert "report-test" in content
        assert "analyzer" in content

    def test_html_report_includes_metrics(self, tmp_path):
        """Test that HTML report includes key metrics."""
        tracer = LCTLClaudeCodeTracer(chain_id="metrics-test")
        tracer.on_task_start(agent_type="test", description="Test", prompt="Test")
        tracer.on_task_complete(agent_type="test", result="Done", success=True, tokens_in=5000, tokens_out=2000)

        output_path = tmp_path / "report.html"
        generate_html_report(tracer.session.chain, str(output_path))

        content = Path(output_path).read_text()
        assert "Tokens" in content
        assert "Duration" in content
        assert "Est. Cost" in content


class TestGetSessionMetadata:
    """Tests for session metadata."""

    def test_get_session_metadata(self):
        """Test session metadata retrieval."""
        metadata = get_session_metadata()

        assert "timestamp" in metadata
        assert "working_dir" in metadata  # working_dir not cwd
        assert "python_version" in metadata
        # platform may not always be present

    def test_get_session_metadata_git(self):
        """Test git info in metadata."""
        metadata = get_session_metadata()

        # Running in a git repo, should have git info
        if "git" in metadata:
            git_info = metadata["git"]
            assert "branch" in git_info
            assert "commit" in git_info


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_cost_default(self):
        """Test cost estimation with default pricing."""
        result = estimate_cost(1_000_000, 500_000)

        assert result["input_cost"] == 3.0  # $3/MTok
        assert result["output_cost"] == 7.5  # $15/MTok * 0.5
        assert result["total_cost"] == 10.5

    def test_estimate_cost_opus(self):
        """Test cost estimation for Opus model."""
        result = estimate_cost(1_000_000, 1_000_000, model="claude-opus-4.5")

        assert result["input_cost"] == 5.0
        assert result["output_cost"] == 25.0
        assert result["total_cost"] == 30.0

    def test_estimate_cost_haiku(self):
        """Test cost estimation for Haiku model."""
        result = estimate_cost(1_000_000, 1_000_000, model="claude-3-haiku")

        assert result["input_cost"] == 0.25
        assert result["output_cost"] == 1.25
        assert result["total_cost"] == 1.5

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model uses default."""
        result = estimate_cost(1_000_000, 1_000_000, model="unknown-model")

        assert result["model"] == "unknown-model"
        assert result["pricing"] == MODEL_PRICING["default"]


class TestModelPricing:
    """Tests for model pricing dict."""

    def test_model_pricing_has_major_models(self):
        """Test that pricing includes all major models."""
        expected_models = [
            "claude-opus-4.5",
            "claude-sonnet-4",
            "claude-haiku-4.5",
            "claude-3-opus",
            "claude-3-haiku",
            "default",
        ]

        for model in expected_models:
            assert model in MODEL_PRICING
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]

    def test_opus_pricing_higher_than_sonnet(self):
        """Test that Opus pricing is higher than Sonnet."""
        opus = MODEL_PRICING["claude-opus-4"]
        sonnet = MODEL_PRICING["claude-sonnet-4"]

        assert opus["input"] > sonnet["input"]
        assert opus["output"] > sonnet["output"]

    def test_haiku_pricing_lowest(self):
        """Test that Haiku has lowest pricing."""
        haiku = MODEL_PRICING["claude-3-haiku"]
        sonnet = MODEL_PRICING["claude-sonnet-4"]

        assert haiku["input"] < sonnet["input"]
        assert haiku["output"] < sonnet["output"]


class TestIsAvailable:
    """Tests for availability check."""

    def test_is_available(self):
        """Test that Claude Code integration is always available."""
        assert is_available() is True


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, tmp_path):
        """Test a complete multi-agent workflow."""
        tracer = LCTLClaudeCodeTracer(
            chain_id="complete-workflow",
            output_dir=str(tmp_path),
        )

        # Start main task
        tracer.on_task_start(
            agent_type="Plan",
            description="Plan implementation",
            prompt="Create implementation plan for user auth",
        )

        # Plan discovers requirements
        tracer.on_fact_discovered("req-1", "Need JWT tokens", confidence=0.8)
        tracer.on_fact_discovered("req-2", "Need refresh tokens", confidence=0.7)

        # Plan completes
        tracer.on_task_complete(
            agent_type="Plan",
            result="Created implementation plan",
            success=True,
            agent_id="plan-001",
        )

        # Spawn implementor
        tracer.on_task_start(
            agent_type="implementor",
            description="Implement authentication",
            prompt="Implement JWT auth per plan",
        )

        # Implementor uses tools
        tracer.on_tool_call("Write", {"file_path": "/auth.py"}, {"success": True})
        tracer.on_file_change("/auth.py", "create", lines_added=100)

        # Implementor completes
        tracer.on_task_complete(
            agent_type="implementor",
            result="Implemented JWT authentication",
            success=True,
            tokens_in=5000,
            tokens_out=3000,
        )

        # Create checkpoint
        tracer.checkpoint("After implementation")

        # Get summary
        summary = tracer.get_summary()

        assert summary["event_count"] > 10
        assert len(summary["agent_stats"]) == 2
        assert summary["fact_count"] >= 2
        assert summary["file_changes"] == 1

        # Export and verify
        path = tracer.export()
        assert Path(path).exists()

        # Generate report
        report_path = tmp_path / "report.html"
        generate_html_report(tracer.session.chain, str(report_path))
        assert report_path.exists()

    def test_parallel_agents(self, tmp_path):
        """Test parallel agent execution tracking."""
        tracer = LCTLClaudeCodeTracer(
            chain_id="parallel-test",
            output_dir=str(tmp_path),
        )

        # Start parallel group
        group_id = tracer.start_parallel_group()

        # Start multiple agents in parallel
        tracer.on_task_start(
            agent_type="Explore",
            description="Search backend",
            prompt="Find backend code",
            parallel_group=group_id,
        )
        tracer.on_task_start(
            agent_type="Explore",
            description="Search frontend",
            prompt="Find frontend code",
            parallel_group=group_id,
        )

        # Complete them
        tracer.on_task_complete(agent_type="Explore", result="Found backend", success=True)
        tracer.on_task_complete(agent_type="Explore", result="Found frontend", success=True)

        # End parallel group
        tracer.end_parallel_group()

        summary = tracer.get_summary()
        assert "Explore" in summary["agent_stats"]

    def test_error_recovery(self, tmp_path):
        """Test workflow with errors and recovery."""
        tracer = LCTLClaudeCodeTracer(
            chain_id="error-recovery",
            output_dir=str(tmp_path),
        )

        # First attempt fails
        tracer.on_task_start(agent_type="builder", description="Build", prompt="Build project")
        tracer.on_task_complete(
            agent_type="builder",
            result="Build failed",
            success=False,
            error_message="Missing dependency",
        )

        # User fixes issue
        tracer.on_user_interaction(
            question="Should I install the missing dependency?",
            response="Yes, install it",
        )

        # Second attempt succeeds
        tracer.on_task_start(agent_type="builder", description="Build", prompt="Build project again")
        tracer.on_task_complete(
            agent_type="builder",
            result="Build successful",
            success=True,
        )

        summary = tracer.get_summary()
        assert summary["error_count"] >= 1
        assert summary["user_interactions"] >= 1
