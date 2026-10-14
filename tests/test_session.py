"""Tests for LCTL session management (lctl/core/session.py)."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest

from lctl.core.events import Chain, Event, EventType, ReplayEngine
from lctl.core.session import LCTLSession, traced_step


class TestLCTLSessionBasics:
    """Tests for basic LCTLSession functionality."""

    def test_session_creation(self):
        """Test basic session creation."""
        session = LCTLSession()
        assert session.chain is not None
        assert len(session.chain.id) == 8
        assert session._seq == 0
        assert session._current_agent is None

    def test_session_creation_with_custom_id(self):
        """Test session creation with custom chain ID."""
        session = LCTLSession(chain_id="custom-chain")
        assert session.chain.id == "custom-chain"

    def test_session_context_manager(self):
        """Test session as context manager."""
        with LCTLSession(chain_id="ctx-test") as session:
            assert session.chain.id == "ctx-test"

    def test_session_context_manager_records_exception(self):
        """Test that context manager records exceptions as errors."""
        session = LCTLSession(chain_id="error-test")

        with pytest.raises(ValueError):
            with session:
                raise ValueError("Test error")

        error_events = [e for e in session.chain.events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].data["type"] == "ValueError"
        assert error_events[0].data["message"] == "Test error"
        assert error_events[0].data["recoverable"] is False

    def test_session_context_manager_no_error_on_success(self):
        """Test that context manager does not record error on success."""
        with LCTLSession(chain_id="success-test") as session:
            pass

        error_events = [e for e in session.chain.events if e.type == EventType.ERROR]
        assert len(error_events) == 0


class TestLCTLSessionSequencing:
    """Tests for sequence number management."""

    def test_sequence_numbers_increment(self):
        """Test that sequence numbers increment properly."""
        session = LCTLSession()
        seq1 = session.step_start("agent1", "action1")
        seq2 = session.step_end("agent1")
        seq3 = session.add_fact("F1", "fact text")

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    def test_sequence_numbers_are_unique(self):
        """Test that all events have unique sequence numbers."""
        session = LCTLSession()
        for i in range(10):
            session.step_start(f"agent{i}", f"action{i}")
            session.step_end()

        seqs = [e.seq for e in session.chain.events]
        assert len(seqs) == len(set(seqs))


class TestLCTLSessionStepMethods:
    """Tests for step_start and step_end methods."""

    def test_step_start(self):
        """Test step_start creates correct event."""
        session = LCTLSession()
        seq = session.step_start("planner", "analyze", "input data")

        assert seq == 1
        assert session._current_agent == "planner"

        event = session.chain.events[0]
        assert event.type == EventType.STEP_START
        assert event.agent == "planner"
        assert event.data["intent"] == "analyze"
        assert event.data["input_summary"] == "input data"

    def test_step_start_default_input_summary(self):
        """Test step_start with default input summary."""
        session = LCTLSession()
        session.step_start("agent1", "action1")

        event = session.chain.events[0]
        assert event.data["input_summary"] == ""

    def test_step_end_basic(self):
        """Test step_end creates correct event."""
        session = LCTLSession()
        session.step_start("executor", "execute")
        seq = session.step_end(
            outcome="success",
            output_summary="completed",
            duration_ms=500,
            tokens_in=100,
            tokens_out=50,
        )

        assert seq == 2
        assert session._current_agent is None

        event = session.chain.events[1]
        assert event.type == EventType.STEP_END
        assert event.agent == "executor"
        assert event.data["outcome"] == "success"
        assert event.data["output_summary"] == "completed"
        assert event.data["duration_ms"] == 500
        assert event.data["tokens"]["input"] == 100
        assert event.data["tokens"]["output"] == 50

    def test_step_end_uses_current_agent(self):
        """Test step_end uses current agent if not specified."""
        session = LCTLSession()
        session.step_start("agent1", "action")
        session.step_end()

        event = session.chain.events[1]
        assert event.agent == "agent1"

    def test_step_end_without_start_uses_unknown(self):
        """Test step_end without step_start uses 'unknown' agent."""
        session = LCTLSession()
        session.step_end()

        event = session.chain.events[0]
        assert event.agent == "unknown"

    def test_step_end_with_explicit_agent(self):
        """Test step_end with explicit agent overrides current."""
        session = LCTLSession()
        session.step_start("agent1", "action")
        session.step_end(agent="agent2")

        event = session.chain.events[1]
        assert event.agent == "agent2"


class TestLCTLSessionFactMethods:
    """Tests for add_fact and modify_fact methods."""

    def test_add_fact_basic(self):
        """Test add_fact creates correct event."""
        session = LCTLSession()
        seq = session.add_fact("F1", "Test fact", confidence=0.9)

        assert seq == 1
        event = session.chain.events[0]
        assert event.type == EventType.FACT_ADDED
        assert event.data["id"] == "F1"
        assert event.data["text"] == "Test fact"
        assert event.data["confidence"] == 0.9

    def test_add_fact_default_confidence(self):
        """Test add_fact with default confidence of 1.0."""
        session = LCTLSession()
        session.add_fact("F1", "Certain fact")

        event = session.chain.events[0]
        assert event.data["confidence"] == 1.0

    def test_add_fact_with_source(self):
        """Test add_fact with explicit source."""
        session = LCTLSession()
        session.add_fact("F1", "Fact text", source="external_api")

        event = session.chain.events[0]
        assert event.agent == "external_api"
        assert event.data["source"] == "external_api"

    def test_add_fact_uses_current_agent(self):
        """Test add_fact uses current agent if no source specified."""
        session = LCTLSession()
        session.step_start("analyzer", "analyze")
        session.add_fact("F1", "Fact from analyzer")

        event = session.chain.events[1]
        assert event.agent == "analyzer"

    def test_modify_fact_basic(self):
        """Test modify_fact creates correct event."""
        session = LCTLSession()
        session.add_fact("F1", "Original fact", confidence=0.5)
        seq = session.modify_fact("F1", text="Updated fact", confidence=0.9, reason="verified")

        assert seq == 2
        event = session.chain.events[1]
        assert event.type == EventType.FACT_MODIFIED
        assert event.data["id"] == "F1"
        assert event.data["text"] == "Updated fact"
        assert event.data["confidence"] == 0.9
        assert event.data["reason"] == "verified"

    def test_modify_fact_partial_update(self):
        """Test modify_fact with only confidence update."""
        session = LCTLSession()
        session.add_fact("F1", "Original fact", confidence=0.5)
        session.modify_fact("F1", confidence=0.8)

        event = session.chain.events[1]
        assert event.data["confidence"] == 0.8
        assert "text" not in event.data

    def test_modify_fact_only_text(self):
        """Test modify_fact with only text update."""
        session = LCTLSession()
        session.add_fact("F1", "Original fact")
        session.modify_fact("F1", text="New text")

        event = session.chain.events[1]
        assert event.data["text"] == "New text"
        assert "confidence" not in event.data


class TestLCTLSessionToolCall:
    """Tests for tool_call method."""

    def test_tool_call_basic(self):
        """Test tool_call creates correct event."""
        session = LCTLSession()
        seq = session.tool_call(
            tool="search",
            input_data={"query": "test"},
            output_data={"results": ["a", "b"]},
            duration_ms=150,
        )

        assert seq == 1
        event = session.chain.events[0]
        assert event.type == EventType.TOOL_CALL
        assert event.data["tool"] == "search"
        assert event.data["input"] == {"query": "test"}
        assert event.data["output"] == {"results": ["a", "b"]}
        assert event.data["duration_ms"] == 150

    def test_tool_call_uses_current_agent(self):
        """Test tool_call uses current agent."""
        session = LCTLSession()
        session.step_start("tool_user", "use tools")
        session.tool_call("calc", 5, 25)

        event = session.chain.events[1]
        assert event.agent == "tool_user"

    def test_tool_call_default_duration(self):
        """Test tool_call with default duration of 0."""
        session = LCTLSession()
        session.tool_call("noop", None, None)

        event = session.chain.events[0]
        assert event.data["duration_ms"] == 0


class TestLCTLSessionError:
    """Tests for error recording method."""

    def test_error_basic(self):
        """Test error creates correct event."""
        session = LCTLSession()
        seq = session.error(
            category="validation",
            error_type="ValueError",
            message="Invalid input",
            recoverable=True,
            suggested_action="Fix input",
        )

        assert seq == 1
        event = session.chain.events[0]
        assert event.type == EventType.ERROR
        assert event.data["category"] == "validation"
        assert event.data["type"] == "ValueError"
        assert event.data["message"] == "Invalid input"
        assert event.data["recoverable"] is True
        assert event.data["suggested_action"] == "Fix input"

    def test_error_default_recoverable(self):
        """Test error with default recoverable=True."""
        session = LCTLSession()
        session.error("test", "TestError", "test message")

        event = session.chain.events[0]
        assert event.data["recoverable"] is True

    def test_error_uses_current_agent(self):
        """Test error uses current agent."""
        session = LCTLSession()
        session.step_start("error_prone", "risky action")
        session.error("runtime", "RuntimeError", "Something failed")

        event = session.chain.events[1]
        assert event.agent == "error_prone"


class TestLCTLSessionCheckpoint:
    """Tests for checkpoint method."""

    def test_checkpoint_basic(self):
        """Test checkpoint creates correct event."""
        session = LCTLSession()
        session.add_fact("F1", "Fact 1", confidence=0.9)
        seq = session.checkpoint()

        assert seq == 2
        event = session.chain.events[1]
        assert event.type == EventType.CHECKPOINT
        assert event.agent == "system"
        assert "state_hash" in event.data
        assert "facts_snapshot" in event.data

    def test_checkpoint_with_custom_snapshot(self):
        """Test checkpoint with custom facts snapshot."""
        session = LCTLSession()
        custom_snapshot = {"F_CUSTOM": {"text": "custom", "confidence": 1.0}}
        seq = session.checkpoint(facts_snapshot=custom_snapshot)

        event = session.chain.events[0]
        assert event.data["facts_snapshot"] == custom_snapshot

    def test_checkpoint_computes_state_from_events(self):
        """Test checkpoint computes state from existing events."""
        session = LCTLSession()
        session.add_fact("F1", "First fact", confidence=0.8)
        session.add_fact("F2", "Second fact", confidence=0.9)
        session.checkpoint()

        event = session.chain.events[2]
        snapshot = event.data["facts_snapshot"]
        assert "F1" in snapshot
        assert "F2" in snapshot


class TestLCTLSessionExport:
    """Tests for export functionality."""

    def test_export_json(self, tmp_path: Path):
        """Test exporting session to JSON file."""
        session = LCTLSession(chain_id="export-test")
        session.step_start("agent1", "action1")
        session.add_fact("F1", "test fact")
        session.step_end()

        export_path = tmp_path / "export.lctl.json"
        session.export(str(export_path))

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["chain"]["id"] == "export-test"
        assert len(data["events"]) == 3

    def test_export_yaml(self, tmp_path: Path):
        """Test exporting session to YAML file."""
        import yaml

        session = LCTLSession(chain_id="yaml-export")
        session.add_fact("F1", "yaml fact")

        export_path = tmp_path / "export.lctl.yaml"
        session.export(str(export_path))

        assert export_path.exists()
        data = yaml.safe_load(export_path.read_text())
        assert data["chain"]["id"] == "yaml-export"

    def test_to_dict(self):
        """Test to_dict returns chain dictionary."""
        session = LCTLSession(chain_id="dict-test")
        session.add_fact("F1", "fact")

        result = session.to_dict()
        assert result["chain"]["id"] == "dict-test"
        assert len(result["events"]) == 1


class TestTracedStep:
    """Tests for traced_step context manager."""

    def test_traced_step_basic(self):
        """Test traced_step records start and end events."""
        session = LCTLSession()

        with traced_step(session, "worker", "process", "input data"):
            pass

        assert len(session.chain.events) == 2
        assert session.chain.events[0].type == EventType.STEP_START
        assert session.chain.events[1].type == EventType.STEP_END

    def test_traced_step_records_intent(self):
        """Test traced_step records intent."""
        session = LCTLSession()

        with traced_step(session, "analyzer", "analyze_code", "main.py"):
            pass

        start_event = session.chain.events[0]
        assert start_event.data["intent"] == "analyze_code"
        assert start_event.data["input_summary"] == "main.py"

    def test_traced_step_records_success_outcome(self):
        """Test traced_step records success outcome."""
        session = LCTLSession()

        with traced_step(session, "agent", "action"):
            pass

        end_event = session.chain.events[1]
        assert end_event.data["outcome"] == "success"

    def test_traced_step_records_duration(self):
        """Test traced_step records duration."""
        session = LCTLSession()

        with traced_step(session, "agent", "sleep"):
            time.sleep(0.05)

        end_event = session.chain.events[1]
        assert end_event.data["duration_ms"] >= 40

    def test_traced_step_on_exception(self):
        """Test traced_step handles exceptions properly."""
        session = LCTLSession()

        with pytest.raises(RuntimeError):
            with traced_step(session, "failing_agent", "fail"):
                raise RuntimeError("Intentional failure")

        assert len(session.chain.events) == 3
        start_event = session.chain.events[0]
        assert start_event.type == EventType.STEP_START

        end_event = session.chain.events[1]
        assert end_event.data["outcome"] == "error"

        error_event = session.chain.events[2]
        assert error_event.type == EventType.ERROR
        assert error_event.data["type"] == "RuntimeError"
        assert error_event.data["message"] == "Intentional failure"

    def test_traced_step_propagates_exception(self):
        """Test traced_step re-raises exceptions."""
        session = LCTLSession()

        with pytest.raises(ValueError, match="test error"):
            with traced_step(session, "agent", "action"):
                raise ValueError("test error")


class TestLCTLSessionIntegration:
    """Integration tests for LCTLSession."""

    def test_full_workflow(self, tmp_path: Path):
        """Test complete workflow: create, record, export, load, replay."""
        with LCTLSession(chain_id="integration-test") as session:
            session.step_start("planner", "plan", "requirements")
            session.add_fact("REQ1", "Must handle edge cases", confidence=0.9)
            session.step_end(outcome="success", duration_ms=200, tokens_in=50, tokens_out=30)

            session.step_start("executor", "execute", "plan")
            session.tool_call("code_gen", {"template": "class"}, {"code": "class X: pass"}, 100)
            session.modify_fact("REQ1", confidence=1.0, reason="implemented")
            session.step_end(outcome="success", duration_ms=500, tokens_in=100, tokens_out=200)

            session.checkpoint()

        export_path = tmp_path / "workflow.lctl.json"
        session.export(str(export_path))

        loaded_chain = Chain.load(export_path)
        assert loaded_chain.id == "integration-test"
        assert len(loaded_chain.events) == 8

        engine = ReplayEngine(loaded_chain)
        state = engine.replay_all()

        assert "REQ1" in state.facts
        assert state.facts["REQ1"]["confidence"] == 1.0
        assert state.metrics["total_duration_ms"] == 800
        assert state.metrics["total_tokens_in"] == 150

    def test_multiple_agents_workflow(self):
        """Test workflow with multiple agents."""
        session = LCTLSession()

        for agent in ["agent1", "agent2", "agent3"]:
            session.step_start(agent, f"{agent}_action")
            session.add_fact(f"F_{agent}", f"Fact from {agent}")
            session.step_end()

        engine = ReplayEngine(session.chain)
        state = engine.replay_all()

        assert len(state.facts) == 3
        assert all(f"F_agent{i}" in state.facts for i in [1, 2, 3])

    def test_error_recovery_workflow(self):
        """Test workflow with error recovery."""
        session = LCTLSession()

        session.step_start("validator", "validate")
        session.error("validation", "InputError", "Invalid format", recoverable=True)
        session.step_end(outcome="partial")

        session.step_start("validator", "retry_validate")
        session.add_fact("V1", "Validation passed")
        session.step_end(outcome="success")

        engine = ReplayEngine(session.chain)
        state = engine.replay_all()

        assert len(state.errors) == 1
        assert state.errors[0]["recoverable"] is True
        assert "V1" in state.facts
