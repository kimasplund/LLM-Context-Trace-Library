"""Tests for DSPy integration (lctl/integrations/dspy.py)."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from lctl.core.events import Chain, Event, EventType, ReplayEngine
from lctl.core.session import LCTLSession

sys.modules["dspy"] = MagicMock()


class MockSignature:
    """Mock DSPy Signature for testing."""

    def __init__(
        self,
        input_fields: Optional[Dict[str, Any]] = None,
        output_fields: Optional[Dict[str, Any]] = None,
    ):
        self.input_fields = input_fields or {"question": "str"}
        self.output_fields = output_fields or {"answer": "str"}
        self.__name__ = "question -> answer"


class MockPrediction:
    """Mock DSPy Prediction for testing."""

    def __init__(self, answer: str = "Test answer"):
        self.answer = answer
        self.completions = [self]

    def __str__(self) -> str:
        return self.answer


class MockModule:
    """Mock DSPy Module for testing."""

    def __init__(
        self,
        name: str = "MockModule",
        signature: Optional[MockSignature] = None,
    ):
        self.name = name
        self.signature = signature or MockSignature()

    def __call__(self, *args, **kwargs) -> MockPrediction:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> MockPrediction:
        return MockPrediction(answer=f"Answer to: {kwargs.get('question', 'unknown')}")


class MockPredict(MockModule):
    """Mock DSPy Predict module for testing."""

    def __init__(self, signature: Optional[MockSignature] = None):
        super().__init__(name="Predict", signature=signature)


class MockChainOfThought(MockModule):
    """Mock DSPy ChainOfThought module for testing."""

    def __init__(self, signature: Optional[MockSignature] = None):
        super().__init__(name="ChainOfThought", signature=signature)


class MockTeleprompter:
    """Mock DSPy Teleprompter for testing."""

    def __init__(self, metric: Any = None):
        self.metric = metric
        self._compile_count = 0

    def compile(
        self,
        student: Any,
        trainset: Optional[List[Any]] = None,
        **kwargs,
    ) -> Any:
        self._compile_count += 1
        return student


@pytest.fixture
def mock_dspy():
    """Fixture to mock DSPy imports."""
    with patch.dict(
        "sys.modules",
        {
            "dspy": MagicMock(
                Module=MockModule,
                Predict=MockPredict,
                Signature=MockSignature,
            ),
        },
    ):
        yield


class TestDSPyAvailability:
    """Tests for DSPy availability checking."""

    def test_is_available_function_exists(self):
        """Test that is_available function exists."""
        import lctl.integrations.dspy as dspy_module

        assert hasattr(dspy_module, "is_available")
        assert callable(dspy_module.is_available)

    def test_dspy_not_available_error(self):
        """Test DSPyNotAvailableError is defined."""
        import lctl.integrations.dspy as dspy_module

        error = dspy_module.DSPyNotAvailableError()
        assert "DSPy is not installed" in str(error)
        assert "pip install" in str(error)


class TestLCTLDSPyCallbackBasics:
    """Tests for basic LCTLDSPyCallback functionality."""

    def test_callback_creation(self, mock_dspy):
        """Test basic callback creation."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()
            assert callback.session is not None
            assert callback.chain is not None
            assert len(callback._module_stack) == 0

    def test_callback_creation_with_chain_id(self, mock_dspy):
        """Test callback creation with custom chain ID."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback(chain_id="custom-chain")
            assert callback.session.chain.id == "custom-chain"

    def test_callback_creation_with_existing_session(self, mock_dspy):
        """Test callback creation with existing session."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            session = LCTLSession(chain_id="existing-session")
            callback = LCTLDSPyCallback(session=session)
            assert callback.session is session
            assert callback.session.chain.id == "existing-session"

    def test_callback_verbose_mode(self, mock_dspy):
        """Test callback with verbose mode enabled."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback(verbose=True)
            assert callback._verbose is True


class TestLCTLDSPyCallbackModuleEvents:
    """Tests for module start/end event handling."""

    def test_on_module_start(self, mock_dspy):
        """Test on_module_start records step_start event."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()
            module = MockModule(name="TestModule")

            callback.on_module_start(module, {"question": "What is LCTL?"})

            assert len(callback._module_stack) == 1
            assert callback._current_module == "TestModule"

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            assert len(step_starts) == 1
            assert step_starts[0].agent == "TestModule"
            assert step_starts[0].data["intent"] == "module_forward"

    def test_on_module_end_success(self, mock_dspy):
        """Test on_module_end records step_end event on success."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()
            module = MockModule(name="TestModule")
            prediction = MockPrediction(answer="The answer is 42")

            callback.on_module_start(module, {"question": "test"})
            callback.on_module_end(module, prediction)

            assert len(callback._module_stack) == 0
            assert callback._current_module is None

            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_ends) == 1
            assert step_ends[0].data["outcome"] == "success"
            assert "42" in step_ends[0].data["output_summary"]

    def test_on_module_end_error(self, mock_dspy):
        """Test on_module_end records error event on failure."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()
            module = MockModule(name="TestModule")
            error = ValueError("Test error")

            callback.on_module_start(module, {"question": "test"})
            callback.on_module_end(module, None, error=error)

            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_ends) == 1
            assert step_ends[0].data["outcome"] == "error"

            error_events = [
                e for e in callback.chain.events if e.type == EventType.ERROR
            ]
            assert len(error_events) == 1
            assert error_events[0].data["type"] == "ValueError"

    def test_nested_module_execution(self, mock_dspy):
        """Test nested module execution tracking."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()
            outer_module = MockModule(name="OuterModule")
            inner_module = MockModule(name="InnerModule")

            callback.on_module_start(outer_module, {"input": "outer"})
            assert len(callback._module_stack) == 1

            callback.on_module_start(inner_module, {"input": "inner"})
            assert len(callback._module_stack) == 2

            callback.on_module_end(inner_module, MockPrediction("inner result"))
            assert len(callback._module_stack) == 1

            callback.on_module_end(outer_module, MockPrediction("outer result"))
            assert len(callback._module_stack) == 0


class TestLCTLDSPyCallbackLLMCalls:
    """Tests for LLM call tracing."""

    def test_on_llm_call(self, mock_dspy):
        """Test on_llm_call records tool_call event."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()

            callback.on_llm_call(
                prompt="What is 2+2?",
                response="4",
                model="gpt-4",
                tokens_in=10,
                tokens_out=5,
                duration_ms=100,
            )

            assert callback._llm_call_count == 1

            tool_events = [
                e for e in callback.chain.events if e.type == EventType.TOOL_CALL
            ]
            assert len(tool_events) == 1
            assert tool_events[0].data["tool"] == "llm:gpt-4"
            assert tool_events[0].data["duration_ms"] == 100

    def test_on_llm_call_records_usage_fact(self, mock_dspy):
        """Test on_llm_call records token usage as fact."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()

            callback.on_llm_call(
                prompt="test",
                response="response",
                tokens_in=100,
                tokens_out=50,
            )

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            usage_facts = [
                e for e in fact_events if "LLM call" in e.data.get("text", "")
            ]
            assert len(usage_facts) == 1
            assert "100 tokens in" in usage_facts[0].data["text"]
            assert "50 tokens out" in usage_facts[0].data["text"]


class TestLCTLDSPyCallbackPredictions:
    """Tests for prediction tracking."""

    def test_on_prediction(self, mock_dspy):
        """Test on_prediction records fact."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()
            prediction = MockPrediction(answer="Test prediction result")

            callback.on_prediction(prediction, source="TestModule", confidence=0.9)

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            assert len(fact_events) == 1
            assert "Test prediction result" in fact_events[0].data["text"]
            assert fact_events[0].data["confidence"] == 0.9


class TestLCTLDSPyCallbackOptimization:
    """Tests for optimization/teleprompter tracing."""

    def test_on_optimization_start(self, mock_dspy):
        """Test on_optimization_start records step_start."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()

            callback.on_optimization_start(
                "BootstrapFewShot",
                {"max_bootstrapped_demos": 4, "max_labeled_demos": 8},
            )

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            assert len(step_starts) == 1
            assert step_starts[0].agent == "BootstrapFewShot"
            assert step_starts[0].data["intent"] == "optimization"

    def test_on_optimization_iteration(self, mock_dspy):
        """Test on_optimization_iteration records checkpoint."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()

            callback.on_optimization_iteration(
                iteration=5,
                score=0.85,
                best_score=0.90,
                metrics={"accuracy": 0.88},
            )

            assert callback._optimization_iteration == 5

            checkpoint_events = [
                e for e in callback.chain.events if e.type == EventType.CHECKPOINT
            ]
            assert len(checkpoint_events) == 1

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            iteration_facts = [
                e for e in fact_events if "iteration 5" in e.data.get("text", "")
            ]
            assert len(iteration_facts) == 1

    def test_on_optimization_end(self, mock_dspy):
        """Test on_optimization_end records step_end."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback()

            callback.on_optimization_start("BootstrapFewShot", {})
            callback.on_optimization_end(
                "BootstrapFewShot",
                best_score=0.92,
                total_iterations=10,
            )

            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_ends) == 1
            assert step_ends[0].data["outcome"] == "success"
            assert "10 iterations" in step_ends[0].data["output_summary"]


class TestLCTLDSPyCallbackExport:
    """Tests for export functionality."""

    def test_export_json(self, tmp_path: Path, mock_dspy):
        """Test exporting callback to JSON file."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback(chain_id="export-test")
            callback.on_module_start(MockModule(), {"question": "test"})
            callback.on_module_end(MockModule(), MockPrediction())

            export_path = tmp_path / "export.lctl.json"
            callback.export(str(export_path))

            assert export_path.exists()
            data = json.loads(export_path.read_text())
            assert data["chain"]["id"] == "export-test"
            assert len(data["events"]) >= 2

    def test_to_dict(self, mock_dspy):
        """Test to_dict returns chain dictionary."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback

            callback = LCTLDSPyCallback(chain_id="dict-test")

            result = callback.to_dict()
            assert result["chain"]["id"] == "dict-test"
            assert "events" in result


class TestTracedDSPyModule:
    """Tests for TracedDSPyModule wrapper."""

    def test_traced_module_creation(self, mock_dspy):
        """Test TracedDSPyModule creation."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, TracedDSPyModule

            callback = LCTLDSPyCallback(chain_id="traced-test")
            module = MockModule(name="TestModule")

            traced = TracedDSPyModule(module, callback)

            assert traced.module is module
            assert traced.callback is callback

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            assert len(fact_events) == 1
            assert "TestModule" in fact_events[0].data["text"]

    def test_traced_module_call(self, mock_dspy):
        """Test TracedDSPyModule __call__ traces execution."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, TracedDSPyModule

            callback = LCTLDSPyCallback(chain_id="call-test")
            module = MockModule(name="TestModule")
            traced = TracedDSPyModule(module, callback)

            result = traced(question="What is LCTL?")

            assert "LCTL" in result.answer

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_starts) >= 1
            assert len(step_ends) >= 1

    def test_traced_module_forward(self, mock_dspy):
        """Test TracedDSPyModule forward() traces execution."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, TracedDSPyModule

            callback = LCTLDSPyCallback()
            module = MockModule()
            traced = TracedDSPyModule(module, callback)

            result = traced.forward(question="test")

            assert result is not None

    def test_traced_module_error_handling(self, mock_dspy):
        """Test TracedDSPyModule handles errors correctly."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, TracedDSPyModule

            callback = LCTLDSPyCallback()

            class FailingModule(MockModule):
                def forward(self, *args, **kwargs):
                    raise RuntimeError("Module failed")

            module = FailingModule()
            traced = TracedDSPyModule(module, callback)

            with pytest.raises(RuntimeError, match="Module failed"):
                traced(question="test")

            error_events = [
                e for e in callback.chain.events if e.type == EventType.ERROR
            ]
            assert len(error_events) == 1

    def test_traced_module_export(self, tmp_path: Path, mock_dspy):
        """Test TracedDSPyModule export."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, TracedDSPyModule

            callback = LCTLDSPyCallback(chain_id="module-export")
            traced = TracedDSPyModule(MockModule(), callback)

            export_path = tmp_path / "module.lctl.json"
            traced.export(str(export_path))

            assert export_path.exists()

    def test_traced_module_proxies_attributes(self, mock_dspy):
        """Test TracedDSPyModule proxies attribute access."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, TracedDSPyModule

            callback = LCTLDSPyCallback()
            module = MockModule(name="ProxyTest")
            traced = TracedDSPyModule(module, callback)

            assert traced.name == "ProxyTest"
            assert traced.signature is not None


class TestTraceModuleFunction:
    """Tests for trace_module decorator/wrapper function."""

    def test_trace_module_with_instance(self, mock_dspy):
        """Test trace_module wrapping a module instance."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

            callback = LCTLDSPyCallback()
            module = MockModule(name="InstanceModule")

            traced = trace_module(module, callback)

            assert hasattr(traced, "module")
            assert traced.module is module

    def test_trace_module_as_decorator(self, mock_dspy):
        """Test trace_module as a decorator."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

            callback = LCTLDSPyCallback()

            @trace_module(callback)
            class DecoratedModule(MockModule):
                def forward(self, **kwargs):
                    return MockPrediction("decorated result")

            assert hasattr(DecoratedModule, "_lctl_callback")

    def test_trace_module_requires_callback(self, mock_dspy):
        """Test trace_module requires callback for instance wrapping."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import trace_module

            module = MockModule()

            with pytest.raises(ValueError, match="callback is required"):
                trace_module(module, None)


class TestLCTLDSPyTeleprompter:
    """Tests for LCTLDSPyTeleprompter tracer."""

    def test_teleprompter_creation(self, mock_dspy):
        """Test LCTLDSPyTeleprompter creation."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, LCTLDSPyTeleprompter

            callback = LCTLDSPyCallback(chain_id="teleprompter-test")
            teleprompter = MockTeleprompter()

            tracer = LCTLDSPyTeleprompter(teleprompter, callback)

            assert tracer.teleprompter is teleprompter
            assert tracer.callback is callback

            fact_events = [
                e for e in callback.chain.events if e.type == EventType.FACT_ADDED
            ]
            assert len(fact_events) == 1
            assert "MockTeleprompter" in fact_events[0].data["text"]

    def test_teleprompter_compile(self, mock_dspy):
        """Test LCTLDSPyTeleprompter compile traces optimization."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, LCTLDSPyTeleprompter

            callback = LCTLDSPyCallback()
            teleprompter = MockTeleprompter()
            tracer = LCTLDSPyTeleprompter(teleprompter, callback)

            student = MockModule(name="StudentModule")
            trainset = [{"question": "q1"}, {"question": "q2"}]

            result = tracer.compile(student, trainset=trainset)

            assert result is student

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_starts) >= 1
            assert len(step_ends) >= 1

    def test_teleprompter_compile_error(self, mock_dspy):
        """Test LCTLDSPyTeleprompter handles compile errors."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, LCTLDSPyTeleprompter

            callback = LCTLDSPyCallback()

            class FailingTeleprompter(MockTeleprompter):
                def compile(self, student, **kwargs):
                    raise RuntimeError("Optimization failed")

            teleprompter = FailingTeleprompter()
            tracer = LCTLDSPyTeleprompter(teleprompter, callback)

            with pytest.raises(RuntimeError, match="Optimization failed"):
                tracer.compile(MockModule())

            error_events = [
                e for e in callback.chain.events if e.type == EventType.ERROR
            ]
            assert len(error_events) == 1

    def test_teleprompter_record_iteration(self, mock_dspy):
        """Test LCTLDSPyTeleprompter record_iteration."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, LCTLDSPyTeleprompter

            callback = LCTLDSPyCallback()
            teleprompter = MockTeleprompter()
            tracer = LCTLDSPyTeleprompter(teleprompter, callback)

            tracer.record_iteration(5, score=0.85, best_score=0.90)

            checkpoint_events = [
                e for e in callback.chain.events if e.type == EventType.CHECKPOINT
            ]
            assert len(checkpoint_events) >= 1

    def test_teleprompter_export(self, tmp_path: Path, mock_dspy):
        """Test LCTLDSPyTeleprompter export."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, LCTLDSPyTeleprompter

            callback = LCTLDSPyCallback(chain_id="tp-export")
            tracer = LCTLDSPyTeleprompter(MockTeleprompter(), callback)

            export_path = tmp_path / "tp.lctl.json"
            tracer.export(str(export_path))

            assert export_path.exists()

    def test_teleprompter_proxies_attributes(self, mock_dspy):
        """Test LCTLDSPyTeleprompter proxies attribute access."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, LCTLDSPyTeleprompter

            callback = LCTLDSPyCallback()
            teleprompter = MockTeleprompter(metric="accuracy")
            tracer = LCTLDSPyTeleprompter(teleprompter, callback)

            assert tracer.metric == "accuracy"


class TestDSPyModuleContext:
    """Tests for DSPyModuleContext context manager."""

    def test_context_manager_records_start_end(self, mock_dspy):
        """Test context manager records start and end events."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, DSPyModuleContext

            callback = LCTLDSPyCallback()

            with DSPyModuleContext(callback, "TestModule", {"input": "test"}):
                pass

            step_starts = [
                e for e in callback.chain.events if e.type == EventType.STEP_START
            ]
            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_starts) >= 1
            assert len(step_ends) >= 1

    def test_context_manager_set_result(self, mock_dspy):
        """Test context manager set_result."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, DSPyModuleContext

            callback = LCTLDSPyCallback()

            with DSPyModuleContext(callback, "TestModule") as ctx:
                ctx.set_result(MockPrediction("my result"))

            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            assert len(step_ends) >= 1

    def test_context_manager_handles_error(self, mock_dspy):
        """Test context manager handles errors."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import LCTLDSPyCallback, DSPyModuleContext

            callback = LCTLDSPyCallback()

            with pytest.raises(ValueError):
                with DSPyModuleContext(callback, "FailingModule"):
                    raise ValueError("Test error")

            step_ends = [
                e for e in callback.chain.events if e.type == EventType.STEP_END
            ]
            error_ends = [e for e in step_ends if e.data.get("outcome") == "error"]
            assert len(error_ends) >= 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        """Test _truncate with short text."""
        from lctl.integrations.dspy import _truncate

        text = "Short text"
        result = _truncate(text, max_length=200)
        assert result == text

    def test_truncate_long_text(self):
        """Test _truncate with long text."""
        from lctl.integrations.dspy import _truncate

        text = "A" * 300
        result = _truncate(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_extract_signature_info(self):
        """Test _extract_signature_info."""
        from lctl.integrations.dspy import _extract_signature_info

        sig = MockSignature(
            input_fields={"question": "str", "context": "str"},
            output_fields={"answer": "str"},
        )

        result = _extract_signature_info(sig)
        assert "question" in result["input_fields"]
        assert "context" in result["input_fields"]
        assert "answer" in result["output_fields"]

    def test_extract_signature_info_none(self):
        """Test _extract_signature_info with None."""
        from lctl.integrations.dspy import _extract_signature_info

        result = _extract_signature_info(None)
        assert result["input_fields"] == []
        assert result["output_fields"] == []

    def test_extract_prediction_text_string(self):
        """Test _extract_prediction_text with string."""
        from lctl.integrations.dspy import _extract_prediction_text

        result = _extract_prediction_text("direct string")
        assert result == "direct string"

    def test_extract_prediction_text_prediction(self):
        """Test _extract_prediction_text with prediction object."""
        from lctl.integrations.dspy import _extract_prediction_text

        prediction = MockPrediction(answer="The answer")
        result = _extract_prediction_text(prediction)
        assert result == "The answer"

    def test_extract_prediction_text_none(self):
        """Test _extract_prediction_text with None."""
        from lctl.integrations.dspy import _extract_prediction_text

        result = _extract_prediction_text(None)
        assert result == ""

    def test_get_module_name_with_name(self):
        """Test _get_module_name with named module."""
        from lctl.integrations.dspy import _get_module_name

        module = MockModule(name="CustomName")
        result = _get_module_name(module)
        assert result == "CustomName"

    def test_get_module_name_with_signature(self):
        """Test _get_module_name with signature."""
        from lctl.integrations.dspy import _get_module_name

        class NamedModule:
            def __init__(self):
                self.signature = type("Sig", (), {"__name__": "q->a"})()

        module = NamedModule()
        result = _get_module_name(module)
        assert "NamedModule" in result

    def test_get_module_name_none(self):
        """Test _get_module_name with None."""
        from lctl.integrations.dspy import _get_module_name

        result = _get_module_name(None)
        assert result == "unknown"


class TestIntegration:
    """Integration tests for the DSPy integration."""

    def test_full_workflow(self, tmp_path: Path, mock_dspy):
        """Test complete workflow: create, trace, export."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import (
                LCTLDSPyCallback,
                TracedDSPyModule,
                trace_module,
            )

            callback = LCTLDSPyCallback(chain_id="integration-test")

            module = MockModule(name="QA")
            traced = trace_module(module, callback)

            result = traced(question="What is LCTL?")
            assert result is not None

            callback.on_llm_call(
                prompt="What is LCTL?",
                response="LCTL is a tracing format",
                model="gpt-4",
                tokens_in=50,
                tokens_out=25,
            )

            export_path = tmp_path / "workflow.lctl.json"
            callback.export(str(export_path))

            assert export_path.exists()
            data = json.loads(export_path.read_text())
            assert data["chain"]["id"] == "integration-test"
            assert len(data["events"]) >= 3

    def test_optimization_workflow(self, mock_dspy):
        """Test optimization workflow with teleprompter."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import (
                LCTLDSPyCallback,
                LCTLDSPyTeleprompter,
            )

            callback = LCTLDSPyCallback(chain_id="optimization-test")
            teleprompter = MockTeleprompter()
            tracer = LCTLDSPyTeleprompter(teleprompter, callback)

            student = MockModule(name="Student")
            trainset = [{"q": f"question_{i}"} for i in range(10)]

            optimized = tracer.compile(student, trainset=trainset)
            assert optimized is student

            engine = ReplayEngine(callback.chain)
            state = engine.replay_all()
            assert state.metrics["event_count"] > 0

    def test_error_recovery_workflow(self, mock_dspy):
        """Test workflow with error recovery."""
        with patch(
            "lctl.integrations.dspy.DSPY_AVAILABLE", True
        ):
            from lctl.integrations.dspy import (
                LCTLDSPyCallback,
                TracedDSPyModule,
            )

            callback = LCTLDSPyCallback(chain_id="error-recovery")

            class FailingModule(MockModule):
                def __init__(self):
                    super().__init__(name="FailingModule")
                    self._call_count = 0

                def forward(self, **kwargs):
                    self._call_count += 1
                    if self._call_count == 1:
                        raise ValueError("First call fails")
                    return MockPrediction("Recovered")

            failing = FailingModule()
            traced = TracedDSPyModule(failing, callback)

            with pytest.raises(ValueError):
                traced(question="test")

            result = traced(question="retry")
            assert "Recovered" in result.answer

            data = callback.to_dict()
            error_events = [e for e in data["events"] if e["type"] == "error"]
            assert len(error_events) == 1


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        """Test that all exported items exist."""
        from lctl.integrations import dspy

        for name in dspy.__all__:
            assert hasattr(dspy, name), f"Missing export: {name}"

    def test_expected_exports(self):
        """Test that expected items are exported."""
        from lctl.integrations.dspy import __all__

        expected = [
            "DSPY_AVAILABLE",
            "DSPyNotAvailableError",
            "LCTLDSPyCallback",
            "TracedDSPyModule",
            "LCTLDSPyTeleprompter",
            "DSPyModuleContext",
            "trace_module",
            "is_available",
        ]

        for item in expected:
            assert item in __all__, f"Missing from __all__: {item}"
