"""Example: Using LCTL with DSPy.

This example demonstrates how to trace DSPy module executions,
LLM calls, and teleprompter optimization runs with LCTL for
time-travel debugging and observability.

Prerequisites:
    pip install lctl dspy-ai

Usage:
    python dspy_example.py
"""

import os

try:
    import dspy
    from dspy import ChainOfThought, Module, Predict, Signature
    from dspy.teleprompt import BootstrapFewShot

    DSPY_INSTALLED = True
except ImportError:
    DSPY_INSTALLED = False
    print("DSPy is not installed. Install with: pip install dspy-ai")
    print("This example shows the intended usage pattern.\n")

from lctl.integrations.dspy import (
    DSPY_AVAILABLE,
    DSPyModuleContext,
    LCTLDSPyCallback,
    LCTLDSPyTeleprompter,
    TracedDSPyModule,
    is_available,
    trace_module,
)


def example_basic_callback():
    """Example 1: Using LCTLDSPyCallback directly.

    This approach gives you full control over the callback
    and allows manual event recording.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="basic-example")

    print("Callback created with chain ID:", callback.chain.id)
    print()

    callback.on_module_start(
        module=type("MockModule", (), {"signature": None})(),
        inputs={"question": "What is DSPy?"},
    )

    callback.on_llm_call(
        prompt="What is DSPy?",
        response="DSPy is a framework for programming LLMs.",
        model="gpt-4",
        tokens_in=10,
        tokens_out=15,
        duration_ms=200,
    )

    class MockPrediction:
        answer = "DSPy is a framework for programming LLMs declaratively."

    callback.on_module_end(
        module=type("MockModule", (), {})(),
        prediction=MockPrediction(),
    )

    callback.export("basic_dspy_trace.lctl.json")
    print(f"Recorded {len(callback.chain.events)} events")
    print("Exported trace to: basic_dspy_trace.lctl.json")


def example_traced_module():
    """Example 2: Using trace_module wrapper.

    This approach wraps your DSPy module and automatically
    traces all forward() calls.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="traced-module-example")

    cot = ChainOfThought("question -> answer")

    traced_cot = trace_module(cot, callback)

    print(f"Wrapped module: {type(traced_cot.module).__name__}")
    print()

    print("To run the traced module:")
    print('  result = traced_cot(question="What is the meaning of life?")')
    print()

    traced_cot.export("traced_module.lctl.json")
    print("Exported trace to: traced_module.lctl.json")


def example_decorator_usage():
    """Example 3: Using trace_module as a decorator.

    This approach is useful when defining custom DSPy modules.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="decorator-example")

    @trace_module(callback)
    class QAModule(Module):
        """Simple Q&A module with tracing."""

        def __init__(self):
            super().__init__()
            self.predictor = Predict("question -> answer")

        def forward(self, question: str):
            return self.predictor(question=question)

    print("Decorated module class: QAModule")
    print("All forward() calls will be automatically traced")
    print()

    callback.export("decorator_example.lctl.json")
    print("Exported trace to: decorator_example.lctl.json")


def example_manual_tracing():
    """Example 4: Manual tracing with context manager.

    This approach is useful when you need fine-grained control
    or when integrating with existing code.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="manual-example")

    with DSPyModuleContext(callback, "CustomQA", {"question": "What is LCTL?"}):
        print("Simulating module execution...")

        callback.on_llm_call(
            prompt="What is LCTL?",
            response="LCTL is a tracing format for LLM applications.",
            model="gpt-4",
            tokens_in=8,
            tokens_out=12,
            duration_ms=150,
        )

    print(f"Recorded {len(callback.chain.events)} events")
    callback.export("manual_dspy_trace.lctl.json")
    print("Exported trace to: manual_dspy_trace.lctl.json")


def example_teleprompter_tracing():
    """Example 5: Tracing teleprompter optimization runs.

    LCTL captures optimization iterations and records checkpoints
    for time-travel debugging of optimization runs.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="optimization-example")

    def simple_metric(example, prediction, trace=None):
        return prediction.answer == example.answer

    try:
        teleprompter = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=4)
    except Exception:
        print("Could not create teleprompter. Using mock for demonstration.")

        class MockTeleprompter:
            def compile(self, student, trainset=None, **kwargs):
                return student

        teleprompter = MockTeleprompter()

    tracer = LCTLDSPyTeleprompter(teleprompter, callback, verbose=True)

    print(f"Teleprompter tracer created: {type(teleprompter).__name__}")
    print()

    tracer.record_iteration(1, score=0.65, best_score=0.65)
    tracer.record_iteration(2, score=0.72, best_score=0.72)
    tracer.record_iteration(3, score=0.68, best_score=0.72)
    tracer.record_iteration(4, score=0.78, best_score=0.78)

    print(f"Recorded {len(callback.chain.events)} events including iterations")
    tracer.export("optimization_trace.lctl.json")
    print("Exported trace to: optimization_trace.lctl.json")


def example_chain_of_thought():
    """Example 6: Tracing Chain-of-Thought reasoning.

    This example shows how LCTL captures the reasoning steps
    in a ChainOfThought module.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="cot-example", verbose=True)

    cot = ChainOfThought("question -> rationale, answer")

    traced = trace_module(cot, callback)

    print("Chain-of-Thought module wrapped for tracing")
    print("Signature: question -> rationale, answer")
    print()

    callback.on_prediction(
        prediction=type("P", (), {"rationale": "Step 1...", "answer": "42"})(),
        source="ChainOfThought",
        confidence=0.95,
    )

    traced.export("cot_trace.lctl.json")
    print("Exported trace to: cot_trace.lctl.json")


def example_error_handling():
    """Example 7: Tracing errors in DSPy execution.

    LCTL captures errors and provides context for debugging.
    """
    if not DSPY_AVAILABLE:
        print("DSPy not installed. Skipping example.")
        return

    callback = LCTLDSPyCallback(chain_id="errors-example")

    callback.on_module_start(
        module=type("FailingModule", (), {})(),
        inputs={"question": "test"},
    )
    callback.on_module_end(
        module=type("FailingModule", (), {})(),
        prediction=None,
        error=RuntimeError("Simulated module failure"),
    )

    print(f"Recorded {len(callback.chain.events)} events including error")

    data = callback.to_dict()
    error_events = [e for e in data["events"] if e["type"] == "error"]
    print(f"Error events: {len(error_events)}")

    callback.export("errors_dspy_trace.lctl.json")
    print("Exported trace to: errors_dspy_trace.lctl.json")


def example_analyze_trace():
    """Example 8: Analyzing a trace with LCTL replay engine.

    After recording a trace, you can analyze it with LCTL's
    replay engine and analysis tools.
    """
    from pathlib import Path

    from lctl import Chain, ReplayEngine

    trace_path = Path("manual_dspy_trace.lctl.json")
    if not trace_path.exists():
        print("Run example_manual_tracing() first to generate a trace.")
        return

    chain = Chain.load(trace_path)
    engine = ReplayEngine(chain)

    state = engine.replay_all()
    print(f"Total events: {state.metrics['event_count']}")
    print(f"Total duration: {state.metrics['total_duration_ms']}ms")
    print(f"Tokens in: {state.metrics['total_tokens_in']}")
    print(f"Tokens out: {state.metrics['total_tokens_out']}")
    print(f"Errors: {state.metrics['error_count']}")

    print("\nFacts discovered:")
    for fact_id, fact in state.facts.items():
        print(f"  {fact_id}: {fact['text'][:50]}...")

    bottlenecks = engine.find_bottlenecks()
    if bottlenecks:
        print("\nBottlenecks:")
        for b in bottlenecks[:3]:
            print(f"  {b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")


def print_example_code():
    """Print example code when DSPy is not installed."""
    code = '''
import dspy
from dspy import ChainOfThought, Module, Predict
from dspy.teleprompt import BootstrapFewShot
from lctl.integrations.dspy import (
    LCTLDSPyCallback,
    LCTLDSPyTeleprompter,
    trace_module,
)

# Configure DSPy with your LLM
dspy.configure(lm=dspy.LM("openai/gpt-4"))

# Option 1: Wrap existing modules
callback = LCTLDSPyCallback(chain_id="my-dspy-chain")

cot = ChainOfThought("question -> answer")
traced_cot = trace_module(cot, callback)

result = traced_cot(question="What is machine learning?")
print(result.answer)

callback.export("trace.lctl.json")

# Option 2: Use as decorator for custom modules
@trace_module(callback)
class QAModule(Module):
    def __init__(self):
        super().__init__()
        self.predictor = Predict("question -> answer")

    def forward(self, question: str):
        return self.predictor(question=question)

qa = QAModule()
result = qa(question="What is the capital of France?")

# Option 3: Trace teleprompter optimization
def accuracy_metric(example, prediction, trace=None):
    return prediction.answer.lower() == example.answer.lower()

teleprompter = BootstrapFewShot(metric=accuracy_metric)
tracer = LCTLDSPyTeleprompter(teleprompter, callback)

trainset = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="What is 3+3?", answer="6"),
]

optimized = tracer.compile(qa, trainset=trainset)
tracer.export("optimization_trace.lctl.json")

# Analyze the trace
from lctl import Chain, ReplayEngine

chain = Chain.load("trace.lctl.json")
engine = ReplayEngine(chain)
state = engine.replay_all()

print(f"Total events: {state.metrics['event_count']}")
print(f"Total duration: {state.metrics['total_duration_ms']}ms")
print(f"Facts discovered: {len(state.facts)}")
'''
    print(code)


def main():
    """Run all examples."""
    print("LCTL DSPy Integration Examples")
    print("=" * 50)

    if not DSPY_AVAILABLE:
        print("\nDSPy is not installed.")
        print("Install with: pip install dspy-ai")
        print("\nThe integration gracefully degrades when DSPy is not available.")
        print("\nExample code showing intended usage:")
        print("-" * 50)
        print_example_code()
        return

    print("\nDSPy is available!")
    print("\nRunning examples...")
    print()

    print("-" * 50)
    print("Example 1: Basic Callback")
    print("-" * 50)
    example_basic_callback()
    print()

    print("-" * 50)
    print("Example 2: Traced Module Wrapper")
    print("-" * 50)
    example_traced_module()
    print()

    print("-" * 50)
    print("Example 3: Decorator Usage")
    print("-" * 50)
    example_decorator_usage()
    print()

    print("-" * 50)
    print("Example 4: Manual Tracing")
    print("-" * 50)
    example_manual_tracing()
    print()

    print("-" * 50)
    print("Example 5: Teleprompter Optimization")
    print("-" * 50)
    example_teleprompter_tracing()
    print()

    print("-" * 50)
    print("Example 6: Chain-of-Thought Tracing")
    print("-" * 50)
    example_chain_of_thought()
    print()

    print("-" * 50)
    print("Example 7: Error Handling")
    print("-" * 50)
    example_error_handling()
    print()

    print("-" * 50)
    print("Example 8: Analyze Trace")
    print("-" * 50)
    example_analyze_trace()


if __name__ == "__main__":
    main()
