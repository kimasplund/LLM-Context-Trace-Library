# Using LCTL with DSPy

This tutorial shows you how to integrate LCTL (LLM Context Trace Library) with DSPy for time-travel debugging and observability of module executions, LLM calls, and teleprompter optimization runs.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.11+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **DSPy** installed:
   ```bash
   pip install dspy
   ```
4. **OpenAI API key** (or another LLM provider) set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

First, verify that both LCTL and DSPy are properly installed:

```python
from lctl.integrations.dspy import dspy_available

if dspy_available():
    print("DSPy integration is ready!")
else:
    print("Please install DSPy: pip install dspy")
```

### Step 2: Import Required Modules

```python
import dspy
from lctl.integrations.dspy import (
    LCTLDSPyCallback,
    trace_module,
)
```

## Complete Working Examples

### Example 1: Using LCTLDSPyCallback Directly

The callback approach gives you full control and works with any DSPy module.

```python
import dspy
from lctl.integrations.dspy import LCTLDSPyCallback

# Configure DSPy with your LLM
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

# Create the callback handler
callback = LCTLDSPyCallback(chain_id="qa-dspy")


# Define a simple module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


# Wrap with tracing
qa = SimpleQA()
traced_qa = trace_module(qa, callback)

# Run the module
result = traced_qa(question="What is the capital of France?")
print(f"Answer: {result.answer}")
print(f"Recorded {len(callback.chain.events)} events")

# Export the trace
callback.export("dspy_qa_trace.lctl.json")
```

### Example 2: Tracing Chain of Thought

LCTL captures the reasoning steps in Chain of Thought modules:

```python
import dspy
from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

# Create callback
callback = LCTLDSPyCallback(chain_id="cot-reasoning")

# Use Chain of Thought for reasoning
cot = dspy.ChainOfThought("question -> answer")
traced_cot = trace_module(cot, callback)

# Ask a question requiring reasoning
result = traced_cot(question="If a train travels 60 mph for 2.5 hours, how far does it go?")
print(f"Reasoning: {result.rationale}")
print(f"Answer: {result.answer}")

callback.export("cot_trace.lctl.json")
```

### Example 3: Tracing ReAct Agents

LCTL captures tool calls and reasoning in ReAct modules:

```python
import dspy
from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


# Define tools
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Python is a programming language."


def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Create ReAct module with tools
react = dspy.ReAct(signature="question -> answer", tools=[search, calculate])

# Trace it
callback = LCTLDSPyCallback(chain_id="react-agent")
traced_react = trace_module(react, callback)

# Run
result = traced_react(question="What is Python and what is 15 * 4?")
print(f"Answer: {result.answer}")
print(f"Events: {len(callback.chain.events)}")

callback.export("react_trace.lctl.json")
```

### Example 4: Tracing Custom Modules

Trace your own DSPy modules:

```python
import dspy
from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


class MultiStepReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought("problem -> analysis")
        self.solve = dspy.ChainOfThought("problem, analysis -> solution")
        self.verify = dspy.Predict("problem, solution -> verification")

    def forward(self, problem):
        # Step 1: Analyze
        analysis = self.analyze(problem=problem)

        # Step 2: Solve
        solution = self.solve(problem=problem, analysis=analysis.analysis)

        # Step 3: Verify
        verification = self.verify(problem=problem, solution=solution.solution)

        return dspy.Prediction(
            analysis=analysis.analysis,
            solution=solution.solution,
            verification=verification.verification,
        )


# Trace the multi-step module
callback = LCTLDSPyCallback(chain_id="multi-step-reasoner")
reasoner = MultiStepReasoner()
traced_reasoner = trace_module(reasoner, callback)

result = traced_reasoner(problem="Design a sorting algorithm for a linked list")
print(f"Analysis: {result.analysis[:100]}...")
print(f"Solution: {result.solution[:100]}...")

callback.export("multistep_trace.lctl.json")
```

### Example 5: Tracing Teleprompter Optimization

LCTL can trace teleprompter (optimizer) runs:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


# Define your module
class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


# Create training examples
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What color is the sky?", answer="Blue").with_inputs("question"),
]


# Define metric
def exact_match(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()


# Trace the optimization
callback = LCTLDSPyCallback(chain_id="teleprompter-optimization")

# Create optimizer
optimizer = BootstrapFewShot(metric=exact_match)

# Compile (optimize) the module
qa = QAModule()
traced_qa = trace_module(qa, callback)
optimized_qa = optimizer.compile(traced_qa, trainset=trainset)

# Test optimized module
result = optimized_qa(question="What is 3+3?")
print(f"Answer: {result.answer}")

callback.export("optimization_trace.lctl.json")
```

### Example 6: Using the Decorator Pattern

Use the decorator for cleaner code:

```python
import dspy
from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

callback = LCTLDSPyCallback(chain_id="decorator-demo")


@trace_module(callback)
class SummarizeModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.Predict("text -> summary")

    def forward(self, text):
        return self.summarize(text=text)


# Use the decorated module
summarizer = SummarizeModule()
result = summarizer(text="LCTL is a time-travel debugging library for LLM workflows...")
print(f"Summary: {result.summary}")

callback.export("decorator_trace.lctl.json")
```

## How to View Traces in the Dashboard

After exporting your traces, analyze them using LCTL's CLI tools:

### View Execution Flow

```bash
lctl trace dspy_qa_trace.lctl.json
```

Output:
```
Execution trace for chain: qa-dspy
  [1] step_start: SimpleQA (intent: forward)
  [2] step_start: Predict (intent: predict)
  [3] step_end: Predict (outcome: success, 850ms)
  [4] step_end: SimpleQA (outcome: success, 900ms)
```

### Time-Travel Replay

Step through your module execution:

```bash
lctl replay dspy_qa_trace.lctl.json
```

Replay to a specific event:

```bash
lctl replay --to-seq 3 dspy_qa_trace.lctl.json
```

### Performance Statistics

```bash
lctl stats cot_trace.lctl.json
```

Output:
```
Duration: 1.85s | Tokens: 245 in, 128 out | Cost: $0.004
```

### Find Bottlenecks

```bash
lctl bottleneck react_trace.lctl.json
```

### Compare Traces

```bash
lctl diff trace_v1.lctl.json trace_v2.lctl.json
```

### Launch Web UI

```bash
lctl debug multistep_trace.lctl.json
```

## What LCTL Captures

The DSPy integration automatically captures:

| Event Type | Description |
|------------|-------------|
| `step_start` | Module/predictor execution begins |
| `step_end` | Execution completes with outcome and duration |
| `tool_call` | Tool invocation with input/output (ReAct) |
| `fact_added` | Signature fields and predictions |
| `error` | Exceptions with context |

### DSPy-Specific Events

| Event | Data Captured |
|-------|---------------|
| Module call | Module name, input fields, signature |
| Predictor call | Predictor type (Predict, CoT, ReAct), input/output |
| Tool call | Tool name, arguments, result |
| Optimization step | Metric scores, example data |

## Troubleshooting

### Issue: DSPy Not Found

**Error**: `ImportError: DSPy is not installed`

**Solution**: Install DSPy:
```bash
pip install dspy
```

### Issue: No Events Recorded

**Problem**: Trace file has 0 events.

**Solution**: Ensure you wrap the module with `trace_module`:
```python
# Correct
traced_module = trace_module(my_module, callback)
result = traced_module(question="...")

# Wrong - not traced
result = my_module(question="...")
```

### Issue: Token Counts Show 0

**Problem**: Token usage shows 0 in traces.

**Solution**: This depends on the LLM provider. Ensure you're using a provider that reports token usage (like OpenAI).

### Issue: Tool Calls Not Captured

**Problem**: ReAct tool invocations aren't showing in traces.

**Solution**: Ensure tools are properly defined with docstrings:
```python
def my_tool(arg: str) -> str:
    """Description of what this tool does."""  # Required!
    return result
```

## Next Steps

- Learn about [LCTL CLI commands](../cli/README.md)
- Explore [LangChain integration](./langchain-tutorial.md)
- Check out [CrewAI integration](./crewai-tutorial.md)
- See [OpenAI Agents integration](./openai-agents-tutorial.md)
