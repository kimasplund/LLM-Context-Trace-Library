"""DSPy example: dspy.Predict with DummyLM wrapped by trace_module (no API key).

Run it in a scratch directory; it writes ``dspy_trace.lctl.json`` there. View with:

    lctl trace --tree dspy_trace.lctl.json
"""

import sys

try:
    import dspy
except ImportError:
    print("DSPy is not installed. Install it with:")
    print(
        '  pip install "lctl[dspy] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

from pathlib import Path

import dspy
from dspy.utils import DummyLM

from lctl.integrations.dspy import LCTLDSPyCallback, is_available, trace_module


def main() -> None:
    if not is_available():
        print('LCTL DSPy integration unavailable: pip install "lctl[dspy]"')
        sys.exit(0)

    callback = LCTLDSPyCallback(chain_id="dspy-example")
    with dspy.context(lm=DummyLM([{"answer": "Paris"}, {"answer": "Berlin"}])):
        traced = trace_module(dspy.Predict("question -> answer"), callback)
        for question in ("Capital of France?", "Capital of Germany?"):
            print(f"{question} -> {traced(question=question).answer}")

    path = Path("dspy_trace.lctl.json")
    callback.export(str(path))
    print(f"Trace written to {path.resolve()} ({len(callback.session.chain.events)} events)")


if __name__ == "__main__":
    main()
