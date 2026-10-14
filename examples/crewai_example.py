"""CrewAI example: a two-agent crew driven by a scripted stub LLM (no API key).

Run it in a scratch directory; it writes ``crewai_trace.lctl.json`` there. View with:

    lctl trace --tree crewai_trace.lctl.json
"""

import os

# Telemetry must be off before crewai is imported, so nothing tries the network.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-fake")  # never used: the LLM below is a stub

import sys

try:
    import crewai  # noqa: F401
except ImportError:
    print("CrewAI is not installed. Install it with:")
    print(
        '  pip install "lctl[crewai] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

import time
from pathlib import Path

from crewai import BaseLLM
from crewai.tools import tool

from lctl.integrations.crewai import LCTLAgent, LCTLCrew, LCTLTask, is_available


class ScriptedLLM(BaseLLM):
    """Replays canned ReAct text, one reply per call."""

    def __init__(self, replies):
        super().__init__(model="stub")
        self.replies = list(replies)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        time.sleep(0.01)
        return self.replies.pop(0)

    def supports_function_calling(self):
        return False


FINAL = "Thought: I now know the final answer\nFinal Answer: {}"


@tool("Search Database")
def search_database(query: str) -> str:
    """Search the internal database for information."""
    return f"Found 3 results for: {query}"


def main() -> None:
    if not is_available():
        print('LCTL CrewAI integration unavailable: pip install "lctl[crewai]"')
        sys.exit(0)

    researcher = LCTLAgent(
        role="researcher",
        goal="Find the facts",
        backstory="A careful analyst.",
        tools=[search_database],
        llm=ScriptedLLM(
            [
                'Thought: use the tool\nAction: Search Database\nAction Input: {"query": "orders"}',
                FINAL.format("three results"),
            ]
        ),
    )
    writer = LCTLAgent(
        role="writer",
        goal="Write it up",
        backstory="A concise writer.",
        llm=ScriptedLLM([FINAL.format("a short report")]),
    )
    tasks = [
        LCTLTask(description="Research {topic}", expected_output="notes", agent=researcher),
        LCTLTask(description="Write it up", expected_output="report", agent=writer),
    ]
    crew = LCTLCrew(agents=[researcher, writer], tasks=tasks, chain_id="crewai-example")
    result = crew.kickoff(inputs={"topic": "orders"})
    print(f"Crew finished: {result}")

    path = Path("crewai_trace.lctl.json")
    crew.export_trace(str(path))
    print(f"Trace written to {path.resolve()} ({len(crew.session.chain.events)} events)")


if __name__ == "__main__":
    main()
