"""Span API demo (core only, no framework, no API key).

Writes ``spans_demo.lctl.json`` into the current directory. View it with:

    lctl trace --tree spans_demo.lctl.json
"""

import asyncio
from pathlib import Path

from lctl import LCTLSession


async def worker(session: LCTLSession, name: str, delay: float) -> None:
    with session.span(name, "work", f"{name} job"):
        await asyncio.sleep(delay)
        session.tool_call("fetch", {"source": name}, {"rows": 3}, duration_ms=int(delay * 1000))


async def run(session: LCTLSession) -> None:
    with session.span("planner", "plan", "build the report"):
        with session.span("researcher", "research", "collect inputs"):
            session.tool_call("search", {"q": "sales"}, {"hits": 12}, duration_ms=40)
        with session.span("writer", "draft", "write the summary"):
            session.add_fact("F1", "Sales grew 12%", confidence=0.85)

        # Two workers in parallel: siblings under the planner, in separate lanes.
        await asyncio.gather(worker(session, "worker-a", 0.02), worker(session, "worker-b", 0.03))

        # An error caught inside a span: the span ends in error, the planner carries on.
        try:
            with session.span("validator", "validate", "check totals"):
                raise ValueError("totals do not match")
        except ValueError:
            session.tool_call("log", {"msg": "validation failed, using fallback"}, {"ok": True})


def main() -> None:
    with LCTLSession(chain_id="spans-demo") as session:
        asyncio.run(run(session))
    path = Path("spans_demo.lctl.json")
    session.export(str(path))
    print(f"Trace written to {path.resolve()}")
    print(f"View it with: lctl trace --tree {path}")


if __name__ == "__main__":
    main()
