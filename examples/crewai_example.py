"""Example: Using LCTL with CrewAI for traced crew execution.

This example demonstrates how to use LCTL to trace a CrewAI crew
for time-travel debugging and observability.

Prerequisites:
    pip install lctl crewai

Usage:
    python crewai_example.py
"""

import os

# Check if CrewAI is available
try:
    from crewai import Task

    CREWAI_INSTALLED = True
except ImportError:
    CREWAI_INSTALLED = False
    print("CrewAI is not installed. Install with: pip install crewai")
    print("This example shows the intended usage pattern.\n")

from lctl.integrations.crewai import (
    CREWAI_AVAILABLE,
    LCTLAgent,
    LCTLCrew,
    LCTLTask,
    trace_crew,
)


def main() -> None:
    """Run a sample CrewAI crew with LCTL tracing."""

    if not CREWAI_AVAILABLE:
        print("=" * 60)
        print("CrewAI Integration Example (Mock Mode)")
        print("=" * 60)
        print()
        print("CrewAI is not installed. Below is how the code would look:")
        print()
        print_example_code()
        return

    # Example 1: Using LCTL wrappers directly
    print("=" * 60)
    print("Example 1: Using LCTL Wrappers")
    print("=" * 60)

    # Create traced agents
    researcher = LCTLAgent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You work at a leading tech think tank. Your expertise lies
        in identifying emerging trends. You have a knack for dissecting complex
        data and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
    )

    writer = LCTLAgent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your
        insightful and engaging articles. You transform complex concepts into
        compelling narratives.""",
        verbose=True,
        allow_delegation=True,
    )

    # Create traced tasks
    research_task = LCTLTask(
        description="""Conduct a comprehensive analysis of the latest
        advancements in AI in 2024. Identify key trends, breakthrough
        technologies, and potential industry impacts.""",
        expected_output="""Full analysis report with detailed findings on
        AI advancements, including a summary of key trends and technologies.""",
        agent=researcher,
    )

    write_task = LCTLTask(
        description="""Using the insights provided, develop an engaging blog
        post that highlights the most significant AI advancements. Your post
        should be informative yet accessible.""",
        expected_output="""Full blog post of at least 4 paragraphs covering
        the major AI trends and their implications.""",
        agent=writer,
        context=[research_task],
    )

    # Create traced crew
    crew = LCTLCrew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process="sequential",
        verbose=True,
        chain_id="ai-research-crew",
    )

    print("\nStarting crew execution with LCTL tracing...")
    print("(Note: This requires an LLM API key to actually run)")
    print()

    # In a real scenario, you would run:
    # result = crew.kickoff()

    # For demo purposes, we'll just show the setup
    print("Crew configured with:")
    print(f"  - Chain ID: {crew.session.chain.id}")
    print(f"  - Process: sequential")
    print(f"  - Agents: {len(crew.crew.agents)}")
    print(f"  - Tasks: {len(crew.crew.tasks)}")
    print()

    # Export what we have so far (just the setup events)
    crew.export_trace("ai_research_crew.lctl.json")
    print("Trace exported to: ai_research_crew.lctl.json")
    print()

    # Example 2: Wrapping an existing crew
    print("=" * 60)
    print("Example 2: Wrapping an Existing Crew")
    print("=" * 60)
    print()
    print("You can also wrap an existing CrewAI Crew:")
    print()
    print("    from crewai import Agent, Crew, Task")
    print("    from lctl.integrations.crewai import trace_crew")
    print()
    print("    # Your existing crew")
    print("    crew = Crew(agents=[...], tasks=[...])")
    print()
    print("    # Add LCTL tracing")
    print("    traced_crew = trace_crew(crew, chain_id='my-crew')")
    print("    result = traced_crew.kickoff()")
    print("    traced_crew.export_trace('trace.lctl.json')")
    print()


def print_example_code() -> None:
    """Print example code when CrewAI is not installed."""
    code = """
from lctl.integrations.crewai import LCTLAgent, LCTLCrew, LCTLTask

# Create agents with automatic tracing
researcher = LCTLAgent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert at identifying emerging trends...",
    verbose=True,
)

writer = LCTLAgent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="Known for insightful and engaging articles...",
    verbose=True,
)

# Create tasks
research_task = LCTLTask(
    description="Analyze latest AI advancements...",
    expected_output="Full analysis report...",
    agent=researcher,
)

write_task = LCTLTask(
    description="Write an engaging blog post...",
    expected_output="Full blog post...",
    agent=writer,
    context=[research_task],
)

# Create and run the crew
crew = LCTLCrew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process="sequential",
    verbose=True,
)

# Execute with tracing
result = crew.kickoff()

# Export the trace for debugging
crew.export_trace("research_crew.lctl.json")

# Or get trace as dict
trace_data = crew.get_trace()
"""
    print(code)
    print()
    print("The trace captures:")
    print("  - Crew kickoff and completion")
    print("  - Agent steps and reasoning")
    print("  - Task execution with timing")
    print("  - Tool usage")
    print("  - Delegation events")
    print("  - Errors with context")
    print()


if __name__ == "__main__":
    main()
