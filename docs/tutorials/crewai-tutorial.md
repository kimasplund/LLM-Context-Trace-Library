# Using LCTL with CrewAI

This tutorial shows you how to integrate LCTL (LLM Context Transfer Language) v4.0 with CrewAI crews for time-travel debugging and observability of multi-agent workflows.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **CrewAI** installed:
   ```bash
   pip install crewai
   ```
4. **OpenAI API key** (or another LLM provider) set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

First, verify that both LCTL and CrewAI are properly installed:

```python
from lctl.integrations.crewai import CREWAI_AVAILABLE

if CREWAI_AVAILABLE:
    print("CrewAI integration is ready!")
else:
    print("Please install CrewAI: pip install crewai")
```

### Step 2: Import Required Modules

```python
from lctl.integrations.crewai import (
    LCTLAgent,
    LCTLTask,
    LCTLCrew,
    trace_crew,
)
```

## Complete Working Examples

### Example 1: Building a Traced Crew from Scratch

Create agents and tasks with built-in LCTL tracing:

```python
from lctl.integrations.crewai import LCTLAgent, LCTLTask, LCTLCrew

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
    context=[research_task],  # This task depends on research_task
)

# Create the traced crew
crew = LCTLCrew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process="sequential",
    verbose=True,
    chain_id="ai-research-crew",
)

# Execute the crew
result = crew.kickoff()
print(f"Result: {result}")

# Export the trace
crew.export_trace("ai_research_crew.lctl.json")
```

### Example 2: Wrapping an Existing Crew

If you already have a CrewAI crew, wrap it with LCTL tracing:

```python
from crewai import Agent, Crew, Task
from lctl.integrations.crewai import trace_crew

# Your existing CrewAI setup
agent1 = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with attention to detail",
)

agent2 = Agent(
    role="Writer",
    goal="Create clear documentation",
    backstory="Technical writer with clarity focus",
)

task1 = Task(
    description="Research the topic thoroughly",
    expected_output="Detailed research notes",
    agent=agent1,
)

task2 = Task(
    description="Write documentation from research",
    expected_output="Clear documentation",
    agent=agent2,
)

# Create the crew
crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])

# Wrap with LCTL tracing
traced_crew = trace_crew(crew, chain_id="documentation-crew")

# Execute with tracing
result = traced_crew.kickoff()

# Export trace
traced_crew.export_trace("documentation_trace.lctl.json")
```

### Example 3: Hierarchical Process with Manager

Trace a hierarchical crew with a manager agent:

```python
from lctl.integrations.crewai import LCTLAgent, LCTLTask, LCTLCrew

# Create specialist agents
analyst = LCTLAgent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="Expert in statistical analysis and data visualization",
    verbose=True,
)

developer = LCTLAgent(
    role="Python Developer",
    goal="Write clean, efficient code",
    backstory="Senior developer with expertise in data processing",
    verbose=True,
)

reviewer = LCTLAgent(
    role="Code Reviewer",
    goal="Ensure code quality and best practices",
    backstory="Tech lead with focus on maintainability",
    verbose=True,
)

# Create tasks
analysis_task = LCTLTask(
    description="Analyze the dataset and identify key patterns",
    expected_output="Analysis report with key findings",
    agent=analyst,
)

coding_task = LCTLTask(
    description="Implement data processing pipeline based on analysis",
    expected_output="Python code for data processing",
    agent=developer,
)

review_task = LCTLTask(
    description="Review the code for quality and suggest improvements",
    expected_output="Code review with recommendations",
    agent=reviewer,
)

# Create hierarchical crew (manager coordinates agents)
crew = LCTLCrew(
    agents=[analyst, developer, reviewer],
    tasks=[analysis_task, coding_task, review_task],
    process="hierarchical",
    verbose=True,
    chain_id="data-pipeline-team",
)

result = crew.kickoff()
crew.export_trace("hierarchical_crew.lctl.json")
```

### Example 4: Crew with Custom Tools

Trace agents that use custom tools:

```python
from crewai_tools import tool
from lctl.integrations.crewai import LCTLAgent, LCTLTask, LCTLCrew

# Define custom tools
@tool("Search Database")
def search_database(query: str) -> str:
    """Search the internal database for information."""
    # Simulated database search
    return f"Found 3 results for: {query}"

@tool("Send Notification")
def send_notification(message: str) -> str:
    """Send a notification to the team."""
    return f"Notification sent: {message}"

# Create agent with tools
support_agent = LCTLAgent(
    role="Customer Support Specialist",
    goal="Help customers resolve their issues efficiently",
    backstory="Experienced support agent with deep product knowledge",
    tools=[search_database, send_notification],
    verbose=True,
)

# Create task
support_task = LCTLTask(
    description="""Help the customer find information about their order
    and notify the shipping team if needed.""",
    expected_output="Resolution summary with actions taken",
    agent=support_agent,
)

# Create crew
crew = LCTLCrew(
    agents=[support_agent],
    tasks=[support_task],
    chain_id="support-workflow",
    verbose=True,
)

# The trace will capture tool invocations
result = crew.kickoff()
crew.export_trace("support_trace.lctl.json")
```

### Example 5: Passing Inputs to Crew

Trace crews that accept dynamic inputs:

```python
from lctl.integrations.crewai import LCTLAgent, LCTLTask, LCTLCrew

translator = LCTLAgent(
    role="Professional Translator",
    goal="Provide accurate translations",
    backstory="Multilingual expert with cultural awareness",
)

translation_task = LCTLTask(
    description="Translate the following text to {target_language}: {text}",
    expected_output="Accurate translation preserving meaning and tone",
    agent=translator,
)

crew = LCTLCrew(
    agents=[translator],
    tasks=[translation_task],
    chain_id="translation-service",
)

# Pass inputs at runtime
result = crew.kickoff(inputs={
    "target_language": "Spanish",
    "text": "Hello, how can I help you today?"
})

crew.export_trace("translation_trace.lctl.json")
```

## How to View Traces in the Dashboard

After exporting your traces, analyze them using LCTL's CLI tools:

### View Execution Flow

```bash
lctl trace ai_research_crew.lctl.json
```

Output:
```
Execution trace for chain: ai-research-crew
  [1] step_start: crew-manager (intent: kickoff)
  [2] fact_added: crew-config
  [3] fact_added: agent-0 (Senior Research Analyst)
  [4] fact_added: agent-1 (Tech Content Strategist)
  [5] step_start: senior-research-analyst (intent: execute_step)
  [6] tool_call: search_tool
  [7] step_end: senior-research-analyst (outcome: success)
  ...
```

### Time-Travel Replay

Replay the crew execution step by step:

```bash
lctl replay ai_research_crew.lctl.json
```

Replay to a specific point (before the writer started):

```bash
lctl replay --to-seq 15 ai_research_crew.lctl.json
```

### Performance Statistics

```bash
lctl stats ai_research_crew.lctl.json
```

Output:
```
Duration: 45.2s | Tokens: 2,340 in, 890 out | Cost: $0.047
Agents: 2 | Tasks: 2 | Steps: 12
```

### Find Bottlenecks

```bash
lctl bottleneck ai_research_crew.lctl.json
```

Output:
```
senior-research-analyst: 60% of time (27.1s)
  - Consider caching research results
tech-content-strategist: 35% of time (15.8s)
crew-manager: 5% of time (2.3s)
```

### Compare Runs

Compare two crew executions to see what changed:

```bash
lctl diff crew_run_v1.lctl.json crew_run_v2.lctl.json
```

### Launch Web UI

```bash
lctl debug ai_research_crew.lctl.json
```

## Common Patterns and Best Practices

### 1. Use Descriptive Chain IDs

```python
# Good - descriptive and versioned
crew = LCTLCrew(..., chain_id="content-pipeline-v2")
crew = LCTLCrew(..., chain_id="customer-onboarding-flow")

# Avoid - generic names
crew = LCTLCrew(..., chain_id="test")
crew = LCTLCrew(...)  # Auto-generates UUID
```

### 2. Access Raw Session for Custom Events

Add custom facts or events during execution:

```python
crew = LCTLCrew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    chain_id="custom-events-demo",
)

# Access the LCTL session for custom tracing
session = crew.session

# Add a custom fact before kickoff
session.add_fact(
    fact_id="initial-context",
    text="Processing batch #12345 with 500 records",
    confidence=1.0,
    source="orchestrator",
)

result = crew.kickoff()

# Add completion fact
session.add_fact(
    fact_id="batch-complete",
    text="Batch #12345 processed successfully",
    confidence=1.0,
    source="orchestrator",
)

crew.export_trace("custom_events.lctl.json")
```

### 3. Analyze Traces Programmatically

```python
from lctl import Chain, ReplayEngine

# Load the trace
chain = Chain.load("ai_research_crew.lctl.json")
engine = ReplayEngine(chain)

# Replay all events
state = engine.replay_all()

# Access metrics
print(f"Total events: {state.metrics['event_count']}")
print(f"Duration: {state.metrics['total_duration_ms']}ms")
print(f"Errors: {state.metrics['error_count']}")

# Get discovered facts
print("\nFacts discovered during execution:")
for fact_id, fact in state.facts.items():
    print(f"  {fact_id}: {fact['text'][:60]}...")

# Find execution bottlenecks
bottlenecks = engine.find_bottlenecks()
print("\nTop bottlenecks:")
for b in bottlenecks[:3]:
    print(f"  {b['agent']}: {b['duration_ms']}ms ({b['percentage']:.1f}%)")
```

### 4. Get Trace as Dictionary

For integration with monitoring systems:

```python
trace_data = crew.get_trace()

# Example: Send to monitoring service
import json
print(json.dumps(trace_data, indent=2))

# Or extract specific information
events = trace_data.get("events", [])
errors = [e for e in events if e.get("type") == "error"]
if errors:
    print(f"Warning: {len(errors)} errors occurred")
```

### 5. Compare Sequential vs Hierarchical

Run the same tasks with different processes and compare:

```python
# Sequential execution
seq_crew = LCTLCrew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process="sequential",
    chain_id="comparison-sequential",
)
seq_crew.kickoff()
seq_crew.export_trace("sequential_run.lctl.json")

# Hierarchical execution
hier_crew = LCTLCrew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process="hierarchical",
    chain_id="comparison-hierarchical",
)
hier_crew.kickoff()
hier_crew.export_trace("hierarchical_run.lctl.json")

# Compare the traces
# $ lctl diff sequential_run.lctl.json hierarchical_run.lctl.json
```

## Troubleshooting

### Issue: CrewAI Not Found

**Error**: `CrewAINotAvailableError: CrewAI is not installed`

**Solution**: Install CrewAI:
```bash
pip install crewai
```

### Issue: No Step Events in Trace

**Problem**: Trace only shows kickoff and completion, no intermediate steps.

**Solutions**:

1. Ensure `verbose=True` on the crew:
   ```python
   crew = LCTLCrew(..., verbose=True)
   ```

2. Enable verbose on agents:
   ```python
   agent = LCTLAgent(..., verbose=True)
   ```

3. CrewAI step callbacks require certain configurations - check that your agents are executing properly.

### Issue: Trace Export Fails

**Problem**: `export_trace()` raises an error.

**Solutions**:

1. Ensure the directory exists:
   ```python
   import os
   os.makedirs("traces", exist_ok=True)
   crew.export_trace("traces/my_trace.lctl.json")
   ```

2. Check file permissions for the target directory.

### Issue: Task Context Not Captured

**Problem**: Task dependencies (context) don't appear in traces.

**Solution**: Ensure tasks with context use `LCTLTask`:
```python
# Both tasks should be LCTLTask for full tracing
task1 = LCTLTask(...)
task2 = LCTLTask(..., context=[task1])  # Context properly traced
```

### Issue: Tool Calls Missing

**Problem**: Tool invocations don't appear in the trace.

**Solution**: Ensure you're using `LCTLAgent` which properly hooks into tool execution:
```python
# Tools will be traced
agent = LCTLAgent(
    role="...",
    tools=[my_tool],
    ...
)
```

### Issue: Async Kickoff Incomplete Traces

**Problem**: `kickoff_async()` produces incomplete traces.

**Solution**: Async tracing starts but may not capture all events if the process completes before the script ends. Ensure you await completion:
```python
import asyncio

async def run_crew():
    result = await crew.kickoff_async()
    # Wait for async operations to complete
    crew.export_trace("async_trace.lctl.json")
    return result

asyncio.run(run_crew())
```

## What LCTL Captures

The CrewAI integration automatically captures:

| Event Type | Description |
|------------|-------------|
| `step_start` | Crew kickoff, agent step execution |
| `step_end` | Step completion with outcome and duration |
| `tool_call` | Tool invocations with input/output |
| `fact_added` | Crew config, agent roles, task completions |
| `error` | Execution errors with context |

## Trace Structure Example

Here's what a typical crew trace looks like:

```json
{
  "lctl": "4.0",
  "chain": {
    "id": "ai-research-crew"
  },
  "events": [
    {
      "seq": 1,
      "type": "step_start",
      "agent": "crew-manager",
      "data": {"intent": "kickoff", "input_summary": "Starting sequential crew with 2 agents"}
    },
    {
      "seq": 2,
      "type": "fact_added",
      "agent": "crew-manager",
      "data": {"id": "crew-config", "text": "Crew configuration: sequential process, 2 agents, 2 tasks"}
    },
    {
      "seq": 3,
      "type": "fact_added",
      "agent": "crew-manager",
      "data": {"id": "agent-0", "text": "Agent 'Senior Research Analyst': Uncover cutting-edge developments..."}
    }
  ]
}
```

## Next Steps

- Learn about [LCTL CLI commands](../cli/README.md)
- Explore [LangChain integration](./langchain-tutorial.md)
- Check out [AutoGen integration](./autogen-tutorial.md)
- See [OpenAI Agents integration](./openai-agents-tutorial.md)
