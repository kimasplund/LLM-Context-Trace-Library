# Using LCTL with AutoGen/AG2

This tutorial shows you how to integrate LCTL (LLM Context Transfer Language) v4.0 with AutoGen (or AG2) agent conversations and GroupChat for time-travel debugging and observability.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed
2. **LCTL** installed:
   ```bash
   pip install lctl
   ```
3. **AutoGen** installed (choose one):
   ```bash
   pip install autogen-agentchat   # Original AutoGen
   # or
   pip install ag2                  # AG2 fork
   ```
4. **OpenAI API key** (or another LLM provider) set as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Step-by-Step Setup

### Step 1: Verify Installation

First, verify that both LCTL and AutoGen are properly installed:

```python
from lctl.integrations.autogen import AUTOGEN_AVAILABLE

if AUTOGEN_AVAILABLE:
    print("AutoGen integration is ready!")
else:
    print("Please install AutoGen: pip install autogen-agentchat")
```

### Step 2: Import Required Modules

```python
# For original AutoGen
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Or for AG2
# from ag2 import ConversableAgent, GroupChat, GroupChatManager

from lctl.integrations.autogen import (
    LCTLAutogenCallback,
    LCTLConversableAgent,
    LCTLGroupChatManager,
    trace_agent,
    trace_group_chat,
)
```

## Complete Working Examples

### Example 1: Using LCTL Wrapper Classes

The simplest approach - use LCTL wrapper classes for automatic tracing:

```python
from lctl.integrations.autogen import LCTLConversableAgent

# Create agents with built-in tracing
assistant = LCTLConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant. Provide concise, accurate answers.",
    llm_config={"model": "gpt-4"},
    chain_id="simple-chat",  # All agents share this chain ID
)

user_proxy = LCTLConversableAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Automated user for testing
    max_consecutive_auto_reply=3,
    chain_id="simple-chat",  # Same chain ID to combine traces
)

# Run the conversation
result = assistant.initiate_chat(
    user_proxy,
    message="What are three key benefits of using Python for data science?"
)

# Export the trace
assistant.export_trace("assistant_chat.lctl.json")
```

### Example 2: Attaching Callback to Existing Agents

If you have existing agents, attach LCTL tracing via callback:

```python
from autogen import ConversableAgent
from lctl.integrations.autogen import LCTLAutogenCallback

# Your existing agents
researcher = ConversableAgent(
    name="researcher",
    system_message="You are a research expert. Find and verify information.",
    llm_config={"model": "gpt-4"},
)

writer = ConversableAgent(
    name="writer",
    system_message="You are a technical writer. Create clear documentation.",
    llm_config={"model": "gpt-4"},
)

# Create callback and attach to agents
callback = LCTLAutogenCallback(chain_id="research-workflow")
callback.attach(researcher)
callback.attach(writer)

# Run conversation - tracing happens automatically
researcher.initiate_chat(
    writer,
    message="I've researched the topic. Here are the key points: ..."
)

# Export trace
callback.export("research_workflow.lctl.json")
```

### Example 3: Tracing GroupChat Conversations

Trace multi-agent GroupChat with speaker transitions:

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager
from lctl.integrations.autogen import LCTLGroupChatManager

# Create specialist agents
planner = ConversableAgent(
    name="planner",
    system_message="You plan and coordinate tasks. Break down problems into steps.",
    llm_config={"model": "gpt-4"},
)

coder = ConversableAgent(
    name="coder",
    system_message="You write clean, efficient Python code.",
    llm_config={"model": "gpt-4"},
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="You review code for bugs, security issues, and best practices.",
    llm_config={"model": "gpt-4"},
)

# Create GroupChat
group_chat = GroupChat(
    agents=[planner, coder, reviewer],
    messages=[],
    max_round=10,
)

# Create traced GroupChatManager
manager = LCTLGroupChatManager(
    groupchat=group_chat,
    name="dev_manager",
    chain_id="dev-team-chat",
)

# Start the group discussion
planner.initiate_chat(
    manager,
    message="We need to implement a function to validate email addresses."
)

# Export trace with all agent interactions
manager.export_trace("dev_team_chat.lctl.json")
```

### Example 4: Using trace_agent Helper

Quick tracing for a single agent:

```python
from autogen import ConversableAgent
from lctl.integrations.autogen import trace_agent

# Create your agent
agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={"model": "gpt-4"},
)

# Add tracing with one line
callback = trace_agent(agent, chain_id="quick-trace")

# Create a user proxy for the conversation
user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
)
callback.attach(user)  # Attach user to same trace

# Run conversation
agent.initiate_chat(user, message="Hello! How can I help you today?")

# Export
callback.export("quick_trace.lctl.json")
```

### Example 5: Tracing GroupChat with Helper

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager
from lctl.integrations.autogen import trace_group_chat

# Create agents
agent1 = ConversableAgent(
    name="analyst",
    system_message="You analyze data and provide insights.",
    llm_config={"model": "gpt-4"},
)

agent2 = ConversableAgent(
    name="strategist",
    system_message="You develop strategies based on analysis.",
    llm_config={"model": "gpt-4"},
)

# Create group chat and manager
group_chat = GroupChat(agents=[agent1, agent2], messages=[], max_round=5)
manager = GroupChatManager(groupchat=group_chat)

# Add tracing
callback = trace_group_chat(group_chat, manager, chain_id="strategy-discussion")

# Run discussion
agent1.initiate_chat(manager, message="Let's analyze the Q4 results.")

# Export
callback.export("strategy_trace.lctl.json")
```

### Example 6: Tracking Nested Conversations

AutoGen supports nested chats. Track these hierarchical conversations:

```python
from autogen import ConversableAgent
from lctl.integrations.autogen import LCTLAutogenCallback

# Create agents
orchestrator = ConversableAgent(
    name="orchestrator",
    system_message="You coordinate work between specialists.",
    llm_config={"model": "gpt-4"},
)

specialist1 = ConversableAgent(
    name="data_specialist",
    system_message="You handle data processing tasks.",
    llm_config={"model": "gpt-4"},
)

specialist2 = ConversableAgent(
    name="ml_specialist",
    system_message="You handle machine learning tasks.",
    llm_config={"model": "gpt-4"},
)

# Setup tracing
callback = LCTLAutogenCallback(chain_id="nested-workflow")
callback.attach(orchestrator)
callback.attach(specialist1)
callback.attach(specialist2)

# Track nested conversation explicitly
callback.start_nested_chat("orchestrator", "Delegating data processing")

# Nested conversation with specialist1
orchestrator.initiate_chat(
    specialist1,
    message="Process the raw data files."
)

callback.end_nested_chat("Data processing complete", outcome="success")

# Another nested conversation
callback.start_nested_chat("orchestrator", "Delegating ML training")

orchestrator.initiate_chat(
    specialist2,
    message="Train the model on processed data."
)

callback.end_nested_chat("Model training complete", outcome="success")

# Export full trace with nested structure
callback.export("nested_workflow.lctl.json")
```

### Example 7: Agents with Function Calling

Trace agents that use tool/function calls:

```python
from autogen import ConversableAgent
from lctl.integrations.autogen import LCTLAutogenCallback

# Define tools/functions
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72F"

def search_database(query: str) -> str:
    """Search the database."""
    return f"Found 3 results for: {query}"

# Create agent with tools
assistant = ConversableAgent(
    name="assistant",
    system_message="You help users with weather and database queries.",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "search_database",
                "description": "Search the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]
    },
)

# Register function executors
assistant.register_function(
    function_map={
        "get_weather": get_weather,
        "search_database": search_database,
    }
)

user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
)

# Setup tracing
callback = LCTLAutogenCallback(chain_id="function-calling-demo")
callback.attach(assistant)
callback.attach(user)

# Run - tool calls will be traced
assistant.initiate_chat(
    user,
    message="What's the weather in Paris? Also search for 'sales reports'."
)

callback.export("function_calls_trace.lctl.json")
```

## How to View Traces in the Dashboard

After exporting your traces, analyze them using LCTL's CLI tools:

### View Execution Flow

```bash
lctl trace dev_team_chat.lctl.json
```

Output:
```
Execution trace for chain: dev-team-chat
  [1] fact_added: groupchat-config (3 agents, max_round=10)
  [2] fact_added: groupchat-agents (planner, coder, reviewer)
  [3] step_start: planner (intent: send_message)
  [4] step_end: planner (outcome: success)
  [5] step_start: coder (intent: generate_reply)
  [6] step_end: coder (outcome: success)
  ...
```

### Time-Travel Replay

```bash
lctl replay dev_team_chat.lctl.json
```

Replay to a specific point:

```bash
lctl replay --to-seq 10 dev_team_chat.lctl.json
```

### Performance Statistics

```bash
lctl stats dev_team_chat.lctl.json
```

Output:
```
Duration: 32.5s | Tokens: 1,890 in, 720 out | Cost: $0.038
Messages: 8 | Agents: 3 | Rounds: 4
```

### Find Communication Bottlenecks

```bash
lctl bottleneck dev_team_chat.lctl.json
```

Output:
```
coder: 45% of time (14.6s)
  - Consider breaking into smaller tasks
reviewer: 35% of time (11.4s)
planner: 20% of time (6.5s)
```

### Compare Conversations

```bash
lctl diff conversation_v1.lctl.json conversation_v2.lctl.json
```

### Launch Web UI

```bash
lctl debug dev_team_chat.lctl.json
```

## Common Patterns and Best Practices

### 1. Share Chain IDs for Related Agents

```python
# All related agents should share the same chain ID
chain_id = "customer-support-v2"

assistant = LCTLConversableAgent(name="assistant", chain_id=chain_id, ...)
support = LCTLConversableAgent(name="support", chain_id=chain_id, ...)
escalation = LCTLConversableAgent(name="escalation", chain_id=chain_id, ...)
```

### 2. Record Custom Tool Results

When tools return results outside the standard message flow:

```python
callback = LCTLAutogenCallback(chain_id="custom-tools")
callback.attach(agent)

# After a tool executes
callback.record_tool_result(
    tool_name="database_query",
    result={"rows": 150, "status": "success"},
    duration_ms=250,
    agent="assistant"
)
```

### 3. Handle Errors Gracefully

```python
callback = LCTLAutogenCallback(chain_id="error-handling")
callback.attach(agent)

try:
    agent.initiate_chat(other_agent, message="Process this request")
except Exception as e:
    # Record the error in the trace
    callback.record_error(e, agent="agent", recoverable=False)
    raise
finally:
    # Always export, even on errors
    callback.export("error_trace.lctl.json")
```

### 4. Analyze Traces Programmatically

```python
from lctl import Chain, ReplayEngine

# Load and analyze
chain = Chain.load("dev_team_chat.lctl.json")
engine = ReplayEngine(chain)

# Replay all events
state = engine.replay_all()

# Get metrics
print(f"Total events: {state.metrics['event_count']}")
print(f"Duration: {state.metrics['total_duration_ms']}ms")
print(f"Errors: {state.metrics['error_count']}")

# Extract facts (key observations)
print("\nDiscovered facts:")
for fact_id, fact in state.facts.items():
    print(f"  {fact_id}: {fact['text'][:60]}...")

# Get execution trace
trace = engine.get_trace()
for step in trace:
    print(f"[{step['seq']}] {step['type']}: {step['agent']}")
```

### 5. Export as Dictionary for Monitoring

```python
trace_data = callback.to_dict()

# Send to monitoring service
import json
import requests

requests.post(
    "https://monitoring.example.com/traces",
    json=trace_data
)
```

### 6. Combine with Human Input Mode

```python
# For conversations with human input
user = LCTLConversableAgent(
    name="human",
    human_input_mode="ALWAYS",  # or "TERMINATE"
    chain_id="human-in-loop",
)

assistant = LCTLConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    chain_id="human-in-loop",
)

# Human inputs will be traced as messages
result = assistant.initiate_chat(
    user,
    message="How can I help you today?"
)

assistant.export_trace("human_in_loop.lctl.json")
```

## Troubleshooting

### Issue: AutoGen Not Found

**Error**: `AutogenNotAvailableError: AutoGen is not installed`

**Solution**: Install AutoGen:
```bash
pip install autogen-agentchat
# or
pip install ag2
```

### Issue: No Messages in Trace

**Problem**: Trace shows agent attachments but no message events.

**Solutions**:

1. Ensure agents are attached before the conversation starts:
   ```python
   callback = LCTLAutogenCallback(chain_id="my-chat")
   callback.attach(agent1)
   callback.attach(agent2)
   # Now start the conversation
   agent1.initiate_chat(agent2, message="Hello")
   ```

2. Verify the conversation actually runs (check for LLM errors).

### Issue: GroupChat Agents Not Traced

**Problem**: Individual agent messages in GroupChat not appearing.

**Solution**: Use `attach_group_chat` or attach all agents:
```python
callback = LCTLAutogenCallback(chain_id="group-chat")

# Option 1: Attach entire group chat
callback.attach_group_chat(group_chat, manager)

# Option 2: Attach each agent manually
for agent in group_chat.agents:
    callback.attach(agent)
callback.attach(manager)
```

### Issue: Tool Calls Not Captured

**Problem**: Function/tool calls don't appear in trace.

**Solution**: The LCTL callback captures tool calls from message content. Ensure:
1. Functions are properly registered with the agent
2. The LLM actually makes function calls (check raw responses)
3. Use `record_tool_result()` for manual tracking if needed

### Issue: Nested Chat Tracking Incorrect

**Problem**: Nested conversations appear flat in the trace.

**Solution**: Explicitly mark nested chat boundaries:
```python
callback.start_nested_chat("orchestrator", "Delegating to specialist")
# ... nested conversation ...
callback.end_nested_chat("Completed", outcome="success")
```

### Issue: Multiple Sessions Conflict

**Problem**: Events from different conversations mix together.

**Solution**: Use separate callbacks or chain IDs:
```python
# Option 1: Separate callbacks
callback1 = LCTLAutogenCallback(chain_id="session-1")
callback2 = LCTLAutogenCallback(chain_id="session-2")

# Option 2: Same callback, different chain IDs
# (Note: Create new callback for each session)
```

### Issue: Trace File Empty

**Problem**: Exported file has no events.

**Solution**:
1. Verify conversation completed:
   ```python
   result = agent.initiate_chat(other, message="Hello")
   print(f"Conversation finished: {result}")
   ```
2. Check that `export()` is called after the conversation:
   ```python
   agent.initiate_chat(...)
   callback.export("trace.lctl.json")  # Must be after
   ```

## What LCTL Captures

The AutoGen integration automatically captures:

| Event Type | Description |
|------------|-------------|
| `step_start` | Message send, reply generation starts |
| `step_end` | Operation completes with outcome |
| `tool_call` | Function/tool invocations |
| `fact_added` | Agent attachment, GroupChat config, tool responses |
| `error` | Exceptions with context |

## Trace Structure Example

Here's what a typical AutoGen conversation trace looks like:

```json
{
  "lctl": "4.0",
  "chain": {
    "id": "dev-team-chat"
  },
  "events": [
    {
      "seq": 1,
      "type": "fact_added",
      "agent": "lctl-autogen",
      "data": {
        "id": "groupchat-config",
        "text": "GroupChat configured with 3 agents, max_round=10"
      }
    },
    {
      "seq": 2,
      "type": "step_start",
      "agent": "planner",
      "data": {
        "intent": "send_message",
        "input_summary": "To dev_manager: We need to implement..."
      }
    },
    {
      "seq": 3,
      "type": "step_end",
      "agent": "planner",
      "data": {
        "outcome": "success",
        "output_summary": "Message sent to dev_manager"
      }
    }
  ]
}
```

## Next Steps

- Learn about [LCTL CLI commands](../cli/README.md)
- Explore [LangChain integration](./langchain-tutorial.md)
- Check out [CrewAI integration](./crewai-tutorial.md)
- See [OpenAI Agents integration](./openai-agents-tutorial.md)
