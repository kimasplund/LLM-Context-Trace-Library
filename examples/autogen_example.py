"""Example: Using LCTL with AutoGen/AG2 for traced agent conversations.

This example demonstrates how to use LCTL to trace AutoGen agent
conversations for time-travel debugging and observability.

Prerequisites:
    pip install lctl autogen-agentchat
    # or
    pip install lctl ag2

Usage:
    python autogen_example.py
"""

import os

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_INSTALLED = True
except ImportError:
    try:
        from ag2 import ConversableAgent, GroupChat, GroupChatManager
        AUTOGEN_INSTALLED = True
    except ImportError:
        AUTOGEN_INSTALLED = False
        print("AutoGen is not installed. Install with: pip install autogen-agentchat")
        print("Or for AG2: pip install ag2")
        print("This example shows the intended usage pattern.\n")

from lctl.integrations.autogen import (
    AUTOGEN_AVAILABLE,
    LCTLAutogenCallback,
    LCTLConversableAgent,
    LCTLGroupChatManager,
    trace_agent,
    trace_group_chat,
)


def main() -> None:
    """Run sample AutoGen conversations with LCTL tracing."""

    if not AUTOGEN_AVAILABLE:
        print("=" * 60)
        print("AutoGen Integration Example (Mock Mode)")
        print("=" * 60)
        print()
        print("AutoGen is not installed. Below is how the code would look:")
        print()
        print_example_code()
        return

    print("=" * 60)
    print("Example 1: Using LCTL Wrappers")
    print("=" * 60)
    print()

    assistant = LCTLConversableAgent(
        name="assistant",
        system_message="You are a helpful AI assistant. Respond concisely.",
        llm_config={"model": "gpt-4"},
        chain_id="simple-chat",
    )

    user_proxy = LCTLConversableAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        chain_id="simple-chat",
    )

    print("Agents configured with LCTL tracing:")
    print(f"  - Chain ID: {assistant.session.chain.id}")
    print(f"  - Assistant: {assistant.agent.name}")
    print(f"  - User Proxy: {user_proxy.agent.name}")
    print()

    assistant.export_trace("simple_chat.lctl.json")
    print("Trace exported to: simple_chat.lctl.json")
    print()

    print("=" * 60)
    print("Example 2: Using Callback with Existing Agents")
    print("=" * 60)
    print()

    agent1 = ConversableAgent(
        name="researcher",
        system_message="You are a research expert.",
        llm_config={"model": "gpt-4"},
    )

    agent2 = ConversableAgent(
        name="writer",
        system_message="You are a technical writer.",
        llm_config={"model": "gpt-4"},
    )

    callback = LCTLAutogenCallback(chain_id="research-workflow")
    callback.attach(agent1)
    callback.attach(agent2)

    print("Callback attached to agents:")
    print(f"  - Chain ID: {callback.session.chain.id}")
    print(f"  - Attached agents: {len(callback._attached_agents)}")
    print()

    callback.export("research_workflow.lctl.json")
    print("Trace exported to: research_workflow.lctl.json")
    print()

    print("=" * 60)
    print("Example 3: GroupChat Tracing")
    print("=" * 60)
    print()

    planner = ConversableAgent(
        name="planner",
        system_message="You plan and coordinate tasks.",
        llm_config={"model": "gpt-4"},
    )

    coder = ConversableAgent(
        name="coder",
        system_message="You write clean, efficient code.",
        llm_config={"model": "gpt-4"},
    )

    reviewer = ConversableAgent(
        name="reviewer",
        system_message="You review code for quality and bugs.",
        llm_config={"model": "gpt-4"},
    )

    group_chat = GroupChat(
        agents=[planner, coder, reviewer],
        messages=[],
        max_round=10,
    )

    manager = LCTLGroupChatManager(
        groupchat=group_chat,
        name="dev_manager",
        chain_id="dev-team-chat",
    )

    print("GroupChat configured with LCTL tracing:")
    print(f"  - Chain ID: {manager.session.chain.id}")
    print(f"  - Agents: {len(group_chat.agents)}")
    print(f"  - Max rounds: {group_chat.max_round}")
    print()

    manager.export_trace("dev_team_chat.lctl.json")
    print("Trace exported to: dev_team_chat.lctl.json")
    print()

    print("=" * 60)
    print("Example 4: Nested Conversation Tracking")
    print("=" * 60)
    print()

    callback = LCTLAutogenCallback(chain_id="nested-workflow")

    callback.start_nested_chat("orchestrator", "Delegating to sub-agents")
    callback.session.step_start("sub_agent_1", "process", "Handling subtask 1")
    callback.session.step_end("sub_agent_1", outcome="success", output_summary="Subtask 1 done")
    callback.session.step_start("sub_agent_2", "process", "Handling subtask 2")
    callback.session.step_end("sub_agent_2", outcome="success", output_summary="Subtask 2 done")
    callback.end_nested_chat("Both subtasks completed", outcome="success")

    print("Nested conversation tracked:")
    print(f"  - Events recorded: {len(callback.session.chain.events)}")
    print()

    callback.export("nested_workflow.lctl.json")
    print("Trace exported to: nested_workflow.lctl.json")
    print()

    print("=" * 60)
    print("Example 5: Using trace_agent and trace_group_chat helpers")
    print("=" * 60)
    print()
    print("For quick tracing of existing agents:")
    print()
    print("    from autogen import ConversableAgent, GroupChat, GroupChatManager")
    print("    from lctl.integrations.autogen import trace_agent, trace_group_chat")
    print()
    print("    # Trace a single agent")
    print("    agent = ConversableAgent(name='assistant', ...)")
    print("    callback = trace_agent(agent)")
    print("    agent.initiate_chat(other_agent, message='Hello')")
    print("    callback.export('trace.lctl.json')")
    print()
    print("    # Trace a group chat")
    print("    group_chat = GroupChat(agents=[...], messages=[], max_round=10)")
    print("    manager = GroupChatManager(groupchat=group_chat)")
    print("    callback = trace_group_chat(group_chat, manager)")
    print("    agent.initiate_chat(manager, message='Start discussion')")
    print("    callback.export('groupchat.lctl.json')")
    print()


def print_example_code() -> None:
    """Print example code when AutoGen is not installed."""
    code = '''
from autogen import ConversableAgent, GroupChat, GroupChatManager
from lctl.integrations.autogen import (
    LCTLAutogenCallback,
    LCTLConversableAgent,
    LCTLGroupChatManager,
    trace_agent,
)

# Option 1: Use LCTL wrapper classes for automatic tracing
assistant = LCTLConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"model": "gpt-4"},
    chain_id="my-conversation",
)

user = LCTLConversableAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
)

# Run conversation with automatic tracing
result = assistant.initiate_chat(user, message="Hello!")

# Export trace for debugging
assistant.export_trace("conversation.lctl.json")


# Option 2: Use callback with existing agents
agent1 = ConversableAgent(name="agent1", ...)
agent2 = ConversableAgent(name="agent2", ...)

callback = LCTLAutogenCallback(chain_id="workflow")
callback.attach(agent1)
callback.attach(agent2)

agent1.initiate_chat(agent2, message="Let's work together")
callback.export("workflow.lctl.json")


# Option 3: Trace GroupChat conversations
group_chat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=10,
)

manager = LCTLGroupChatManager(
    groupchat=group_chat,
    chain_id="team-discussion",
)

agent1.initiate_chat(manager, message="Team meeting starts")
manager.export_trace("team_discussion.lctl.json")


# Option 4: Track nested conversations
callback = LCTLAutogenCallback()
callback.attach(orchestrator)

callback.start_nested_chat("orchestrator", "Delegating work")
# ... nested agent interactions ...
callback.end_nested_chat("Work completed")

callback.export("nested.lctl.json")
'''
    print(code)
    print()
    print("The trace captures:")
    print("  - Agent-to-agent message passing")
    print("  - Tool/function calls and responses")
    print("  - GroupChat speaker transitions")
    print("  - Nested conversation hierarchy")
    print("  - Errors with context")
    print()


if __name__ == "__main__":
    main()
