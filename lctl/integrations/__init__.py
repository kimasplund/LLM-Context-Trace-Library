"""LCTL Integrations - Framework integrations for automatic tracing."""

from .langchain import (
    LCTLCallbackHandler,
    LCTLChain,
    trace_chain,
    is_available as langchain_available,
)

from .openai_agents import (
    LCTLOpenAIAgentTracer,
    LCTLRunHooks,
    TracedAgent,
    trace_agent,
    is_available as openai_agents_available,
)

from .autogen import (
    LCTLAutogenCallback,
    LCTLConversableAgent,
    LCTLGroupChatManager,
    trace_agent as trace_autogen_agent,
    trace_group_chat,
    is_available as autogen_available,
)

__all__ = [
    # LangChain integration
    "LCTLCallbackHandler",
    "LCTLChain",
    "trace_chain",
    "langchain_available",
    # OpenAI Agents SDK integration
    "LCTLOpenAIAgentTracer",
    "LCTLRunHooks",
    "TracedAgent",
    "trace_agent",
    "openai_agents_available",
    # AutoGen/AG2 integration
    "LCTLAutogenCallback",
    "LCTLConversableAgent",
    "LCTLGroupChatManager",
    "trace_autogen_agent",
    "trace_group_chat",
    "autogen_available",
]
