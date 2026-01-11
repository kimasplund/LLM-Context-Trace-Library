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

from .llamaindex import (
    LCTLLlamaIndexCallback,
    LCTLQueryEngine,
    LCTLChatEngine,
    trace_query_engine,
    trace_chat_engine,
    is_available as llamaindex_available,
)

from .dspy import (
    LCTLDSPyCallback,
    TracedDSPyModule,
    LCTLDSPyTeleprompter,
    DSPyModuleContext,
    trace_module,
    is_available as dspy_available,
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
    # LlamaIndex integration
    "LCTLLlamaIndexCallback",
    "LCTLQueryEngine",
    "LCTLChatEngine",
    "trace_query_engine",
    "trace_chat_engine",
    "llamaindex_available",
    # DSPy integration
    "LCTLDSPyCallback",
    "TracedDSPyModule",
    "LCTLDSPyTeleprompter",
    "DSPyModuleContext",
    "trace_module",
    "dspy_available",
]
