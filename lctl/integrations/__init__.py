"""LCTL Integrations - Framework integrations for automatic tracing."""

from .autogen import (
    LCTLAutogenCallback,
    LCTLConversableAgent,
    LCTLGroupChatManager,
    trace_group_chat,
)
from .autogen import (
    is_available as autogen_available,
)
from .autogen import (
    trace_agent as trace_autogen_agent,
)
from .dspy import (
    DSPyModuleContext,
    LCTLDSPyCallback,
    LCTLDSPyTeleprompter,
    TracedDSPyModule,
    trace_module,
)
from .dspy import (
    is_available as dspy_available,
)
from .langchain import (
    LCTLCallbackHandler,
    LCTLChain,
    trace_chain,
)
from .langchain import (
    is_available as langchain_available,
)
from .llamaindex import (
    LCTLChatEngine,
    LCTLLlamaIndexCallback,
    LCTLQueryEngine,
    trace_chat_engine,
    trace_query_engine,
)
from .llamaindex import (
    is_available as llamaindex_available,
)
from .openai_agents import (
    LCTLOpenAIAgentTracer,
    LCTLRunHooks,
    TracedAgent,
    trace_agent,
)
from .openai_agents import (
    is_available as openai_agents_available,
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
