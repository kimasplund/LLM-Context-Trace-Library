"""LCTL Integrations - Framework integrations for automatic tracing."""

from .claude_code import (
    LCTLClaudeCodeTracer,
    generate_hooks as generate_claude_code_hooks,
)
from .claude_code import (
    is_available as claude_code_available,
)
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
from .crewai import (
    LCTLAgent,
    LCTLTask,
    LCTLCrew,
    trace_crew,
)
from .crewai import (
    is_available as crewai_available,
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
from .semantic_kernel import (
    LCTLSemanticKernelTracer,
    trace_kernel,
)
from .semantic_kernel import (
    is_available as semantic_kernel_available,
)

__all__ = [
    # Claude Code integration
    "LCTLClaudeCodeTracer",
    "generate_claude_code_hooks",
    "claude_code_available",
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
    # CrewAI integration
    "LCTLAgent",
    "LCTLTask",
    "LCTLCrew",
    "trace_crew",
    "crewai_available",
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
    # Semantic Kernel integration
    "LCTLSemanticKernelTracer",
    "trace_kernel",
    "semantic_kernel_available",
]
