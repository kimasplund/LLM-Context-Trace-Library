"""LangChain integration for LCTL.

Provides automatic tracing of LangChain chains, LLMs, and tools with LCTL.

Usage:
    from lctl.integrations.langchain import LCTLCallbackHandler

    handler = LCTLCallbackHandler()
    chain.invoke(input, config={"callbacks": [handler]})
    handler.export("trace.lctl.json")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from ..core.session import LCTLSession

try:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import AgentAction, AgentFinish, LLMResult
        from langchain.schema.messages import BaseMessage

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseCallbackHandler = object


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _extract_token_counts(response: Any) -> Dict[str, int]:
    """Extract token counts from LLM response."""
    tokens = {"input": 0, "output": 0}

    if response is None:
        return tokens

    if hasattr(response, "llm_output") and response.llm_output:
        llm_output = response.llm_output

        if "token_usage" in llm_output:
            usage = llm_output["token_usage"]
            tokens["input"] = usage.get("prompt_tokens", 0)
            tokens["output"] = usage.get("completion_tokens", 0)

        elif "usage" in llm_output:
            usage = llm_output["usage"]
            tokens["input"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            tokens["output"] = usage.get("completion_tokens", usage.get("output_tokens", 0))

    return tokens


class LCTLCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records events to LCTL.

    Captures:
    - LLM calls (start/end with token counts)
    - Chain execution (start/end)
    - Tool invocations
    - Agent actions
    - Errors

    Example:
        handler = LCTLCallbackHandler(chain_id="my-chain")
        result = chain.invoke(
            {"input": "Hello"},
            config={"callbacks": [handler]}
        )
        handler.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
    ) -> None:
        """Initialize the callback handler.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain-core"
            )

        super().__init__()
        self.session = session or LCTLSession(chain_id=chain_id)

        self._run_stack: Dict[UUID, Dict[str, Any]] = {}
        self._chain_depth = 0

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self.session.chain

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self.session.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL chain as a dictionary."""
        return self.session.to_dict()

    def _get_agent_name(self, serialized: Dict[str, Any], default: str = "langchain") -> str:
        """Extract agent name from serialized data."""
        if serialized:
            if "name" in serialized:
                return serialized["name"]
            if "id" in serialized:
                id_parts = serialized["id"]
                if isinstance(id_parts, list) and id_parts:
                    return id_parts[-1]
        return default

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        agent_name = self._get_agent_name(serialized, "llm")
        prompt_summary = _truncate(prompts[0]) if prompts else ""

        self._run_stack[run_id] = {
            "type": "llm",
            "agent": agent_name,
            "start_time": time.time(),
            "prompt_summary": prompt_summary,
        }

        self.session.step_start(
            agent=agent_name,
            intent="llm_call",
            input_summary=prompt_summary,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends."""
        run_info = self._run_stack.pop(run_id, {})
        agent_name = run_info.get("agent", "llm")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        tokens = _extract_token_counts(response)

        output_text = ""
        if response.generations and response.generations[0]:
            first_gen = response.generations[0][0]
            if hasattr(first_gen, "text"):
                output_text = first_gen.text
            elif hasattr(first_gen, "message") and hasattr(first_gen.message, "content"):
                output_text = first_gen.message.content

        self.session.step_end(
            agent=agent_name,
            outcome="success",
            output_summary=_truncate(output_text),
            duration_ms=duration_ms,
            tokens_in=tokens["input"],
            tokens_out=tokens["output"],
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        run_info = self._run_stack.pop(run_id, {})
        agent_name = run_info.get("agent", "llm")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        self.session.step_end(
            agent=agent_name,
            outcome="error",
            duration_ms=duration_ms,
        )
        self.session.error(
            category="llm_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=False,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts."""
        agent_name = self._get_agent_name(serialized, "chat_model")

        message_summary = ""
        if messages and messages[0]:
            last_msg = messages[0][-1]
            if hasattr(last_msg, "content"):
                content = last_msg.content
                if isinstance(content, str):
                    message_summary = _truncate(content)
                elif isinstance(content, list) and content:
                    first_part = content[0]
                    if isinstance(first_part, dict) and "text" in first_part:
                        message_summary = _truncate(first_part["text"])
                    elif isinstance(first_part, str):
                        message_summary = _truncate(first_part)

        self._run_stack[run_id] = {
            "type": "chat_model",
            "agent": agent_name,
            "start_time": time.time(),
            "message_summary": message_summary,
        }

        self.session.step_start(
            agent=agent_name,
            intent="chat_model_call",
            input_summary=message_summary,
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        self._chain_depth += 1
        agent_name = self._get_agent_name(serialized, f"chain_{self._chain_depth}")

        input_summary = ""
        if inputs:
            if isinstance(inputs, dict):
                keys = list(inputs.keys())[:3]
                input_summary = f"keys: {keys}"
            else:
                input_summary = _truncate(str(inputs))

        self._run_stack[run_id] = {
            "type": "chain",
            "agent": agent_name,
            "start_time": time.time(),
            "depth": self._chain_depth,
            "input_summary": input_summary,
        }

        self.session.step_start(
            agent=agent_name,
            intent="chain_execution",
            input_summary=input_summary,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        run_info = self._run_stack.pop(run_id, {})
        agent_name = run_info.get("agent", "chain")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        output_summary = ""
        if outputs:
            if isinstance(outputs, dict):
                keys = list(outputs.keys())[:3]
                output_summary = f"keys: {keys}"
            else:
                output_summary = _truncate(str(outputs))

        self.session.step_end(
            agent=agent_name,
            outcome="success",
            output_summary=output_summary,
            duration_ms=duration_ms,
        )

        if self._chain_depth > 0:
            self._chain_depth -= 1

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        run_info = self._run_stack.pop(run_id, {})
        agent_name = run_info.get("agent", "chain")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        self.session.step_end(
            agent=agent_name,
            outcome="error",
            duration_ms=duration_ms,
        )
        self.session.error(
            category="chain_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=False,
        )

        if self._chain_depth > 0:
            self._chain_depth -= 1

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        tool_name = self._get_agent_name(serialized, "tool")

        self._run_stack[run_id] = {
            "type": "tool",
            "tool_name": tool_name,
            "start_time": time.time(),
            "input": input_str,
        }

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        run_info = self._run_stack.pop(run_id, {})
        tool_name = run_info.get("tool_name", "tool")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)
        input_data = run_info.get("input", "")

        self.session.tool_call(
            tool=tool_name,
            input_data=_truncate(input_data),
            output_data=_truncate(output) if isinstance(output, str) else _truncate(str(output)),
            duration_ms=duration_ms,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        run_info = self._run_stack.pop(run_id, {})
        tool_name = run_info.get("tool_name", "tool")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)
        input_data = run_info.get("input", "")

        self.session.tool_call(
            tool=tool_name,
            input_data=_truncate(input_data),
            output_data=f"ERROR: {type(error).__name__}: {error}",
            duration_ms=duration_ms,
        )
        self.session.error(
            category="tool_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=True,
        )

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        self.session.add_fact(
            fact_id=f"action_{run_id.hex[:8]}",
            text=f"Agent action: {action.tool} with input: {_truncate(str(action.tool_input))}",
            confidence=1.0,
            source="agent",
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        output = finish.return_values.get("output", "")
        self.session.add_fact(
            fact_id=f"finish_{run_id.hex[:8]}",
            text=f"Agent finished: {_truncate(output)}",
            confidence=1.0,
            source="agent",
        )

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts."""
        retriever_name = self._get_agent_name(serialized, "retriever")

        self._run_stack[run_id] = {
            "type": "retriever",
            "name": retriever_name,
            "start_time": time.time(),
            "query": query,
        }

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever ends."""
        run_info = self._run_stack.pop(run_id, {})
        retriever_name = run_info.get("name", "retriever")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)
        query = run_info.get("query", "")

        self.session.tool_call(
            tool=retriever_name,
            input_data=_truncate(query),
            output_data=f"Retrieved {len(documents)} documents",
            duration_ms=duration_ms,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        run_info = self._run_stack.pop(run_id, {})
        retriever_name = run_info.get("name", "retriever")
        query = run_info.get("query", "")
        start_time = run_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        self.session.tool_call(
            tool=retriever_name,
            input_data=_truncate(query),
            output_data=f"ERROR: {type(error).__name__}: {error}",
            duration_ms=duration_ms,
        )
        self.session.error(
            category="retriever_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=True,
        )


class LCTLChain:
    """Wrapper that adds LCTL tracing to any LangChain chain.

    Example:
        from lctl.integrations.langchain import LCTLChain

        traced_chain = LCTLChain(my_chain, chain_id="my-analysis")
        result = traced_chain.invoke({"input": "Hello"})
        traced_chain.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain: Any,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
    ) -> None:
        """Initialize the traced chain wrapper.

        Args:
            chain: The LangChain chain to wrap.
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain-core"
            )

        self._chain = chain
        self.handler = LCTLCallbackHandler(chain_id=chain_id, session=session)

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self.handler.chain

    @property
    def session(self):
        """Access the LCTL session."""
        return self.handler.session

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Invoke the chain with LCTL tracing.

        Args:
            input: Input to the chain.
            config: Optional config dict. Callbacks will be merged.
            **kwargs: Additional arguments passed to the chain.

        Returns:
            The chain's output.
        """
        config = config or {}
        existing_callbacks = config.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            existing_callbacks = [existing_callbacks]
        config["callbacks"] = existing_callbacks + [self.handler]

        return self._chain.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Async invoke the chain with LCTL tracing.

        Args:
            input: Input to the chain.
            config: Optional config dict. Callbacks will be merged.
            **kwargs: Additional arguments passed to the chain.

        Returns:
            The chain's output.
        """
        config = config or {}
        existing_callbacks = config.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            existing_callbacks = [existing_callbacks]
        config["callbacks"] = existing_callbacks + [self.handler]

        return await self._chain.ainvoke(input, config=config, **kwargs)

    def stream(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Stream the chain with LCTL tracing.

        Args:
            input: Input to the chain.
            config: Optional config dict. Callbacks will be merged.
            **kwargs: Additional arguments passed to the chain.

        Yields:
            Chunks from the chain.
        """
        config = config or {}
        existing_callbacks = config.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            existing_callbacks = [existing_callbacks]
        config["callbacks"] = existing_callbacks + [self.handler]

        yield from self._chain.stream(input, config=config, **kwargs)

    async def astream(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Async stream the chain with LCTL tracing.

        Args:
            input: Input to the chain.
            config: Optional config dict. Callbacks will be merged.
            **kwargs: Additional arguments passed to the chain.

        Yields:
            Chunks from the chain.
        """
        config = config or {}
        existing_callbacks = config.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            existing_callbacks = [existing_callbacks]
        config["callbacks"] = existing_callbacks + [self.handler]

        async for chunk in self._chain.astream(input, config=config, **kwargs):
            yield chunk

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self.handler.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL trace as a dictionary."""
        return self.handler.to_dict()


def trace_chain(
    chain: Any,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
) -> LCTLChain:
    """Convenience function to wrap a chain with LCTL tracing.

    Args:
        chain: The LangChain chain to trace.
        chain_id: Optional chain ID for the LCTL session.
        session: Optional existing LCTL session to use.

    Returns:
        LCTLChain wrapper with tracing enabled.

    Example:
        from lctl.integrations.langchain import trace_chain

        traced = trace_chain(my_chain)
        result = traced.invoke(input)
        traced.export("trace.lctl.json")
    """
    return LCTLChain(chain, chain_id=chain_id, session=session)


def is_available() -> bool:
    """Check if LangChain integration is available.

    Returns:
        True if LangChain is installed, False otherwise.
    """
    return LANGCHAIN_AVAILABLE
