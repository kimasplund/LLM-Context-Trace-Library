"""LlamaIndex integration for LCTL.

Provides automatic tracing of LlamaIndex query engines, chat engines,
retrievers, and LLM calls with LCTL for time-travel debugging.

Usage:
    from lctl.integrations.llamaindex import LCTLLlamaIndexCallback, trace_query_engine

    # Option 1: Use callback directly
    callback = LCTLLlamaIndexCallback()
    index.as_query_engine(callback_manager=callback.get_callback_manager())

    # Option 2: Use helper to wrap existing engine
    traced = trace_query_engine(query_engine)
    response = traced.query("What is the capital of France?")
    traced.export("trace.lctl.json")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..core.session import LCTLSession

try:
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        from llama_index.callbacks import CallbackManager
        from llama_index.callbacks.base import BaseCallbackHandler
        from llama_index.callbacks.schema import CBEventType, EventPayload

        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        LLAMAINDEX_AVAILABLE = False
        BaseCallbackHandler = object
        CallbackManager = None
        CBEventType = None
        EventPayload = None


class LlamaIndexNotAvailableError(ImportError):
    """Raised when LlamaIndex is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "LlamaIndex is not installed. Install with: pip install llama-index"
        )


def _check_llamaindex_available() -> None:
    """Check if LlamaIndex is available, raise error if not."""
    if not LLAMAINDEX_AVAILABLE:
        raise LlamaIndexNotAvailableError()


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _extract_token_counts(payload: Dict[str, Any]) -> Dict[str, int]:
    """Extract token counts from LlamaIndex payload."""
    tokens = {"input": 0, "output": 0}

    if not payload:
        return tokens

    if EventPayload is not None:
        response = payload.get(EventPayload.RESPONSE)
        if response and hasattr(response, "raw"):
            raw = response.raw
            if hasattr(raw, "usage"):
                usage = raw.usage
                if hasattr(usage, "prompt_tokens"):
                    tokens["input"] = usage.prompt_tokens
                if hasattr(usage, "completion_tokens"):
                    tokens["output"] = usage.completion_tokens

    return tokens


def _get_event_name(event_type: Any) -> str:
    """Get a readable name for an event type."""
    if event_type is None:
        return "unknown"
    return str(event_type.name if hasattr(event_type, "name") else event_type).lower()


class LCTLLlamaIndexCallback(BaseCallbackHandler):
    """LlamaIndex callback handler that records events to LCTL.

    Captures:
    - LLM calls (start/end with token counts)
    - Query execution (start/end)
    - Chat messages (start/end)
    - Retrieval operations
    - Embedding generation
    - Errors

    Example:
        callback = LCTLLlamaIndexCallback(chain_id="my-query")
        query_engine = index.as_query_engine(
            callback_manager=callback.get_callback_manager()
        )
        response = query_engine.query("What is the capital of France?")
        callback.export("trace.lctl.json")
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
        _check_llamaindex_available()

        self.session = session or LCTLSession(chain_id=chain_id)
        self._event_stack: Dict[str, Dict[str, Any]] = {}
        self._query_depth = 0

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self.session.chain

    def get_callback_manager(self) -> "CallbackManager":
        """Get a CallbackManager configured with this handler.

        Returns:
            A LlamaIndex CallbackManager with this handler attached.
        """
        _check_llamaindex_available()
        return CallbackManager([self])

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self.session.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL chain as a dictionary."""
        return self.session.to_dict()

    def on_event_start(
        self,
        event_type: "CBEventType",
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Handle event start."""
        payload = payload or {}
        event_name = _get_event_name(event_type)

        self._event_stack[event_id] = {
            "type": event_type,
            "start_time": time.time(),
            "parent_id": parent_id,
        }

        if CBEventType is None:
            return event_id

        if event_type == CBEventType.LLM:
            self._on_llm_start(event_id, payload)
        elif event_type == CBEventType.QUERY:
            self._on_query_start(event_id, payload)
        elif event_type == CBEventType.RETRIEVE:
            self._on_retrieval_start(event_id, payload)
        elif event_type == CBEventType.EMBEDDING:
            self._on_embedding_start(event_id, payload)
        elif event_type == CBEventType.SYNTHESIZE:
            self._on_synthesize_start(event_id, payload)
        elif event_type == CBEventType.TEMPLATING:
            self._on_templating_start(event_id, payload)
        elif event_type == CBEventType.CHUNKING:
            self._on_chunking_start(event_id, payload)
        elif event_type == CBEventType.RERANKING:
            self._on_reranking_start(event_id, payload)
        else:
            self.session.step_start(
                agent=f"llamaindex_{event_name}",
                intent=event_name,
                input_summary=f"Event: {event_name}",
            )

        return event_id

    def on_event_end(
        self,
        event_type: "CBEventType",
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Handle event end."""
        payload = payload or {}
        event_info = self._event_stack.pop(event_id, {})
        start_time = event_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        if CBEventType is None:
            return

        if event_type == CBEventType.LLM:
            self._on_llm_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.QUERY:
            self._on_query_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.RETRIEVE:
            self._on_retrieval_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.EMBEDDING:
            self._on_embedding_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.SYNTHESIZE:
            self._on_synthesize_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.TEMPLATING:
            self._on_templating_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.CHUNKING:
            self._on_chunking_end(event_id, payload, duration_ms)
        elif event_type == CBEventType.RERANKING:
            self._on_reranking_end(event_id, payload, duration_ms)
        else:
            event_name = _get_event_name(event_type)
            self.session.step_end(
                agent=f"llamaindex_{event_name}",
                outcome="success",
                output_summary=f"Completed: {event_name}",
                duration_ms=duration_ms,
            )

    def _on_llm_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle LLM start event."""
        messages = payload.get(EventPayload.MESSAGES, []) if EventPayload else []
        prompt = payload.get(EventPayload.PROMPT, "") if EventPayload else ""

        input_summary = ""
        if messages:
            if isinstance(messages, list) and messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    input_summary = _truncate(str(last_msg.content))
                elif isinstance(last_msg, dict):
                    input_summary = _truncate(str(last_msg.get("content", "")))
                else:
                    input_summary = _truncate(str(last_msg))
        elif prompt:
            input_summary = _truncate(str(prompt))

        self._event_stack[event_id]["input_summary"] = input_summary

        self.session.step_start(
            agent="llm",
            intent="llm_call",
            input_summary=input_summary,
        )

    def _on_llm_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle LLM end event."""
        tokens = _extract_token_counts(payload)

        response_text = ""
        if EventPayload is not None:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                if hasattr(response, "text"):
                    response_text = response.text
                elif hasattr(response, "message") and hasattr(response.message, "content"):
                    response_text = str(response.message.content)
                else:
                    response_text = str(response)

            completion = payload.get(EventPayload.COMPLETION)
            if completion and not response_text:
                response_text = str(completion)

        self.session.step_end(
            agent="llm",
            outcome="success",
            output_summary=_truncate(response_text),
            duration_ms=duration_ms,
            tokens_in=tokens["input"],
            tokens_out=tokens["output"],
        )

    def _on_query_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle query start event."""
        self._query_depth += 1
        query_str = ""
        if EventPayload is not None:
            query_str = str(payload.get(EventPayload.QUERY_STR, ""))

        self._event_stack[event_id]["query"] = query_str

        self.session.step_start(
            agent="query_engine",
            intent="query",
            input_summary=_truncate(query_str) if query_str else "Query execution",
        )

    def _on_query_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle query end event."""
        response_text = ""
        if EventPayload is not None:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                if hasattr(response, "response"):
                    response_text = str(response.response)
                else:
                    response_text = str(response)

        self.session.step_end(
            agent="query_engine",
            outcome="success",
            output_summary=_truncate(response_text),
            duration_ms=duration_ms,
        )

        if self._query_depth > 0:
            self._query_depth -= 1

    def _on_retrieval_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle retrieval start event."""
        query_str = ""
        if EventPayload is not None:
            query_str = str(payload.get(EventPayload.QUERY_STR, ""))

        self._event_stack[event_id]["query"] = query_str

        self.session.step_start(
            agent="retriever",
            intent="retrieve",
            input_summary=_truncate(query_str) if query_str else "Retrieval",
        )

    def _on_retrieval_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle retrieval end event."""
        num_nodes = 0
        if EventPayload is not None:
            nodes = payload.get(EventPayload.NODES, [])
            num_nodes = len(nodes) if nodes else 0

        event_info = self._event_stack.get(event_id, {})
        query = event_info.get("query", "")

        self.session.tool_call(
            tool="retriever",
            input_data=_truncate(query) if query else "retrieval query",
            output_data=f"Retrieved {num_nodes} nodes",
            duration_ms=duration_ms,
        )

        self.session.step_end(
            agent="retriever",
            outcome="success",
            output_summary=f"Retrieved {num_nodes} nodes",
            duration_ms=duration_ms,
        )

        if num_nodes > 0 and EventPayload is not None:
            nodes = payload.get(EventPayload.NODES, [])
            for i, node in enumerate(nodes[:3]):
                node_text = ""
                if hasattr(node, "text"):
                    node_text = node.text
                elif hasattr(node, "node") and hasattr(node.node, "text"):
                    node_text = node.node.text
                elif hasattr(node, "get_content"):
                    node_text = node.get_content()

                if node_text:
                    self.session.add_fact(
                        fact_id=f"retrieved-{event_id[:8]}-{i}",
                        text=_truncate(node_text, 300),
                        confidence=1.0,
                        source="retriever",
                    )

    def _on_embedding_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle embedding start event."""
        self.session.step_start(
            agent="embedding",
            intent="generate_embedding",
            input_summary="Generating embeddings",
        )

    def _on_embedding_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle embedding end event."""
        num_chunks = 0
        if EventPayload is not None:
            chunks = payload.get(EventPayload.CHUNKS, [])
            num_chunks = len(chunks) if chunks else 0

        self.session.step_end(
            agent="embedding",
            outcome="success",
            output_summary=f"Generated embeddings for {num_chunks} chunks",
            duration_ms=duration_ms,
        )

    def _on_synthesize_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle synthesize start event."""
        query_str = ""
        if EventPayload is not None:
            query_str = str(payload.get(EventPayload.QUERY_STR, ""))

        self.session.step_start(
            agent="synthesizer",
            intent="synthesize",
            input_summary=_truncate(query_str) if query_str else "Response synthesis",
        )

    def _on_synthesize_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle synthesize end event."""
        response_text = ""
        if EventPayload is not None:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                if hasattr(response, "response"):
                    response_text = str(response.response)
                else:
                    response_text = str(response)

        self.session.step_end(
            agent="synthesizer",
            outcome="success",
            output_summary=_truncate(response_text),
            duration_ms=duration_ms,
        )

    def _on_templating_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle templating start event."""
        template = ""
        if EventPayload is not None:
            template = str(payload.get(EventPayload.TEMPLATE, ""))

        self.session.step_start(
            agent="templater",
            intent="template",
            input_summary=_truncate(template) if template else "Template processing",
        )

    def _on_templating_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle templating end event."""
        self.session.step_end(
            agent="templater",
            outcome="success",
            output_summary="Template processed",
            duration_ms=duration_ms,
        )

    def _on_chunking_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle chunking start event."""
        self.session.step_start(
            agent="chunker",
            intent="chunk",
            input_summary="Document chunking",
        )

    def _on_chunking_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle chunking end event."""
        num_chunks = 0
        if EventPayload is not None:
            chunks = payload.get(EventPayload.CHUNKS, [])
            num_chunks = len(chunks) if chunks else 0

        self.session.step_end(
            agent="chunker",
            outcome="success",
            output_summary=f"Created {num_chunks} chunks",
            duration_ms=duration_ms,
        )

    def _on_reranking_start(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle reranking start event."""
        query_str = ""
        if EventPayload is not None:
            query_str = str(payload.get(EventPayload.QUERY_STR, ""))

        self.session.step_start(
            agent="reranker",
            intent="rerank",
            input_summary=_truncate(query_str) if query_str else "Reranking nodes",
        )

    def _on_reranking_end(
        self, event_id: str, payload: Dict[str, Any], duration_ms: int
    ) -> None:
        """Handle reranking end event."""
        num_nodes = 0
        if EventPayload is not None:
            nodes = payload.get(EventPayload.NODES, [])
            num_nodes = len(nodes) if nodes else 0

        self.session.step_end(
            agent="reranker",
            outcome="success",
            output_summary=f"Reranked {num_nodes} nodes",
            duration_ms=duration_ms,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace (required by BaseCallbackHandler interface)."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace (required by BaseCallbackHandler interface)."""
        pass


class LCTLQueryEngine:
    """Wrapper around LlamaIndex QueryEngine with built-in LCTL tracing.

    Example:
        traced_engine = LCTLQueryEngine(query_engine, chain_id="my-qa")
        response = traced_engine.query("What is the capital of France?")
        traced_engine.export("trace.lctl.json")
    """

    def __init__(
        self,
        query_engine: Any,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
    ) -> None:
        """Initialize an LCTL-traced QueryEngine.

        Args:
            query_engine: The LlamaIndex query engine to wrap.
            chain_id: Optional chain ID for LCTL session.
            session: Optional existing LCTL session to use.
        """
        _check_llamaindex_available()

        self._query_engine = query_engine
        self._callback = LCTLLlamaIndexCallback(
            chain_id=chain_id or f"query-{str(uuid4())[:8]}",
            session=session,
        )

        if hasattr(query_engine, "callback_manager"):
            existing_handlers = []
            if query_engine.callback_manager:
                existing_handlers = list(query_engine.callback_manager.handlers)
            query_engine.callback_manager = CallbackManager(
                existing_handlers + [self._callback]
            )

    @property
    def query_engine(self) -> Any:
        """Get the underlying query engine."""
        return self._query_engine

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session."""
        return self._callback.session

    @property
    def callback(self) -> LCTLLlamaIndexCallback:
        """Get the LCTL callback handler."""
        return self._callback

    def query(self, query_str: str, **kwargs: Any) -> Any:
        """Execute a query with LCTL tracing.

        Args:
            query_str: The query string.
            **kwargs: Additional arguments passed to the query engine.

        Returns:
            The query response.
        """
        return self._query_engine.query(query_str, **kwargs)

    async def aquery(self, query_str: str, **kwargs: Any) -> Any:
        """Execute an async query with LCTL tracing.

        Args:
            query_str: The query string.
            **kwargs: Additional arguments passed to the query engine.

        Returns:
            The query response.
        """
        return await self._query_engine.aquery(query_str, **kwargs)

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self._callback.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Get the LCTL trace as a dictionary."""
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying query engine."""
        return getattr(self._query_engine, name)


class LCTLChatEngine:
    """Wrapper around LlamaIndex ChatEngine with built-in LCTL tracing.

    Example:
        traced_chat = LCTLChatEngine(chat_engine, chain_id="my-chat")
        response = traced_chat.chat("Hello!")
        traced_chat.export("chat_trace.lctl.json")
    """

    def __init__(
        self,
        chat_engine: Any,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
    ) -> None:
        """Initialize an LCTL-traced ChatEngine.

        Args:
            chat_engine: The LlamaIndex chat engine to wrap.
            chain_id: Optional chain ID for LCTL session.
            session: Optional existing LCTL session to use.
        """
        _check_llamaindex_available()

        self._chat_engine = chat_engine
        self._callback = LCTLLlamaIndexCallback(
            chain_id=chain_id or f"chat-{str(uuid4())[:8]}",
            session=session,
        )

        if hasattr(chat_engine, "callback_manager"):
            existing_handlers = []
            if chat_engine.callback_manager:
                existing_handlers = list(chat_engine.callback_manager.handlers)
            chat_engine.callback_manager = CallbackManager(
                existing_handlers + [self._callback]
            )

    @property
    def chat_engine(self) -> Any:
        """Get the underlying chat engine."""
        return self._chat_engine

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session."""
        return self._callback.session

    @property
    def callback(self) -> LCTLLlamaIndexCallback:
        """Get the LCTL callback handler."""
        return self._callback

    def chat(self, message: str, **kwargs: Any) -> Any:
        """Send a chat message with LCTL tracing.

        Args:
            message: The chat message.
            **kwargs: Additional arguments passed to the chat engine.

        Returns:
            The chat response.
        """
        return self._chat_engine.chat(message, **kwargs)

    async def achat(self, message: str, **kwargs: Any) -> Any:
        """Send an async chat message with LCTL tracing.

        Args:
            message: The chat message.
            **kwargs: Additional arguments passed to the chat engine.

        Returns:
            The chat response.
        """
        return await self._chat_engine.achat(message, **kwargs)

    def stream_chat(self, message: str, **kwargs: Any) -> Any:
        """Stream a chat message with LCTL tracing.

        Args:
            message: The chat message.
            **kwargs: Additional arguments passed to the chat engine.

        Returns:
            The streaming response.
        """
        return self._chat_engine.stream_chat(message, **kwargs)

    async def astream_chat(self, message: str, **kwargs: Any) -> Any:
        """Async stream a chat message with LCTL tracing.

        Args:
            message: The chat message.
            **kwargs: Additional arguments passed to the chat engine.

        Returns:
            The streaming response.
        """
        return await self._chat_engine.astream_chat(message, **kwargs)

    def reset(self) -> None:
        """Reset the chat history."""
        if hasattr(self._chat_engine, "reset"):
            self._chat_engine.reset()

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self._callback.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Get the LCTL trace as a dictionary."""
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying chat engine."""
        return getattr(self._chat_engine, name)


def trace_query_engine(
    query_engine: Any,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
) -> LCTLQueryEngine:
    """Wrap a LlamaIndex query engine with LCTL tracing.

    Args:
        query_engine: The LlamaIndex query engine to trace.
        chain_id: Optional chain ID for tracing.
        session: Optional existing LCTL session to use.

    Returns:
        An LCTLQueryEngine wrapper with tracing enabled.

    Example:
        from llama_index.core import VectorStoreIndex
        from lctl.integrations.llamaindex import trace_query_engine

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        traced = trace_query_engine(query_engine)
        response = traced.query("What is the main topic?")
        traced.export("trace.lctl.json")
    """
    return LCTLQueryEngine(query_engine, chain_id=chain_id, session=session)


def trace_chat_engine(
    chat_engine: Any,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
) -> LCTLChatEngine:
    """Wrap a LlamaIndex chat engine with LCTL tracing.

    Args:
        chat_engine: The LlamaIndex chat engine to trace.
        chain_id: Optional chain ID for tracing.
        session: Optional existing LCTL session to use.

    Returns:
        An LCTLChatEngine wrapper with tracing enabled.

    Example:
        from llama_index.core import VectorStoreIndex
        from lctl.integrations.llamaindex import trace_chat_engine

        index = VectorStoreIndex.from_documents(documents)
        chat_engine = index.as_chat_engine()

        traced = trace_chat_engine(chat_engine)
        response = traced.chat("Tell me about the document")
        traced.export("chat_trace.lctl.json")
    """
    return LCTLChatEngine(chat_engine, chain_id=chain_id, session=session)


def is_available() -> bool:
    """Check if LlamaIndex integration is available.

    Returns:
        True if LlamaIndex is installed, False otherwise.
    """
    return LLAMAINDEX_AVAILABLE


__all__ = [
    "LLAMAINDEX_AVAILABLE",
    "LlamaIndexNotAvailableError",
    "LCTLLlamaIndexCallback",
    "LCTLQueryEngine",
    "LCTLChatEngine",
    "trace_query_engine",
    "trace_chat_engine",
    "is_available",
]
