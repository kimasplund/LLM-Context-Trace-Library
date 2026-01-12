"""Semantic Kernel integration for LCTL.

This module provides automatic tracing for Semantic Kernel (v1.x).
It uses the Filter API to intercept function invocations.

Requires:
    - semantic-kernel>=1.0.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from ..core.session import LCTLSession

try:
    # Check availability
    import semantic_kernel  # noqa: F401
    # Use direct imports from filters package for better compatibility
    from semantic_kernel.filters import (
        FilterTypes, 
        FunctionInvocationContext,
        PromptRenderContext
    )
    from semantic_kernel.kernel import Kernel

    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class SemanticKernelNotAvailableError(ImportError):
    """Raised when Semantic Kernel is not installed."""
    def __init__(self) -> None:
        super().__init__(
            "Semantic Kernel is not installed. Install with: pip install semantic-kernel"
        )


def _check_sk_available() -> None:
    if not SK_AVAILABLE:
        raise SemanticKernelNotAvailableError()


class LCTLSemanticKernelTracer:
    """Tracer for Semantic Kernel."""

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ):
        """Initialize tracer.

        Args:
            chain_id: Optional unique identifier for the trace chain
            session: Optional existing LCTLSession
            verbose: Enable verbose logging
        """
        _check_sk_available()
        self.session = session or LCTLSession(
            chain_id=chain_id or f"sk-{id(self)}"
        )
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)
        self._traced_kernels: set = set()  # Track kernels to prevent double-tracing

    @property
    def chain(self):
        return self.session.chain

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file."""
        self.session.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL chain as a dictionary."""
        return self.session.to_dict()

    def trace_kernel(self, kernel: Kernel) -> Kernel:
        """Attach tracing to a Semantic Kernel instance.

        Adds filters to the kernel.
        """
        kernel_id = id(kernel)
        if kernel_id in self._traced_kernels:
            self._logger.warning("Kernel already traced, skipping to prevent double-tracing")
            return kernel

        self._traced_kernels.add(kernel_id)
        kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, self._function_invocation_filter)
        kernel.add_filter(FilterTypes.PROMPT_RENDERING, self._prompt_render_filter)
        return kernel

    async def _prompt_render_filter(self, context: PromptRenderContext, next: Callable[..., Any]):
        """Filter for prompt rendering."""
        # Execute render
        await next(context)
        
        # Capture rendered prompt
        if context.rendered_prompt:
            function_name = context.function.name
            plugin_name = context.function.plugin_name or "Global"
            full_name = f"{plugin_name}.{function_name}"
            
            # Record as a fact or part of step?
            # Since render happens before invocation filter (usually), or as part of it.
            # We can log it as a fact associated with the function.
            # Or use 'tool_call' semantics? No, prompt is internal.
            # Adding as a high-confidence fact about the agent's "thought" process.
            
            self.session.add_fact(
                fact_id=f"prompt-{full_name}-{hash(context.rendered_prompt)}",
                text=f"Rendered Prompt for {full_name}: {context.rendered_prompt}",
                confidence=1.0,
                source="sk-prompt-filter"
            )

    async def _function_invocation_filter(self, context: FunctionInvocationContext, next: Callable[..., Any]):
        """Filter for function invocations."""

        # 1. Step Start
        function_name = context.function.name
        plugin_name = context.function.plugin_name or "Global"
        full_name = f"{plugin_name}.{function_name}"

        # Summarize inputs
        input_summary = _truncate(str(context.arguments))

        self.session.step_start(
            agent=full_name,
            intent="execute_function",
            input_summary=input_summary
        )

        start_time = time.time()

        try:
            # 2. Execute next filter/function directly
            await next(context)

            # 3. Handle Result (Streaming vs Non-Streaming)
            result = context.result

            # Check for streaming (AsyncGenerator) in result.value
            if result and hasattr(result, "value") and hasattr(result.value, "__aiter__"):
                original_stream = result.value
                stream_start_time = time.time()

                async def stream_wrapper():
                    accumulated_content = ""
                    tokens_in = 0
                    tokens_out = 0
                    stream_completed = False

                    try:
                        async for chunk in original_stream:
                            # Accumulate content
                            chunk_str = str(chunk)
                            accumulated_content += chunk_str

                            # Check for usage in chunk metadata
                            if hasattr(chunk, "metadata") and chunk.metadata:
                                usage = chunk.metadata.get("usage")
                                if usage:
                                    if hasattr(usage, "prompt_tokens"):
                                        tokens_in = usage.prompt_tokens
                                    elif isinstance(usage, dict):
                                        tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)

                                    if hasattr(usage, "completion_tokens"):
                                        tokens_out = usage.completion_tokens
                                    elif isinstance(usage, dict):
                                        tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

                            yield chunk

                        # Stream finished successfully
                        stream_completed = True
                        duration_ms = int((time.time() - stream_start_time) * 1000)
                        self.session.step_end(
                            agent=full_name,
                            outcome="success",
                            output_summary=_truncate(accumulated_content),
                            duration_ms=duration_ms,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out
                        )

                    except Exception as e:
                        # Stream error
                        self.session.error(
                            category="execution_error",
                            error_type=type(e).__name__,
                            message=str(e),
                            recoverable=False
                        )
                        duration_ms = int((time.time() - stream_start_time) * 1000)
                        self.session.step_end(
                            agent=full_name,
                            outcome="error",
                            duration_ms=duration_ms
                        )
                        raise
                    finally:
                        # Ensure step_end is called even if stream is abandoned
                        if not stream_completed:
                            try:
                                duration_ms = int((time.time() - stream_start_time) * 1000)
                                self.session.step_end(
                                    agent=full_name,
                                    outcome="abandoned",
                                    output_summary=_truncate(accumulated_content),
                                    duration_ms=duration_ms,
                                    tokens_in=tokens_in,
                                    tokens_out=tokens_out
                                )
                            except Exception:
                                pass  # Don't fail if cleanup fails

                # Replace the stream with our wrapper
                result.value = stream_wrapper()
                return

            # Non-streaming handling
            duration_ms = int((time.time() - start_time) * 1000)
            output_summary = _truncate(str(result)) if result else ""

            # Extract usage if available
            tokens_in = 0
            tokens_out = 0
            if result and hasattr(result, "metadata") and result.metadata:
                usage = result.metadata.get("usage", None)
                if usage:
                    # Attempt to read common usage patterns (OpenAI, etc)
                    # Usage might be an object or dict
                    if hasattr(usage, "prompt_tokens"):
                        tokens_in = usage.prompt_tokens
                    elif isinstance(usage, dict):
                        tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)

                    if hasattr(usage, "completion_tokens"):
                        tokens_out = usage.completion_tokens
                    elif isinstance(usage, dict):
                        tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

            self.session.step_end(
                agent=full_name,
                outcome="success",
                output_summary=output_summary,
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out
            )

        except Exception as e:
            # 4. Step End (Error) - only catches sync/setup errors, stream errors caught in wrapper
            duration_ms = int((time.time() - start_time) * 1000)
            self.session.error(
                category="execution_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False
            )
            self.session.step_end(
                agent=full_name,
                outcome="error",
                duration_ms=duration_ms
            )
            raise


__all__ = [
    "SK_AVAILABLE",
    "LCTLSemanticKernelTracer",
    "trace_kernel",
    "is_available",
]


def trace_kernel(
    kernel: Kernel,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False
) -> LCTLSemanticKernelTracer:
    """Convenience function to trace a Semantic Kernel.

    Args:
        kernel: The Kernel instance.
        chain_id: Optional chain ID.
        session: Optional session.
        verbose: Enable verbose logging.
        
    Returns:
        The tracer instance.
    """
    tracer = LCTLSemanticKernelTracer(chain_id, session, verbose)
    tracer.trace_kernel(kernel)
    return tracer

def is_available() -> bool:
    """Check if Semantic Kernel is available."""
    return SK_AVAILABLE
