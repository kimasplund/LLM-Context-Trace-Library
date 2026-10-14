"""Semantic Kernel integration for LCTL.

This module provides automatic tracing for Semantic Kernel (v1.x).
It uses the Filter API to intercept function invocations.

Requires:
    - semantic-kernel>=1.0.0
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ..core.session import LCTLSession
from .base import BaseTracer, truncate

try:
    # Check availability - import used for availability check
    import semantic_kernel as _sk

    del _sk  # Remove from namespace after check
    # Use direct imports from filters package for better compatibility
    from semantic_kernel.filters import FilterTypes, FunctionInvocationContext, PromptRenderContext
    from semantic_kernel.kernel import Kernel

    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


class SemanticKernelNotAvailableError(ImportError):
    """Raised when Semantic Kernel is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Semantic Kernel is not installed. Install with: pip install semantic-kernel"
        )


def _check_sk_available() -> None:
    if not SK_AVAILABLE:
        raise SemanticKernelNotAvailableError()


class LCTLSemanticKernelTracer(BaseTracer):
    """Tracer for Semantic Kernel.

    Extends BaseTracer for standardized session management, thread safety,
    and automatic stale entry cleanup.
    """

    def __init__(
        self,
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        verbose: bool = False,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0,
    ):
        """Initialize tracer.

        Args:
            chain_id: Optional unique identifier for the trace chain
            session: Optional existing LCTLSession
            verbose: Enable verbose logging
            auto_cleanup: Whether to auto-cleanup stale entries
            cleanup_interval: Cleanup interval in seconds (default 1 hour)
        """
        _check_sk_available()
        super().__init__(
            chain_id=chain_id or f"sk-{id(self)}",
            session=session,
            auto_cleanup=auto_cleanup,
            cleanup_interval=cleanup_interval,
        )
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)
        self._traced_kernels: set = set()  # Track kernels to prevent double-tracing

    def trace_kernel(self, kernel: Kernel) -> Kernel:
        """Attach tracing to a Semantic Kernel instance.

        Adds filters to the kernel.
        """
        kernel_id = id(kernel)
        with self._lock:
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

            try:
                self.session.add_fact(
                    fact_id=f"prompt-{full_name}-{hash(context.rendered_prompt)}",
                    text=f"Rendered Prompt for {full_name}: {context.rendered_prompt}",
                    confidence=1.0,
                    source="sk-prompt-filter",
                )
            except Exception:
                pass  # Don't break execution if tracing fails

    async def _function_invocation_filter(
        self, context: FunctionInvocationContext, next: Callable[..., Any]
    ):
        """Filter for function invocations."""

        # 1. Step Start
        function_name = context.function.name
        plugin_name = context.function.plugin_name or "Global"
        full_name = f"{plugin_name}.{function_name}"

        # Summarize inputs
        input_summary = truncate(str(context.arguments))

        try:
            self.session.step_start(
                agent=full_name, intent="execute_function", input_summary=input_summary
            )
        except Exception:
            pass  # Don't break execution if tracing fails

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
                                        tokens_in = usage.get("prompt_tokens", 0) or usage.get(
                                            "input_tokens", 0
                                        )

                                    if hasattr(usage, "completion_tokens"):
                                        tokens_out = usage.completion_tokens
                                    elif isinstance(usage, dict):
                                        tokens_out = usage.get("completion_tokens", 0) or usage.get(
                                            "output_tokens", 0
                                        )

                            yield chunk

                        # Stream finished successfully
                        stream_completed = True
                        duration_ms = int((time.time() - stream_start_time) * 1000)
                        try:
                            self.session.step_end(
                                agent=full_name,
                                outcome="success",
                                output_summary=truncate(accumulated_content),
                                duration_ms=duration_ms,
                                tokens_in=tokens_in,
                                tokens_out=tokens_out,
                            )
                        except Exception:
                            pass  # Don't break execution if tracing fails

                    except Exception as e:
                        # Stream error
                        try:
                            self.session.error(
                                category="execution_error",
                                error_type=type(e).__name__,
                                message=str(e),
                                recoverable=False,
                            )
                        except Exception:
                            pass  # Don't break execution if tracing fails
                        duration_ms = int((time.time() - stream_start_time) * 1000)
                        try:
                            self.session.step_end(
                                agent=full_name, outcome="error", duration_ms=duration_ms
                            )
                        except Exception:
                            pass  # Don't break execution if tracing fails
                        raise
                    finally:
                        # Ensure step_end is called even if stream is abandoned
                        if not stream_completed:
                            try:
                                duration_ms = int((time.time() - stream_start_time) * 1000)
                                self.session.step_end(
                                    agent=full_name,
                                    outcome="abandoned",
                                    output_summary=truncate(accumulated_content),
                                    duration_ms=duration_ms,
                                    tokens_in=tokens_in,
                                    tokens_out=tokens_out,
                                )
                            except Exception:
                                pass  # Don't fail if cleanup fails

                # Replace the stream with our wrapper
                result.value = stream_wrapper()
                return

            # Non-streaming handling
            duration_ms = int((time.time() - start_time) * 1000)
            output_summary = truncate(str(result)) if result else ""

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
                        tokens_out = usage.get("completion_tokens", 0) or usage.get(
                            "output_tokens", 0
                        )

            try:
                self.session.step_end(
                    agent=full_name,
                    outcome="success",
                    output_summary=output_summary,
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            except Exception:
                pass  # Don't break execution if tracing fails

        except Exception as e:
            # 4. Step End (Error) - only catches sync/setup errors, stream errors caught in wrapper
            duration_ms = int((time.time() - start_time) * 1000)
            try:
                self.session.error(
                    category="execution_error",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=False,
                )
            except Exception:
                pass  # Don't break execution if tracing fails
            try:
                self.session.step_end(agent=full_name, outcome="error", duration_ms=duration_ms)
            except Exception:
                pass  # Don't break execution if tracing fails
            raise


__all__ = [
    "SK_AVAILABLE",
    "LCTLSemanticKernelTracer",
    "is_available",
    "trace_kernel",
]


def trace_kernel(
    kernel: Kernel,
    chain_id: str | None = None,
    session: LCTLSession | None = None,
    verbose: bool = False,
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
