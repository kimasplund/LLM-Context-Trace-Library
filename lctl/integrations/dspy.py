"""DSPy integration for LCTL.

Provides automatic tracing of DSPy module executions, LLM calls,
and teleprompter optimization runs with LCTL for time-travel debugging.

Usage:
    from lctl.integrations.dspy import LCTLDSPyCallback, trace_module

    # Create callback and configure DSPy
    callback = LCTLDSPyCallback(chain_id="my-dspy-chain")

    # Use trace_module decorator
    @trace_module(callback)
    class MyModule(dspy.Module):
        def forward(self, question):
            return self.predictor(question=question)

    # Or wrap existing modules
    traced_cot = trace_module(dspy.ChainOfThought("question -> answer"), callback)

    # Export trace
    callback.export("trace.lctl.json")
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

from ..core.session import LCTLSession

try:
    import dspy
    from dspy import Module, Predict, Signature

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    Module = object
    Predict = None
    Signature = None
    dspy = None


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for summaries."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _extract_signature_info(signature: Any) -> Dict[str, Any]:
    """Extract information from a DSPy signature."""
    info = {"input_fields": [], "output_fields": []}

    if signature is None:
        return info

    if hasattr(signature, "input_fields"):
        input_fields = signature.input_fields
        if isinstance(input_fields, dict):
            info["input_fields"] = list(input_fields.keys())
        elif hasattr(input_fields, "__iter__"):
            info["input_fields"] = [str(f) for f in input_fields]

    if hasattr(signature, "output_fields"):
        output_fields = signature.output_fields
        if isinstance(output_fields, dict):
            info["output_fields"] = list(output_fields.keys())
        elif hasattr(output_fields, "__iter__"):
            info["output_fields"] = [str(f) for f in output_fields]

    return info


def _extract_prediction_text(prediction: Any) -> str:
    """Extract text representation from a DSPy prediction."""
    if prediction is None:
        return ""

    if isinstance(prediction, str):
        return prediction

    if hasattr(prediction, "answer"):
        return str(prediction.answer)

    if hasattr(prediction, "output"):
        return str(prediction.output)

    if hasattr(prediction, "completions") and prediction.completions:
        first_completion = prediction.completions[0]
        if hasattr(first_completion, "answer"):
            return str(first_completion.answer)
        return str(first_completion)

    if hasattr(prediction, "__dict__"):
        for key in ["response", "result", "text", "content"]:
            if key in prediction.__dict__:
                return str(prediction.__dict__[key])

    return str(prediction)[:500]


def _get_module_name(module: Any) -> str:
    """Get a readable name for a DSPy module."""
    if module is None:
        return "unknown"

    if hasattr(module, "name") and module.name:
        return str(module.name)

    if hasattr(module, "__class__"):
        class_name = module.__class__.__name__

        if hasattr(module, "signature") and module.signature:
            sig = module.signature
            if hasattr(sig, "__name__"):
                return f"{class_name}[{sig.__name__}]"
            elif isinstance(sig, str):
                return f"{class_name}[{_truncate(sig, 30)}]"

        return class_name

    return "dspy_module"


class DSPyNotAvailableError(ImportError):
    """Raised when DSPy is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "DSPy is not installed. Install with: pip install dspy-ai"
        )


def _check_dspy_available() -> None:
    """Check if DSPy is available, raise error if not."""
    if not DSPY_AVAILABLE:
        raise DSPyNotAvailableError()


class LCTLDSPyCallback:
    """DSPy callback handler that records events to LCTL.

    Captures:
    - Module forward() calls (start/end with timing)
    - LLM calls with token counts
    - Predictions as facts
    - Optimization iterations as checkpoints
    - Errors

    Example:
        callback = LCTLDSPyCallback(chain_id="my-chain")
        result = traced_module(question="What is LCTL?")
        callback.export("trace.lctl.json")
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the callback handler.

        Args:
            chain_id: Optional chain ID for the LCTL session.
            session: Optional existing LCTL session to use.
            verbose: Enable verbose output.
        """
        _check_dspy_available()

        self.session = session or LCTLSession(chain_id=chain_id or f"dspy-{str(uuid4())[:8]}")
        self._verbose = verbose
        self._module_stack: List[Dict[str, Any]] = []
        self._llm_call_count = 0
        self._optimization_iteration = 0
        self._current_module: Optional[str] = None

    @property
    def chain(self):
        """Access the underlying LCTL chain."""
        return self.session.chain

    def on_module_start(
        self,
        module: Any,
        inputs: Dict[str, Any],
    ) -> None:
        """Called when a DSPy module starts execution.

        Args:
            module: The DSPy module being executed.
            inputs: Input arguments to the module.
        """
        module_name = _get_module_name(module)
        self._current_module = module_name

        input_summary = ""
        if inputs:
            keys = list(inputs.keys())[:3]
            values_preview = []
            for k in keys:
                v = inputs[k]
                v_str = str(v)[:50] if v else "(empty)"
                values_preview.append(f"{k}={v_str}")
            input_summary = ", ".join(values_preview)

        self._module_stack.append({
            "module": module_name,
            "start_time": time.time(),
            "inputs": inputs,
        })

        self.session.step_start(
            agent=module_name,
            intent="module_forward",
            input_summary=_truncate(input_summary, 200),
        )

        sig_info = _extract_signature_info(getattr(module, "signature", None))
        if sig_info["input_fields"] or sig_info["output_fields"]:
            self.session.add_fact(
                fact_id=f"signature_{module_name}_{len(self.chain.events)}",
                text=f"Signature: {sig_info['input_fields']} -> {sig_info['output_fields']}",
                confidence=1.0,
                source=module_name,
            )

        if self._verbose:
            print(f"[LCTL] Module started: {module_name}")

    def on_module_end(
        self,
        module: Any,
        prediction: Any,
        error: Optional[BaseException] = None,
    ) -> None:
        """Called when a DSPy module completes execution.

        Args:
            module: The DSPy module that completed.
            prediction: The prediction result (if successful).
            error: The exception (if failed).
        """
        module_name = _get_module_name(module)

        stack_info = {}
        if self._module_stack:
            stack_info = self._module_stack.pop()

        start_time = stack_info.get("start_time", time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        if error is not None:
            self.session.step_end(
                agent=module_name,
                outcome="error",
                output_summary=f"Error: {type(error).__name__}: {str(error)[:100]}",
                duration_ms=duration_ms,
            )
            self.session.error(
                category="module_error",
                error_type=type(error).__name__,
                message=str(error),
                recoverable=False,
                suggested_action="Check module configuration and inputs",
            )
        else:
            prediction_text = _extract_prediction_text(prediction)
            self.session.step_end(
                agent=module_name,
                outcome="success",
                output_summary=_truncate(prediction_text, 200),
                duration_ms=duration_ms,
            )

            self.session.add_fact(
                fact_id=f"prediction_{module_name}_{len(self.chain.events)}",
                text=f"Prediction: {_truncate(prediction_text, 300)}",
                confidence=1.0,
                source=module_name,
            )

        self._current_module = None

        if self._verbose:
            status = "error" if error else "success"
            print(f"[LCTL] Module ended: {module_name} ({status}, {duration_ms}ms)")

    def on_llm_call(
        self,
        prompt: str,
        response: str,
        model: str = "unknown",
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """Called when an LLM call is made.

        Args:
            prompt: The prompt sent to the LLM.
            response: The response from the LLM.
            model: The model name/identifier.
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.
            duration_ms: Duration of the LLM call.
        """
        self._llm_call_count += 1
        agent = self._current_module or "llm"

        self.session.tool_call(
            tool=f"llm:{model}",
            input_data=_truncate(prompt, 300),
            output_data=_truncate(response, 300),
            duration_ms=duration_ms,
        )

        if tokens_in > 0 or tokens_out > 0:
            self.session.add_fact(
                fact_id=f"llm_usage_{self._llm_call_count}",
                text=f"LLM call #{self._llm_call_count}: {tokens_in} tokens in, {tokens_out} tokens out",
                confidence=1.0,
                source=agent,
            )

        if self._verbose:
            print(f"[LCTL] LLM call #{self._llm_call_count}: {model} ({tokens_in}+{tokens_out} tokens)")

    def on_prediction(
        self,
        prediction: Any,
        source: str = "unknown",
        confidence: float = 1.0,
    ) -> None:
        """Called when a prediction is generated.

        Args:
            prediction: The prediction object.
            source: Source module name.
            confidence: Confidence score (0-1).
        """
        prediction_text = _extract_prediction_text(prediction)

        self.session.add_fact(
            fact_id=f"prediction_{source}_{len(self.chain.events)}",
            text=prediction_text,
            confidence=confidence,
            source=source,
        )

    def on_optimization_start(
        self,
        teleprompter_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when an optimization run starts.

        Args:
            teleprompter_name: Name of the teleprompter.
            config: Optimization configuration.
        """
        self._optimization_iteration = 0

        config_summary = ""
        if config:
            config_items = [f"{k}={v}" for k, v in list(config.items())[:5]]
            config_summary = ", ".join(config_items)

        self.session.step_start(
            agent=teleprompter_name,
            intent="optimization",
            input_summary=_truncate(config_summary, 200),
        )

        self.session.add_fact(
            fact_id=f"optimization_config_{teleprompter_name}",
            text=f"Optimization started: {teleprompter_name} with config: {config_summary}",
            confidence=1.0,
            source=teleprompter_name,
        )

        if self._verbose:
            print(f"[LCTL] Optimization started: {teleprompter_name}")

    def on_optimization_iteration(
        self,
        iteration: int,
        score: Optional[float] = None,
        best_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called for each optimization iteration.

        Args:
            iteration: Current iteration number.
            score: Score for this iteration.
            best_score: Best score so far.
            metrics: Additional metrics.
        """
        self._optimization_iteration = iteration

        metrics_summary = f"iteration={iteration}"
        if score is not None:
            metrics_summary += f", score={score:.4f}"
        if best_score is not None:
            metrics_summary += f", best={best_score:.4f}"

        if metrics:
            for k, v in list(metrics.items())[:3]:
                metrics_summary += f", {k}={v}"

        self.session.checkpoint()

        self.session.add_fact(
            fact_id=f"optimization_iteration_{iteration}",
            text=f"Optimization iteration {iteration}: {metrics_summary}",
            confidence=1.0,
            source="teleprompter",
        )

        if self._verbose:
            print(f"[LCTL] Optimization iteration {iteration}: score={score}")

    def on_optimization_end(
        self,
        teleprompter_name: str,
        best_score: Optional[float] = None,
        total_iterations: int = 0,
    ) -> None:
        """Called when an optimization run completes.

        Args:
            teleprompter_name: Name of the teleprompter.
            best_score: Best score achieved.
            total_iterations: Total number of iterations.
        """
        output_summary = f"Optimization complete: {total_iterations} iterations"
        if best_score is not None:
            output_summary += f", best_score={best_score:.4f}"

        self.session.step_end(
            agent=teleprompter_name,
            outcome="success",
            output_summary=output_summary,
        )

        self.session.checkpoint()

        if self._verbose:
            print(f"[LCTL] Optimization ended: {teleprompter_name} ({total_iterations} iterations)")

    def record_error(
        self,
        error: BaseException,
        module_name: Optional[str] = None,
        recoverable: bool = True,
    ) -> None:
        """Record an error during DSPy execution.

        Args:
            error: The exception that occurred.
            module_name: Optional module name where error occurred.
            recoverable: Whether the error is recoverable.
        """
        self.session.error(
            category="dspy_error",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=recoverable,
            suggested_action="Check module configuration and LLM settings",
        )

    def export(self, path: str) -> None:
        """Export the LCTL chain to a file.

        Args:
            path: File path to export to (JSON or YAML).
        """
        self.session.export(path)
        if self._verbose:
            event_count = len(self.chain.events)
            print(f"[LCTL] Exported {event_count} events to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL chain as a dictionary."""
        return self.session.to_dict()


T = TypeVar("T")


def trace_module(
    module_or_callback: Union[Any, LCTLDSPyCallback],
    callback: Optional[LCTLDSPyCallback] = None,
) -> Union[Callable, Any]:
    """Decorator/wrapper to trace a DSPy module with LCTL.

    Can be used in two ways:

    1. As a decorator:
        @trace_module(callback)
        class MyModule(dspy.Module):
            ...

    2. As a wrapper:
        traced = trace_module(my_module, callback)

    Args:
        module_or_callback: Either a DSPy module to wrap, or a callback
            when used as a decorator.
        callback: The LCTLDSPyCallback to use (when wrapping a module).

    Returns:
        Traced module or decorator function.
    """
    _check_dspy_available()

    if isinstance(module_or_callback, LCTLDSPyCallback):
        cb = module_or_callback

        def decorator(module_class_or_instance: T) -> T:
            return _wrap_module(module_class_or_instance, cb)

        return decorator
    else:
        module = module_or_callback
        if callback is None:
            raise ValueError("callback is required when wrapping a module instance")
        return _wrap_module(module, callback)


def _wrap_module(module: Any, callback: LCTLDSPyCallback) -> Any:
    """Wrap a DSPy module with LCTL tracing.

    Args:
        module: The DSPy module to wrap.
        callback: The callback for recording events.

    Returns:
        Wrapped module with tracing.
    """
    if isinstance(module, type):
        return _wrap_module_class(module, callback)
    else:
        return _wrap_module_instance(module, callback)


def _wrap_module_class(module_class: type, callback: LCTLDSPyCallback) -> type:
    """Wrap a DSPy module class with LCTL tracing.

    Args:
        module_class: The module class to wrap.
        callback: The callback for recording events.

    Returns:
        Wrapped module class.
    """
    original_forward = module_class.forward

    @functools.wraps(original_forward)
    def traced_forward(self, *args, **kwargs):
        callback.on_module_start(self, kwargs)
        try:
            result = original_forward(self, *args, **kwargs)
            callback.on_module_end(self, result)
            return result
        except Exception as e:
            callback.on_module_end(self, None, error=e)
            raise

    module_class.forward = traced_forward
    module_class._lctl_callback = callback

    return module_class


def _wrap_module_instance(module: Any, callback: LCTLDSPyCallback) -> "TracedDSPyModule":
    """Wrap a DSPy module instance with LCTL tracing.

    Args:
        module: The module instance to wrap.
        callback: The callback for recording events.

    Returns:
        Wrapped module instance.
    """
    return TracedDSPyModule(module, callback)


class TracedDSPyModule:
    """Wrapper around a DSPy module with built-in LCTL tracing.

    Example:
        cot = dspy.ChainOfThought("question -> answer")
        traced_cot = TracedDSPyModule(cot, callback)
        result = traced_cot(question="What is LCTL?")
    """

    def __init__(
        self,
        module: Any,
        callback: LCTLDSPyCallback,
    ) -> None:
        """Initialize the traced module wrapper.

        Args:
            module: The DSPy module to wrap.
            callback: The LCTL callback for recording events.
        """
        _check_dspy_available()

        self._module = module
        self._callback = callback

        module_name = _get_module_name(module)
        callback.session.add_fact(
            fact_id=f"module_wrapped_{module_name}",
            text=f"Module '{module_name}' wrapped for tracing",
            confidence=1.0,
            source="lctl-dspy",
        )

    @property
    def module(self) -> Any:
        """Get the underlying DSPy module."""
        return self._module

    @property
    def callback(self) -> LCTLDSPyCallback:
        """Get the LCTL callback."""
        return self._callback

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the module with tracing.

        Args:
            *args: Positional arguments to the module.
            **kwargs: Keyword arguments to the module.

        Returns:
            The module's prediction result.
        """
        self._callback.on_module_start(self._module, kwargs)
        try:
            result = self._module(*args, **kwargs)
            self._callback.on_module_end(self._module, result)
            return result
        except Exception as e:
            self._callback.on_module_end(self._module, None, error=e)
            raise

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with tracing.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The module's prediction result.
        """
        return self.__call__(*args, **kwargs)

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self._callback.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL trace as a dictionary."""
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying module."""
        return getattr(self._module, name)


class LCTLDSPyTeleprompter:
    """Tracer for DSPy teleprompter optimization runs.

    Wraps a teleprompter to automatically trace optimization iterations
    and record performance metrics.

    Example:
        from dspy.teleprompt import BootstrapFewShot

        callback = LCTLDSPyCallback(chain_id="optimization")
        teleprompter = BootstrapFewShot(metric=my_metric)
        tracer = LCTLDSPyTeleprompter(teleprompter, callback)

        optimized_module = tracer.compile(my_module, trainset=train_data)
        callback.export("optimization_trace.lctl.json")
    """

    def __init__(
        self,
        teleprompter: Any,
        callback: LCTLDSPyCallback,
        verbose: bool = False,
    ) -> None:
        """Initialize the teleprompter tracer.

        Args:
            teleprompter: The DSPy teleprompter to wrap.
            callback: The LCTL callback for recording events.
            verbose: Enable verbose output.
        """
        _check_dspy_available()

        self._teleprompter = teleprompter
        self._callback = callback
        self._verbose = verbose
        self._teleprompter_name = type(teleprompter).__name__

        callback.session.add_fact(
            fact_id=f"teleprompter_{self._teleprompter_name}",
            text=f"Teleprompter '{self._teleprompter_name}' configured for tracing",
            confidence=1.0,
            source="lctl-dspy",
        )

    @property
    def teleprompter(self) -> Any:
        """Get the underlying teleprompter."""
        return self._teleprompter

    @property
    def callback(self) -> LCTLDSPyCallback:
        """Get the LCTL callback."""
        return self._callback

    def compile(
        self,
        student: Any,
        trainset: Optional[List[Any]] = None,
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Compile/optimize a module with tracing.

        Args:
            student: The student module to optimize.
            trainset: Training examples.
            valset: Validation examples.
            **kwargs: Additional arguments for the teleprompter.

        Returns:
            The optimized module.
        """
        config = {
            "student": _get_module_name(student),
            "trainset_size": len(trainset) if trainset else 0,
            "valset_size": len(valset) if valset else 0,
            **{k: str(v)[:50] for k, v in list(kwargs.items())[:5]},
        }

        self._callback.on_optimization_start(self._teleprompter_name, config)

        start_time = time.time()
        try:
            original_compile = self._teleprompter.compile

            result = original_compile(student, trainset=trainset, **kwargs)

            total_time_ms = int((time.time() - start_time) * 1000)

            self._callback.on_optimization_end(
                self._teleprompter_name,
                total_iterations=self._callback._optimization_iteration + 1,
            )

            self._callback.session.add_fact(
                fact_id=f"optimization_complete_{self._teleprompter_name}",
                text=f"Optimization completed in {total_time_ms}ms",
                confidence=1.0,
                source=self._teleprompter_name,
            )

            return result

        except Exception as e:
            self._callback.session.step_end(
                agent=self._teleprompter_name,
                outcome="error",
                output_summary=f"Optimization failed: {type(e).__name__}: {str(e)[:100]}",
            )
            self._callback.record_error(e, self._teleprompter_name, recoverable=False)
            raise

    def record_iteration(
        self,
        iteration: int,
        score: Optional[float] = None,
        best_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually record an optimization iteration.

        Use this when you need fine-grained control over iteration recording.

        Args:
            iteration: Current iteration number.
            score: Score for this iteration.
            best_score: Best score so far.
            metrics: Additional metrics.
        """
        self._callback.on_optimization_iteration(
            iteration, score, best_score, metrics
        )

    def export(self, path: str) -> None:
        """Export the LCTL trace to a file."""
        self._callback.export(path)

    def to_dict(self) -> Dict[str, Any]:
        """Export the LCTL trace as a dictionary."""
        return self._callback.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying teleprompter."""
        return getattr(self._teleprompter, name)


class DSPyModuleContext:
    """Context manager for tracing DSPy module execution.

    Example:
        with DSPyModuleContext(callback, "MyModule", {"question": "What is LCTL?"}) as ctx:
            result = module(question="What is LCTL?")
            ctx.set_result(result)
    """

    def __init__(
        self,
        callback: LCTLDSPyCallback,
        module_name: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the context manager.

        Args:
            callback: The LCTL callback.
            module_name: Name of the module being traced.
            inputs: Input arguments to the module.
        """
        self._callback = callback
        self._module_name = module_name
        self._inputs = inputs or {}
        self._result: Any = None
        self._error: Optional[BaseException] = None

    def __enter__(self) -> "DSPyModuleContext":
        """Start tracing the module execution."""

        class MockModule:
            def __init__(self, name: str):
                self.name = name

            def __class__(self):
                return type("MockModule", (), {"__name__": self._module_name})

        mock = MockModule(self._module_name)
        mock.__class__ = type(self._module_name, (), {})

        self._callback.on_module_start(mock, self._inputs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracing the module execution."""

        class MockModule:
            def __init__(self, name: str):
                self.name = name

        mock = MockModule(self._module_name)
        mock.__class__ = type(self._module_name, (), {})

        if exc_type is not None:
            self._callback.on_module_end(mock, None, error=exc_val)
        else:
            self._callback.on_module_end(mock, self._result)

        return False

    def set_result(self, result: Any) -> None:
        """Set the result of the module execution.

        Args:
            result: The prediction result.
        """
        self._result = result


def is_available() -> bool:
    """Check if DSPy integration is available.

    Returns:
        True if DSPy is installed, False otherwise.
    """
    return DSPY_AVAILABLE


__all__ = [
    "DSPY_AVAILABLE",
    "DSPyNotAvailableError",
    "LCTLDSPyCallback",
    "TracedDSPyModule",
    "LCTLDSPyTeleprompter",
    "DSPyModuleContext",
    "trace_module",
    "is_available",
]
