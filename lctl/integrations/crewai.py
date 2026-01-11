"""CrewAI integration for LCTL.

Provides wrappers around CrewAI components to automatically trace crew execution
with LCTL for time-travel debugging.

Usage:
    from lctl.integrations.crewai import LCTLCrew, LCTLAgent

    # Create agents with tracing
    researcher = LCTLAgent(
        role="Senior Researcher",
        goal="Research topics thoroughly",
        backstory="Expert researcher with years of experience"
    )

    # Create crew with tracing
    crew = LCTLCrew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    result = crew.kickoff()
    crew.export_trace("research_crew.lctl.json")
"""

import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..core.session import LCTLSession

# Try to import CrewAI - graceful degradation if not available
try:
    from crewai import Agent, Crew, Task
    from crewai.agents.agent_builder.base_agent import BaseAgent
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Crew = None
    Task = None
    BaseAgent = None


class CrewAINotAvailableError(ImportError):
    """Raised when CrewAI is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "CrewAI is not installed. Install it with: pip install crewai"
        )


def _check_crewai_available() -> None:
    """Check if CrewAI is available, raise error if not."""
    if not CREWAI_AVAILABLE:
        raise CrewAINotAvailableError()


class LCTLAgent:
    """Wrapper around CrewAI Agent with LCTL tracing.

    Captures agent role, goal, and backstory as metadata.
    Tracks all agent actions during crew execution.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[Any]] = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
        allow_delegation: bool = True,
        max_iter: int = 25,
        max_rpm: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize an LCTL-traced CrewAI agent.

        Args:
            role: The agent's role description.
            goal: The agent's primary goal.
            backstory: Background context for the agent.
            tools: List of tools available to the agent.
            llm: Language model to use.
            verbose: Enable verbose output.
            allow_delegation: Allow task delegation to other agents.
            max_iter: Maximum iterations for task completion.
            max_rpm: Maximum requests per minute.
            max_execution_time: Maximum execution time in seconds.
            **kwargs: Additional arguments passed to CrewAI Agent.
        """
        _check_crewai_available()

        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.allow_delegation = allow_delegation
        self.verbose = verbose

        # Store metadata for tracing
        self._metadata: Dict[str, Any] = {
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "allow_delegation": allow_delegation,
            "tool_count": len(self.tools),
            "tool_names": [getattr(t, "name", str(t)) for t in self.tools],
        }

        # Create the underlying CrewAI agent
        self._agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            llm=llm,
            verbose=verbose,
            allow_delegation=allow_delegation,
            max_iter=max_iter,
            max_rpm=max_rpm,
            max_execution_time=max_execution_time,
            **kwargs
        )

    @property
    def agent(self) -> "Agent":
        """Get the underlying CrewAI Agent."""
        return self._agent

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get agent metadata for tracing."""
        return self._metadata

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying agent."""
        return getattr(self._agent, name)


class LCTLTask:
    """Wrapper around CrewAI Task with LCTL tracing.

    Captures task description, expected output, and assigned agent.
    """

    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: Optional[Union[LCTLAgent, "Agent"]] = None,
        tools: Optional[List[Any]] = None,
        async_execution: bool = False,
        context: Optional[List["LCTLTask"]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize an LCTL-traced CrewAI task.

        Args:
            description: Task description.
            expected_output: Expected output format/content.
            agent: Agent assigned to this task.
            tools: Task-specific tools.
            async_execution: Enable async execution.
            context: Context from other tasks.
            **kwargs: Additional arguments passed to CrewAI Task.
        """
        _check_crewai_available()

        self.description = description
        self.expected_output = expected_output
        self._lctl_agent = agent if isinstance(agent, LCTLAgent) else None

        # Get the underlying CrewAI agent
        underlying_agent = agent.agent if isinstance(agent, LCTLAgent) else agent

        # Get underlying tasks for context
        underlying_context = None
        if context:
            underlying_context = [
                t.task if isinstance(t, LCTLTask) else t for t in context
            ]

        # Store metadata for tracing
        self._metadata: Dict[str, Any] = {
            "description": description[:200],  # Truncate for readability
            "expected_output": expected_output[:200],
            "async_execution": async_execution,
            "has_context": context is not None and len(context) > 0,
        }

        # Create the underlying CrewAI task
        self._task = Task(
            description=description,
            expected_output=expected_output,
            agent=underlying_agent,
            tools=tools,
            async_execution=async_execution,
            context=underlying_context,
            **kwargs
        )

    @property
    def task(self) -> "Task":
        """Get the underlying CrewAI Task."""
        return self._task

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get task metadata for tracing."""
        return self._metadata

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying task."""
        return getattr(self._task, name)


class LCTLCrew:
    """Wrapper around CrewAI Crew with LCTL tracing.

    Automatically traces:
    - Crew kickoff and completion
    - Agent steps (which agent, what task)
    - Task execution (start, end, duration)
    - Agent delegation events
    - Tool usage
    - Errors

    Works with both sequential and hierarchical processes.
    """

    def __init__(
        self,
        agents: List[Union[LCTLAgent, "Agent"]],
        tasks: List[Union[LCTLTask, "Task"]],
        process: Optional[str] = None,
        verbose: bool = False,
        manager_llm: Optional[Any] = None,
        manager_agent: Optional[Union[LCTLAgent, "Agent"]] = None,
        chain_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize an LCTL-traced CrewAI crew.

        Args:
            agents: List of agents in the crew.
            tasks: List of tasks to execute.
            process: Process type ('sequential' or 'hierarchical').
            verbose: Enable verbose output.
            manager_llm: LLM for hierarchical process manager.
            manager_agent: Custom manager agent for hierarchical process.
            chain_id: Custom chain ID for LCTL tracing.
            **kwargs: Additional arguments passed to CrewAI Crew.
        """
        _check_crewai_available()

        self._lctl_agents = [a for a in agents if isinstance(a, LCTLAgent)]
        self._lctl_tasks = [t for t in tasks if isinstance(t, LCTLTask)]

        # Extract underlying CrewAI objects
        underlying_agents = [
            a.agent if isinstance(a, LCTLAgent) else a for a in agents
        ]
        underlying_tasks = [
            t.task if isinstance(t, LCTLTask) else t for t in tasks
        ]
        underlying_manager = (
            manager_agent.agent if isinstance(manager_agent, LCTLAgent) else manager_agent
        )

        # Initialize LCTL session
        self._session = LCTLSession(chain_id=chain_id or f"crew-{str(uuid4())[:8]}")
        self._process = process or "sequential"
        self._verbose = verbose

        # Store crew metadata
        self._metadata: Dict[str, Any] = {
            "process": self._process,
            "agent_count": len(agents),
            "task_count": len(tasks),
            "agent_roles": [
                a.role if isinstance(a, LCTLAgent) else getattr(a, "role", "unknown")
                for a in agents
            ],
        }

        # Create the underlying CrewAI crew
        crew_kwargs: Dict[str, Any] = {
            "agents": underlying_agents,
            "tasks": underlying_tasks,
            "verbose": verbose,
            **kwargs
        }

        if process:
            crew_kwargs["process"] = process
        if manager_llm:
            crew_kwargs["manager_llm"] = manager_llm
        if underlying_manager:
            crew_kwargs["manager_agent"] = underlying_manager

        self._crew = Crew(**crew_kwargs)

        # Set up callbacks for tracing
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Set up CrewAI callbacks for LCTL tracing."""
        # Store original callbacks if any
        self._original_step_callback = getattr(self._crew, "step_callback", None)
        self._original_task_callback = getattr(self._crew, "task_callback", None)

        # Install LCTL callbacks
        self._crew.step_callback = self._on_step
        self._crew.task_callback = self._on_task_complete

    def _get_agent_name(self, agent: Any) -> str:
        """Get a readable name for an agent."""
        if hasattr(agent, "role"):
            return str(agent.role).replace(" ", "-").lower()[:30]
        return "unknown-agent"

    def _on_step(self, step_output: Any) -> None:
        """Callback for each agent step."""
        try:
            agent_name = "unknown"
            thought = ""
            tool_name = None
            tool_input = None
            tool_output = None

            # Extract step information based on CrewAI output format
            if hasattr(step_output, "agent"):
                agent_name = self._get_agent_name(step_output.agent)

            if hasattr(step_output, "thought"):
                thought = str(step_output.thought)[:500]

            if hasattr(step_output, "tool"):
                tool_name = str(step_output.tool)

            if hasattr(step_output, "tool_input"):
                tool_input = step_output.tool_input

            if hasattr(step_output, "result"):
                tool_output = str(step_output.result)[:500] if step_output.result else None

            # Record step start
            self._session.step_start(
                agent=agent_name,
                intent="execute_step",
                input_summary=thought[:100] if thought else "Agent step"
            )

            # Record tool call if applicable
            if tool_name:
                self._session.tool_call(
                    tool=tool_name,
                    input_data=tool_input,
                    output_data=tool_output,
                    duration_ms=0  # Duration not available at step level
                )

            # Record step end
            self._session.step_end(
                agent=agent_name,
                outcome="success",
                output_summary=tool_output[:100] if tool_output else "Step completed"
            )

        except Exception as e:
            # Don't let tracing errors break execution
            if self._verbose:
                print(f"[LCTL] Warning: Error in step callback: {e}")

        # Call original callback if present
        if self._original_step_callback:
            self._original_step_callback(step_output)

    def _on_task_complete(self, task_output: Any) -> None:
        """Callback for task completion."""
        try:
            task_description = ""

            if hasattr(task_output, "description"):
                task_description = str(task_output.description)[:200]

            if hasattr(task_output, "raw"):
                str(task_output.raw)[:500]
            elif hasattr(task_output, "output"):
                str(task_output.output)[:500]

            # Add fact about task completion
            fact_id = f"task-{len(self._session.chain.events)}"
            self._session.add_fact(
                fact_id=fact_id,
                text=f"Task completed: {task_description}",
                confidence=1.0,
                source="crew-execution"
            )

        except Exception as e:
            if self._verbose:
                print(f"[LCTL] Warning: Error in task callback: {e}")

        # Call original callback if present
        if self._original_task_callback:
            self._original_task_callback(task_output)

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the crew with LCTL tracing.

        Args:
            inputs: Optional inputs to pass to the crew.

        Returns:
            The crew execution result.
        """
        start_time = time.time()

        # Record crew kickoff
        self._session.step_start(
            agent="crew-manager",
            intent="kickoff",
            input_summary=f"Starting {self._process} crew with {len(self._crew.agents)} agents"
        )

        # Add crew metadata as facts
        self._session.add_fact(
            fact_id="crew-config",
            text=f"Crew configuration: {self._process} process, "
                 f"{len(self._crew.agents)} agents, {len(self._crew.tasks)} tasks",
            confidence=1.0,
            source="crew-manager"
        )

        # Record agent information
        for i, agent in enumerate(self._crew.agents):
            role = getattr(agent, "role", f"Agent-{i}")
            goal = getattr(agent, "goal", "")
            self._session.add_fact(
                fact_id=f"agent-{i}",
                text=f"Agent '{role}': {goal[:100]}",
                confidence=1.0,
                source="crew-manager"
            )

        try:
            # Execute the crew
            result = self._crew.kickoff(inputs=inputs)

            duration_ms = int((time.time() - start_time) * 1000)

            # Record successful completion
            self._session.step_end(
                agent="crew-manager",
                outcome="success",
                output_summary="Crew execution completed successfully",
                duration_ms=duration_ms
            )

            # Add final result as fact
            result_text = str(result)[:500] if result else "No output"
            self._session.add_fact(
                fact_id="crew-result",
                text=f"Final result: {result_text}",
                confidence=1.0,
                source="crew-manager"
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Record error
            self._session.step_end(
                agent="crew-manager",
                outcome="error",
                output_summary=f"Crew execution failed: {str(e)[:200]}",
                duration_ms=duration_ms
            )

            self._session.error(
                category="crew_execution_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False,
                suggested_action="Check agent configurations and task definitions"
            )

            raise

    def kickoff_async(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the crew asynchronously with LCTL tracing.

        Args:
            inputs: Optional inputs to pass to the crew.

        Returns:
            Async result of crew execution.
        """
        # Record async kickoff
        self._session.step_start(
            agent="crew-manager",
            intent="kickoff_async",
            input_summary=f"Starting async {self._process} crew"
        )

        try:
            result = self._crew.kickoff_async(inputs=inputs)
            return result
        except Exception as e:
            self._session.error(
                category="crew_async_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False
            )
            raise

    def export_trace(self, path: str) -> None:
        """Export the LCTL trace to a file.

        Args:
            path: File path for the trace (supports .json and .yaml).
        """
        self._session.export(path)
        if self._verbose:
            event_count = len(self._session.chain.events)
            print(f"[LCTL] Exported {event_count} events to {path}")

    def get_trace(self) -> Dict[str, Any]:
        """Get the LCTL trace as a dictionary.

        Returns:
            The trace data as a dictionary.
        """
        return self._session.to_dict()

    @property
    def session(self) -> LCTLSession:
        """Get the LCTL session for advanced operations."""
        return self._session

    @property
    def crew(self) -> "Crew":
        """Get the underlying CrewAI Crew."""
        return self._crew

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying crew."""
        return getattr(self._crew, name)


def trace_crew(
    crew: "Crew",
    chain_id: Optional[str] = None,
    verbose: bool = False
) -> LCTLCrew:
    """Wrap an existing CrewAI Crew with LCTL tracing.

    This is useful when you have an existing Crew instance and want to
    add tracing without modifying the original code.

    Args:
        crew: Existing CrewAI Crew instance.
        chain_id: Custom chain ID for tracing.
        verbose: Enable verbose output.

    Returns:
        An LCTLCrew wrapper around the existing crew.

    Example:
        crew = Crew(agents=[...], tasks=[...])
        traced = trace_crew(crew)
        result = traced.kickoff()
        traced.export_trace("trace.lctl.json")
    """
    _check_crewai_available()

    # Create wrapper that uses the existing crew
    wrapper = object.__new__(LCTLCrew)
    wrapper._crew = crew
    wrapper._session = LCTLSession(chain_id=chain_id or f"crew-{str(uuid4())[:8]}")
    wrapper._process = getattr(crew, "process", "sequential")
    wrapper._verbose = verbose
    wrapper._lctl_agents = []
    wrapper._lctl_tasks = []
    wrapper._metadata = {
        "process": wrapper._process,
        "agent_count": len(crew.agents),
        "task_count": len(crew.tasks),
    }
    wrapper._original_step_callback = None
    wrapper._original_task_callback = None
    wrapper._setup_callbacks()

    return wrapper


__all__ = [
    "CREWAI_AVAILABLE",
    "CrewAINotAvailableError",
    "LCTLAgent",
    "LCTLTask",
    "LCTLCrew",
    "trace_crew",
]
