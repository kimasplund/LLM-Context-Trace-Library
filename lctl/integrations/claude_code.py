"""Claude Code integration for LCTL.

This module provides tracing for Claude Code's multi-agent workflows using
the Task tool. It captures agent spawning, tool calls, and agent completions
as LCTL events for debugging and analysis.

Usage with Claude Code hooks:

    1. Create hook scripts in your project's .claude/hooks/ directory
    2. Use the LCTLClaudeCodeTracer to record events

Example hook script (.claude/hooks/PostToolUse.sh):

    #!/bin/bash
    # Trace Task tool completions
    if [ "$CLAUDE_TOOL_NAME" = "Task" ]; then
        python -c "
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer
    import os, json

    tracer = LCTLClaudeCodeTracer.get_or_create()
    tracer.on_task_complete(
        agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
        description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
        result=os.environ.get('CLAUDE_TOOL_RESULT', '')[:1000]
    )
    "
    fi

Programmatic usage:

    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    # Start tracing a workflow
    tracer = LCTLClaudeCodeTracer(chain_id="my-workflow")

    # Record agent spawning
    tracer.on_task_start(
        agent_type="implementor",
        description="Implement authentication",
        prompt="Add JWT auth to the API..."
    )

    # Record agent completion
    tracer.on_task_complete(
        agent_type="implementor",
        description="Implement authentication",
        result="Successfully implemented JWT authentication..."
    )

    # Record tool calls within agents
    tracer.on_tool_call(
        tool_name="Bash",
        input_data={"command": "pytest"},
        output_data={"exit_code": 0},
        agent="implementor"
    )

    # Export trace
    tracer.export("workflow-trace.lctl.json")
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.session import LCTLSession

# Singleton instance for hook-based usage
_tracer_instance: Optional["LCTLClaudeCodeTracer"] = None
_tracer_file: Optional[Path] = None


class LCTLClaudeCodeTracer:
    """Tracer for Claude Code multi-agent workflows.

    This tracer captures events from Claude Code's Task tool usage,
    enabling time-travel debugging of multi-agent workflows.

    Attributes:
        session: The underlying LCTL session
        agent_stack: Stack of currently active agents (for nested spawning)
        tool_counts: Count of tool calls per agent
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize the Claude Code tracer.

        Args:
            chain_id: Identifier for the trace chain
            session: Existing LCTL session to use
            output_dir: Directory for trace output (default: .claude/traces/)
        """
        self.session = session or LCTLSession(
            chain_id=chain_id or f"claude-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.output_dir = Path(output_dir) if output_dir else Path(".claude/traces")
        self.agent_stack: List[Dict[str, Any]] = []
        self.tool_counts: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}
        self._background_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> info
        self._agent_ids: Dict[str, str] = {}  # agent_type -> last agent_id
        self._parallel_group: Optional[str] = None  # Current parallel execution group
        self._file_changes: List[Dict[str, Any]] = []  # Track file modifications

    @classmethod
    def get_or_create(
        cls,
        chain_id: Optional[str] = None,
        state_file: Optional[str] = None,
    ) -> "LCTLClaudeCodeTracer":
        """Get existing tracer instance or create new one.

        This is useful for hook scripts that need to maintain state
        across multiple invocations.

        Args:
            chain_id: Chain ID for new tracer
            state_file: File to persist tracer state

        Returns:
            Tracer instance
        """
        global _tracer_instance, _tracer_file

        state_path = Path(state_file) if state_file else Path(".claude/traces/.lctl-state.json")

        if _tracer_instance is not None:
            return _tracer_instance

        # Try to restore from state file
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)

                # Load existing chain
                chain_path = Path(state.get("chain_path", ""))
                if chain_path.exists():
                    from ..core.events import Chain
                    chain = Chain.load(chain_path)
                    session = LCTLSession(chain_id=chain.id)
                    session.chain = chain
                    session._seq = len(chain.events)

                    _tracer_instance = cls(session=session)
                    _tracer_instance.agent_stack = state.get("agent_stack", [])
                    _tracer_instance.tool_counts = state.get("tool_counts", {})
                    _tracer_instance._start_times = state.get("start_times", {})
                    _tracer_instance._background_tasks = state.get("background_tasks", {})
                    _tracer_instance._agent_ids = state.get("agent_ids", {})
                    _tracer_instance._parallel_group = state.get("parallel_group")
                    _tracer_instance._file_changes = state.get("file_changes", [])
                    _tracer_file = state_path
                    return _tracer_instance
            except Exception:
                pass  # Fall through to create new

        # Create new tracer
        _tracer_instance = cls(chain_id=chain_id)
        _tracer_file = state_path
        _tracer_instance._save_state()
        return _tracer_instance

    def _save_state(self) -> None:
        """Persist tracer state for hook continuity."""
        if _tracer_file is None:
            return

        _tracer_file.parent.mkdir(parents=True, exist_ok=True)

        # Save chain
        chain_path = self.output_dir / f"{self.session.chain.id}.lctl.json"
        chain_path.parent.mkdir(parents=True, exist_ok=True)
        self.session.export(str(chain_path))

        # Save state
        state = {
            "chain_path": str(chain_path),
            "agent_stack": self.agent_stack,
            "tool_counts": self.tool_counts,
            "start_times": self._start_times,
            "background_tasks": self._background_tasks,
            "agent_ids": self._agent_ids,
            "parallel_group": self._parallel_group,
            "file_changes": self._file_changes,
        }
        with open(_tracer_file, "w") as f:
            json.dump(state, f)

    def on_task_start(
        self,
        agent_type: str,
        description: str,
        prompt: str = "",
        model: Optional[str] = None,
        run_in_background: bool = False,
        resume_agent_id: Optional[str] = None,
        parallel_group: Optional[str] = None,
    ) -> None:
        """Record a Task tool invocation (agent spawn).

        Args:
            agent_type: The subagent_type parameter (e.g., "implementor", "Explore")
            description: The task description
            prompt: The full prompt sent to the agent
            model: Optional model override
            run_in_background: Whether this is a background task
            resume_agent_id: Agent ID if resuming a previous agent
            parallel_group: Group ID if part of parallel execution
        """
        self._start_times[agent_type] = time.time()

        # Track parallel execution
        if parallel_group:
            self._parallel_group = parallel_group

        # Track nested agents
        agent_info = {
            "agent_type": agent_type,
            "description": description,
            "start_time": self._start_times[agent_type],
            "background": run_in_background,
            "resume_from": resume_agent_id,
            "parallel_group": parallel_group,
        }
        self.agent_stack.append(agent_info)

        # Build input summary with context
        input_parts = []
        if resume_agent_id:
            input_parts.append(f"[RESUME:{resume_agent_id}]")
        if run_in_background:
            input_parts.append("[BACKGROUND]")
        if parallel_group:
            input_parts.append(f"[PARALLEL:{parallel_group}]")
        input_parts.append(prompt[:400] if prompt else description)
        input_summary = " ".join(input_parts)

        self.session.step_start(
            agent=agent_type,
            intent=description[:100],
            input_summary=input_summary,
        )

        # Add fact about agent spawn
        spawn_text = f"Spawned {agent_type}: {description}"
        if resume_agent_id:
            spawn_text += f" (resuming {resume_agent_id})"
        if run_in_background:
            spawn_text += " [background]"

        self.session.add_fact(
            fact_id=f"spawn-{agent_type}-{len(self.session.chain.events)}",
            text=spawn_text,
            confidence=1.0,
            source="claude-code",
        )

        if model:
            self.session.add_fact(
                fact_id=f"model-{agent_type}-{len(self.session.chain.events)}",
                text=f"Using model: {model}",
                confidence=1.0,
                source="claude-code",
            )

        self._save_state()

    def on_task_complete(
        self,
        agent_type: str,
        description: str = "",
        result: str = "",
        success: bool = True,
        error_message: Optional[str] = None,
        agent_id: Optional[str] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record a Task tool completion.

        Args:
            agent_type: The agent that completed
            description: Task description
            result: Result summary from the agent
            success: Whether the agent succeeded
            error_message: Error message if failed
            agent_id: The agent ID returned (for resume capability)
            tokens_in: Input tokens used
            tokens_out: Output tokens generated
        """
        duration_ms = 0
        if agent_type in self._start_times:
            duration_ms = int((time.time() - self._start_times[agent_type]) * 1000)
            del self._start_times[agent_type]

        # Store agent_id for potential resume tracking
        if agent_id:
            self._agent_ids[agent_type] = agent_id

        # Pop from agent stack
        if self.agent_stack and self.agent_stack[-1]["agent_type"] == agent_type:
            self.agent_stack.pop()

        outcome = "success" if success else "failure"

        self.session.step_end(
            agent=agent_type,
            outcome=outcome,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # Add fact about result
        if result:
            result_text = f"{agent_type} result: {result[:500]}"
            if agent_id:
                result_text += f" [agent_id: {agent_id}]"

            self.session.add_fact(
                fact_id=f"result-{agent_type}-{len(self.session.chain.events)}",
                text=result_text,
                confidence=0.9 if success else 0.5,
                source=agent_type,
            )

        # Record error if failed
        if not success and error_message:
            self.session.error(
                category="agent_failure",
                error_type="TaskError",
                message=error_message,
                recoverable=True,
            )

        self._save_state()

    def on_tool_call(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
        agent: Optional[str] = None,
    ) -> None:
        """Record a tool call within an agent.

        Args:
            tool_name: Name of the tool (Bash, Read, Write, etc.)
            input_data: Tool input parameters
            output_data: Tool output
            duration_ms: Tool execution time
            agent: Agent that made the call (inferred from stack if not provided)
        """
        # Infer agent from stack
        if agent is None and self.agent_stack:
            agent = self.agent_stack[-1]["agent_type"]

        # Track tool counts
        key = f"{agent or 'main'}:{tool_name}"
        self.tool_counts[key] = self.tool_counts.get(key, 0) + 1

        # Truncate large data
        def truncate(data: Any, max_len: int = 500) -> Any:
            if isinstance(data, str):
                return data[:max_len] + "..." if len(data) > max_len else data
            elif isinstance(data, dict):
                return {k: truncate(v, max_len) for k, v in list(data.items())[:10]}
            return data

        self.session.tool_call(
            tool=tool_name,
            input_data=truncate(input_data),
            output_data=truncate(output_data or {}),
            duration_ms=duration_ms,
        )

        self._save_state()

    def on_fact_discovered(
        self,
        fact_id: str,
        text: str,
        confidence: float = 0.8,
        agent: Optional[str] = None,
    ) -> None:
        """Record a fact discovered by an agent.

        Args:
            fact_id: Unique fact identifier
            text: Fact content
            confidence: Confidence score (0.0-1.0)
            agent: Agent that discovered the fact
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        self.session.add_fact(
            fact_id=fact_id,
            text=text,
            confidence=confidence,
            source=source or "claude-code",
        )
        self._save_state()

    def on_fact_updated(
        self,
        fact_id: str,
        confidence: Optional[float] = None,
        text: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """Update a fact's confidence or text.

        Args:
            fact_id: Fact to update
            confidence: New confidence score
            text: New text (optional)
            reason: Reason for update
        """
        self.session.modify_fact(
            fact_id=fact_id,
            confidence=confidence,
            text=text,
            reason=reason,
        )
        self._save_state()

    def on_user_interaction(
        self,
        question: str,
        response: str,
        options: Optional[List[str]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Record a human-in-the-loop interaction (AskUserQuestion).

        Args:
            question: The question asked to user
            response: User's response
            options: Available options presented
            agent: Agent that asked (inferred from stack if not provided)
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        # Record as tool call
        self.session.tool_call(
            tool="AskUserQuestion",
            input_data={
                "question": question[:500],
                "options": options or [],
            },
            output_data={"response": response[:500]},
            duration_ms=0,  # User think time not tracked
        )

        # Add fact about user decision
        self.session.add_fact(
            fact_id=f"user-decision-{len(self.session.chain.events)}",
            text=f"User responded to '{question[:100]}': {response[:200]}",
            confidence=1.0,  # User decisions are ground truth
            source="human",
        )

        self._save_state()

    def on_file_change(
        self,
        file_path: str,
        change_type: str,  # "create", "edit", "delete"
        agent: Optional[str] = None,
        lines_added: int = 0,
        lines_removed: int = 0,
    ) -> None:
        """Record a file modification.

        Args:
            file_path: Path to the modified file
            change_type: Type of change (create, edit, delete)
            agent: Agent that made the change
            lines_added: Lines added
            lines_removed: Lines removed
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        # Track file change
        change_info = {
            "file_path": file_path,
            "change_type": change_type,
            "agent": source,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "timestamp": time.time(),
        }
        self._file_changes.append(change_info)

        # Add fact about file change
        change_text = f"File {change_type}: {file_path}"
        if lines_added or lines_removed:
            change_text += f" (+{lines_added}/-{lines_removed})"

        self.session.add_fact(
            fact_id=f"file-{change_type}-{len(self.session.chain.events)}",
            text=change_text,
            confidence=1.0,
            source=source or "claude-code",
        )

        self._save_state()

    def on_web_fetch(
        self,
        url: str,
        prompt: str,
        result_summary: str,
        agent: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        """Record a web fetch operation.

        Args:
            url: URL fetched
            prompt: Prompt used for extraction
            result_summary: Summary of fetched content
            agent: Agent that made the request
            duration_ms: Request duration
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        self.session.tool_call(
            tool="WebFetch",
            input_data={"url": url, "prompt": prompt[:200]},
            output_data={"summary": result_summary[:500]},
            duration_ms=duration_ms,
        )

        # Add fact about external data
        self.session.add_fact(
            fact_id=f"web-{len(self.session.chain.events)}",
            text=f"Fetched from {url}: {result_summary[:200]}",
            confidence=0.7,  # External data has lower confidence
            source=source or "web",
        )

        self._save_state()

    def on_web_search(
        self,
        query: str,
        results_count: int,
        top_result: str = "",
        agent: Optional[str] = None,
    ) -> None:
        """Record a web search operation.

        Args:
            query: Search query
            results_count: Number of results returned
            top_result: Summary of top result
            agent: Agent that made the search
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        self.session.tool_call(
            tool="WebSearch",
            input_data={"query": query},
            output_data={"results_count": results_count, "top": top_result[:200]},
            duration_ms=0,
        )

        self.session.add_fact(
            fact_id=f"search-{len(self.session.chain.events)}",
            text=f"Searched '{query}': {results_count} results. Top: {top_result[:150]}",
            confidence=0.75,
            source=source or "web",
        )

        self._save_state()

    def start_parallel_group(self, group_id: Optional[str] = None) -> str:
        """Start a parallel execution group.

        Args:
            group_id: Optional group identifier

        Returns:
            The group ID
        """
        self._parallel_group = group_id or f"parallel-{len(self.session.chain.events)}"
        return self._parallel_group

    def end_parallel_group(self) -> None:
        """End the current parallel execution group."""
        if self._parallel_group:
            self.session.add_fact(
                fact_id=f"parallel-end-{self._parallel_group}",
                text=f"Parallel group {self._parallel_group} completed",
                confidence=1.0,
                source="claude-code",
            )
            self._parallel_group = None
            self._save_state()

    def get_file_changes(self) -> List[Dict[str, Any]]:
        """Get list of file changes made during the workflow.

        Returns:
            List of file change records
        """
        return self._file_changes.copy()

    def get_agent_ids(self) -> Dict[str, str]:
        """Get mapping of agent types to their last agent IDs.

        Returns:
            Dict mapping agent_type to agent_id (for resume)
        """
        return self._agent_ids.copy()

    def checkpoint(self, description: str = "") -> None:
        """Create a checkpoint for fast replay.

        Args:
            description: Checkpoint description
        """
        self.session.checkpoint(description=description)
        self._save_state()

    def export(self, path: Optional[str] = None) -> str:
        """Export the trace to a file.

        Args:
            path: Output path (default: auto-generated in output_dir)

        Returns:
            Path to exported file
        """
        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = str(self.output_dir / f"{self.session.chain.id}.lctl.json")

        self.session.export(path)
        return path

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the traced workflow.

        Returns:
            Summary dict with agent stats, tool counts, facts, etc.
        """
        from ..core.events import ReplayEngine

        engine = ReplayEngine(self.session.chain)
        state = engine.replay_all()
        trace = engine.get_trace()

        # Agent stats
        agent_stats: Dict[str, Dict[str, Any]] = {}
        for step in trace:
            agent = step["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {"steps": 0, "duration_ms": 0, "tokens": 0}
            agent_stats[agent]["steps"] += 1
            agent_stats[agent]["duration_ms"] += step.get("duration_ms", 0)
            agent_stats[agent]["tokens"] += step.get("tokens_in", 0) + step.get("tokens_out", 0)

        # Count user interactions
        user_interactions = sum(
            1 for e in self.session.chain.events
            if e.type.value == "tool_call" and e.data.get("tool") == "AskUserQuestion"
        )

        return {
            "chain_id": self.session.chain.id,
            "event_count": len(self.session.chain.events),
            "agent_stats": agent_stats,
            "tool_counts": self.tool_counts,
            "fact_count": len(state.facts),
            "error_count": len(state.errors),
            "total_duration_ms": state.metrics.get("total_duration_ms", 0),
            "total_tokens_in": state.metrics.get("total_tokens_in", 0),
            "total_tokens_out": state.metrics.get("total_tokens_out", 0),
            "file_changes": len(self._file_changes),
            "user_interactions": user_interactions,
            "agent_ids": self._agent_ids,
        }

    def reset(self) -> None:
        """Reset the tracer for a new workflow."""
        global _tracer_instance, _tracer_file

        self.session = LCTLSession(
            chain_id=f"claude-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.agent_stack = []
        self.tool_counts = {}
        self._start_times = {}
        self._background_tasks = {}
        self._agent_ids = {}
        self._parallel_group = None
        self._file_changes = []

        # Clear state file
        if _tracer_file and _tracer_file.exists():
            _tracer_file.unlink()

        _tracer_instance = None


def generate_hooks(output_dir: str = ".claude/hooks") -> Dict[str, str]:
    """Generate Claude Code hook scripts for LCTL tracing.

    Args:
        output_dir: Directory to write hook scripts

    Returns:
        Dict mapping hook name to file path
    """
    hooks_dir = Path(output_dir)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hooks = {}

    # PreToolUse hook for Task tool
    pre_tool_hook = '''#!/bin/bash
# LCTL Tracing Hook - Pre Tool Use
# Records Task tool invocations as STEP_START events

if [ "$CLAUDE_TOOL_NAME" = "Task" ]; then
    python3 -c "
import os
import sys
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()
    tracer.on_task_start(
        agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
        description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
        prompt=os.environ.get('CLAUDE_TOOL_INPUT_prompt', ''),
        model=os.environ.get('CLAUDE_TOOL_INPUT_model'),
    )
except Exception as e:
    # Don't break Claude Code if tracing fails
    print(f'LCTL trace warning: {e}', file=sys.stderr)
"
fi
'''

    pre_hook_path = hooks_dir / "PreToolUse.sh"
    pre_hook_path.write_text(pre_tool_hook)
    pre_hook_path.chmod(0o755)
    hooks["PreToolUse"] = str(pre_hook_path)

    # PostToolUse hook for Task tool
    post_tool_hook = '''#!/bin/bash
# LCTL Tracing Hook - Post Tool Use
# Records Task tool completions as STEP_END events

if [ "$CLAUDE_TOOL_NAME" = "Task" ]; then
    python3 -c "
import os
import sys
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()

    # Check for error in result
    result = os.environ.get('CLAUDE_TOOL_RESULT', '')
    success = 'error' not in result.lower()[:100]

    tracer.on_task_complete(
        agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
        description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
        result=result[:2000],
        success=success,
    )
except Exception as e:
    print(f'LCTL trace warning: {e}', file=sys.stderr)
"
fi

# Also trace other tool calls
if [ "$CLAUDE_TOOL_NAME" != "Task" ]; then
    python3 -c "
import os
import sys
import json
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()

    # Get tool input (simplified - full parsing would need more logic)
    tool_name = os.environ.get('CLAUDE_TOOL_NAME', 'unknown')

    # Skip high-frequency read-only tools to reduce noise
    if tool_name not in ('Read', 'Glob', 'Grep', 'LS'):
        tracer.on_tool_call(
            tool_name=tool_name,
            input_data={'raw': os.environ.get('CLAUDE_TOOL_INPUT', '')[:500]},
            output_data={'raw': os.environ.get('CLAUDE_TOOL_RESULT', '')[:500]},
        )
except Exception as e:
    pass  # Silent fail for non-Task tools
"
fi
'''

    post_hook_path = hooks_dir / "PostToolUse.sh"
    post_hook_path.write_text(post_tool_hook)
    post_hook_path.chmod(0o755)
    hooks["PostToolUse"] = str(post_hook_path)

    # Stop hook to export final trace
    stop_hook = '''#!/bin/bash
# LCTL Tracing Hook - Stop
# Exports final trace when Claude Code session ends

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer
    from pathlib import Path

    state_file = Path('.claude/traces/.lctl-state.json')
    if state_file.exists():
        tracer = LCTLClaudeCodeTracer.get_or_create()
        path = tracer.export()
        summary = tracer.get_summary()

        print(f'LCTL Trace exported: {path}')
        print(f'  Events: {summary[\"event_count\"]}')
        print(f'  Agents: {list(summary[\"agent_stats\"].keys())}')
        print(f'  Facts: {summary[\"fact_count\"]}')

        # Clean up state file
        state_file.unlink()
except Exception as e:
    print(f'LCTL export warning: {e}', file=sys.stderr)
"
'''

    stop_hook_path = hooks_dir / "Stop.sh"
    stop_hook_path.write_text(stop_hook)
    stop_hook_path.chmod(0o755)
    hooks["Stop"] = str(stop_hook_path)

    return hooks


def is_available() -> bool:
    """Check if Claude Code tracing is available.

    Returns:
        True (always available - no external dependencies)
    """
    return True


__all__ = [
    "LCTLClaudeCodeTracer",
    "generate_hooks",
    "is_available",
]
