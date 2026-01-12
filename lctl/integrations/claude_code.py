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

    def on_todo_write(
        self,
        todos: List[Dict[str, Any]],
        previous_todos: Optional[List[Dict[str, Any]]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Record a TodoWrite tool call (task list update).

        Args:
            todos: Current todo list
            previous_todos: Previous todo list (for diff)
            agent: Agent that made the change
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        # Analyze todo changes
        completed = 0
        in_progress = 0
        pending = 0
        for todo in todos:
            status = todo.get("status", "pending")
            if status == "completed":
                completed += 1
            elif status == "in_progress":
                in_progress += 1
            else:
                pending += 1

        self.session.tool_call(
            tool="TodoWrite",
            input_data={
                "todo_count": len(todos),
                "completed": completed,
                "in_progress": in_progress,
                "pending": pending,
            },
            output_data={"todos": [t.get("content", "")[:50] for t in todos[:5]]},
            duration_ms=0,
        )

        # Add fact about task progress
        self.session.add_fact(
            fact_id=f"todo-{len(self.session.chain.events)}",
            text=f"Task list: {completed} done, {in_progress} active, {pending} pending",
            confidence=1.0,
            source=source or "claude-code",
        )

        self._save_state()

    def on_skill_invoke(
        self,
        skill_name: str,
        args: Optional[str] = None,
        result_summary: str = "",
        agent: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        """Record a Skill tool invocation.

        Args:
            skill_name: Name of the invoked skill
            args: Arguments passed to the skill
            result_summary: Summary of skill result
            agent: Agent that invoked the skill
            duration_ms: Skill execution time
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        self.session.tool_call(
            tool="Skill",
            input_data={"skill": skill_name, "args": args or ""},
            output_data={"summary": result_summary[:500]},
            duration_ms=duration_ms,
        )

        # Skills are important workflow events
        self.session.add_fact(
            fact_id=f"skill-{skill_name}-{len(self.session.chain.events)}",
            text=f"Invoked skill '{skill_name}': {result_summary[:200]}",
            confidence=0.9,
            source=source or "skill",
        )

        self._save_state()

    def on_mcp_tool_call(
        self,
        server_name: str,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        """Record an MCP (Model Context Protocol) tool call.

        Args:
            server_name: Name of the MCP server (e.g., "sql", "chroma")
            tool_name: Specific tool within the server
            input_data: Tool input parameters
            output_data: Tool output
            agent: Agent that made the call
            duration_ms: Tool execution time
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        full_tool_name = f"mcp__{server_name}__{tool_name}"

        # Track tool counts
        key = f"{source or 'main'}:{full_tool_name}"
        self.tool_counts[key] = self.tool_counts.get(key, 0) + 1

        # Truncate large data
        def truncate(data: Any, max_len: int = 300) -> Any:
            if isinstance(data, str):
                return data[:max_len] + "..." if len(data) > max_len else data
            elif isinstance(data, dict):
                return {k: truncate(v, max_len) for k, v in list(data.items())[:8]}
            return data

        self.session.tool_call(
            tool=full_tool_name,
            input_data=truncate(input_data),
            output_data=truncate(output_data or {}),
            duration_ms=duration_ms,
        )

        # MCP tools often return important data
        self.session.add_fact(
            fact_id=f"mcp-{server_name}-{len(self.session.chain.events)}",
            text=f"MCP {server_name}.{tool_name}: {str(output_data)[:150]}",
            confidence=0.85,
            source=source or f"mcp-{server_name}",
        )

        self._save_state()

    def on_git_commit(
        self,
        commit_hash: str,
        message: str,
        files_changed: int = 0,
        insertions: int = 0,
        deletions: int = 0,
        agent: Optional[str] = None,
    ) -> None:
        """Record a git commit linked to the current workflow.

        Args:
            commit_hash: Git commit hash (short or full)
            message: Commit message
            files_changed: Number of files changed
            insertions: Lines added
            deletions: Lines removed
            agent: Agent that created the commit
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        # Record as tool call
        self.session.tool_call(
            tool="GitCommit",
            input_data={"message": message[:200]},
            output_data={
                "commit": commit_hash[:12],
                "files": files_changed,
                "insertions": insertions,
                "deletions": deletions,
            },
            duration_ms=0,
        )

        # Commit facts are high confidence anchors
        self.session.add_fact(
            fact_id=f"commit-{commit_hash[:8]}",
            text=f"Git commit {commit_hash[:8]}: {message[:100]} (+{insertions}/-{deletions})",
            confidence=1.0,  # Commits are ground truth
            source=source or "git",
        )

        # Create checkpoint at commit for fast replay
        self.checkpoint(description=f"After commit {commit_hash[:8]}")

        self._save_state()

    def link_to_git_history(self, repo_path: str = ".") -> Dict[str, Any]:
        """Link workflow file changes to git history.

        Args:
            repo_path: Path to git repository

        Returns:
            Dict with git linkage info
        """
        import subprocess

        result = {"commits": [], "uncommitted_changes": []}

        try:
            # Get recent commits during workflow
            cmd = ["git", "-C", repo_path, "log", "--oneline", "-20"]
            output = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if output.returncode == 0:
                for line in output.stdout.strip().split("\n")[:10]:
                    if line:
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            result["commits"].append({
                                "hash": parts[0],
                                "message": parts[1][:80],
                            })

            # Check for uncommitted changes in tracked files
            for fc in self._file_changes:
                file_path = fc.get("file_path", "")
                cmd = ["git", "-C", repo_path, "status", "--porcelain", file_path]
                output = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if output.stdout.strip():
                    result["uncommitted_changes"].append({
                        "file": file_path,
                        "status": output.stdout.strip()[:50],
                    })

        except Exception:
            pass  # Git operations are optional

        return result

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
            description: Checkpoint description (stored as fact)
        """
        # Add description as a fact before checkpoint
        if description:
            self.session.add_fact(
                fact_id=f"checkpoint-{len(self.session.chain.events)}",
                text=f"Checkpoint: {description}",
                confidence=1.0,
                source="system",
            )
        self.session.checkpoint()
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

    # PostToolUse hook for Task tool and other tracked tools
    post_tool_hook = '''#!/bin/bash
# LCTL Tracing Hook - Post Tool Use
# Records tool completions including Task, TodoWrite, Skill, MCP tools, and Git

TOOL="$CLAUDE_TOOL_NAME"

python3 -c "
import os
import sys
import json
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()
    tool_name = os.environ.get('CLAUDE_TOOL_NAME', 'unknown')
    result = os.environ.get('CLAUDE_TOOL_RESULT', '')

    # Task tool - agent completions
    if tool_name == 'Task':
        success = 'error' not in result.lower()[:100]
        tracer.on_task_complete(
            agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
            description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
            result=result[:2000],
            success=success,
        )

    # TodoWrite - task list updates
    elif tool_name == 'TodoWrite':
        try:
            todos_str = os.environ.get('CLAUDE_TOOL_INPUT_todos', '[]')
            todos = json.loads(todos_str) if todos_str else []
            tracer.on_todo_write(todos=todos)
        except:
            tracer.on_todo_write(todos=[])

    # Skill invocations
    elif tool_name == 'Skill':
        skill_name = os.environ.get('CLAUDE_TOOL_INPUT_skill', 'unknown')
        args = os.environ.get('CLAUDE_TOOL_INPUT_args', '')
        tracer.on_skill_invoke(
            skill_name=skill_name,
            args=args,
            result_summary=result[:500],
        )

    # MCP tool calls (mcp__server__tool pattern)
    elif tool_name.startswith('mcp__'):
        parts = tool_name.split('__')
        if len(parts) >= 3:
            server_name = parts[1]
            mcp_tool = '__'.join(parts[2:])
            try:
                input_str = os.environ.get('CLAUDE_TOOL_INPUT', '{}')
                input_data = json.loads(input_str) if input_str else {}
            except:
                input_data = {'raw': input_str[:300]}
            try:
                output_data = json.loads(result) if result else {}
            except:
                output_data = {'raw': result[:300]}
            tracer.on_mcp_tool_call(
                server_name=server_name,
                tool_name=mcp_tool,
                input_data=input_data,
                output_data=output_data,
            )

    # Bash tool - detect git commits
    elif tool_name == 'Bash':
        cmd = os.environ.get('CLAUDE_TOOL_INPUT_command', '')
        if 'git commit' in cmd and 'Successfully' not in result:
            # Try to extract commit info from result
            import re
            commit_match = re.search(r'\\[\\w+\\s+([a-f0-9]+)\\]', result)
            if commit_match:
                commit_hash = commit_match.group(1)
                # Extract message from command
                msg_match = re.search(r'-m\\s+[\"\\']([^\"\\']+)[\"\\']', cmd)
                message = msg_match.group(1) if msg_match else 'Commit'
                # Extract stats
                stats_match = re.search(r'(\\d+)\\s+file.*?(\\d+)\\s+insertion.*?(\\d+)\\s+deletion', result)
                files = int(stats_match.group(1)) if stats_match else 0
                insertions = int(stats_match.group(2)) if stats_match else 0
                deletions = int(stats_match.group(3)) if stats_match else 0
                tracer.on_git_commit(
                    commit_hash=commit_hash,
                    message=message,
                    files_changed=files,
                    insertions=insertions,
                    deletions=deletions,
                )

    # File changes - Write, Edit, MultiEdit
    elif tool_name in ('Write', 'Edit', 'MultiEdit'):
        file_path = os.environ.get('CLAUDE_TOOL_INPUT_file_path', '')
        change_type = 'create' if tool_name == 'Write' else 'edit'
        tracer.on_file_change(file_path=file_path, change_type=change_type)

    # User interaction
    elif tool_name == 'AskUserQuestion':
        question = os.environ.get('CLAUDE_TOOL_INPUT_question', '')
        tracer.on_user_interaction(question=question, response=result[:500])

    # Web fetch/search
    elif tool_name == 'WebFetch':
        url = os.environ.get('CLAUDE_TOOL_INPUT_url', '')
        prompt = os.environ.get('CLAUDE_TOOL_INPUT_prompt', '')
        tracer.on_web_fetch(url=url, prompt=prompt, result_summary=result[:500])

    elif tool_name == 'WebSearch':
        query = os.environ.get('CLAUDE_TOOL_INPUT_query', '')
        tracer.on_web_search(query=query, results_count=result.count('http'), top_result=result[:200])

    # Skip high-frequency read-only tools to reduce noise
    elif tool_name not in ('Read', 'Glob', 'Grep', 'LS', 'TaskOutput', 'KillShell'):
        tracer.on_tool_call(
            tool_name=tool_name,
            input_data={'raw': os.environ.get('CLAUDE_TOOL_INPUT', '')[:500]},
            output_data={'raw': result[:500]},
        )

except Exception as e:
    # Don't break Claude Code if tracing fails
    pass
"
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


def validate_hooks(hooks_dir: str = ".claude/hooks") -> Dict[str, Any]:
    """Validate Claude Code hook installation.

    Args:
        hooks_dir: Directory containing hooks

    Returns:
        Validation result dict with 'valid', 'hooks', and 'warnings' keys
    """
    import stat

    hooks_path = Path(hooks_dir)
    result = {
        "valid": True,
        "hooks": {},
        "warnings": [],
    }

    expected_hooks = ["PreToolUse.sh", "PostToolUse.sh", "Stop.sh"]

    for hook_name in expected_hooks:
        hook_path = hooks_path / hook_name
        hook_status = {
            "exists": hook_path.exists(),
            "executable": False,
            "contains_lctl": False,
        }

        if hook_path.exists():
            # Check if executable
            mode = hook_path.stat().st_mode
            hook_status["executable"] = bool(mode & stat.S_IXUSR)

            # Check if it references LCTL
            content = hook_path.read_text()
            hook_status["contains_lctl"] = "lctl" in content.lower() or "LCTL" in content

            if not hook_status["executable"]:
                result["valid"] = False
                result["warnings"].append(f"{hook_name} is not executable")

            if not hook_status["contains_lctl"]:
                result["warnings"].append(f"{hook_name} may not be an LCTL hook")
        else:
            result["valid"] = False

        result["hooks"][hook_name.replace(".sh", "")] = hook_status

    # Check for traces directory
    traces_dir = Path(".claude/traces")
    if not traces_dir.exists():
        result["warnings"].append("Traces directory does not exist (will be created on first trace)")

    return result


def generate_html_report(chain: Any, output_path: str) -> str:
    """Generate an HTML report for a Claude Code trace.

    Args:
        chain: LCTL Chain object
        output_path: Path to write HTML report

    Returns:
        Path to generated report
    """
    from ..core.events import ReplayEngine

    engine = ReplayEngine(chain)
    state = engine.replay_all()
    trace = engine.get_trace()
    bottlenecks = engine.find_bottlenecks()

    # Calculate metrics
    total_duration = state.metrics.get("total_duration_ms", 0)
    total_tokens_in = state.metrics.get("total_tokens_in", 0)
    total_tokens_out = state.metrics.get("total_tokens_out", 0)

    # Estimate cost (Claude pricing)
    cost_estimate = (total_tokens_in / 1_000_000) * 3 + (total_tokens_out / 1_000_000) * 15

    # Build agent timeline data
    agents = {}
    for step in trace:
        agent = step["agent"]
        if agent not in agents:
            agents[agent] = {"steps": 0, "duration_ms": 0, "tokens": 0}
        agents[agent]["steps"] += 1
        agents[agent]["duration_ms"] += step.get("duration_ms", 0)

    # Build events list for timeline
    events_data = []
    for event in chain.events:
        events_data.append({
            "seq": event.seq,
            "type": event.type.value,
            "agent": event.agent or "system",
            "timestamp": event.timestamp.isoformat() if event.timestamp else "",
            "data": str(event.data)[:200],
        })

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCTL Report - {chain.id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .subtitle {{ opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h3 {{ color: #667eea; margin-bottom: 15px; font-size: 1.1em; }}
        .metric {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .timeline {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
        .event {{ display: flex; padding: 10px; border-left: 3px solid #667eea; margin-left: 20px; margin-bottom: 10px; background: #f9f9f9; border-radius: 0 5px 5px 0; }}
        .event-seq {{ width: 40px; font-weight: bold; color: #667eea; }}
        .event-type {{ width: 120px; font-size: 0.85em; padding: 2px 8px; border-radius: 3px; background: #e0e7ff; color: #4338ca; }}
        .event-type.step_start {{ background: #dcfce7; color: #166534; }}
        .event-type.step_end {{ background: #fef3c7; color: #92400e; }}
        .event-type.fact_added {{ background: #dbeafe; color: #1e40af; }}
        .event-type.error {{ background: #fee2e2; color: #991b1b; }}
        .event-agent {{ width: 120px; font-weight: 500; }}
        .event-data {{ flex: 1; color: #666; font-size: 0.9em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .agents {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .agent-chip {{ padding: 8px 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 20px; font-size: 0.9em; }}
        .bottleneck {{ display: flex; align-items: center; margin-bottom: 10px; }}
        .bottleneck-bar {{ height: 20px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 3px; margin-right: 10px; }}
        .bottleneck-label {{ min-width: 100px; }}
        .facts {{ max-height: 300px; overflow-y: auto; }}
        .fact {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .fact-id {{ font-weight: bold; color: #667eea; }}
        .fact-conf {{ float: right; padding: 2px 8px; background: #e0e7ff; border-radius: 10px; font-size: 0.8em; }}
        footer {{ text-align: center; padding: 20px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LCTL Workflow Report</h1>
            <p class="subtitle">Chain: {chain.id} | Version: {chain.version}</p>
        </header>

        <div class="grid">
            <div class="card">
                <h3>Events</h3>
                <div class="metric">{len(chain.events)}</div>
                <div class="metric-label">Total events recorded</div>
            </div>
            <div class="card">
                <h3>Duration</h3>
                <div class="metric">{total_duration / 1000:.1f}s</div>
                <div class="metric-label">Total execution time</div>
            </div>
            <div class="card">
                <h3>Tokens</h3>
                <div class="metric">{(total_tokens_in + total_tokens_out):,}</div>
                <div class="metric-label">In: {total_tokens_in:,} | Out: {total_tokens_out:,}</div>
            </div>
            <div class="card">
                <h3>Est. Cost</h3>
                <div class="metric">${cost_estimate:.4f}</div>
                <div class="metric-label">Based on Claude pricing</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Agents ({len(agents)})</h3>
                <div class="agents">
                    {"".join(f'<span class="agent-chip">{a} ({s["steps"]} steps)</span>' for a, s in agents.items())}
                </div>
            </div>
            <div class="card">
                <h3>Facts ({len(state.facts)})</h3>
                <div class="facts">
                    {"".join(f'<div class="fact"><span class="fact-id">{fid}</span><span class="fact-conf">{f.get("confidence", 1.0):.0%}</span><br>{f.get("text", "")[:100]}</div>' for fid, f in list(state.facts.items())[:10])}
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Bottleneck Analysis</h3>
            {"".join(f'<div class="bottleneck"><span class="bottleneck-label">{b["agent"]}</span><div class="bottleneck-bar" style="width: {b.get("percentage", 0)}%"></div>{b.get("percentage", 0):.1f}% ({b["duration_ms"]}ms)</div>' for b in bottlenecks[:5])}
        </div>

        <div class="timeline card">
            <h3>Event Timeline (first 50)</h3>
            {"".join(f'<div class="event"><span class="event-seq">{e["seq"]}</span><span class="event-type {e["type"]}">{e["type"]}</span><span class="event-agent">{e["agent"]}</span><span class="event-data">{e["data"]}</span></div>' for e in events_data[:50])}
        </div>

        <footer>
            <p>Generated by LCTL (LLM Context Transfer Language) | <a href="https://github.com/kimasplund/LLM-Context-Transfer-Language">GitHub</a></p>
        </footer>
    </div>
</body>
</html>'''

    Path(output_path).write_text(html)
    return output_path


def get_session_metadata() -> Dict[str, Any]:
    """Get metadata about the current session environment.

    Returns:
        Dict with working_dir, git_branch, git_commit, project_name, etc.
    """
    import subprocess

    metadata = {
        "working_dir": str(Path.cwd()),
        "project_name": Path.cwd().name,
        "git_branch": None,
        "git_commit": None,
        "git_dirty": False,
        "python_version": None,
        "timestamp": datetime.now().isoformat(),
    }

    # Get Python version
    import sys
    metadata["python_version"] = sys.version.split()[0]

    # Get git info
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            metadata["git_branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            metadata["git_commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            metadata["git_dirty"] = bool(result.stdout.strip())
    except Exception:
        pass  # Git not available

    return metadata


# Model pricing for cost estimation (per 1M tokens)
# Source: https://platform.claude.com/docs/en/about-claude/pricing
MODEL_PRICING = {
    # Claude 4.5 series
    "claude-opus-4-5-20251101": {"input": 5.0, "output": 25.0},
    "claude-opus-4.5": {"input": 5.0, "output": 25.0},
    "opus-4.5": {"input": 5.0, "output": 25.0},
    "opus": {"input": 5.0, "output": 25.0},  # Default opus to latest
    "claude-sonnet-4-5-20241022": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
    "sonnet-4.5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20241022": {"input": 1.0, "output": 5.0},
    "claude-haiku-4.5": {"input": 1.0, "output": 5.0},
    "haiku-4.5": {"input": 1.0, "output": 5.0},
    # Claude 4.1 series
    "claude-opus-4-1-20250414": {"input": 15.0, "output": 75.0},
    "claude-opus-4.1": {"input": 15.0, "output": 75.0},
    "opus-4.1": {"input": 15.0, "output": 75.0},
    # Claude 4 series
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "sonnet-4": {"input": 3.0, "output": 15.0},
    "sonnet": {"input": 3.0, "output": 15.0},  # Default sonnet to latest
    # Claude 3.7 series (deprecated but still used)
    "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
    "claude-sonnet-3.7": {"input": 3.0, "output": 15.0},
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.0},
    # Claude 3 series (some deprecated)
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "haiku": {"input": 0.25, "output": 1.25},  # Default haiku to cheapest
    # Default for unknown models (assume Sonnet pricing)
    "default": {"input": 3.0, "output": 15.0},
}


def estimate_cost(
    tokens_in: int,
    tokens_out: int,
    model: str = "default"
) -> Dict[str, float]:
    """Estimate cost based on token usage.

    Args:
        tokens_in: Input tokens
        tokens_out: Output tokens
        model: Model name for pricing

    Returns:
        Dict with input_cost, output_cost, total_cost
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    input_cost = (tokens_in / 1_000_000) * pricing["input"]
    output_cost = (tokens_out / 1_000_000) * pricing["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model,
        "pricing": pricing,
    }


__all__ = [
    "LCTLClaudeCodeTracer",
    "generate_hooks",
    "validate_hooks",
    "generate_html_report",
    "get_session_metadata",
    "estimate_cost",
    "MODEL_PRICING",
    "is_available",
]
