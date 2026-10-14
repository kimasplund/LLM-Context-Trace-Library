"""Claude Code integration for LCTL.

This module provides tracing for Claude Code's multi-agent workflows using
the Task tool. It captures agent spawning, tool calls, and agent completions
as LCTL events for debugging and analysis.

Usage with Claude Code hooks:

    lctl claude init        # writes .claude/hooks/{PreToolUse,PostToolUse,Stop}.sh
    lctl claude validate    # checks they are installed and executable

Claude Code passes each hook its payload as JSON on stdin (``tool_name``,
``tool_input``, ...). The generated scripts (see ``generate_hooks``) capture
stdin, base64-encode it into ``HOOK_INPUT_B64``, and run an embedded Python
snippet that decodes the payload and calls the matching
``LCTLClaudeCodeTracer`` method.

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

import fcntl
import html
import json
import logging
import os
import shlex
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.session import LCTLSession

# Logger for debugging
logger = logging.getLogger("lctl.claude_code")

# Thread-safe singleton support
_tracer_lock = threading.Lock()
HOOK_LOCK_TIMEOUT_SECONDS = 10.0
_hook_lock_file: Any = None  # held until the hook process exits


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a same-directory temp file and ``os.replace``."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def acquire_hook_lock(traces_dir: str | Path = ".claude/traces") -> None:
    """Take the exclusive trace lock for the rest of this hook process.

    Hooks run concurrently (parallel subagents); every read-modify-write of the state and
    chain files must happen under this lock. It is released when the process exits. If it
    cannot be acquired within ``LCTL_HOOK_LOCK_TIMEOUT`` seconds (default 10) the hook logs
    the problem and exits 0 without writing, so Claude Code is never blocked.
    """
    global _hook_lock_file
    try:
        timeout = float(os.environ.get("LCTL_HOOK_LOCK_TIMEOUT", HOOK_LOCK_TIMEOUT_SECONDS))
    except ValueError:
        timeout = HOOK_LOCK_TIMEOUT_SECONDS
    directory = Path(traces_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_file = open(directory / ".lctl.lock", "a")  # noqa: SIM115  # held until process exit
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                lock_file.close()
                try:
                    with open("/tmp/lctl-hook-error.log", "a") as log:  # noqa: S108
                        log.write(
                            f"lock timeout after {timeout}s: {directory / '.lctl.lock'}\n---\n"
                        )
                except OSError:
                    pass
                sys.exit(0)
            time.sleep(0.01)
    _hook_lock_file = lock_file


_tracer_instance: LCTLClaudeCodeTracer | None = None
_tracer_file: Path | None = None


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
        chain_id: str | None = None,
        session: LCTLSession | None = None,
        output_dir: str | None = None,
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
        self.agent_stack: list[dict[str, Any]] = []
        self.tool_counts: dict[str, int] = {}
        self._start_times: dict[str, float] = {}
        self._agent_ids: dict[str, str] = {}  # agent_type -> last agent_id
        self._parallel_group: str | None = None  # Current parallel execution group
        self._file_changes: list[dict[str, Any]] = []  # Track file modifications

    @classmethod
    def get_or_create(
        cls,
        chain_id: str | None = None,
        state_file: str | None = None,
    ) -> LCTLClaudeCodeTracer:
        """Get existing tracer instance or create new one.

        This is useful for hook scripts that need to maintain state
        across multiple invocations. Thread-safe via lock.

        Args:
            chain_id: Chain ID for new tracer
            state_file: File to persist tracer state

        Returns:
            Tracer instance
        """
        global _tracer_instance, _tracer_file

        with _tracer_lock:
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
                        _tracer_instance._agent_ids = state.get("agent_ids", {})
                        _tracer_instance._parallel_group = state.get("parallel_group")
                        _tracer_instance._file_changes = state.get("file_changes", [])
                        _tracer_file = state_path
                        return _tracer_instance
                except Exception as e:
                    logger.debug(f"State restoration failed: {e}")
                    # Fall through to create new

            # Create new tracer
            _tracer_instance = cls(chain_id=chain_id)
            _tracer_file = state_path
            _tracer_instance._save_state()
            return _tracer_instance

    def _save_state(self) -> None:
        """Persist tracer state for hook continuity.

        Written via temp file + ``os.replace`` so readers never see a torn file.
        Hook scripts serialise whole read-modify-write cycles with ``acquire_hook_lock``.
        """
        if _tracer_file is None:
            return

        try:
            _tracer_file.parent.mkdir(parents=True, exist_ok=True)

            # Save chain
            chain_path = self.output_dir / f"{self.session.chain.id}.lctl.json"
            chain_path.parent.mkdir(parents=True, exist_ok=True)
            self.session.export(str(chain_path))

            # Save state with file locking
            state = {
                "chain_path": str(chain_path),
                "agent_stack": self.agent_stack,
                "tool_counts": self.tool_counts,
                "start_times": self._start_times,
                "agent_ids": self._agent_ids,
                "parallel_group": self._parallel_group,
                "file_changes": self._file_changes,
            }
            _atomic_write_text(_tracer_file, json.dumps(state))
        except Exception as e:
            logger.debug(f"State save failed: {e}")

    def on_task_start(
        self,
        agent_type: str,
        description: str,
        prompt: str = "",
        model: str | None = None,
        run_in_background: bool = False,
        resume_agent_id: str | None = None,
        parallel_group: str | None = None,
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
        error_message: str | None = None,
        agent_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        launched: bool = False,
        tokens_detail: dict[str, int] | None = None,
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
            launched: The call only started a background agent (outcome ``"launched"``,
                no result fact); real completion arrives later
            tokens_detail: Raw input-token components (plain, cache creation, cache read)
        """
        duration_ms = 0
        if agent_type in self._start_times:
            duration_ms = int((time.time() - self._start_times[agent_type]) * 1000)
            del self._start_times[agent_type]

        # Store agent_id for potential resume tracking
        if agent_id:
            self._agent_ids[agent_type] = agent_id

        # Pop from agent stack - search for matching agent (handles out-of-order completion)
        for i in range(len(self.agent_stack) - 1, -1, -1):
            if self.agent_stack[i]["agent_type"] == agent_type:
                self.agent_stack.pop(i)
                break

        outcome = "launched" if launched else ("success" if success else "failure")

        extra: dict[str, Any] = {}
        if agent_id:
            extra["agent_id"] = agent_id
        if tokens_detail:
            extra["tokens_detail"] = tokens_detail
        self.session.step_end(
            agent=agent_type,
            outcome=outcome,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            extra=extra or None,
        )

        # Add fact about result (launch metadata is not a result)
        if result and not launched:
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
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None = None,
        duration_ms: int = 0,
        agent: str | None = None,
    ) -> None:
        """Record a tool call within an agent.

        Args:
            tool_name: Name of the tool (Bash, Read, Write, etc.)
            input_data: Tool input parameters
            output_data: Tool output
            duration_ms: Tool execution time
            agent: Agent that made the call (inferred from stack if not provided)
        """
        # Infer agent from stack (default to claude-code for main session)
        if agent is None:
            agent = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        # Track tool counts
        key = f"{agent}:{tool_name}"
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
            agent=agent,
        )

        self._save_state()

    def on_fact_discovered(
        self,
        fact_id: str,
        text: str,
        confidence: float = 0.8,
        agent: str | None = None,
    ) -> None:
        """Record a fact discovered by an agent.

        Args:
            fact_id: Unique fact identifier
            text: Fact content
            confidence: Confidence score (0.0-1.0)
            agent: Agent that discovered the fact
        """
        source = agent
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        self.session.add_fact(
            fact_id=fact_id,
            text=text,
            confidence=confidence,
            source=source,
        )
        self._save_state()

    def on_fact_updated(
        self,
        fact_id: str,
        confidence: float | None = None,
        text: str | None = None,
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
        options: list[str] | None = None,
        agent: str | None = None,
    ) -> None:
        """Record a human-in-the-loop interaction (AskUserQuestion).

        Args:
            question: The question asked to user
            response: User's response
            options: Available options presented
            agent: Agent that asked (inferred from stack if not provided)
        """
        source = agent
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        # Record as tool call
        self.session.tool_call(
            tool="AskUserQuestion",
            input_data={
                "question": question[:500],
                "options": options or [],
            },
            output_data={"response": response[:500]},
            duration_ms=0,  # User think time not tracked
            agent=source,
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
        agent: str | None = None,
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
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

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
            source=source,
        )

        self._save_state()

    def on_web_fetch(
        self,
        url: str,
        prompt: str,
        result_summary: str,
        agent: str | None = None,
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
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        self.session.tool_call(
            tool="WebFetch",
            input_data={"url": url, "prompt": prompt[:200]},
            output_data={"summary": result_summary[:500]},
            duration_ms=duration_ms,
            agent=source,
        )

        # Add fact about external data
        self.session.add_fact(
            fact_id=f"web-{len(self.session.chain.events)}",
            text=f"Fetched from {url}: {result_summary[:200]}",
            confidence=0.7,  # External data has lower confidence
            source=source,
        )

        self._save_state()

    def on_web_search(
        self,
        query: str,
        results_count: int,
        top_result: str = "",
        agent: str | None = None,
    ) -> None:
        """Record a web search operation.

        Args:
            query: Search query
            results_count: Number of results returned
            top_result: Summary of top result
            agent: Agent that made the search
        """
        source = agent
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        self.session.tool_call(
            tool="WebSearch",
            input_data={"query": query},
            output_data={"results_count": results_count, "top": top_result[:200]},
            duration_ms=0,
            agent=source,
        )

        self.session.add_fact(
            fact_id=f"search-{len(self.session.chain.events)}",
            text=f"Searched '{query}': {results_count} results. Top: {top_result[:150]}",
            confidence=0.75,
            source=source,
        )

        self._save_state()

    def start_parallel_group(self, group_id: str | None = None) -> str:
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
        todos: list[dict[str, Any]],
        previous_todos: list[dict[str, Any]] | None = None,
        agent: str | None = None,
    ) -> None:
        """Record a TodoWrite tool call (task list update).

        Args:
            todos: Current todo list
            previous_todos: Previous todo list (for diff)
            agent: Agent that made the change
        """
        source = agent
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

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
            agent=source,
        )

        # Add fact about task progress
        self.session.add_fact(
            fact_id=f"todo-{len(self.session.chain.events)}",
            text=f"Task list: {completed} done, {in_progress} active, {pending} pending",
            confidence=1.0,
            source=source,
        )

        self._save_state()

    def on_skill_invoke(
        self,
        skill_name: str,
        args: str | None = None,
        result_summary: str = "",
        agent: str | None = None,
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
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        self.session.tool_call(
            tool="Skill",
            input_data={"skill": skill_name, "args": args or ""},
            output_data={"summary": result_summary[:500]},
            duration_ms=duration_ms,
            agent=source,
        )

        # Skills are important workflow events
        self.session.add_fact(
            fact_id=f"skill-{skill_name}-{len(self.session.chain.events)}",
            text=f"Invoked skill '{skill_name}': {result_summary[:200]}",
            confidence=0.9,
            source=source,
        )

        self._save_state()

    def on_mcp_tool_call(
        self,
        server_name: str,
        tool_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None = None,
        agent: str | None = None,
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
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

        full_tool_name = f"mcp__{server_name}__{tool_name}"

        # Track tool counts
        key = f"{source}:{full_tool_name}"
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
            agent=source,
        )

        # MCP tools often return important data
        self.session.add_fact(
            fact_id=f"mcp-{server_name}-{len(self.session.chain.events)}",
            text=f"MCP {server_name}.{tool_name}: {str(output_data)[:150]}",
            confidence=0.85,
            source=source,
        )

        self._save_state()

    def on_git_commit(
        self,
        commit_hash: str,
        message: str,
        files_changed: int = 0,
        insertions: int = 0,
        deletions: int = 0,
        agent: str | None = None,
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
        if source is None:
            source = self.agent_stack[-1]["agent_type"] if self.agent_stack else "claude-code"

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
            agent=source,
        )

        # Commit facts are high confidence anchors
        self.session.add_fact(
            fact_id=f"commit-{commit_hash[:8]}",
            text=f"Git commit {commit_hash[:8]}: {message[:100]} (+{insertions}/-{deletions})",
            confidence=1.0,  # Commits are ground truth
            source=source,
        )

        # Create checkpoint at commit for fast replay
        self.checkpoint(description=f"After commit {commit_hash[:8]}")

        self._save_state()

    def link_to_git_history(self, repo_path: str = ".") -> dict[str, Any]:
        """Link workflow file changes to git history.

        Args:
            repo_path: Path to git repository

        Returns:
            Dict with git linkage info
        """
        import subprocess

        result: dict[str, Any] = {"commits": [], "uncommitted_changes": []}

        try:
            # Get recent commits during workflow
            cmd = ["git", "-C", repo_path, "log", "--oneline", "-20"]
            output = subprocess.run(cmd, capture_output=True, text=True, timeout=5)  # noqa: S603  # list argv, no shell; paths passed after --
            if output.returncode == 0:
                for line in output.stdout.strip().split("\n")[:10]:
                    if line:
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            result["commits"].append(
                                {
                                    "hash": parts[0],
                                    "message": parts[1][:80],
                                }
                            )

            # Check for uncommitted changes in tracked files
            for fc in self._file_changes:
                file_path = fc.get("file_path", "")
                cmd = ["git", "-C", repo_path, "status", "--porcelain", "--", file_path]
                output = subprocess.run(cmd, capture_output=True, text=True, timeout=2)  # noqa: S603  # list argv, no shell; paths passed after --
                if output.stdout.strip():
                    result["uncommitted_changes"].append(
                        {
                            "file": file_path,
                            "status": output.stdout.strip()[:50],
                        }
                    )

        except Exception:  # noqa: S110  # tracing must never break the host application
            pass  # Git operations are optional

        return result

    def get_file_changes(self) -> list[dict[str, Any]]:
        """Get list of file changes made during the workflow.

        Returns:
            List of file change records
        """
        return self._file_changes.copy()

    def get_agent_ids(self) -> dict[str, str]:
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

    def export(self, path: str | None = None) -> str:
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

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the traced workflow.

        Returns:
            Summary dict with agent stats, tool counts, facts, etc.
        """
        from ..core.events import ReplayEngine

        engine = ReplayEngine(self.session.chain)
        state = engine.replay_all()
        trace = engine.get_trace()

        # Agent stats
        agent_stats: dict[str, dict[str, Any]] = {}
        for step in trace:
            agent = step["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {"steps": 0, "duration_ms": 0, "tokens": 0}
            agent_stats[agent]["steps"] += 1
            agent_stats[agent]["duration_ms"] += step.get("duration_ms", 0)
            agent_stats[agent]["tokens"] += step.get("tokens_in", 0) + step.get("tokens_out", 0)

        # Count user interactions
        user_interactions = sum(
            1
            for e in self.session.chain.events
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

    def cleanup_stale_start_times(self, max_age_seconds: float = 3600.0) -> int:
        """Remove stale start times that are older than max_age_seconds.

        This is useful for cleaning up orphaned agent starts that never completed,
        which can happen if an agent crashes or is interrupted.

        Args:
            max_age_seconds: Maximum age in seconds before a start time is
                considered stale. Defaults to 1 hour.

        Returns:
            Number of stale entries removed.
        """
        current_time = time.time()
        stale_agents = [
            agent_type
            for agent_type, start_time in self._start_times.items()
            if (current_time - start_time) > max_age_seconds
        ]

        for agent_type in stale_agents:
            del self._start_times[agent_type]
            logger.debug(f"Cleaned up stale start time for agent: {agent_type}")

        if stale_agents:
            self._save_state()

        return len(stale_agents)

    def reset(self) -> None:
        """Reset the tracer for a new workflow."""
        global _tracer_instance, _tracer_file

        self.session = LCTLSession(
            chain_id=f"claude-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.agent_stack = []
        self.tool_counts = {}
        self._start_times = {}
        self._agent_ids = {}
        self._parallel_group = None
        self._file_changes = []

        # Clear state file
        if _tracer_file and _tracer_file.exists():
            try:
                _tracer_file.unlink()
            except Exception as e:
                logger.debug(f"Failed to remove state file: {e}")

        _tracer_instance = None


HOOK_MARKER = "LCTL Tracing Hook"  # first-line marker of every generated script

HOOK_EVENTS = ("PreToolUse", "PostToolUse", "Stop", "SessionStart", "SessionEnd")

# Settings matcher per event; the subagent tool is "Agent" in Claude Code 2.1.x, "Task" before.
HOOK_MATCHERS = {
    "PreToolUse": "Task|Agent",
    "PostToolUse": "*",
    "Stop": "",
    "SessionStart": "",
    "SessionEnd": "",
}

_HOOK_PYTHON_PRELUDE = """#!/bin/bash
# LCTL Tracing Hook - {title}
# {summary}

# Find working Python (prefer venv, fallback to system)
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
elif command -v /usr/bin/python3 &> /dev/null; then
    PYTHON="/usr/bin/python3"
else
    PYTHON="python3"
fi

# Capture stdin to temp file and encode as base64
TMPFILE=$(mktemp)
cat > "$TMPFILE"
INPUT_B64=$(base64 -w0 "$TMPFILE")
rm -f "$TMPFILE"

# Run Python with base64-encoded input (quoted heredoc: bash never touches the code)
HOOK_INPUT_B64="$INPUT_B64" $PYTHON << 'PYTHON_EOF'
import base64
import json
import os
import sys
sys.path.insert(0, '.')

try:
    from pathlib import Path

    from lctl.integrations.claude_code import LCTLClaudeCodeTracer, acquire_hook_lock

    acquire_hook_lock()

    hook_input = {{}}
    input_b64 = os.environ.get('HOOK_INPUT_B64', '')
    if input_b64:
        hook_input = json.loads(base64.b64decode(input_b64).decode('utf-8'))
    session_id = str(hook_input.get('session_id') or '')
    state_file = Path('.claude/traces/.lctl-state.json')

    def report(tracer, label):
        path = tracer.export()
        summary = tracer.get_summary()
        print(f'LCTL Trace {{label}}: {{path}}')
        print(f'  Events: {{summary["event_count"]}}')
        print(f'  Agents: {{list(summary["agent_stats"].keys())}}')
        print(f'  Facts: {{summary["fact_count"]}}')

{body}
except Exception as e:
    # Don't break Claude Code if tracing fails
    import traceback
    with open('/tmp/lctl-hook-error.log', 'a') as f:
        f.write(f'{title}: {{e}}\\n{{traceback.format_exc()}}\\n---\\n')
PYTHON_EOF
"""

_STOP_BODY = """    # Stop fires after every turn: snapshot only, never delete state.
    if state_file.exists():
        report(LCTLClaudeCodeTracer.get_or_create(), 'snapshot')
"""

_SESSION_END_BODY = """    # The session is over: write the final trace and drop the state.
    if state_file.exists():
        report(LCTLClaudeCodeTracer.get_or_create(), 'exported')
        state_file.unlink()
"""

_SESSION_START_BODY = """    # Compaction restarts the hook inside the same session: keep the live trace.
    if hook_input.get('source') == 'compact':
        sys.exit(0)
    # Never append to another session's trace: set any leftover state aside as-is
    # (its chain file is already on disk) and start a fresh chain.
    from datetime import datetime
    now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    safe_sid = ''.join(c for c in session_id if c.isalnum() or c in '-_')[:12]
    if state_file.exists():
        import uuid
        archive = f'.lctl-state-{now}-{safe_sid}-{uuid.uuid4().hex[:8]}.json.archived'
        state_file.rename(state_file.with_name(archive))
    chain_id = __CHAIN_PREFIX__ + '-' + now
    if safe_sid:
        chain_id += '-' + safe_sid
    LCTLClaudeCodeTracer.get_or_create(chain_id=chain_id)
"""


def _session_hook_script(title: str, summary: str, body: str) -> str:
    return _HOOK_PYTHON_PRELUDE.format(title=title, summary=summary, body=body.rstrip("\n"))


def _is_lctl_script(path: Path) -> bool:
    try:
        return HOOK_MARKER in path.read_text()
    except (OSError, UnicodeDecodeError):
        return False


def foreign_hook_scripts(output_dir: str = ".claude/hooks") -> list[Path]:
    """Scripts at LCTL's target paths that are not LCTL's (a user's own hooks)."""
    return [
        path
        for event in HOOK_EVENTS
        if (path := Path(output_dir) / f"{event}.sh").exists() and not _is_lctl_script(path)
    ]


def _write_hook(path: Path, content: str, overwrite: bool) -> None:
    """Write a hook script; an existing one is kept unless ``overwrite``.

    A script that is not LCTL's is copied to ``<script>.bak`` before it is replaced.
    """
    if overwrite or not path.exists():
        if path.exists() and not _is_lctl_script(path):
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        path.write_text(content)
    path.chmod(0o755)


def generate_hooks(
    output_dir: str = ".claude/hooks",
    chain_id: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    """Generate Claude Code hook scripts for LCTL tracing.

    Args:
        output_dir: Directory to write hook scripts
        chain_id: Prefix for the chain IDs SessionStart creates (default
            ``claude-code``); giving one also regenerates SessionStart.sh
        overwrite: Replace scripts that already exist (a non-LCTL script is backed
            up to ``<script>.bak`` first). When False they are left untouched and
            only missing scripts are written.

    Returns:
        Dict mapping hook name to file path
    """
    hooks_dir = Path(output_dir)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hooks = {}

    # PreToolUse hook for Task tool
    pre_tool_hook = """#!/bin/bash
# LCTL Tracing Hook - Pre Tool Use
# Records Task/Agent (subagent) tool invocations as STEP_START events

# Find working Python (prefer venv, fallback to system)
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
elif command -v /usr/bin/python3 &> /dev/null; then
    PYTHON="/usr/bin/python3"
else
    PYTHON="python3"
fi

# Capture stdin to temp file and encode as base64
TMPFILE=$(mktemp)
cat > "$TMPFILE"
INPUT_B64=$(base64 -w0 "$TMPFILE")
rm -f "$TMPFILE"

# Run Python with base64-encoded input
HOOK_INPUT_B64="$INPUT_B64" $PYTHON << 'PYTHON_EOF'
import os
import sys
import json
import base64
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer, acquire_hook_lock

    # Decode JSON from base64
    input_b64 = os.environ.get('HOOK_INPUT_B64', '')
    if not input_b64:
        sys.exit(0)

    hook_input = json.loads(base64.b64decode(input_b64).decode('utf-8'))
    tool_name = hook_input.get('tool_name', '')

    # Only trace subagent spawns ("Agent" in Claude Code 2.1.x, "Task" before)
    if tool_name in ('Task', 'Agent'):
        acquire_hook_lock()
        tracer = LCTLClaudeCodeTracer.get_or_create()
        tool_input = hook_input.get('tool_input', {})

        tracer.on_task_start(
            agent_type=tool_input.get('subagent_type', 'unknown'),
            description=tool_input.get('description', ''),
            prompt=tool_input.get('prompt', ''),
            model=tool_input.get('model'),
            run_in_background=tool_input.get('run_in_background', False),
            resume_agent_id=tool_input.get('resume'),
        )

except Exception as e:
    # Don't break Claude Code if tracing fails
    import traceback
    with open('/tmp/lctl-hook-error.log', 'a') as f:
        f.write(f'PreToolUse: {e}\\n{traceback.format_exc()}\\n---\\n')
PYTHON_EOF
"""

    pre_hook_path = hooks_dir / "PreToolUse.sh"
    _write_hook(pre_hook_path, pre_tool_hook, overwrite)
    hooks["PreToolUse"] = str(pre_hook_path)

    # PostToolUse hook for Task tool and other tracked tools
    post_tool_hook = """#!/bin/bash
# LCTL Tracing Hook - Post Tool Use
# Records tool completions including Task, TodoWrite, Skill, MCP tools, and Git

# Find working Python (prefer venv, fallback to system)
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
elif command -v /usr/bin/python3 &> /dev/null; then
    PYTHON="/usr/bin/python3"
else
    PYTHON="python3"
fi

# Capture stdin to temp file and encode as base64
TMPFILE=$(mktemp)
cat > "$TMPFILE"
INPUT_B64=$(base64 -w0 "$TMPFILE")
rm -f "$TMPFILE"

# Run Python with base64-encoded input
HOOK_INPUT_B64="$INPUT_B64" $PYTHON << 'PYTHON_EOF'
import os
import sys
import json
import base64
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer, acquire_hook_lock

    # Decode JSON from base64
    input_b64 = os.environ.get('HOOK_INPUT_B64', '')
    if not input_b64:
        sys.exit(0)

    hook_input = json.loads(base64.b64decode(input_b64).decode('utf-8'))

    acquire_hook_lock()
    tracer = LCTLClaudeCodeTracer.get_or_create()
    tool_name = hook_input.get('tool_name', 'unknown')
    tool_input = hook_input.get('tool_input', {})
    tool_response = hook_input.get('tool_response', {})

    # Convert tool_response to string for result
    if isinstance(tool_response, dict):
        result = json.dumps(tool_response)
    else:
        result = str(tool_response)

    # Task/Agent tool - subagent completions
    if tool_name in ('Task', 'Agent'):
        if not isinstance(tool_response, dict):
            tool_response = {'content': [{'type': 'text', 'text': str(tool_response)}]}
        status = tool_response.get('status', 'completed')
        # Background / remote / teammate spawns only launch the agent: not a failure.
        launched = bool(tool_response.get('isAsync')) or status in (
            'async_launched', 'remote_launched', 'teammate_spawned',
        )
        success = launched or status == 'completed'
        content = tool_response.get('content', [])
        result_text = ''
        if content and isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get('type') == 'text':
                    result_text += c.get('text', '')

        # Input tokens = plain + cache creation + cache read (cached prompts dominate the cost).
        usage = tool_response.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        tokens_detail = None
        tokens_in = 0
        if usage:
            tokens_detail = {
                k: int(usage.get(k) or 0)
                for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')
            }
            tokens_in = sum(tokens_detail.values())
        elif isinstance(tool_response.get('totalTokens'), int):
            # No breakdown: record the total as input, output unknown.
            tokens_in = tool_response['totalTokens']
        tokens_out = int(usage.get('output_tokens') or 0)

        tracer.on_task_complete(
            agent_type=tool_input.get('subagent_type', 'unknown'),
            description=tool_input.get('description', ''),
            result=result_text[:2000] if result_text else result[:2000],
            success=success,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            agent_id=tool_response.get('agentId'),
            launched=launched,
            tokens_detail=tokens_detail,
        )

    # TodoWrite - task list updates
    elif tool_name == 'TodoWrite':
        todos = tool_input.get('todos', [])
        tracer.on_todo_write(todos=todos)

    # Skill invocations
    elif tool_name == 'Skill':
        skill_name = tool_input.get('skill', 'unknown')
        args = tool_input.get('args', '')
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
            tracer.on_mcp_tool_call(
                server_name=server_name,
                tool_name=mcp_tool,
                input_data=tool_input,
                output_data=tool_response,
            )

    # Bash tool - detect git commits
    elif tool_name == 'Bash':
        cmd = tool_input.get('command', '')
        stdout = tool_response.get('stdout', '') if isinstance(tool_response, dict) else str(tool_response)
        if 'git commit' in cmd:
            import re
            commit_match = re.search(r'\\[\\w+\\s+([a-f0-9]+)\\]', stdout)
            if commit_match:
                commit_hash = commit_match.group(1)
                msg_match = re.search(r'-m\\s+[\"\\x27]([^\"\\x27]+)[\"\\x27]', cmd)
                message = msg_match.group(1) if msg_match else 'Commit'
                stats_match = re.search(r'(\\d+)\\s+file.*?(\\d+)\\s+insertion.*?(\\d+)\\s+deletion', stdout)
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
        file_path = tool_input.get('file_path', '')
        change_type = 'create' if tool_name == 'Write' else 'edit'
        tracer.on_file_change(file_path=file_path, change_type=change_type)

    # User interaction
    elif tool_name == 'AskUserQuestion':
        questions = tool_input.get('questions', [])
        question = questions[0].get('question', '') if questions else ''
        tracer.on_user_interaction(question=question, response=result[:500])

    # Web fetch/search
    elif tool_name == 'WebFetch':
        url = tool_input.get('url', '')
        prompt = tool_input.get('prompt', '')
        tracer.on_web_fetch(url=url, prompt=prompt, result_summary=result[:500])

    elif tool_name == 'WebSearch':
        query = tool_input.get('query', '')
        tracer.on_web_search(query=query, results_count=result.count('http'), top_result=result[:200])

    # Skip high-frequency read-only tools to reduce noise
    elif tool_name not in ('Read', 'Glob', 'Grep', 'LS', 'TaskOutput', 'KillShell'):
        tracer.on_tool_call(
            tool_name=tool_name,
            input_data=tool_input,
            output_data=tool_response if isinstance(tool_response, dict) else {'raw': str(tool_response)[:500]},
        )

except Exception as e:
    # Don't break Claude Code if tracing fails
    import traceback
    with open('/tmp/lctl-hook-error.log', 'a') as f:
        f.write(f'{e}\\n{traceback.format_exc()}\\n---\\n')
PYTHON_EOF
"""

    post_hook_path = hooks_dir / "PostToolUse.sh"
    _write_hook(post_hook_path, post_tool_hook, overwrite)
    hooks["PostToolUse"] = str(post_hook_path)

    session_hooks = {
        "Stop": (
            "Stop",
            "Exports a snapshot of the trace after each turn (state is kept)",
            _STOP_BODY,
        ),
        "SessionStart": (
            "Session Start",
            "Starts a fresh trace so sessions never share a chain",
            _SESSION_START_BODY,
        ),
        "SessionEnd": (
            "Session End",
            "Exports the final trace and removes the session state",
            _SESSION_END_BODY,
        ),
    }
    for hook_name, (title, summary, body) in session_hooks.items():
        hook_path = hooks_dir / f"{hook_name}.sh"
        if hook_name == "SessionStart":
            body = body.replace("__CHAIN_PREFIX__", repr(chain_id or "claude-code"))
        _write_hook(
            hook_path,
            _session_hook_script(title, summary, body),
            overwrite or (hook_name == "SessionStart" and chain_id is not None),
        )
        hooks[hook_name] = str(hook_path)

    return hooks


def _normalize_hooks_dir(hooks_dir: str) -> str:
    return hooks_dir.replace("\\", "/").rstrip("/")


def is_lctl_hook_command(
    command: Any,
    event: str,
    hooks_dir: str = ".claude/hooks",
    project_root: Path | None = None,
) -> bool:
    """True if ``command`` runs ``<hooks_dir>/<event>.sh`` (an LCTL-installed hook).

    The command is parsed shell-style, so entries written before the project root
    was quoted and current quoted ones are both recognised. With ``project_root``,
    absolute and ``$CLAUDE_PROJECT_DIR``-relative paths to the same script also match.
    """
    if not isinstance(command, str):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    relative = f"{_normalize_hooks_dir(hooks_dir)}/{event}.sh"
    if tokens[-1] == relative:
        return True
    if project_root is None:
        return False
    root = str(project_root)
    script = tokens[-1].replace("${CLAUDE_PROJECT_DIR}", root).replace("$CLAUDE_PROJECT_DIR", root)
    try:
        return (project_root / script).resolve() == (project_root / relative).resolve()
    except OSError:
        return False


def build_hook_settings(project_root: Path, hooks_dir: str = ".claude/hooks") -> dict[str, Any]:
    """The ``hooks`` section of ``.claude/settings.json`` that enables LCTL tracing."""
    hooks_dir = _normalize_hooks_dir(hooks_dir)
    root = shlex.quote(str(project_root))
    return {
        event: [
            {
                "matcher": HOOK_MATCHERS[event],
                "hooks": [
                    {
                        "type": "command",
                        "command": f"cd {root} && bash {shlex.quote(f'{hooks_dir}/{event}.sh')}",
                    }
                ],
            }
        ]
        for event in HOOK_EVENTS
    }


def _redirect_to_backup(command: str, event: str, hooks_dir: str) -> str:
    """Rewrite a command running ``<hooks_dir>/<event>.sh`` to run ``....sh.bak`` instead.

    ``.bak`` is inserted right after the script name in the original text, so the
    rest of the command (a leading ``cd``, quoting, ``$CLAUDE_PROJECT_DIR``) is kept.
    """
    name = f"{event}.sh"
    at = command.rfind(name)
    if at != -1:
        rewritten = command[: at + len(name)] + ".bak" + command[at + len(name) :]
        try:
            if shlex.split(rewritten)[-1] == shlex.split(command)[-1] + ".bak":
                return rewritten
        except (ValueError, IndexError):
            pass
    script = f"{_normalize_hooks_dir(hooks_dir)}/{event}.sh"
    return f"bash {shlex.quote(script + '.bak')}"


def merge_hook_settings(
    existing: dict[str, Any],
    project_root: Path,
    hooks_dir: str = ".claude/hooks",
    backed_up: Sequence[str] = (),
    redirected: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Return ``existing`` settings with LCTL hook entries replaced by the current ones.

    Earlier LCTL entries (commands running ``<hooks_dir>/<Event>.sh``) are removed
    and the current entry appended, so repeated calls never duplicate. Every other
    key, event and user hook is left as it was. ``existing`` is not modified.

    ``backed_up`` names events whose script was not LCTL's and has been moved to
    ``<script>.bak``: entries running that script are the user's, so they are kept and
    rewritten to run the backup. Each such rewrite is appended to ``redirected`` as
    ``(matcher, new_script_path)``.

    Raises:
        ValueError: ``hooks`` or an event's value has an unexpected shape, so that
            nothing user-written is silently dropped.
    """
    merged = dict(existing)
    raw_hooks = merged.get("hooks", {})
    if not isinstance(raw_hooks, dict):
        raise ValueError("settings 'hooks' must be an object")
    hooks_section = dict(raw_hooks)

    for event, groups in build_hook_settings(project_root, hooks_dir).items():
        kept: list[Any] = []
        existing_groups = hooks_section.get(event, [])
        if not isinstance(existing_groups, list):
            raise ValueError(f"settings hooks.{event} must be a list")
        for group in existing_groups:
            entries = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(entries, list):
                kept.append(group)
                continue
            remaining = []
            for h in entries:
                if isinstance(h, dict) and is_lctl_hook_command(
                    h.get("command"), event, hooks_dir, project_root
                ):
                    if event not in backed_up:
                        continue
                    h = {**h, "command": _redirect_to_backup(h["command"], event, hooks_dir)}
                    if redirected is not None:
                        redirected.append(
                            (
                                str(group.get("matcher", "")),
                                f"{_normalize_hooks_dir(hooks_dir)}/{event}.sh.bak",
                            )
                        )
                remaining.append(h)
            if len(remaining) == len(entries) and remaining == entries:
                kept.append(group)
            elif remaining:
                kept.append({**group, "hooks": remaining})
        hooks_section[event] = kept + groups

    merged["hooks"] = hooks_section
    return merged


def is_available() -> bool:
    """Check if Claude Code tracing is available.

    Returns:
        True (always available - no external dependencies)
    """
    return True


def _settings_reasons(hooks_dir: str, settings_path: Path) -> list[str]:
    """Settings-level reasons the install is missing or predates the current hook set."""
    hooks_dir = _normalize_hooks_dir(hooks_dir)
    try:
        settings = json.loads(settings_path.read_text())
    except (OSError, ValueError):
        return [f"{settings_path} is missing or unreadable, so no LCTL hooks are registered"]
    hooks_section = settings.get("hooks") if isinstance(settings, dict) else None
    if not isinstance(hooks_section, dict):
        hooks_section = {}

    def lctl_groups(event: str) -> list[dict[str, Any]]:
        groups = hooks_section.get(event)
        found: list[dict[str, Any]] = []
        for g in groups if isinstance(groups, list) else []:
            entries = g.get("hooks") if isinstance(g, dict) else None
            if isinstance(entries, list) and any(
                isinstance(h, dict) and is_lctl_hook_command(h.get("command"), event, hooks_dir)
                for h in entries
            ):
                found.append(g)
        return found

    reasons: list[str] = []
    if not any(lctl_groups(event) for event in HOOK_EVENTS):
        return [f"{settings_path} registers no LCTL hooks"]
    for group in lctl_groups("PreToolUse"):
        if "Agent" not in str(group.get("matcher", "")).split("|"):
            reasons.append(
                f"PreToolUse matcher {group.get('matcher')!r} does not match the Agent tool"
            )
    for event in HOOK_EVENTS:
        if not lctl_groups(event):
            reasons.append(f"{event} hook is not configured in {settings_path}")
    return reasons


def stale_hook_scripts(hooks_dir: str = ".claude/hooks") -> list[str]:
    """Reasons LCTL-owned scripts in ``hooks_dir`` predate the current generator."""
    hooks_path = Path(hooks_dir)
    checks = {
        "PreToolUse": ("'Agent'", "does not handle the Agent tool"),
        "PostToolUse": ("'Agent'", "does not handle the Agent tool"),
        "Stop": ("PYTHON_EOF", "uses the old, broken Stop script"),
    }
    reasons = []
    for event, (needle, why) in checks.items():
        path = hooks_path / f"{event}.sh"
        if path.exists() and _is_lctl_script(path) and needle not in path.read_text():
            reasons.append(f"{event}.sh {why}")
    return reasons


def validate_hooks(
    hooks_dir: str = ".claude/hooks", settings_path: str = ".claude/settings.json"
) -> dict[str, Any]:
    """Validate Claude Code hook installation.

    Args:
        hooks_dir: Directory containing hooks
        settings_path: Claude Code settings file that registers the hooks

    Returns:
        Validation result dict with 'valid', 'hooks', 'warnings', and
        'outdated' (bool) / 'reasons' (list) / 'needs_force' (bool) keys. An
        outdated or unregistered install should be refreshed with
        ``lctl claude init``, or ``lctl claude init --force`` when
        ``needs_force`` (some LCTL script itself is stale).
    """
    import stat

    hooks_path = Path(hooks_dir)
    result: dict[str, Any] = {
        "valid": True,
        "hooks": {},
        "warnings": [],
        "outdated": False,
        "reasons": [],
        "needs_force": False,
    }

    expected_hooks = [f"{event}.sh" for event in HOOK_EVENTS]

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

    reasons = _settings_reasons(hooks_dir, Path(settings_path))
    script_reasons = stale_hook_scripts(hooks_dir)
    for event in HOOK_EVENTS:
        if not (hooks_path / f"{event}.sh").exists():
            reasons.append(f"{event}.sh is missing")
    result["reasons"] = reasons + script_reasons
    result["outdated"] = bool(result["reasons"])
    result["needs_force"] = bool(script_reasons)

    # Check for traces directory
    traces_dir = Path(".claude/traces")
    if not traces_dir.exists():
        result["warnings"].append(
            "Traces directory does not exist (will be created on first trace)"
        )

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

    # Helper to escape user-controlled data for XSS prevention
    def esc(text: Any) -> str:
        return html.escape(str(text))

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
        events_data.append(
            {
                "seq": event.seq,
                "type": event.type.value,
                "agent": event.agent or "system",
                "timestamp": event.timestamp.isoformat() if event.timestamp else "",
                "data": str(event.data)[:200],
            }
        )

    # Escape chain metadata
    chain_id_esc = esc(chain.id)
    chain_version_esc = esc(chain.version)

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCTL Report - {chain_id_esc}</title>
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
            <p class="subtitle">Chain: {chain_id_esc} | Version: {chain_version_esc}</p>
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
                    {"".join(f'<span class="agent-chip">{esc(a)} ({s["steps"]} steps)</span>' for a, s in agents.items())}
                </div>
            </div>
            <div class="card">
                <h3>Facts ({len(state.facts)})</h3>
                <div class="facts">
                    {"".join(f'<div class="fact"><span class="fact-id">{esc(fid)}</span><span class="fact-conf">{f.get("confidence", 1.0):.0%}</span><br>{esc(f.get("text", "")[:100])}</div>' for fid, f in list(state.facts.items())[:10])}
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Bottleneck Analysis</h3>
            {"".join(f'<div class="bottleneck"><span class="bottleneck-label">{esc(b["agent"])}</span><div class="bottleneck-bar" style="width: {b.get("percentage", 0)}%"></div>{b.get("percentage", 0):.1f}% ({b["duration_ms"]}ms)</div>' for b in bottlenecks[:5])}
        </div>

        <div class="timeline card">
            <h3>Event Timeline (first 50)</h3>
            {"".join(f'<div class="event"><span class="event-seq">{e["seq"]}</span><span class="event-type {esc(e["type"])}">{esc(e["type"])}</span><span class="event-agent">{esc(e["agent"])}</span><span class="event-data">{esc(e["data"])}</span></div>' for e in events_data[:50])}
        </div>

        <footer>
            <p>Generated by LCTL (LLM Context Trace Library) | <a href="https://github.com/kimasplund/LLM-Context-Trace-Library">GitHub</a></p>
        </footer>
    </div>
</body>
</html>"""

    Path(output_path).write_text(html_content)
    return output_path


def get_session_metadata() -> dict[str, Any]:
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
            ["git", "branch", "--show-current"],  # noqa: S607  # resolving git via PATH is intended
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            metadata["git_branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607  # resolving git via PATH is intended
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            metadata["git_commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607  # resolving git via PATH is intended
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            metadata["git_dirty"] = bool(result.stdout.strip())
    except Exception:  # noqa: S110  # tracing must never break the host application
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


def estimate_cost(tokens_in: int, tokens_out: int, model: str = "default") -> dict[str, Any]:
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
    "MODEL_PRICING",
    "LCTLClaudeCodeTracer",
    "estimate_cost",
    "generate_hooks",
    "generate_html_report",
    "get_session_metadata",
    "is_available",
    "validate_hooks",
]
