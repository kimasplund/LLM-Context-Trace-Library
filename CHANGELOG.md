# Changelog

All notable changes to the LCTL (LLM Context Trace Library) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Trace format 4.2: optional UUIDv7 event `id`, `span_id` and `parent_span_id` on events, and a chain-level `trace_id`, giving an exact call tree and parallel lanes. Older traces still load, and LCTL 4.1 reads 4.2 files.
- `LCTLSession.span()` context manager and `current_span_id`. Spans nest per execution context, so concurrent asyncio tasks get separate parents. `step_start`/`step_end`/`tool_call`/`llm_trace`/`error` accept keyword-only span ids.
- `ReplayEngine.get_span_tree()`, with deterministic inference for older traces (orphan step ends kept as `orphan-<seq>` nodes).
- `lctl trace --tree` prints the span call tree.

### Fixed

- Errors recorded by `traced_step` and `@lctl.traced` were attributed to agent "unknown"; they now carry the failing step's agent and span. `@lctl.traced` also no longer breaks the call if tracing itself fails.
- Concurrent agents in one session (e.g. under `asyncio.gather`) no longer overwrite each other's agent attribution.
- `ReplayEngine.diff()` ignores per-run identity fields.
- A failing `EventEmitter` attached to an `LCTLSession` no longer raises out of the recording call; like event listeners, its errors are dropped so tracing never breaks the host application.

- Web dashboard: the Metrics and Evaluation tabs and the Compare button never responded (their handlers were attached by an `init` reassignment that ran after the page-load listener had captured the original function)
- Web dashboard: event details open in a drawer over the current view instead of rendering ~17,000px down the page; panels scroll internally so the page stays about two screens tall; `Esc` closes the drawer
- Web dashboard: the "Duration" stat showed 0ms while scrubbing; it is now **Elapsed** (wall-clock time to the replay position) and long spans format as hours/days
- Web dashboard: static asset URLs carry a content hash, so browsers no longer run stale JavaScript/CSS after an upgrade
- Standalone WebSocket server (`WebSocketServer.start` / `start_websocket_server`) now works with the `websockets` library: connections previously failed immediately (close code 1011), and `stop()` now closes the listener. Events emitted from a thread other than the server's (e.g. with `run_in_thread=True`) are delivered promptly instead of waiting until something else woke the server loop (observed ~20s); `stop()` can be awaited from another thread or event loop and then waits for the server to shut down; a `stop()` awaited right after `asyncio.create_task(server.start(...))` (before the listener is bound) now shuts it down instead of leaving it serving; a second concurrent `start()` raises `RuntimeError` instead of orphaning the running server
- WebSocket send loop no longer recurses on every heartbeat, which silently stopped event delivery after ~1000 idle heartbeats and, on Python 3.11, kept disconnected clients' loops running

### Changed

- Project management with uv and a committed `uv.lock`; dev tools moved to a PEP 735 dependency group
- `ruff format` replaces black; modern ruff rule set (pyupgrade, bugbear, bandit, simplify, async, pytest-style)
- mypy type checking; package ships `py.typed`
- pytest runs with warnings as errors and branch coverage (`fail_under` enforced)
- GitHub Actions CI (Python 3.11-3.14 full suite, 3.15 core suite, wheel check, VS Code extension build) and Dependabot
- pre-commit hooks mirroring CI's lint job
- License metadata follows PEP 639 (`AGPL-3.0-only`, no conflicting classifier)

## [4.1.0] - 2026-01-19

### Added

- Claude Code integration for multi-agent workflow tracing
- Comprehensive Claude Code tracing with advanced features (Task tool invocations, agent completions, tool calls, TodoWrite updates, skill invocations, MCP tool calls, git commits, user interactions)
- `lctl claude init` command to generate hook scripts for Claude Code integration
- `lctl claude validate` command to check hook installation
- `lctl claude status` command to show active session
- `lctl claude report` command to generate HTML reports
- `lctl claude clean` command to clean old traces
- `--chain-id` option to `lctl claude init` for custom chain identification
- Settings.json generation in `lctl claude init`
- React dashboard to VS Code extension
- Streaming API support
- RAG/retriever documentation to LangChain tutorial
- VS Code extension visualization enhancements
- VS Code extension screenshot to README
- DSPy and LlamaIndex integration documentation

### Changed

- Rebranded project to "LLM Context Trace Library"
- Updated all repository URLs to new project name
- Updated Python requirements
- Upgraded VS Code extension major dependencies
- Upgraded @vscode/test-cli from 0.0.4 to 0.0.12
- Upgraded vite to fix esbuild security vulnerability (CVE)
- Improved Claude Code hooks to use stdin JSON and propagate agent names

### Fixed

- Integration module exports and tutorial bugs
- Claude Code hooks to work from any subdirectory
- Comprehensive integration fixes: thread safety, memory leaks, and bugs
- Test collection errors for optional dependencies
- Comprehensive project review fixes
- Version consistency to 4.1.0

### Documentation

- Added comprehensive Claude Code integration documentation
- Added Claude Code tutorial (`docs/tutorials/claude-code-tutorial.md`)
- Updated task.md with license and history rewrite completion
- Added RAG/retriever documentation to LangChain tutorial
- Updated README with DSPy and LlamaIndex integrations
- Added VS Code extension screenshots

## [4.0.0] - 2025-01-15

Initial public release under AGPLv3 license.

### Added

- Core event sourcing system with Chain, Event, State, and ReplayEngine
- LCTLSession for recording events
- Time-travel replay functionality
- Chain comparison and diff tools
- Performance bottleneck analysis
- Confidence tracking for facts
- CLI commands: replay, stats, bottleneck, diff, trace, debug, dashboard
- Web dashboard with FastAPI
- Framework integrations:
  - LangChain (LCTLCallbackHandler, LCTLChain)
  - CrewAI (LCTLCrew, LCTLAgent, LCTLTask)
  - AutoGen/AG2 (LCTLAutogenCallback)
  - OpenAI Agents SDK (LCTLOpenAIAgentTracer, TracedAgent)
- VS Code extension for visualization
- JSON and YAML chain file format support
- Comprehensive test suite

[4.1.0]: https://github.com/kimasplund/LLM-Context-Trace-Library/compare/v4.0.0...v4.1.0
[4.0.0]: https://github.com/kimasplund/LLM-Context-Trace-Library/releases/tag/v4.0.0
