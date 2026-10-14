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
- `Chain.load`/`Chain.save` accept `str` paths; `SchemaVersionError` is a `ValueError`; the `auto_instrument` docstring matches its behaviour
- Claude Code tracing: the `Agent` tool is recorded like `Task` (subagents were missed); `Stop.sh` no longer crashes with a `NameError`; each session gets its own chain via new `SessionStart`/`SessionEnd` hooks, and compaction no longer starts a new chain
- `lctl claude init` merges into an existing `settings.json` (writing `settings.json.bak` first) instead of replacing it, and validates all inputs before writing anything
- Claude Code hook commands shell-quote the project root, so paths with spaces no longer break (and block) the hooks
- `lctl claude validate` flags unregistered, outdated or stale installs instead of printing "valid"
- Web dashboard: `run_dashboard(reload=True)` works (import-string app factory)
- Web dashboard: non-ASCII API keys are rejected with 403 instead of a 500
- RPA webhooks are now actually delivered; previously they were registered but never sent
- LlamaIndex: the callback handler is initialised correctly and wired through the callback manager; `llm_trace` carries the model and LLM agent; the "already being traced" error explains that engines built from the same `Settings` share one callback manager
- PydanticAI: results and usage from 1.x and 2.x are read correctly, `run_sync` is traced, attributes are proxied through the traced agent, and stream deltas are recorded
- OpenAI Agents: `LCTLRunHooks.on_tool_start`/`on_tool_end` match the SDK signatures (the old ones crashed) and take tool input from `context.tool_arguments`
- CrewAI: steps are attributed to agent roles (and the manager), durations are real and clamped to non-negative, no spurious `ToolResult` steps, `kickoff_for_each` is traced, and user step callbacks are chained and always restored; installing the tracer twice is a no-op
- VS Code extension: token and cost come from library traces (`data.tokens`, legacy `tokens_in`/`tokens_out` as fallback); the `lctl.autoRefresh`, `lctl.maxEventsInTree` and `lctl.defaultPricing` settings are honoured; palette commands offer a file picker when run without a selection; `npm test` runs `node:test` and CI runs it
- Claude Code tracing: concurrent hooks no longer lose events or corrupt the chain (hooks hold `.claude/traces/.lctl.lock` and write atomically; a lock that cannot be taken within 10 s skips the write and exits 0); background/remote/teammate agent launches end their step with outcome `launched` (with `agent_id`) instead of a failure and no longer add a bogus result fact; input tokens include cache-creation and cache-read tokens (components kept in `tokens_detail`)
- `lctl claude init --force` keeps a user's own hook running: when it backs a script up to `<script>.bak`, the `settings.json` entries that ran it are kept and now run the backup
- Install hints in error messages, CLI output and examples use `pip install "lctl[<extra>] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"`; `lctl` is not published on PyPI
- PydanticAI: `llm_trace` records the agent's real model name and is emitted inside the step's span on `run`, `run_sync` and streams
- Webhook delivery closes HTTP error responses (no `ResourceWarning` on Python 3.14)
- Examples: every example runs without an API key; duplicates merged; added `spans_demo` and `examples/README.md`

### Changed

- Project management with uv and a committed `uv.lock`; dev tools moved to a PEP 735 dependency group
- `ruff format` replaces black; modern ruff rule set (pyupgrade, bugbear, bandit, simplify, async, pytest-style)
- mypy type checking; package ships `py.typed`
- pytest runs with warnings as errors and branch coverage (`fail_under` enforced)
- GitHub Actions CI (Python 3.11-3.14 full suite, 3.15 core suite, wheel check, VS Code extension build) and Dependabot
- pre-commit hooks mirroring CI's lint job
- License metadata follows PEP 639 (`AGPL-3.0-only`, no conflicting classifier)
- `lctl dashboard`, `WebSocketServer.start()` and `start_websocket_server()` bind to `127.0.0.1` by default; pass a host explicitly to expose them
- Dashboard API keys are enforced on every route except `/` (and `/static`), including `/ws` (checked before accept, close code 1008) and the SSE stream; `?api_key=` is accepted for `/ws` and SSE. Set `LCTL_LOCALHOST_BYPASS=false` to enforce keys for local clients too
- `ReplayEngine.diff()` and `lctl diff` compare content only, ignoring ids, timestamps and `duration_ms`; pass `ignore_volatile=False` / `--strict` to compare every field
- `LCTLOpenAIAgentTracer.run_config` is deprecated and emits a `DeprecationWarning`; use `hooks=tracer.create_hooks()`
- RPA webhooks are HMAC-signed, retried on failure and recorded in a delivery log (`GET /api/rpa/webhooks/{id}/deliveries`). Delivery runs on the dashboard app's lifespan event loop, so an app mounted without its lifespan does not deliver webhooks
- `lctl claude init --chain-id` is now the chain-id prefix used by `SessionStart` (`<prefix>-<timestamp>-<session>`); passing it regenerates `SessionStart.sh` without `--force`
- Minimum supported PydanticAI is `pydantic-ai>=1.0`
- CI gains an `integrations` job running live smoke tests for each extra (new `integration` pytest marker)

### Removed

- The AutoGen integration (`lctl.integrations.autogen`) and the `autogen` extra. AutoGen is in maintenance mode and the integration was broken with its own extra. A Microsoft Agent Framework integration is planned as its successor.

### Breaking

- The dashboard, `WebSocketServer.start()` and `start_websocket_server()` bind to `127.0.0.1` by default; deployments that relied on `0.0.0.0` must pass the host explicitly.
- With API keys configured, the whole API, `/ws` and SSE require a key from remote clients. This includes `/api/health` and `/api/rpa/auth/status`, which were previously open.
- With API keys configured, the web UI only works from localhost (or through an SSH tunnel): remote browser access needs a key-aware UI (planned), because the UI's JavaScript never sends a key and the data routes it calls were open before.
- The FastAPI auto-docs (`/docs`, `/redoc`, `/openapi.json`) are disabled because they exposed the route schema to unauthenticated clients.
- The AutoGen integration and the `autogen` extra are removed.
- `ReplayEngine.diff()` ignores volatile fields by default; use `ignore_volatile=False` (`lctl diff --strict`) for the old behaviour.
- `LCTLRunHooks.on_tool_start`/`on_tool_end` signatures changed to the SDK's, and `tracer.run_config` is deprecated.
- `lctl claude init` refuses to overwrite a same-named script that LCTL did not generate (exit 1, nothing written); `--force` backs it up to `<script>.bak` and replaces it. `--force` only regenerates hook scripts and never replaces `settings.json`.
- `lctl claude init --chain-id` now sets a chain-id prefix rather than a fixed chain id, because each session gets its own chain.
- `lctl claude validate` exits non-zero for invalid or outdated installs and advises `lctl claude init --force` for stale scripts.

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
