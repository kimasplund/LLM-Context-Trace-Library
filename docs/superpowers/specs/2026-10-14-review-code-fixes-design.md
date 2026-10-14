# Review Code Fixes (W1) — Design

- **Date:** 2026-10-14
- **Status:** Awaiting review
- **Branch:** `fix/review-findings`
- **Inputs:** the documentation deep review in `docs/reviews/2026-10-14-docs-audit/` (221 findings). Most of the critical ones turned out to be code bugs.
- **Sequence:** W1 fixes the code (this spec). W2 rewrites the docs against working code. Then sub-project B, the event store.

## 1. Goal
Make the code do what users are told it does. That means:
- no data loss from `lctl claude init`;
- Claude Code self-tracing that records subagents and keeps sessions separate;
- every framework integration working against the versions its extra installs;
- dashboard security that matches its claims, with no dead webhook feature;
- core API rough edges that crash the documented examples are gone;
- a VS Code extension that shows real tokens and cost;
- examples that run.

Each fix is test-first. Integrations are proven against the real packages, not mocks.

## 2. Non-goals
- Rewriting the docs prose. That is W2. W1 updates only docstrings and the CHANGELOG.
- The event store (B) and the UI redesign (C).
- Native DSPy callback support, a Microsoft Agent Framework integration, and OpenTelemetry import/export. These are follow-ups (§10).
- Choosing a release version number. §9 lists the breaking changes so the release can decide.

## 3. Claude Code self-tracing (`lctl/integrations/claude_code.py`, `lctl/cli/main.py`)
1. **Subagent tool name.** Claude Code 2.1.x names the subagent tool `Agent` in hook payloads; older versions use `Task`.
   - Both are spawns. The `PreToolUse` matcher becomes `Task|Agent`, and the hook code tests `tool_name in ("Task", "Agent")`.
   - The fixtures use the real payload shape: `tool_name`, `tool_input.subagent_type`/`description`/`prompt`, `tool_use_id`, `session_id`.
2. **Session boundaries.**
   - `Stop.sh` is generated with a quoted heredoc (`<< 'PYTHON_EOF'`), like the other hooks. That removes the quoting bug: the generator's `\"` turned into bash double quotes, so Python received `summary[event_count]`.
   - `Stop` fires once per turn, so it only exports a snapshot and never deletes state.
   - New `SessionStart.sh` starts a fresh trace, archiving any leftover state as-is.
   - New `SessionEnd.sh` exports the final trace and removes the state file.
   - Result: a session is never appended to another session's trace.
3. **`lctl claude init` never destroys configuration.**
   - It reads the existing `.claude/settings.json`, if any, and writes a byte-identical `settings.json.bak` first.
   - For each LCTL hook event it removes earlier LCTL entries and appends the current one. An earlier LCTL entry is one whose command runs `<hooks_dir>/<Event>.sh`.
   - Every other key and every user hook entry is preserved.
   - `--force` controls only whether existing hook *scripts* are overwritten. It never changes the merge behaviour.
4. **`lctl claude validate`** reports outdated installations: a matcher without `Agent`, or missing `SessionStart`/`SessionEnd` hooks. It exits non-zero and tells the user to re-run `lctl claude init`.

**Tests:**
- Run each generated script with `bash` against stdin payloads. Specifically:
  - `Stop.sh` exits 0, prints the summary and leaves the state file in place.
  - `SessionEnd.sh` removes the state file.
  - An `Agent` pre/post pair in two processes yields one closed span with the right agent type.
- **Settings merge:** a settings file with `permissions`, `env` and a user `PreToolUse` hook survives `init` and `init --force` with those keys and entries intact, LCTL entries are not duplicated, and `.bak` matches the original.
- **`validate`:** it flags an old-style install.

## 4. Framework integrations
Support the current major versions that the extras install. Tests drive each framework for real with keyless fake models (§8).

1. **LlamaIndex** (`llamaindex.py`):
   - `LCTLLlamaIndexCallback.__init__` calls `BaseCallbackHandler.__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])`.
   - Fix the `trace_map` property/setter conflict (`llamaindex.py:745`).
   - `trace_query_engine`/`trace_chat_engine` wire the handler through `Settings.callback_manager` or the component's own manager. They never pass `callback_manager=` to `as_query_engine`, because LlamaIndex already supplies that argument.
   - The supported route documented in the docstrings is `Settings.callback_manager = handler.get_callback_manager()`.
2. **PydanticAI** (`pydantic_ai.py`):
   - Work with both 1.x and 2.x. Usage is `result.usage()` if callable, else `result.usage`. Output is `result.output`, else `result.data`.
   - Streaming records deltas (`stream_text(delta=True)`, or the difference from the previous cumulative text), so no text is stored twice.
   - `TracedAgent` forwards unknown attributes to the wrapped agent via `__getattr__`, and traces `run_sync` as well as `run`/`run_stream`.
   - Bump the extra's lower bound to the oldest version tested.
3. **OpenAI Agents** (`openai_agents.py`):
   - `LCTLRunHooks.on_tool_start(context, agent, tool)` and `on_tool_end(context, agent, tool, result)` match the SDK's `RunHooks` signatures. The current required `input_data` parameter crashes the first real tool call.
   - The `run_config` property keeps its return value but emits a `DeprecationWarning` naming `Runner.run(..., hooks=tracer.create_hooks())`.
   - Docstrings stop presenting `on_error` and `on_tool_error` as SDK hooks.
4. **AutoGen is removed.** The user decided this on 2026-10-14.
   - **Why:** Microsoft AutoGen is in maintenance mode. Its README says it "will not receive new features", and Microsoft points new users to its successor, Microsoft Agent Framework. LCTL's current integration doesn't work with the package its own extra installs (`pyautogen>=0.7` resolves to the new AutoGen, where the legacy wrappers raise). Rewriting it for a frozen framework isn't worth it.
   - **Removed:**
     - `lctl/integrations/autogen.py`;
     - the `autogen` extra, and `autogen` from `all`;
     - the AutoGen exports in `lctl/integrations/__init__.py` (`LCTLAutogenCallback`, `trace_autogen_agent`, `autogen_available`, …);
     - `tests/test_autogen.py`, `tests/test_autogen_modern.py`;
     - `examples/autogen_example.py`, `docs/tutorials/autogen-tutorial.md`;
     - the AutoGen modules in the mypy overrides (`autogen.*`, `autogen_core.*`, `ag2.*`).
   - **What users see:** W2 documents the removal and points to the Microsoft Agent Framework follow-up (§10). The CHANGELOG lists it under Removed.

5. **CrewAI** (`crewai.py`):
   - Step and tool events are attributed to the real agent `role`, not `"unknown"`.
   - Durations are measured between callbacks, not 0 ms.
   - The spurious `ToolResult` step is no longer recorded (`crewai.py:371-428`).
   - Tests use CrewAI's fake LLM or a stub LLM.
6. **DSPy and Semantic Kernel:** no behaviour change. W2 documents what they actually record. The SK smoke test covers `kernel_function` with `add_plugin`.

## 5. Dashboard (`lctl/dashboard/app.py`, `lctl/cli/main.py`, `lctl/streaming/websocket.py`)
1. **Safe bind defaults.** `lctl dashboard --host`, `WebSocketServer.start(host=…)` and `start_websocket_server(host=…)` default to `127.0.0.1`. LAN exposure requires an explicit `--host 0.0.0.0`. This closes tooling-spec follow-up 2.
2. **Auth on every route.** When `LCTL_API_KEYS` is set or `LCTL_REQUIRE_API_KEY=true`, remote clients need a valid key on every route:
   - all `/api/*` routes;
   - `POST /api/events/emit`;
   - the `/ws` WebSocket, with the key passed as the `X-API-Key` header or `?api_key=`;
   - the SSE stream.

   The localhost bypass keeps its current default. `LCTL_LOCALHOST_BYPASS=false` enforces keys for local clients too. The index page and static assets stay public; the data does not.
3. **Webhook delivery**, using the existing `/api/rpa/webhooks` registrations:
   - **What is sent:** every event emitted through the dashboard's emitter is matched against each registration's filters (event types, chain ID). Matches are POSTed as JSON.
   - **Headers:** `X-LCTL-Event: <type>`, `X-LCTL-Delivery: <uuidv7>`, and `X-LCTL-Signature: sha256=<hex>`, an HMAC-SHA256 of the raw body keyed by the registration `secret`, sent when a secret is set.
   - **Transport:** stdlib `urllib.request` in a worker thread, so no new dependency. The timeout is 5 s, with 3 attempts and exponential backoff at 1 s, 2 s and 4 s.
   - **Delivery log:** `GET /api/rpa/webhooks/{id}/deliveries` returns the last 100 delivery records (status, attempts, last error, timestamps), kept in memory.
   - **Constraints:** only `http(s)` URLs are accepted at registration, and registration requires an API key as today. A slow receiver never blocks event emission.
4. **`run_dashboard(reload=True)`** runs uvicorn with an import string plus `factory=True`, passing the working directory and port through `LCTL_DASHBOARD_DIR`.

**Tests:**
- For each route class, a remote request with no key gets 401/403 and one with a key gets 200. The WebSocket gets a 1008 close or a rejected handshake without a key.
- A local request with the bypass on gets 200.
- Default host values.
- Webhooks: a local receiver (`http.server` on a thread) gets the signed POST. The signature verifies with the secret. A failing receiver gets 3 attempts and a logged failure. Emission latency is unaffected.
- `run_dashboard` with `reload=True` builds the right uvicorn call (mocked).

## 6. Core API
1. `Chain.load()` and `Chain.save()` accept `str | os.PathLike`.
2. `ReplayEngine.diff(other, *, ignore_volatile=True)`:
   - The default ignores `timestamp`, `id`, `span_id`, `parent_span_id` and `data.duration_ms`.
   - `ignore_volatile=False` compares everything.
   - `lctl diff --strict` maps to `ignore_volatile=False`.
   - Diff entries keep their shape.
3. `auto_instrument()` keeps its behaviour. Its docstring states that it creates the global session used by `@traced` and does not instrument frameworks.
4. `SchemaVersionError` subclasses `ValueError`, so `Chain.load` raises `ValueError` for every invalid file as documented.

**Tests:** `str` paths round-trip. Two real recordings of the same workflow diff to `[]` by default and to a non-empty list with `ignore_volatile=False`. `lctl diff --strict` works. `isinstance(SchemaVersionError(...), ValueError)` holds.

## 7. VS Code extension (`vscode-lctl/`)
1. **Tokens and cost** come from the library's `data.tokens.input`/`data.tokens.output`, with the old `tokens_in`/`tokens_out` as a fallback.
2. **Implement the three declared settings:**
   - `lctl.autoRefresh` turns the file watchers on and off.
   - `lctl.maxEventsInTree` replaces the hard-coded 50.
   - `lctl.defaultPricing` feeds the cost calculation.
3. **Palette commands** with no chain selected (replay, export HTML/Markdown, statistics) show a QuickPick of workspace `**/*.lctl.json` files instead of erroring.
4. **Tests:** the token/cost/stat calculations move into `src/chainStats.ts`. They are tested with Node's built-in `node:test` against the compiled output, run by `npm test` in the CI extension job, with no new dependencies.

## 8. Examples and verification infrastructure
1. **Examples** (`examples/`):
   - Every script runs with no API key, using each framework's fake or test model where possible. Where a framework can't run keyless, the script prints a clear notice and exits 0.
   - Fix the specific breakages:
     - LangChain: the `is_available` import, and a real `__main__`.
     - OpenAI Agents: `hooks=`.
     - PydanticAI: `TestModel`, `output_type` and `.output`.
     - Semantic Kernel: `kernel_function` with `add_plugin`.
     - CrewAI: actually calls `kickoff()`.
   - Merge `openai_agent.py` into `openai_agents_example.py`, using one `<framework>_example.py` convention.
   - Add `examples/README.md` (framework, extra, how to run).
   - Add `examples/spans_demo.py`, which writes a 4.2 trace with nesting and a parallel `asyncio.gather` fan-out.
   - `examples/sample-chain.lctl.json` is unchanged; tests pin its output.
2. **Integration test marker.** Tests needing a real framework are `@pytest.mark.integration` and `importorskip` their framework. The default `uv run pytest` behaviour and the coverage gate are unchanged.
3. **CI `integrations` job.**
   - A matrix over extras: `langchain`, `crewai`, `openai-agents`, `pydantic-ai`, `semantic-kernel`, `dspy`, `llamaindex`.
   - Each cell runs `uv sync --locked --extra <x>`, then `uv run pytest -m integration -k <x>`, then the matching example.
   - It runs on Python 3.13, with no coverage gate on this job.
   - A cell may be marked `continue-on-error` only with a linked follow-up. None is planned.
   - **Lockfile:** if the universal lock can't satisfy an extra's tested range, the extra's bounds are adjusted and the lock regenerated.

## 9. Breaking changes (for the release notes)
- The dashboard, `WebSocketServer.start()` and `start_websocket_server()` bind to `127.0.0.1` by default.
- With API keys configured, the core API, `/ws` and SSE now require a key from remote clients.
- The AutoGen integration and the `autogen` extra are removed (the framework is in maintenance mode and the integration was broken with its own extra).
- `ReplayEngine.diff()` ignores volatile fields by default.
- `LCTLRunHooks.on_tool_start`/`on_tool_end` signatures changed to the SDK's (the old ones crashed).
- `tracer.run_config` is deprecated.

## 10. Follow-ups (recorded, not in scope)
1. Microsoft Agent Framework integration, the successor to AutoGen and the replacement for the removed AutoGen integration.
2. Native DSPy callback support (`dspy.utils.callback.BaseCallback`) for nested module, LM and tool events.
3. Claude Code tracer: key traces by the payload's `session_id` (more robust than the session hooks), and emit span IDs for subagent spans (UI sub-project A follow-up 2).
4. Webhooks: persistent delivery log and optional blocking of private-network targets (SSRF hardening) once the event store (B) exists.
5. Per-integration native-ID span mapping (A follow-up 1).

## 11. Acceptance
- **Default suite:** passes on 3.11–3.14, the 3.15 core suite passes, coverage is at or above `fail_under`, and ruff, mypy and pre-commit are clean.
- **Integrations:** the CI `integrations` job is green for all seven extras.
- **Examples:** every example in `examples/` exits 0 with no API key in its extra's environment.
- **Claude Code:**
  - A scripted end-to-end run (`lctl claude init` in a temp project with pre-existing settings, then the generated hooks driven with real-shape payloads across separate processes) shows the settings preserved, one trace per session and closed `Agent` spans.
  - `lctl trace --tree` on the result shows the subagent nesting.
- **VS Code extension:** `npm test` passes in CI, and tokens are non-zero for a library-produced trace with token data.
