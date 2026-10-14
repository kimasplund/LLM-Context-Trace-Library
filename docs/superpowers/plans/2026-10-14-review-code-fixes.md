# Review Code Fixes (W1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the code bugs that the documentation deep review uncovered, so the code does what users are told it does.

**Architecture:** Independent, test-first fixes across six areas:
- Claude Code tracer
- framework integrations
- dashboard
- core API
- VS Code extension
- examples

Real-framework behaviour is proven by `@pytest.mark.integration` tests. They run in per-extra environments locally and in a new CI matrix job.

**Tech Stack:** Python ≥3.11, pytest, FastAPI/Starlette, click, bash hooks, TypeScript (VS Code extension), uv, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-14-review-code-fixes-design.md`. It is binding, and its § numbers are referenced below. The evidence for each bug is in `docs/reviews/2026-10-14-docs-audit/*.md`.

## Global Constraints
- **Test-first.** Every behaviour change starts with a test, and you watch it fail for the stated reason. Exceptions are pure deletions (Task 5) and config (CI YAML).
- **Integration tests:**
  - Location: `tests/integration/test_<framework>_live.py`.
  - Each is marked `@pytest.mark.integration` and starts with `pytest.importorskip("<framework module>")`, so the default `uv run pytest` skips them when the framework isn't installed. The default suite and its coverage gate (`fail_under = 78`) stay unchanged.
  - No API keys. Use only fake or test models.
- **Per-extra environment, used locally and mirrored by CI.** `SP=/tmp/claude-1000/-home-kim-projects-llm-context-transfer/783deef4-54f3-49cc-81f2-f407fcbaf8ae/scratchpad`.
  ```bash
  UV_PROJECT_ENVIRONMENT=$SP/venv-<extra> uv sync --locked --extra <extra>
  UV_PROJECT_ENVIRONMENT=$SP/venv-<extra> uv run --no-sync pytest -m integration tests/integration/test_<x>_live.py --no-cov -q
  ```
- **Gates for every task:**
  - `uv run pytest -q` passes with coverage ≥ 78.
  - `uv run ruff check . && uv run ruff format --check . && uv run mypy` is clean.
  - The pre-commit hook passes on commit.
- **Docs prose belongs to W2.** W1 touches only docstrings, CLI help text, `CHANGELOG.md` (Task 13), and anything that would otherwise reference removed code. W2 rewrites the README and tutorials.
- **Commit trailer:** end every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01U7JQRWVBpRadDaH4RESMtZ
  ```
- **Never push.** Integration with main is decided with the user at the end.
- **The docs audit's scratch harnesses are references only.** Rebuild what you need inside `tests/`; the harnesses aren't committed:
  - `$SP/docs-audit/tutorials-a/{oa,lc2}`: the openai-agents and langchain fake-model drivers.
  - `$SP/docs-audit/tutorials-b/`: the dspy, llamaindex, crewai, pydantic-ai and SK runs.

## Review Focus
1. **Destroying user configuration.** `lctl claude init`, with and without `--force`, must preserve every unrelated `settings.json` key and every non-LCTL hook entry, and must write `.bak` first. Pinned in Task 2.
2. **Auth bypass.** Every data-bearing route, including the `/ws` WebSocket and the SSE stream, must refuse a remote client without a key when keys are configured. The index page and static assets stay public. Pinned in Task 3, which enumerates the routes from `app.routes` so new routes can't silently skip auth.
3. **Webhook delivery must never block or crash event emission**, whether the receiver is slow, failing or unreachable. Pinned in Task 4.
4. **Version-compat paths.** PydanticAI 1.x vs 2.x and LlamaIndex 0.12 vs 0.14 must each be exercised, not assumed. Pinned in Tasks 7–8, using the locked version plus a hand-run older version.
5. **Removing AutoGen** must leave no dangling imports, exports, extras, mypy overrides, docs links or CI references. Pinned in Task 5 with a repo-wide grep check.

---

### Task 1: Core API (spec §6)
**Files:** `lctl/core/events.py`, `lctl/core/schema.py`, `lctl/__init__.py`, `lctl/cli/main.py` (`diff` command), `tests/test_events.py`, `tests/test_cli.py`, `tests/test_schema.py`

**Interfaces (produces):**
- `Chain.load(path: str | os.PathLike) -> Chain`
- `Chain.save(path: str | os.PathLike) -> None`
- `ReplayEngine.diff(other, *, ignore_volatile: bool = True)`
- `lctl diff --strict`
- `class SchemaVersionError(ValueError)`

- [ ] **Step 1: Write the failing tests.**
  - `test_load_and_save_accept_str_paths`: round-trip through `str(tmp_path / "c.lctl.json")`.
  - `test_diff_ignores_volatile_fields_by_default`: two `LCTLSession` recordings of the same workflow, recorded with a `time.sleep(0.01)` between them so the timestamps really differ, and with different `duration_ms`. `diff()` returns `[]`, and `diff(ignore_volatile=False)` returns a non-empty list.
  - `test_cli_diff_strict_flag`: on two such files, `lctl diff a b` reports no divergence, and `lctl diff --strict a b` reports divergence.
  - `test_schema_version_error_is_value_error`.
- [ ] **Step 2:** Run the new tests. Expected: FAIL. The str path raises `AttributeError`, and diff reports a divergence on every event.
- [ ] **Step 3: Implement.**
  - Coerce paths with `Path(path)`.
  - `diff` strips `id`, `span_id` and `parent_span_id` (top-level), `timestamp`, and `data["duration_ms"]` from both sides when `ignore_volatile`. Use a shallow copy, and never mutate events.
  - The CLI adds `@click.option("--strict", is_flag=True, help="Compare every field, including timestamps and durations.")`.
  - `SchemaVersionError(ValueError)`.
  - Rewrite the `auto_instrument` docstring to: "Create (or return) the global session used by `@lctl.traced` and `lctl.export()`. It does not instrument frameworks; use the integration for your framework."
- [ ] **Step 4:** Run the new tests and the full suite. Expected: all pass. Existing diff tests may assert divergence on timestamps; any that do get `ignore_volatile=False`, and you note it in the commit.
- [ ] **Step 5: Commit** `fix(core): str paths, content-based diff (--strict for all fields), SchemaVersionError is a ValueError`.

### Task 2: Claude Code self-tracing (spec §3)
**Files:** `lctl/integrations/claude_code.py` (`generate_hooks` ~line 1031, the hook scripts, `validate_hooks` ~1364), `lctl/cli/main.py` (`claude_init` ~899, `claude_validate` ~1013), `tests/test_claude_code.py`, `tests/test_cli.py`

**Interfaces (produces):**
- `generate_hooks(...)` returns keys `PreToolUse`, `PostToolUse`, `Stop`, `SessionStart` and `SessionEnd`.
- `validate_hooks(...)` reports `{"outdated": bool, "reasons": [...]}` in addition to its existing keys.

- [ ] **Step 1: Write the failing tests.** Run the generated scripts with real `bash` in `tmp_path`, as a project directory with `.venv` symlinked to the repo venv's python, feeding JSON payloads on stdin.
  - **`test_agent_tool_spawn_recorded_across_processes`.** Run PreToolUse.sh with `{"session_id":"s1","tool_name":"Agent","tool_use_id":"t1","tool_input":{"subagent_type":"Explore","description":"find x","prompt":"..."}}`, then PostToolUse.sh with the matching `tool_response`, as two separate `subprocess.run` calls. The exported chain's `get_span_tree()` has one closed `Explore` node and no orphan. Also check the same for `"tool_name":"Task"`.
  - **`test_stop_hook_runs_and_keeps_state`.** `Stop.sh` exits 0, its stdout contains `Events:`, and `.claude/traces/.lctl-state.json` still exists.
  - **`test_session_end_finalizes_and_session_start_isolates`.** SessionEnd.sh exports and removes the state file. A later SessionStart.sh, then Pre and Post, produce a *new* chain file, and its events don't include the first session's.
  - **`test_init_preserves_unrelated_settings`.**
    - Seed `.claude/settings.json` with `{"permissions": {...}, "env": {...}, "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type":"command","command":"echo user"}]}]}}`.
    - Run `lctl claude init`, then `lctl claude init` again, then `lctl claude init --force`.
    - After each run: `permissions` and `env` are unchanged, the user's Bash hook is present exactly once, the LCTL entries are present exactly once per event, and `settings.json.bak` is byte-identical to the file as it was before that run.
  - **`test_validate_flags_outdated_install`.** A settings file with the old `"matcher": "Task"` and no session hooks makes `lctl claude validate` exit non-zero, and the output contains `lctl claude init`.
- [ ] **Step 2:** Run the new tests. Expected: FAIL. The `Agent` events aren't recorded, `Stop.sh` hits the `event_count` NameError, there are no session hooks, and init replaces the hooks.
- [ ] **Step 3: Implement** per spec §3.
  - The `PreToolUse` matcher is `"Task|Agent"`.
  - The hook code checks `tool_name in ("Task", "Agent")`.
  - Stop, SessionStart and SessionEnd all use the same `HOOK_INPUT_B64 … << 'PYTHON_EOF'` pattern as PreToolUse.
  - LCTL entries are identified by a command that runs `<hooks_dir>/<Event>.sh`.
  - Merge into existing settings: copy them to `settings.json.bak` with `shutil.copy2` first, remove old LCTL entries per event, then append the new ones.
  - `--force` gates only the overwriting of hook scripts.
- [ ] **Step 4:** Run the new tests and the full suite. Expected: all pass.
- [ ] **Step 5: Commit** `fix(claude-code): record Agent subagents, real session boundaries, non-destructive init`.

### Task 3: Dashboard bind defaults, auth on every route, reload (spec §5.1, §5.2, §5.4)
**Files:** `lctl/dashboard/app.py` (`verify_api_key` ~168, the route registration functions, `/ws` ~674, `/api/events/stream` ~759, `run_dashboard` ~1350), `lctl/cli/main.py` (the `--host` option ~774), `lctl/streaming/websocket.py` (`start` and `start_websocket_server` host defaults), `tests/test_dashboard.py`, `tests/test_dashboard_rpa.py`, `tests/test_websocket_comprehensive.py`

**Interfaces (produces):** a reusable auth dependency for HTTP routes and a WebSocket auth check, both honouring `LCTL_API_KEYS`, `LCTL_REQUIRE_API_KEY` and `LCTL_LOCALHOST_BYPASS`.

- [ ] **Step 1: Write the failing tests.**
  - **`test_every_data_route_requires_key_for_remote_clients`.**
    - Build the app with `LCTL_API_KEYS=k1`.
    - Iterate `app.routes` and select every `APIRoute`/`APIWebSocketRoute` except `/` and `/static`.
    - Use a `TestClient` whose client host is remote, e.g. `TestClient(app, client=("203.0.113.5", 50000))` (this needs Starlette ≥ 0.36 for the `client` argument; otherwise patch `request.client`).
    - For each route, a request without a key gets 401/403, and the same request with `X-API-Key: k1` doesn't get 401/403.
    - For `/ws`: a connection without a key is refused (the handshake fails or it closes with 1008), and `?api_key=k1` connects.
  - **`test_localhost_bypass_default_and_disable`.** A local client without a key gets 200 by default, and 401 with `LCTL_LOCALHOST_BYPASS=false`.
  - **`test_bind_defaults_are_loopback`.** The `lctl dashboard --help` output shows a host default of `127.0.0.1`, and the `WebSocketServer.start` and `start_websocket_server` signatures default to `"127.0.0.1"`.
  - **`test_run_dashboard_reload_uses_import_string`.** Monkeypatch `uvicorn.run` to capture its arguments. `run_dashboard(reload=True, working_dir=tmp_path)` calls it with a string app path and `factory=True`, and sets `LCTL_DASHBOARD_DIR`.
- [ ] **Step 2:** Run the new tests. Expected: FAIL. The core routes, `/ws` and SSE are open, and the host defaults are `0.0.0.0`.
- [ ] **Step 3: Implement.**
  - Apply the auth dependency to every data route; keep `/` and `/static` public.
  - The WebSocket and SSE handlers check the `X-API-Key` header or the `api_key` query parameter before accepting.
  - Change the host defaults and drop the now-obsolete `noqa: S104` comments.
  - Add a module-level `create_app_from_env()` factory that reads `LCTL_DASHBOARD_DIR`, and have the `reload` path use it.
- [ ] **Step 4:** Run the new tests and the full suite. Expected: all pass. Existing dashboard tests that act as remote clients may need a key; update them minimally and list them in the commit.
- [ ] **Step 5: Commit** `fix(dashboard): loopback bind by default, API keys enforced on every route incl. ws/SSE, working reload`.

### Task 4: Webhook delivery (spec §5.3)
**Files:** `lctl/dashboard/webhooks.py` (new: matching, signing, delivery, log), `lctl/dashboard/app.py` (wiring, plus the new `GET /api/rpa/webhooks/{id}/deliveries` route), `tests/test_dashboard_webhooks.py` (new)

**Interfaces (produces):**
- `WebhookDispatcher` with:
  - `.register(reg) -> None`
  - `.unregister(id) -> None`
  - `.dispatch(event: dict) -> None` (non-blocking)
  - `.deliveries(id) -> list[dict]`
- `sign(body: bytes, secret: str) -> str`, which returns `"sha256=<hex>"`.

- [ ] **Step 1: Write the failing tests.** Use a local receiver: `http.server.ThreadingHTTPServer` on port 0 in a thread, recording headers and bodies.
  - **`test_matching_event_is_delivered_signed`.** Register a webhook with `secret="s"` and an event-type filter, then emit a matching event through the app (`POST /api/events/emit` with a key). The receiver gets a POST whose body JSON has the event, with headers `X-LCTL-Event`, `X-LCTL-Delivery` (UUIDv7) and `X-LCTL-Signature == sign(body, "s")`.
  - **`test_non_matching_event_not_delivered`.**
  - **`test_failing_receiver_retried_three_times_and_logged`.** The receiver returns 500. Patch the backoff to 0. It receives 3 attempts, and `/api/rpa/webhooks/{id}/deliveries` shows `status: "failed"`, `attempts: 3` and the last error.
  - **`test_slow_receiver_does_not_block_emission`.** The receiver sleeps 2 s, and `POST /api/events/emit` returns in under 0.5 s.
  - **`test_only_http_urls_accepted`.** Registering a `file:///` or `ftp://` URL gets 422 or 400.
- [ ] **Step 2:** Run the tests. Expected: FAIL; nothing is delivered and the deliveries route doesn't exist.
- [ ] **Step 3: Implement.**
  - **Transport:** stdlib `urllib.request` in `asyncio.to_thread`, scheduled with `asyncio.create_task` and keeping strong references to running tasks.
  - **Retries and timeout:** a 5 s timeout and 3 attempts, with backoff `1, 2, 4` s as a module constant that tests can patch.
  - **Delivery log:** keep the last 100 records per webhook in a `collections.deque(maxlen=100)`.
  - **Matching:** filter on `event_types`, plus `chain_id` if the registration model has it. Add an optional `chain_id` field to the registration model if it doesn't.
  - **Hook point:** hook the dispatcher into the dashboard emitter as an `"all"` listener.
- [ ] **Step 4:** Run the new tests and the full suite. Expected: all pass, with no unawaited-coroutine or ResourceWarning under `filterwarnings=error`.
- [ ] **Step 5: Commit** `feat(dashboard): deliver RPA webhooks (HMAC-signed, retries, delivery log)`.

### Task 5: Remove the AutoGen integration (spec §4.4)
**Files:**
- **Delete:** `lctl/integrations/autogen.py`, `tests/test_autogen.py`, `tests/test_autogen_modern.py`, `examples/autogen_example.py`, `docs/tutorials/autogen-tutorial.md`.
- **Modify:**
  - `lctl/integrations/__init__.py` (the AutoGen imports and `__all__`)
  - `pyproject.toml` (the `autogen` extra, `all`, and the `autogen.*`/`autogen_core.*`/`ag2.*` mypy overrides)
  - `uv.lock`
  - `CLAUDE.md`, but only to drop the lines naming the deleted files (W2 rewrites the rest)

- [ ] **Step 1: Delete and edit** the files listed above.
- [ ] **Step 2:** Run `uv lock && uv sync`. Expected: the lock drops the AutoGen packages.
- [ ] **Step 3: Verify no dangling references.**
  - Run `git grep -n -i -E "autogen|ag2|pyautogen" -- . ':!docs/reviews' ':!docs/superpowers' ':!CHANGELOG.md' ':!docs/archive'`. Expected: only README hits, which W2 rewrites; list them in the commit.
  - Run `uv run python -c "import lctl.integrations as i; print(sorted(n for n in i.__all__ if 'autogen' in n.lower()))"`. Expected: `[]`.
- [ ] **Step 4:** Run the full suite. Expected: all pass, with the test count reduced by the deleted AutoGen tests and coverage ≥ 78.
- [ ] **Step 5: Commit** `feat!: remove the AutoGen integration (framework in maintenance mode; integration broken with its own extra)`.

### Task 6: Integration-test infrastructure, plus live smoke tests for LangChain, DSPy and Semantic Kernel (spec §8.2, §8.3, §4.6)
**Files:** `pyproject.toml` (pytest `markers`), `tests/integration/__init__.py`, `tests/integration/test_langchain_live.py`, `tests/integration/test_dspy_live.py`, `tests/integration/test_semantic_kernel_live.py`, `.github/workflows/ci.yml` (new `integrations` job)

**Interfaces (produces):**
- The `integration` marker, used by Tasks 7–10.
- The CI job matrix: `extra: [langchain, crewai, openai-agents, pydantic-ai, semantic-kernel, dspy, llamaindex]`.
- Each cell runs `uv sync --locked --extra ${{ matrix.extra }}`, then `uv run --no-sync pytest -m integration -q --no-cov tests/integration`, then the extra's example from Task 12.

- [ ] **Step 1:** Register `markers = ["integration: needs a real framework package; run in its extra's environment"]` under `[tool.pytest.ini_options]`.
- [ ] **Step 2: Write the smoke tests.** They should pass in their extra's environment without any LCTL change; if one fails, that's a real bug to fix in this task.
  - **LangChain:** `GenericFakeChatModel` (or `FakeListChatModel`) driven through `LCTLCallbackHandler`. The chain records `step_start`/`step_end` and an `llm_trace`.
  - **DSPy:** `dspy.utils.DummyLM` with `trace_module(dspy.Predict("question -> answer"))`. The outer span is recorded and the answer is returned.
  - **Semantic Kernel:**
    - A plugin class with a `@kernel_function` method, registered with `kernel.add_plugin(...)` and invoked with `kernel.invoke(...)`.
    - The tracer records the function step, and an exception raised inside the function is recorded as an error event.
- [ ] **Step 3:** Run each in its per-extra environment (see Global Constraints). Expected: pass. In the default env, `uv run pytest -q` shows them as skipped and the total is unchanged otherwise.
- [ ] **Step 4:** Add the `integrations` CI job, mirroring the `test` job's hardening:
  - SHA-pinned actions, `permissions: contents: read`, `persist-credentials: false`, setup-uv 0.13.1, Python 3.13, `fail-fast: false`.
  - The example step is added in Task 12.
  - Run `uv run pre-commit run zizmor --all-files`. Expected: no findings.
- [ ] **Step 5: Commit** `test: integration marker, live LangChain/DSPy/SK smoke tests, CI integrations job`.

### Task 7: LlamaIndex (spec §4.1)
**Files:** `lctl/integrations/llamaindex.py` (`__init__` ~91, the `trace_map` handling ~714–745, `trace_query_engine` ~977, `trace_chat_engine` ~1009, the module docstring examples), `tests/integration/test_llamaindex_live.py` (new), `tests/test_llamaindex.py` (keep the unit tests)

- [ ] **Step 1: Write the failing live tests,** using `Settings.llm = MockLLM()` and `Settings.embed_model = MockEmbedding(embed_dim=8)` from `llama_index.core`, and an index from two `Document`s.
  - `test_handler_constructs` creates `LCTLLlamaIndexCallback()` without raising.
  - `test_settings_callback_manager_records_query` sets `Settings.callback_manager = cb.get_callback_manager()`, runs `index.as_query_engine().query("q")`, and expects the chain to have a query span, a retrieve event and an `llm_trace`.
  - `test_trace_query_engine_wraps_and_records` and `test_trace_chat_engine_wraps_and_records` each expect no `trace_map` error and recorded events.
- [ ] **Step 2:** Run them in the `venv-llamaindex` environment. Expected: FAIL with `AttributeError: … event_starts_to_ignore` and `property 'trace_map' … has no setter`.
- [ ] **Step 3: Implement** per spec §4.1.
  - Call `BaseCallbackHandler.__init__(self, event_starts_to_ignore=[], event_ends_to_ignore=[])`.
  - Copy trace state without assigning the read-only property: use the manager's public API or `_trace_map`, version-checked.
  - The trace helpers add the handler to the engine's existing callback manager instead of passing `callback_manager=`.
  - Fix the module docstring example to use the `Settings` route.
- [ ] **Step 4:** Run the live tests on the locked version. Then run them once against llama-index-core 0.12.x in a throwaway venv, `$SP/venv-llamaindex-012` with `uv pip install "llama-index-core<0.13"` plus the repo on `PYTHONPATH`. Expected: pass on both. Then run the default suite.
- [ ] **Step 5: Commit** `fix(llamaindex): initialize BaseCallbackHandler, fix trace_map copy, wire via callback manager`.

### Task 8: PydanticAI (spec §4.2)
**Files:** `lctl/integrations/pydantic_ai.py` (`run` ~265–330, `run_stream`/`stream_text` ~364–490, `__getattr__`), `pyproject.toml` (the `pydantic-ai` lower bound), `uv.lock`, `tests/integration/test_pydantic_ai_live.py` (new)

- [ ] **Step 1: Write the failing live tests,** using `Agent(TestModel())` and wrapping it with `trace_agent` or `TracedAgent`.
  - `test_run_records_output_and_usage` runs `await traced.run("hi")`. It returns `result.output`, and the step end carries non-negative token usage.
  - `test_run_sync_traced`.
  - `test_unknown_attributes_forward`: `traced.name` and `traced.override` are reachable.
  - `test_stream_records_deltas_once`: with `TestModel(custom_output_text="Hello world")`, the recorded `stream_end` text is exactly `"Hello world"`, not repeated.
- [ ] **Step 2:** Run them in `venv-pydantic-ai` (locked 2.x). Expected: FAIL with `TypeError: 'RunUsage' object is not callable`, missing `run_sync`, and repeated text.
- [ ] **Step 3: Implement** per spec §4.2:
  - `usage = result.usage() if callable(result.usage) else result.usage`;
  - `output = getattr(result, "output", None)`, falling back to `.data`;
  - record only new text (the delta against the previous cumulative value);
  - a `__getattr__` proxy on `TracedAgent`;
  - trace `run_sync`.

  Set the extra's lower bound to the oldest version tested in Step 4, e.g. `pydantic-ai>=1.0`.
- [ ] **Step 4:** Run the live tests on the locked 2.x, and once on 1.x in a throwaway venv (`uv pip install "pydantic-ai<2"`). Expected: pass on both. Then `uv lock` and run the default suite.
- [ ] **Step 5: Commit** `fix(pydantic-ai): support 1.x and 2.x results, trace run_sync, proxy attributes, record stream deltas`.

### Task 9: OpenAI Agents (spec §4.3)
**Files:** `lctl/integrations/openai_agents.py` (`LCTLRunHooks.on_tool_start` ~253 and `on_tool_end` ~285, the `run_config` properties ~566 and ~904, docstrings), `tests/integration/test_openai_agents_live.py` (new), `tests/test_openai_agents.py`

- [ ] **Step 1: Write the failing live tests,** using a scripted fake model that subclasses the SDK's `Model` interface. It returns one tool call, then a final message. (See `$SP/docs-audit/tutorials-a/oa` for a working shape.)
  - `test_tool_run_records_tool_events` runs `Runner.run(agent, "go", hooks=tracer.create_hooks())`. It completes, and the chain has the agent span plus a `tool_call` event with the tool name and output.
  - `test_run_config_property_warns` expects `pytest.warns(DeprecationWarning, match="hooks=tracer.create_hooks")`.
- [ ] **Step 2:** Run them in `venv-openai-agents`. Expected: FAIL with a `TypeError` from the `on_tool_start` signature.
- [ ] **Step 3: Implement.** Use the SDK `RunHooks` signatures: `on_tool_start(self, context, agent, tool)` and `on_tool_end(self, context, agent, tool, result)`. `run_config` emits a `DeprecationWarning` and otherwise behaves as before. Fix the docstrings per spec.
- [ ] **Step 4:** Run the live tests and the default suite. Update the unit tests that called the old signature.
- [ ] **Step 5: Commit** `fix(openai-agents): RunHooks tool signatures match the SDK; deprecate run_config`.

### Task 10: CrewAI (spec §4.5)
**Files:** `lctl/integrations/crewai.py` (the step callback ~358–440, with the agent attribution near 378 and the ToolResult handling at 371–428), `tests/integration/test_crewai_live.py` (new), `tests/test_crewai.py`

- [ ] **Step 1: Write the failing live tests,** using a crew with two agents (roles `"researcher"` and `"writer"`) on a stub or fake LLM that needs no network, one tool, and `kickoff()` via `LCTLCrew`. See `$SP/docs-audit/tutorials-b/` for the reviewer's working setup.
  - `test_steps_attributed_to_agent_roles`: no step has agent `"unknown"`, and both roles appear.
  - `test_step_durations_are_measured`: at least one `step_end` has `duration_ms > 0`.
  - `test_no_spurious_tool_result_step`: there is no step whose intent or agent is `ToolResult`.
- [ ] **Step 2:** Run them in `venv-crewai`. Expected: FAIL, as described in the review.
- [ ] **Step 3: Implement** per spec §4.5.
  - Resolve the agent role from the step output, or from the task's assigned agent.
  - Measure durations with `time.monotonic()` between step callbacks for the same agent.
  - Treat `ToolResult` outputs as tool results, attached to the current step, not as steps.
- [ ] **Step 4:** Run the live tests and the default suite.
- [ ] **Step 5: Commit** `fix(crewai): attribute steps to agent roles, real durations, no spurious ToolResult steps`.

### Task 11: VS Code extension (spec §7)
**Files:** `vscode-lctl/src/chainStats.ts` (new: pure helpers), `vscode-lctl/src/extension.ts`, `commands.ts`, `chainProvider.ts`, `webviewPanel.ts` (as needed), `vscode-lctl/test/chainStats.test.mjs` (new), `vscode-lctl/package.json` (the `test` script), `.github/workflows/ci.yml` (the extension job runs `npm test`)

**Interfaces (produces):**
- `chainTokens(events): {input: number, output: number}`, which reads `data.tokens.input`/`output` and falls back to `tokens_in`/`tokens_out`.
- `chainCost(tokens, pricing): number`.
- `limitEvents(events, max): Event[]`.

- [ ] **Step 1: Write the failing tests** with `node:test` and `node:assert` against `out/chainStats.js`. Set `"test": "npm run compile && node --test test/"`, replacing the vscode-test runner, which needs a display.
  - `chainTokens` handles both key shapes and an event with no tokens.
  - `chainCost` uses the given pricing.
  - `limitEvents` respects `max`.
- [ ] **Step 2:** Run `cd vscode-lctl && npm test`. Expected: FAIL; the module doesn't exist yet.
- [ ] **Step 3: Implement** the helpers and use them where tokens and cost are computed.
  - **`lctl.autoRefresh`** toggles the file watchers; listen for `onDidChangeConfiguration`.
  - **`lctl.maxEventsInTree`** replaces the hard-coded 50.
  - **`lctl.defaultPricing`** feeds `chainCost`.
  - **Palette commands with no selection** call `vscode.workspace.findFiles("**/*.lctl.json")` and show `showQuickPick`. If no file is chosen, show an info message instead of an error.
- [ ] **Step 4:** Run `npm test`, `npm run lint`, `npm run compile` and `npm run build:webview`. Expected: all pass and `git status` is clean. Add `- run: npm test` (working directory `vscode-lctl`) to the CI extension job.
- [ ] **Step 5: Commit** `fix(vscode): real token/cost from library traces, working settings, file picker for palette commands`.

### Task 12: Examples (spec §8.1)
**Files:** `examples/*.py` (fix, rename and merge), `examples/README.md` (new), `examples/spans_demo.py` (new), `.github/workflows/ci.yml` (the `integrations` job runs the extra's example)

- [ ] **Step 1: Bring every example to one shape.**
  - The name is `<framework>_example.py`.
  - It runs with no API key using the framework's fake or test model.
  - If the framework is missing, it prints the install command (the git-URL form) and exits 0.
  - It writes `<framework>_trace.lctl.json` into the current directory and prints the path.
- [ ] **Step 2: Fix each example.**
  - `langchain_example.py`: use the `is_available` import, add a real `__main__`, and use a fake chat model.
  - Merge `openai_agent.py` into `openai_agents_example.py`, using `hooks=` and the scripted fake model from Task 9.
  - Rename `pydantic_ai_agent.py` to `pydantic_ai_example.py`, using `TestModel` and `.output`.
  - Rename `semantic_kernel_agent.py` to `semantic_kernel_example.py`, using `kernel_function` with `add_plugin`.
  - `crewai_example.py` actually calls `kickoff()` with the stub LLM from Task 10.
  - `llamaindex_example.py` uses `MockLLM`/`MockEmbedding` and the `Settings` route.
  - `dspy_example.py` runs a real `DummyLM`-backed module instead of hand-made events.
- [ ] **Step 3: Add `spans_demo.py`** (core only). It writes `spans_demo.lctl.json` containing:
  - a planner span with two children;
  - an `asyncio.gather` fan-out of two workers;
  - a tool call;
  - an error caught inside a span.
- [ ] **Step 4: Add `examples/README.md`:** a table of example, extra, install command, run command and what to look at (`lctl trace --tree …`).
- [ ] **Step 5: Verify.**
  - Run each example in its per-extra environment from a scratch dir. Expected: exit 0 and a non-empty trace whose `lctl trace --tree` exits 0.
  - Run `spans_demo.py` in the default env. Expected: a tree with nested and parallel spans.
  - Add the per-cell example step to the CI `integrations` job, using a map from extra to example script in the matrix `include`.
- [ ] **Step 6: Commit** `fix(examples): every example runs keyless; merge duplicates; add spans_demo and examples/README`.

### Task 13: CHANGELOG
**Files:** `CHANGELOG.md`
- [ ] **Step 1: Update `[Unreleased]`.**
  - Under `### Removed`: the AutoGen integration and the `autogen` extra, with the reason.
  - Under `### Fixed`: one line per user-visible fix from Tasks 1–12.
  - Under `### Changed`: the bind default, auth coverage, diff default, the `run_config` deprecation, and webhooks now delivered.
  - Add a `### Breaking` note listing spec §9.
  - Also fix the review nit: the "Fixed" list split by a blank line.
- [ ] **Step 2:** Run `uv run pre-commit run --all-files`, then commit `docs: CHANGELOG for review code fixes`.

### Task 14: Acceptance (spec §11)
- [ ] **Step 1: Default suite.** Run 3.11/3.12/3.14 with `uv run --isolated --python <v> pytest -q`, and 3.13 with `uv run pytest -q`. Expected: all pass and coverage ≥ 78.
- [ ] **Step 2: 3.15 core.** Run the same per-extra-env style command as the CI cell. Expected: pass.
- [ ] **Step 3: Integrations.** For each of the 7 extras, run the integration tests and the example in its env. Expected: all pass, with every example exiting 0.
- [ ] **Step 4: Claude Code end-to-end.** In a temp project with a pre-existing `settings.json`:
  1. Run `lctl claude init`.
  2. Drive the hooks: SessionStart, then Pre and Post `Agent` twice (nested), then Stop, then SessionEnd, as separate processes.

  Expected: the settings are preserved, there is one trace for the session, and `lctl trace --tree` shows the `Agent` spans closed.
- [ ] **Step 5: Extension.** `npm test` passes. Feed a library-produced trace with tokens to `chainTokens`; expected: non-zero.
- [ ] **Step 6: Hooks.** Run `uv run pre-commit run --all-files`; expected: all pass. Then report and invoke superpowers:finishing-a-development-branch.
