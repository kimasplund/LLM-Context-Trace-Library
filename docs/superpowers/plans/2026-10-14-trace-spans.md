# Trace Model 4.2 (Span IDs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every LCTL event a UUIDv7 `id`, every step a `span_id` with a `parent_span_id`, and every chain a `trace_id`. This happens in a backward-compatible format revision, 4.2, which replay can turn into an exact call tree. Older traces get an inferred one.

**Architecture:**
- **Format:** optional typed top-level fields on `Event`, plus `trace_id` on `Chain`.
- **Recording:** a per-session `ContextVar` holding an immutable tuple stack of open spans, so asyncio tasks get isolated parents. `_add_event` is the single place that stamps IDs.
- **Replay:** `ReplayEngine.get_span_tree()` builds nodes from exact IDs or by inference.
- **CLI:** `lctl trace --tree` renders the tree.

**Tech Stack:** Python ≥3.11 (stdlib `uuid`, `contextvars`, `os`), pytest with pytest-asyncio in strict mode, click.

**Spec:** `docs/superpowers/specs/2026-10-14-trace-spans-design.md`. It includes the corrections from commit `218f8eb`. Read it first: §3 is the format, §4 the API, §5 the replay rules.

## Global Constraints
- **Optional fields:** all new fields are optional, and keyword-only where they are parameters. Existing method names, positional parameters and return values (`seq`) do not change.
- **`Event.to_dict()`** emits `id`, `span_id` and `parent_span_id` only when they are not `None`. `Chain.to_dict()` emits `chain.trace_id` only when it is not `None`.
- **Version:** `CURRENT_VERSION = "4.2"`, with `VERSION_HISTORY["4.2"] = "Span IDs (event id, span_id, parent_span_id) and chain trace_id"`. Register a no-op migration `4.1 → 4.2`, mirroring `_migrate_4_to_4_1`.
- **IDs:** `new_id()` returns lowercase, hyphenated UUIDv7. `span_id` and `parent_span_id` accept any UUID string.
- **Redaction** never alters `id`, `span_id` or `parent_span_id`.
- **Tracing never breaks user code.** `session.span()` and `traced_step` must never mask, replace or swallow the user's exception.
- **Event order on the error path** stays `STEP_START`, `STEP_END(outcome="error")`, `ERROR`.
- **Behaviour baseline:**
  - `lctl trace examples/sample-chain.lctl.json` (no `--tree`) must stay byte-identical to `$SP/baseline/trace.before.txt`.
  - `lctl stats --json` must stay identical to `$SP/baseline/stats.before.txt` except `"version": "4.2"`.
  - `SP=/tmp/claude-1000/-home-kim-projects-llm-context-transfer/783deef4-54f3-49cc-81f2-f407fcbaf8ae/scratchpad`.
- **Gates for every task:**
  - `uv run pytest -q` passes with coverage at or above `fail_under` (78).
  - `uv run ruff check . && uv run ruff format --check . && uv run mypy` are clean.
  - The pre-commit hook passes on commit.
- **Commit trailer:** end every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01U7JQRWVBpRadDaH4RESMtZ
  ```
- **Never push.** Integration is decided with the user at the end.

## Review Focus
1. **Concurrency.** Spans opened inside tasks spawned with `asyncio.gather` or `create_task`, and `asyncio.to_thread` (which copies context), must not leak into, or pop, a sibling's stack. A `ContextVar` holding a *mutable* list would. The `gather` test in Task 3 pins this.
2. **A user exception while tracing is broken.** If `session.step_end` or `error` raises inside `span()`'s error path, the user's original exception must still propagate unchanged. Task 4 pins this with a session whose `_add_event` raises.
3. **Malformed old traces:** an `END` without a `START`, a `START` that never ends, two same-name agents interleaved. None of them may crash `get_span_tree()` or drop events. Task 5 pins these.
4. **Redaction is on by default** and must not rewrite the ID fields. Task 3 pins this with an ID-shaped value in the data next to real IDs.
5. **Diffing two runs:** identity fields must not make every event differ. Task 5 pins this.

---

### Task 1: UUIDv7 IDs

**Files:**
- Create: `lctl/core/ids.py`
- Test: `tests/test_ids.py`

**Interfaces:**
- Produces: `lctl.core.ids.new_id() -> str`, plus the private `_uuid7_fallback() -> uuid.UUID` used on Python < 3.14 and in tests.

- [ ] **Step 1: Write the failing tests** in `tests/test_ids.py`:
  - `test_new_id_is_uuid7`: parse with `uuid.UUID(new_id())`. Assert `.version == 7`, `.variant == uuid.RFC_4122`, and `str(u) == new_id_value` (lowercase and hyphenated).
  - `test_new_id_unique`: 10,000 values give 10,000 distinct strings.
  - `test_later_millisecond_sorts_later`: `a = new_id()`, `time.sleep(0.002)`, `b = new_id()`, then `assert a < b`.
  - `test_fallback_generates_uuid7`: `u = _uuid7_fallback()` has `.version == 7` and the RFC 4122 variant, and its top 48 bits are within ±2000 ms of `time.time_ns() // 1_000_000`.
  - `test_fallback_used_without_stdlib_uuid7`: `monkeypatch.delattr(uuid, "uuid7", raising=False)`, then `new_id()` still returns a valid v7.
- [ ] **Step 2:** Run `uv run pytest tests/test_ids.py --no-cov -q`. Expected: FAIL with `ModuleNotFoundError: lctl.core.ids`.
- [ ] **Step 3: Implement `lctl/core/ids.py`:**
  - `new_id()` returns `str(uuid.uuid7())` if `hasattr(uuid, "uuid7")`, otherwise `str(_uuid7_fallback())`.
  - The fallback follows RFC 9562 §5.7: `ms = time.time_ns() // 1_000_000`, `rand = int.from_bytes(os.urandom(10), "big")`, and `value = (ms & (2**48 - 1)) << 80 | 0x7 << 76 | ((rand >> 62) & 0xFFF) << 64 | 0b10 << 62 | (rand & (2**62 - 1))`. Return `uuid.UUID(int=value)`.
- [ ] **Step 4:** Run the same command. Expected: 5 passed.
- [ ] **Step 5: Commit** with `feat(core): Add UUIDv7 id generator`.

### Task 2: Format 4.2: event fields, `trace_id`, schema version

**Files:**
- Modify: `lctl/core/events.py` (`Event` around line 101, `Chain` around line 164), `lctl/core/schema.py`
- Test: `tests/test_schema.py` (update the hard-coded `"4.1"` assertions), `tests/test_events.py` (new tests)

**Interfaces:**
- Consumes: `new_id()` from Task 1.
- Produces:
  - `Event(seq, type, timestamp, agent, data=…, id: str | None = None, span_id: str | None = None, parent_span_id: str | None = None)`. The new fields come after `data`, so positional construction is unchanged.
  - `Chain(id, events=…, version=CURRENT_VERSION, trace_id: str | None = field(default_factory=new_id))`.
  - `Chain.save()` assigns `trace_id` when it is `None`.

- [ ] **Step 1: Write the failing tests** in `tests/test_events.py`, as a new class `TestFormat42`:
  - **`test_event_roundtrip_keeps_span_fields`:** `Event(..., id=new_id(), span_id=X, parent_span_id=Y)` survives `to_dict` then `from_dict` with all three fields equal.
  - **`test_event_to_dict_omits_absent_span_fields`:** an `Event` without them has `to_dict().keys() == {"seq","type","timestamp","agent","data"}`.
  - **`test_new_chain_has_uuid7_trace_id`:** `uuid.UUID(Chain(id="c").trace_id).version == 7`.
  - **`test_chain_json_and_yaml_roundtrip_trace_id`:** do the round-trip through `tmp_path / "c.lctl.json"` and `"c.lctl.yaml"` with `save`/`load`. Assert `trace_id` and the event span fields are preserved.
  - **`test_old_file_loads_without_trace_id_and_save_assigns_one`:** write `{"lctl": "4.1", "chain": {"id": "old"}, "events": []}` and load it. Assert `trace_id is None` and `version == "4.2"`. After `save()`, the reloaded file has a v7 `trace_id`.
  - **`test_forward_compat_unknown_event_keys_ignored`:** `Event.from_dict({...valid 4.1 keys..., "id": "x", "span_id": "y", "future_field": 1})` succeeds. The forward-compatibility guarantee rests on this ignore-unknown-keys behaviour, so it is pinned.
  - **In `tests/test_schema.py`:** change `assert CURRENT_VERSION == "4.1"` to `"4.2"`, add `assert "4.2" in VERSION_HISTORY`, and add `test_migration_path_4_1_to_4_2_is_single_step`. Update the 3.0→current path test to expect the extra `("4.1","4.2")` step. Leave the assertions about the 4.0→4.1 behaviour alone.
- [ ] **Step 2:** Run `uv run pytest tests/test_events.py tests/test_schema.py --no-cov -q`. Expected: FAIL, because the fields, `trace_id` and version 4.2 don't exist yet.
- [ ] **Step 3: Implement:**
  - **`Event`:** add the three fields. In `to_dict`, add the keys when they are not `None`. In `from_dict`, pass `id=d.get("id")`, `span_id=d.get("span_id")` and `parent_span_id=d.get("parent_span_id")`.
  - **`Chain`:** add `trace_id`. In `to_dict`, use `"chain": {"id": ..., **({"trace_id": t} if t else {})}`. In `from_dict`, pass `trace_id=chain_meta.get("trace_id")`. In `save()`, set `self.trace_id = new_id()` first when it is `None`.
  - **`schema.py`:** update `CURRENT_VERSION` and `VERSION_HISTORY`, and register `@SchemaMigrator.register("4.1", "4.2") def _migrate_4_1_to_4_2(data): return data.copy()` with a docstring.
- [ ] **Step 4:** Run the Step 2 command, then the full `uv run pytest -q`. Expected: all pass. If any other test pinned `"4.1"`, it is a version assertion; update it to `CURRENT_VERSION` and note it in the commit.
- [ ] **Step 5: Baseline check:**
  ```bash
  uv run lctl stats --json examples/sample-chain.lctl.json | diff - $SP/baseline/stats.before.txt
  ```
  Expected: only the `"version": "4.1"` → `"4.2"` line differs.
  ```bash
  uv run lctl trace examples/sample-chain.lctl.json | diff - $SP/baseline/trace.before.txt
  ```
  Expected: empty.
- [ ] **Step 6: Commit** with `feat(core): Trace format 4.2 with optional event span fields and chain trace_id`.

### Task 3: Session span tracking

**Files:**
- Modify: `lctl/core/session.py` (`__init__` line 38, `_add_event` line 86, `step_start`/`step_end` around lines 139–168, `tool_call` around 201, `llm_trace` around 218, `error` around 241)
- Test: `tests/test_session_spans.py` (new)

**Interfaces:**
- Consumes: `new_id()`, and the `Event` fields from Task 2.
- Produces:
  - `step_start(agent, intent, input_summary="", *, span_id=None, parent_span_id=None) -> int`
  - `step_end(agent=None, outcome="success", output_summary="", duration_ms=0, tokens_in=0, tokens_out=0, *, span_id=None) -> int`
  - `tool_call(..., agent=None, *, span_id=None)`, `llm_trace(..., *, span_id=None)`, `error(..., *, span_id=None)`
  - the read-only property `current_span_id -> str | None`
  - Task 4 relies on all of these.

- [ ] **Step 1: Write the failing tests** in `tests/test_session_spans.py`:
  - **`test_every_event_gets_uuid7_id`:** each of `step_start`, `tool_call`, `add_fact`, `llm_trace`, `error`, `checkpoint` and `step_end` yields an event whose `id` parses as v7. All the IDs are distinct.
  - **`test_nested_steps_link_parents`:** outer `step_start` A, inner `step_start` B. B's `parent_span_id` is A's `span_id`, and A's is `None`. Inner `step_end` carries B's `span_id`, outer `step_end` carries A's.
  - **`test_non_step_events_carry_enclosing_span`:** a `tool_call`/`add_fact` inside B has B's `span_id`. After B ends and before A ends, the same calls have A's `span_id`. Outside any step, `span_id is None`.
  - **`test_explicit_span_and_parent_ids_are_honoured`:** `step_start(..., span_id=S, parent_span_id=P)` records exactly S and P, and `current_span_id == S`.
  - **`test_step_end_explicit_span_not_on_top`:** open A, then B. `step_end(span_id=A)` closes A and leaves `current_span_id == B`. The A end event's agent is A's agent.
  - **`test_explicit_span_attribution`:** `error(..., span_id=A)` after A has ended has `span_id == A` and `agent == A's agent`.
  - **`test_redaction_keeps_ids`:** with redaction on (the default), `add_fact("f", "api_key=sk-abcdefghijklmnopqrstuvwx")` keeps a v7 `id` and the enclosing `span_id` unchanged.
  - **`test_gather_tasks_get_isolated_spans`** (`@pytest.mark.asyncio`):
    - Open an outer span O.
    - Run `asyncio.gather(work("a"), work("b"))`, where `work(name)` does `step_start(name, ...)`, `await asyncio.sleep(0)`, `tool_call(...)`, `await asyncio.sleep(0)` and `step_end()`.
    - Assert both inner spans have parent O.
    - Assert each tool call carries its own task's `span_id` and agent.
    - Assert each `step_end` closes its own span.
    - Assert `current_span_id == O` afterwards.
  - **`test_fresh_context_starts_without_parent`:** inside `contextvars.Context().run(fn)`, a `step_start` has `parent_span_id is None` even while O is open outside. With an explicit `parent_span_id=O` it is O.
- [ ] **Step 2:** Run `uv run pytest tests/test_session_spans.py --no-cov -q`. Expected: FAIL. Events have no `id` or `span_id`, and the keyword arguments are unknown.
- [ ] **Step 3: Implement** in `session.py`:
  - **`__init__`:**
    - `self._span_stack: contextvars.ContextVar[tuple[tuple[str, str], ...]] = contextvars.ContextVar(f"lctl_span_stack_{id(self)}", default=())`, with entries `(span_id, agent)`.
    - `self._span_agents: dict[str, str] = {}`, which maps span IDs to agents for every span started.
  - **`current_span_id`:** return `stack[-1][0]` if the stack is non-empty, else `None`.
  - **`_add_event(event_type, agent, data, *, span_id=_UNSET, parent_span_id=None)`:** it resolves `span_id` as follows:
    - `_UNSET` means the current span, `self.current_span_id`.
    - An explicit value, including `None`, is used as given.
    - It then sets `id=new_id()`, `span_id` and `parent_span_id` on the `Event`.
    - Redaction stays on `data` only.
  - **`step_start`:** `sid = span_id or new_id()` and `parent = parent_span_id if parent_span_id is not None else self.current_span_id`. Record `_span_agents[sid] = agent`, call `_add_event(..., span_id=sid, parent_span_id=parent)`, then push `(sid, agent)` with `self._span_stack.set(stack + ((sid, agent),))`. Keep setting `_current_agent = agent`.
  - **`step_end`:**
    - Pick the target span: `span_id` if given, else the top of the stack, else `None`.
    - Resolve the agent: explicit `agent`, then `_span_agents.get(target)`, then `_current_agent`, then `"unknown"`.
    - Call `_add_event(..., span_id=target)`. Remove the entry with that span ID from the stack, wherever it sits, by setting a new tuple.
    - Keep `_current_agent = None` as today.
  - **`tool_call`, `llm_trace`, `error`:** when given `span_id`, the agent is the explicit `agent` (for `tool_call`), then `_span_agents.get(span_id)`, then today's rule. Pass `span_id` through.
  - **Agent attribution for every other method:** when no explicit agent is given, use the agent of the top stack entry before falling back to `_current_agent`, then `"unknown"`.
- [ ] **Step 4:** Run the Step 2 command, then the full suite. Expected: all pass, including the existing `test_session.py` and the integration tests, unchanged.
- [ ] **Step 5: Commit** with `feat(session): Track spans per execution context and stamp event ids`.

### Task 4: `session.span()`, `traced_step` and `@traced`

**Files:**
- Modify: `lctl/core/session.py` (new method; `traced_step` around line 368), `lctl/__init__.py` (`traced` around line 62)
- Test: `tests/test_session_spans.py` (append)

**Interfaces:**
- Consumes: the Task 3 API.
- Produces: `LCTLSession.span(agent, intent, input_summary="", *, span_id=None, parent_span_id=None)`, a `contextlib.contextmanager` that yields `str`.

- [ ] **Step 1: Write the failing tests:**
  - **`test_span_success`:** `with session.span("a", "do") as sid:` makes `sid == session.current_span_id` inside the block. Afterwards the events are `[STEP_START(span_id=sid), STEP_END(outcome="success", span_id=sid, data.duration_ms >= 0)]`.
  - **`test_span_error_keeps_order_and_attribution`:** `with pytest.raises(RuntimeError, match="boom"), session.span("a", "do") as sid: raise RuntimeError("boom")`. The event types are `[STEP_START, STEP_END, ERROR]`. The end has `outcome == "error"`, and the `ERROR` has `span_id == sid`, `agent == "a"` and `data["type"] == "RuntimeError"`.
  - **`test_span_never_masks_user_exception_when_tracing_breaks`:** monkeypatch `session._add_event` to raise `OSError("disk full")` after the first call. The user's `ValueError("user")` raised inside the span propagates unchanged; `pytest.raises(ValueError, match="user")`.
  - **`test_span_tracing_failure_on_normal_exit_is_silent`:** the same monkeypatch, but with no user exception. The block completes and nothing propagates.
  - **`test_traced_step_produces_span`:** `traced_step(session, "a", "x")` gives start and end events sharing one `span_id`. The existing `tests/test_session.py::test_traced_step_on_exception` must also still pass unchanged.
  - **`test_traced_decorator_produces_span_and_error_attribution`:** with `lctl.auto_instrument()`, a `@lctl.traced("a", "x")` function that raises gives `STEP_START`, `STEP_END(error)` and `ERROR` with a shared `span_id`, and the `ERROR` agent is `"a"`. Clean up `lctl._global_session` afterwards.
- [ ] **Step 2:** Run `uv run pytest tests/test_session_spans.py --no-cov -q -k "span_ or traced"`. Expected: FAIL. `span` doesn't exist, and `@traced`'s error agent is `"unknown"` with the old `_current_agent` logic.
- [ ] **Step 3: Implement:**
  - **`span()`:**
    1. `start = time.time()`.
    2. Try `sid = self.step_start(...)`. If that raises, yield `span_id or ""` and run the block untraced, then return.
    3. In `try: yield sid`.
    4. On `except BaseException as exc`, with tracing errors suppressed: `step_end(span_id=sid, outcome="error", duration_ms=…)`, then, only if `isinstance(exc, Exception)`, `error(category="execution_error", error_type=type(exc).__name__, message=str(exc), recoverable=False, span_id=sid)`. Then `raise`.
    5. On `else`, with tracing errors suppressed: `step_end(span_id=sid, outcome="success", duration_ms=…)`.
    - `sid = self.current_span_id` immediately after `step_start(...)` returns, since `step_start` has just pushed it in this context.
    - `GeneratorExit` and `KeyboardInterrupt` close the span as `"error"` but record no `ERROR` event.
  - **`traced_step(session, agent, intent, input_summary="")`:** the body becomes `with session.span(agent, intent, input_summary): yield`. Keep its docstring.
  - **`traced`:** the wrapper body becomes `with session.span(agent, intent, input_summary): return func(*args, **kwargs)`. Keep `wrapper.__name__` and `__doc__` as today.
- [ ] **Step 4:** Run `uv run pytest tests/test_session_spans.py tests/test_session.py tests/test_init.py --no-cov -q`, then the full suite. Expected: all pass.
- [ ] **Step 5: Commit** with `feat(session): Add session.span(); traced_step and @traced build on it`.

### Task 5: Span tree, inference and identity-blind diff

**Files:**
- Modify: `lctl/core/events.py` (`ReplayEngine`: new `get_span_tree`; `get_trace`; `diff`)
- Create: `tests/fixtures/spans_v41_orphans.lctl.json`, a 4.1 chain with these events:
  1. `START planner`
  2. `START coder`
  3. `TOOL coder`
  4. `END coder`
  5. `END reviewer` (orphan: no start)
  6. `START tester` (never ends)
  7. `END planner`

  Every event has a timestamp one second after the previous one. The `END coder` event has `duration_ms: 2000`.
- Test: `tests/test_span_tree.py`

**Interfaces:**
- Consumes: the `Event` span fields (Task 2) and the session API (Tasks 3–4) for building exact chains in tests.
- Produces: `ReplayEngine.get_span_tree() -> list[dict]`. Each node has `span_id`, `parent_span_id`, `agent`, `intent`, `start_seq`, `end_seq`, `start_time`, `end_time`, `duration_ms`, `outcome`, `event_count`, `children` and `inferred`. Task 6 relies on these keys.

- [ ] **Step 1: Write the failing tests** in `tests/test_span_tree.py`:
  - **`test_exact_tree_from_session`:** a session records `span(planner)` containing `span(coder)` (with one `tool_call`) and then `span(tester)`, both inside `planner`. The tree has one root, `planner`, with children `[coder, tester]` in `start_seq` order. `coder.event_count == 1`, every node has `inferred is False`, and the `span_id`s match the events.
  - **`test_inferred_tree_for_v41_fixture`:** load the fixture.
    - The roots, ordered by `(start_seq or end_seq)`, are `planner` and the orphan `reviewer`. The orphan has `start_seq is None`, `end_seq == 5`, `span_id == "orphan-5"` and `inferred is True`.
    - `planner` has children `[coder, tester]`. `coder` has `span_id == "inferred-2"`, `duration_ms == 2000` and `event_count == 1`. `tester` has `end_seq is None` and `duration_ms is None`.
  - **`test_inferred_ids_stable_across_loads`:** two `Chain.load` calls of the fixture give an identical `get_span_tree()`.
  - **`test_mixed_chain_uses_exact_ids_where_present`:** a chain with one span recorded through the 4.2 session API, followed by appended 4.1-style START/END events without IDs, gives an exact node and an `inferred-<seq>` node.
  - **`test_interleaved_same_name_agents_v41`:** a 4.1-style chain `START a`(1), `START a`(2), `END a`(3), `END a`(4) infers `inferred-1` containing `inferred-2`. `END` 3 closes span 2, `END` 4 closes span 1, and nothing is dropped or crashes.
  - **`test_duration_falls_back_to_timestamps`:** with an end whose `duration_ms` is 0, `duration_ms` is end time minus start time in ms.
  - **`test_diff_ignores_identity_fields`:** two sessions record the identical workflow, and their timestamps are set equal afterwards. `ReplayEngine(a).diff(ReplayEngine(b)) == []`, even though every `id`, `span_id` and `trace_id` differs.
  - **`test_get_trace_includes_span_fields_when_present`:** a 4.2 chain's `get_trace()` entries have `span_id`, and the fixture's entries don't.
- [ ] **Step 2:** Run `uv run pytest tests/test_span_tree.py --no-cov -q`. Expected: FAIL, because `get_span_tree` doesn't exist.
- [ ] **Step 3: Implement `get_span_tree`** with one pass over `self.chain.events` in `seq` order:
  ```text
  nodes = {}; roots = []; open_ids = []            # open_ids: span ids in open order
  for e in events:
    if START:
      sid = e.span_id or f"inferred-{e.seq}"
      parent = e.parent_span_id if e.span_id else (open_ids[-1] if open_ids else None)
      node = new node(start fields, inferred = e.span_id is None, children=[], event_count=0)
      attach node to nodes[parent].children if parent in nodes else roots
      nodes[sid] = node; open_ids.append(sid)
    elif END:
      sid = e.span_id if (e.span_id in nodes) else last id in open_ids whose node is
            inferred and node.agent == e.agent (search from the end)
      if sid is None: node = orphan root (span_id=f"orphan-{e.seq}", start_* None, inferred=True); roots.append(node)
      else: node = nodes[sid]; open_ids.remove(sid)
      set end_seq/end_time/outcome; duration_ms = data.duration_ms if > 0
            else (end_time - start_time) in ms if start_time else None
    else:
      owner = e.span_id if e.span_id in nodes else (open_ids[-1] if open_ids else None)
      if owner: nodes[owner].event_count += 1
  ```
  - Nodes left open keep `end_seq`, `end_time`, `duration_ms` and `outcome` as `None`.
  - A parent ID not present in the chain makes the node a root, but its `parent_span_id` is kept as recorded.
  - **`diff`:** compare `to_dict()` with the keys `id`, `span_id` and `parent_span_id` removed. The output shape is unchanged.
  - **`get_trace`:** add `span_id`/`parent_span_id` to an entry only when the event has them.
- [ ] **Step 4:** Run the Step 2 command and then the full suite. Expected: all pass.
- [ ] **Step 5: Commit** with `feat(replay): Span tree with inference for older traces; diff ignores identity fields`.

### Task 6: `lctl trace --tree`

**Files:**
- Modify: `lctl/cli/main.py` (the `trace` command at line 269)
- Test: `tests/test_cli.py`, new class `TestTraceTree`

**Interfaces:**
- Consumes: `get_span_tree()` node keys (Task 5).

- [ ] **Step 1: Write the failing tests:**
  - **`test_trace_tree_renders_fixture`:** `runner.invoke(cli, ["trace", "--tree", <fixture path>])` exits 0, and the output contains these lines in order:
    ```text
    planner: <intent>  [6.0s] success
      coder: <intent>  [2.0s] success
      tester: <intent>  (open)
    reviewer  (orphan end)
    ```
    Use the fixture's actual intents, and assert on the stripped lines. `NO_COLOR` is not needed because `CliRunner` strips styles.
  - **`test_trace_without_tree_unchanged`:** `lctl trace` on `examples/sample-chain.lctl.json` produces output equal to `$SP/baseline/trace.before.txt`'s content. Embed that expected text inline in the test rather than reading `$SP`.
- [ ] **Step 2:** Run `uv run pytest tests/test_cli.py -k TraceTree --no-cov -q`. Expected: FAIL with `no such option: --tree`.
- [ ] **Step 3: Implement:**
  - Add `@click.option("--tree", is_flag=True, help="Show the span tree (call hierarchy) instead of the step list.")`.
  - With `--tree`, walk `get_span_tree()` depth-first, printing two spaces per depth level. Each line is `f"{agent}: {intent}  [{duration_ms/1000:.1f}s] {outcome}"`. Styling and markers:
    - **Outcome colours:** `success` is green and `error` is red, as in the existing output.
    - **Open span:** `f"{agent}: {intent}  (open)"`.
    - **Orphan:** `f"{agent}  (orphan end)"`.
    - **No spans:** `No step events found.`
  - Without `--tree`, the code path is untouched.
- [ ] **Step 4:** Run the Step 2 command, then the full suite. Then the Task 2 Step 5 baseline commands. Expected: the same results as Task 2.
- [ ] **Step 5: Commit** with `feat(cli): lctl trace --tree shows the span call tree`.

### Task 7: Documentation

**Files:**
- Modify: `docs/api.md`, `CLAUDE.md`, `README.md`, `CHANGELOG.md`

- [ ] **Step 1: `docs/api.md`:**
  - Add the new `Event` fields, `Chain.trace_id`, `LCTLSession.span()`, `current_span_id`, and the keyword-only `span_id`/`parent_span_id` parameters.
  - Add `ReplayEngine.get_span_tree()` with its node keys, and `lctl trace --tree`.
  - Document the OpenTelemetry mapping convention (spec §3.5): OTel IDs go in `data["otel"]`, LCTL IDs stay LCTL's. This is documentation only.
  - Update the JSON format example to `"lctl": "4.2"`, with `trace_id` and one event showing `id`/`span_id`/`parent_span_id`.
- [ ] **Step 2: `CLAUDE.md`:**
  - In "Key Concepts", add a "Spans" subsection of no more than 15 lines: format 4.2, `session.span()`, automatic parenting per asyncio task, and inference for old traces.
  - Update the "Chain File" JSON example to 4.2. Add `lctl trace --tree` to Quick Commands.
- [ ] **Step 3: `README.md`:** add a short "Spans" example to the Quick Start: nested `with session.span(...)` and then `lctl trace --tree`.
- [ ] **Step 4: `CHANGELOG.md`:** add `[Unreleased] → Added` entries for trace format 4.2 (event `id`, `span_id`, `parent_span_id`, chain `trace_id`), `session.span()`, `get_span_tree()` and `lctl trace --tree`. Add a `Fixed` entry: errors recorded by `traced_step`/`@traced` were attributed to agent "unknown".
- [ ] **Step 5:** Run `uv run pre-commit run --all-files` (it formats Python code blocks in Markdown), then `uv run pytest -q`. Expected: all pass.
- [ ] **Step 6: Commit** with `docs: Document trace format 4.2 spans`.

### Task 8: Acceptance verification

- [ ] **Step 1:** Run `uv run pytest -q`. Expected: everything passes and coverage is at or above 78.
- [ ] **Step 2:** For `v` in 3.11, 3.12 and 3.14, run `uv run --isolated --python $v pytest -q`. Expected: every run passes with coverage at or above 78.
- [ ] **Step 3:** Run the 3.15 core suite with `UV_PROJECT_ENVIRONMENT=$SP/venv315 UV_PYTHON=3.15 uv sync --locked --no-default-groups --group test && UV_PROJECT_ENVIRONMENT=$SP/venv315 UV_PYTHON=3.15 uv run --no-sync pytest --no-cov -q`, from a git checkout. Expected: everything passes.
- [ ] **Step 4:** Run `uv run pre-commit run --all-files`. Expected: all hooks pass.
- [ ] **Step 5:** Repeat the Task 2 Step 5 baseline checks. Expected: unchanged.
- [ ] **Step 6: Real-trace smoke test.**
  - Run `uv run lctl trace --tree` on each loadable file in `$SP/uitraces/`. Expected: exit 0, with orphan and open markers where those traces have them.
  - Run `uv run lctl trace --tree examples/sample-chain.lctl.json`. Expected: an inferred tree.
- [ ] **Step 7:** Report the results, then invoke superpowers:finishing-a-development-branch.
