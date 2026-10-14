# LCTL Trace Model 4.2: Span IDs — Design

- **Date:** 2026-10-14
- **Status:** Awaiting review
- **Branch:** `feat/trace-spans`
- **Scope:** Sub-project A of the UI-rethink roadmap:
  0. Dashboard stopgap (done)
  1. **A: trace model with span IDs (this document)**
  2. B: event store (JSONL by default, optional PostgreSQL 18)
  3. C: shared React UI for the web dashboard and the VS Code extension

## 1. Context and goal

The redesigned UI (C) has to answer four questions about a run:
- What went wrong, and where?
- What did my agents do, including the call tree and what ran in parallel?
- Was it slow or expensive?
- How do two runs differ?

Today's trace format cannot express the structure behind the second question. An event is `seq, type, timestamp, agent, data`:
- **Steps aren't identified.** They pair only by agent name. A `STEP_END` closes the latest `STEP_START` of the same agent, so two concurrent agents with the same name can't be told apart.
- **Nothing links a step to its parent.** There is no "who spawned whom".
- **`LCTLSession` keeps one `_current_agent` for the whole session.** Concurrent agents in one session (for example under `asyncio.gather`) overwrite each other's attribution.
- **Real Claude Code traces contain unpaired `STEP_END` events**, which today's inference drops silently.

**Goal:** give every event and every step a stable identity, plus an explicit parent link. Do it in a backward-compatible format revision, `"4.2"`, so C can render an exact call tree and parallel lanes, and B can store and index events.

### Decisions already made (with the user, 2026-10-14)
| # | Decision |
|---|---|
| D1 | Extend the trace format with optional span IDs. Old traces fall back to inference. |
| D2 | IDs are UUIDv7. The span model is LCTL-native but maps onto OpenTelemetry: OTel trace and span IDs can be stored as attributes, but OTel is not the core model. |
| D3 | Structure is typed top-level fields on events (one append-only stream), not a separate spans table and not IDs inside `data`. |
| D4 | A chain-level `trace_id` (UUIDv7) sits next to the human-readable `chain.id`. |
| D5 | `session.span()` is a new context manager; `traced_step` becomes a thin wrapper around it. |
| D6 | `lctl trace --tree` is opt-in. Default `lctl trace` output is unchanged. |

## 2. Non-goals
- Storage, streaming or PostgreSQL. That is sub-project B.
- Any UI change. That is C; the stopgap dashboard keeps working unchanged.
- Mapping per-integration native IDs, such as LangChain `run_id`/`parent_run_id`, OpenAI Agents spans, or Claude Agent SDK `parent_tool_use_id`. These are follow-ups. The API in §4 accepts external IDs so they can be added without changing the format.
- Cleaning up the Claude Code tracer: its fact noise, unpaired steps, corrupt JSON writes and merged sessions.
- Importing or exporting OpenTelemetry.
- Any change to state, metrics or bottleneck semantics.

## 3. Data model (format 4.2)

### 3.1 Event fields
All new fields are optional, top-level and nullable. They are written only when present.

| Field | Present on | Meaning |
|---|---|---|
| `id` | every event created by a 4.2 session | UUIDv7 string, unique per event |
| `span_id` | every event created by a 4.2 session | The span the event belongs to. `step_start`: the span it opens. `step_end`: the span it closes. Every other event type: the innermost open span in the recording context, or `null` outside any step. |
| `parent_span_id` | `step_start` only | The enclosing span, or `null` for a root span |

- `span_id` and `parent_span_id` accept **any** UUID string, so integrations can pass native IDs. LCTL itself generates UUIDv7.
- Events in a chain stay ordered by `seq`. UUIDv7's time ordering is for storage and indexing (B), not for ordering events.
- `Event.to_dict()` emits the three keys only when they are not `None`. A chain written without spans is byte-identical to today's output, apart from the header (§3.2).

### 3.2 Chain header
```json
{"lctl": "4.2", "chain": {"id": "my-workflow", "trace_id": "0199e1c2-…"}, "events": [...]}
```
- `Chain.trace_id: str | None`. A new `Chain(...)` gets a UUIDv7 `trace_id`.
- Loading a file without one leaves it `None`. `Chain.save()` assigns one when it is `None`, so every 4.2 file on disk has a `trace_id`. `Chain.load()` never mutates.

### 3.3 Versioning and compatibility
- `CURRENT_VERSION = "4.2"`. Add `VERSION_HISTORY["4.2"] = "Span IDs (event id, span_id, parent_span_id) and chain trace_id"`. The 4.1 → 4.2 migration needs no transformation because the fields are optional.
- **Backward compatibility (4.2 code reading older files):** 3.0–4.1 files load as today. Their structure is inferred (§5.2).
- **Forward compatibility (4.1 code reading 4.2 files):** this already works. `validate_version` rejects only versions below `MIN_SUPPORTED_VERSION`, and `Event.from_dict` reads known keys and ignores the rest. A test pins this behaviour (§7).

### 3.4 IDs: `lctl/core/ids.py`
- `new_id() -> str` returns a lowercase, hyphenated UUIDv7 string.
- On Python 3.14 and later it uses `uuid.uuid7()`.
- On 3.11–3.13 it uses an RFC 9562 §5.7 implementation: a 48-bit Unix-millisecond timestamp, version `0b0111`, 12 random bits, variant `0b10` and 62 random bits, all from `os.urandom`. No new dependency is added.

### 3.5 OpenTelemetry mapping (documentation only)
On a future import or export, OTel `trace_id` and `span_id` are stored in `data["otel"] = {"trace_id": …, "span_id": …}`. LCTL IDs stay LCTL IDs. Nothing in this sub-project implements that.

## 4. Recording API (`LCTLSession`)

### 4.1 Span tracking
- Each session holds a `contextvars.ContextVar` whose value is an **immutable tuple** of open spans. A stack entry is `(span_id, agent)`. Pushing or popping sets a new tuple; a mutable list would be shared by reference across copied contexts and defeat isolation.
- Each asyncio task runs in a copy of its creator's context. Concurrent coroutines therefore see the spans open when they were created, and each gets its own stack after that.
- Code running in a fresh context starts with an empty stack and can pass `parent_span_id` explicitly. On standard CPython builds a new thread is such a context; free-threaded 3.14 builds may make threads inherit context.
- The stack is per session. Two sessions never share one.

### 4.2 Method changes
Every change is a keyword-only addition with a default, so existing names, positional parameters and return values are unchanged.

| Method | Change |
|---|---|
| `step_start(agent, intent, input_summary="", *, span_id=None, parent_span_id=None) -> int` | Opens a span: `span_id or new_id()`, with parent `parent_span_id` or the innermost open span. Pushes it and returns `seq`, as today. |
| `step_end(agent=None, outcome=…, …, *, span_id=None) -> int` | Closes `span_id` if given, otherwise the innermost open span in the context, and pops it (searching the stack when an explicit ID isn't on top). With no open span, it records the event with `span_id=None`, which matches today's tolerance. |
| `tool_call(…, *, span_id=None)`, `llm_trace(…, *, span_id=None)` | Attach to the given span, or to the innermost open span. |
| `add_fact`, `modify_fact`, `error`, `checkpoint`, `stream_start`, `stream_chunk`, `stream_end` | Attach to the innermost open span automatically. Signatures are unchanged. |
| `current_span_id` (new read-only property) | The innermost open span in the current context, or `None`. |
| `span(agent, intent, input_summary="", *, span_id=None, parent_span_id=None)` (new context manager) | Yields `span_id` (see the behaviour below). |

`session.span()` behaves as follows:
- **On entry:** it calls `step_start(...)` and yields the span ID.
- **On a normal exit:** it calls `step_end(outcome="success")` with the measured `duration_ms`.
- **If an exception escapes:** it records `error(...)` and then `step_end(outcome="error")`, and re-raises the original exception.
- **If tracing itself fails:** the failure is suppressed, so it never masks or replaces the user's exception, nor breaks a normal exit.

### 4.3 Attribution
When a method gets no explicit `agent`, the agent comes from the innermost open span in the current context. The session-wide `_current_agent` stays as a fallback only, so sequential single-context use attributes exactly as today.

### 4.4 Single stamping point
`_add_event` stamps `id` (always) and `span_id` (resolved per §4.2) on every event, so redaction, the streaming emitter and the event listeners all see the new fields. Redaction never touches `id`, `span_id` or `parent_span_id`.

### 4.5 Existing helpers and integrations
- `traced_step(session, agent, intent, input_summary="")` becomes a wrapper around `session.span(...)`. Its outward behaviour is unchanged.
- `@traced` keeps using `traced_step`.
- All nine integrations already call `step_start`/`step_end`, so they get spans and automatic parenting without code changes.

## 5. Replay and analysis (`ReplayEngine`)

### 5.1 `get_span_tree() -> list[dict]` (new)
It returns root nodes. Each node has:
- `span_id`, `parent_span_id`, `agent`, `intent`
- `start_seq`, `end_seq` (`null` if the span never closed), `start_time`, `end_time` (ISO)
- `duration_ms`: the step's recorded `duration_ms`, otherwise end minus start time, otherwise `null`
- `outcome`, `event_count` (events carrying this `span_id`), `children` (ordered by `start_seq`), `inferred: bool`

### 5.2 Inference for traces without span IDs
- A `STEP_START` opens a span with ID `inferred-<start_seq>`. Its parent is the innermost open span at that point.
- A `STEP_END` closes the most recent open span of the same agent, which is today's rule.
- An unmatched `STEP_END` becomes an **orphan node**: a root node with `start_seq: null`, `end_seq` set and `inferred: true`. It is never dropped.
- Inferred IDs are deterministic, so they are stable across loads.
- A mixed chain (some events with `span_id`) uses exact IDs where present and inference for the rest.

### 5.3 Unchanged and adjusted
- **`diff()`** excludes `id`, `span_id` and `parent_span_id` when comparing events. Two runs of the same workflow therefore don't differ on every event.
- **`get_trace()`** keeps its keys and adds the span fields when present.
- **Unchanged:** `replay_to`, `replay_all`, `State`, metrics, `find_bottlenecks` and `get_confidence_timeline`.

## 6. CLI and dashboard
- **`lctl trace --tree`** prints the span tree, one line per span: `<indent><agent>: <intent>  [<duration>] <outcome>`. The marker is `(open)` for an unclosed span and `(orphan end)` for an orphan.
- **Without `--tree`**, `lctl trace` output is byte-identical to today's.
- **Dashboard:** the API returns the new fields automatically through `Event.to_dict()`. The dashboard has no UI changes.

## 7. Testing and acceptance
Tests are written first and each is seen to fail before its implementation.

1. **IDs** (`tests/test_ids.py`):
   - version-7 and RFC 9562 variant bits;
   - 10,000 unique IDs;
   - an ID created in a later millisecond sorts after an earlier one;
   - the stdlib path and the forced fallback path both produce valid UUIDv7.
2. **Session spans** (`tests/test_session_spans.py`):
   - nesting and parent links;
   - explicit `span_id` and `parent_span_id`;
   - `step_end` with an explicit ID that is not on top of the stack;
   - every event carries `id`, and non-step events carry the enclosing `span_id`;
   - `span()` success path; `span()` error path (one `ERROR` event, an `outcome="error"` end, the original exception re-raised);
   - `traced_step` and `@traced` produce spans;
   - **two coroutines under `asyncio.gather`** inside an outer span get distinct spans, both parented to the outer span, each with the right agent attribution;
   - code run in a fresh `contextvars.Context()` (as on a new thread) starts with no parent; an explicit `parent_span_id` works.
3. **Span tree** (`tests/test_span_tree.py`):
   - an exact tree for a 4.2 chain;
   - an inferred tree for a 4.1 fixture, including an orphan `STEP_END` modelled on the real Claude Code traces;
   - an unclosed span is kept with `end_seq: null`;
   - inferred IDs are equal across two loads;
   - a mixed chain.
4. **Diff:** two chains equal except for IDs produce `[]`.
5. **Format:**
   - JSON and YAML round-trips keep `id`, `span_id`, `parent_span_id` and `trace_id`;
   - a 4.1 fixture loads with `trace_id is None`, and `save()` assigns one;
   - `VERSION_HISTORY` has `"4.2"`;
   - a 4.2 event dict passed through the 4.1-era `Event.from_dict` path ignores the new keys (forward compatibility).
6. **CLI:**
   - `lctl trace --tree` output on a fixture;
   - `lctl trace` and `lctl stats --json` on `examples/sample-chain.lctl.json` are byte-identical to the pre-change baseline.
7. **Gates:**
   - all existing tests pass, and coverage stays at or above `fail_under`;
   - CI is green on 3.11–3.14 and on the 3.15 core suite;
   - ruff, ruff format and mypy are clean.

**Performance:** one UUIDv7 and one ContextVar lookup per event, a few microseconds. No benchmark gate.

## 8. Documentation
- `docs/api.md`: the new event fields, `trace_id`, `session.span()`, `current_span_id`, `get_span_tree()` and `lctl trace --tree`.
- `CLAUDE.md`: Key Concepts gains a "Spans" subsection; update the Event JSON example.
- `README.md`: one short "Spans" example.
- `CHANGELOG.md`: an `[Unreleased] → Added` entry.

## 9. Follow-ups (recorded, not in scope)
1. Native-ID mapping per integration: LangChain `run_id`/`parent_run_id`, OpenAI Agents spans, CrewAI, the Claude Agent SDK (`parent_tool_use_id`, `agent_id`).
2. Claude Code tracer: emit spans for Task subagents, stop recording file edits and todos as facts, pair START/END, fix the corrupt-JSON writes, and stop appending a later session into the same chain.
3. OpenTelemetry import and export using §3.5.
4. Sub-project B: store and index on `trace_id`, `span_id` and `id`.
