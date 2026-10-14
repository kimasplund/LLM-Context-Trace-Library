# Documentation Deep Review — 2026-10-14

**Scope:** README, CHANGELOG, LLM-CONTEXT-TRANSFER.md, docs/api.md, CLAUDE.md, the VS Code extension docs, all 9 tutorials, and examples/.

**Method:** five independent reviewers checked every claim against the source and ran the examples. They also ran the tutorial code against the real frameworks, installed in scratch venvs with fake or test models: openai-agents 0.23.1, langchain 1.4.5, ag2 0.14/1.1, dspy 3.4.1, llama-index-core 0.12/0.14, crewai 1.15, pydantic-ai 1.107/2.56 and semantic-kernel 1.45. Detailed per-area tables with file:line evidence: [readme](readme.md) · [api](api.md) · [devdocs](devdocs.md) · [tutorials-a](tutorials-a.md) · [tutorials-b](tutorials-b.md).

| Area | Findings | Critical | Major | Minor |
|---|---|---|---|---|
| README, CHANGELOG, spec | 44 | 3 | 21 | 19 |
| docs/api.md | 46 | 1 | 21 | 24 |
| CLAUDE.md, extension docs | 37 | 4 | 17 | 16 |
| Tutorials: claude-code, langchain, openai-agents, autogen | 54 | 8 | 22 | 24 |
| Tutorials: crewai, llamaindex, dspy, pydantic-ai, semantic-kernel; examples | 40 | 11 | 20 | 9 |
| **Total** | **221** | **27** | **101** | **92** |

## Key conclusion
**Most critical findings are code bugs, not prose.** The docs cannot become correct until the code they describe works. Fix order: code first (W1), then docs (W2).

## W1 — Code bugs found by the review (fix first)
| # | Area | Bug | Effect |
|---|---|---|---|
| 1 | Claude Code tracer | Generated hooks match `tool_name == "Task"`, but current Claude Code (2.1.299) names the subagent tool **`Agent`** (verified by a live hook-payload probe) | No subagent spans or completions are recorded at all |
| 2 | Claude Code tracer | `Stop.sh` crashes on every run (`name 'event_count' is not defined`, a bash quoting bug), and Stop fires per turn, not at session end | The state file is never cleaned up |
| 3 | `lctl claude init` | `--force` replaces the whole `.claude/settings.json`, deleting permissions and other keys; plain `init` drops the user's own hooks | **Destroys user configuration** |
| 4 | LlamaIndex | `LCTLLlamaIndexCallback` never calls `BaseCallbackHandler.__init__` (`AttributeError`). `trace_query_engine`/`trace_chat_engine` raise a `trace_map` setter error. The tests mock the base class away | The integration is unusable |
| 5 | PydanticAI | `TracedAgent.run()` raises `TypeError` on pydantic-ai 2.x (`usage` became a property; `.data` became `.output`). Streaming stores cumulative text repeatedly. There is no `__getattr__` proxy | The integration is unusable on current pydantic-ai |
| 6 | OpenAI Agents | `LCTLRunHooks.on_tool_start` signature crashes on the first real tool call. The documented `run_config=tracer.run_config` raises `TypeError` (`hooks=tracer.create_hooks()` is the working form) | Tool runs crash |
| 7 | AutoGen | Installing `lctl[autogen]` or `autogen-agentchat` selects "modern" mode, where the documented wrappers raise. `pip install ag2` gives 1.x (unsupported). A shared `chain_id` doesn't merge sessions | The documented path fails |
| 8 | VS Code extension | It reads `tokens_in`/`tokens_out`, but the library writes `tokens.{input,output}`. Three declared settings are never read. Palette commands error with "no chain selected" | Tokens and cost always show 0 |
| 9 | Dashboard | `LCTL_REQUIRE_API_KEY` protects only `/api/rpa/*`. Core endpoints, `/ws` and SSE stay open, while the README claims dashboard auth. `lctl dashboard` binds `0.0.0.0` by default. Webhooks are stored but never sent. `run_dashboard(reload=True)` exits with code 3 | Security overclaim and a dead feature |
| 10 | Core API | `Chain.load()`/`Chain.save()` crash on a `str` path. Docs and tutorials use strings in at least 8 places | Many examples crash |
| 11 | Core API | `ReplayEngine.diff()` still compares timestamps and durations, so any two real runs differ on every event | Diff is useless for its main purpose |
| 12 | Core API | `auto_instrument()` instruments nothing; it only creates a global session. The docs claim zero-config framework auto-instrumentation | Overclaim |
| 13 | Examples | `langchain_example.py` dies on import (`langchain_available`). Three of four `*_agent.py` scripts crash with the framework installed. `crewai_example.py` exports 0 events | Broken examples |

## W2 — Documentation rewrite (after W1)
- **Everywhere:**
  - **Stale facts:** "Python 3.9+" should be ≥3.11. `pip install lctl` doesn't work (not on PyPI; use the git URL). "v4.0" should be package 4.1.0 / trace format 4.2.
  - **Invented CLI output:** `trace`/`stats`/`bottleneck`/`replay`/`validate`/`status` samples in the README and all tutorials don't match real output. Regenerate them from real runs.
  - **Dead link** `../cli/README.md`; "Dashboard" headings over CLI-only sections.
- **README:**
  - Restructure: prerequisites; quick start ending in a real CLI step; an integrations table (status, extra, tutorial, example) linking all tutorials and examples; a dashboard section split into features, running, security and the RPA API.
  - Correct the auth claims and the license wording ("-only").
- **docs/api.md:**
  - Fix the crashing examples.
  - Document the missing areas: streaming, metrics, evaluation, redaction (**on by default**), schema/ids, 8 of 13 CLI commands, about 20 dashboard endpoints, and DSPy/LlamaIndex.
  - Fix the ToC anchors.
  - Split into per-area pages.
- **CLAUDE.md:**
  - Cut to about 350 lines.
  - Fix the Integration Guide template (it fails ruff and imports undefined names), the nonexistent `_truncate` pattern, the unimplemented confidence-threshold table, and the 7-of-13 event types.
- **LLM-CONTEXT-TRANSFER.md:** a frozen v4.0 design. It documents commands and modules that don't exist (`lctl validate`/`convert`, `lctl.claude`) and unimplemented features (decay, consensus, contracts). Split it into a normative format spec (4.0 → 4.2) and a clearly labelled roadmap.
- **CHANGELOG:** backfill the omitted 2026-01-19 and 2026-01-23 work (RPA API, redaction, LRU cache, schema versioning, BaseTracer migration, security fixes); create the missing `v4.0.0`/`v4.1.0` tags so the compare links resolve; fix the headings and dates; note the removed `[dev]` extra.
- **VS Code docs:** fix the build recipe (it produces no .vsix) and the replay description. Document the real UI: ReplayBar, Minimap and keyboard shortcuts including `j`/`k`/`e`.
- **Tutorials and examples:**
  - Regenerate each tutorial with keyless fake-model runs, and run every snippet in CI.
  - Share one "view, replay, diff your trace" page instead of five invented copies.
  - Add an `examples/README.md`, merge the duplicate openai examples, and add a 4.2 spans sample.

## W3 — Process
- Add a CI job that executes doc snippets (core snippets with the default env; framework snippets with fake models per extra).
- Generate the CLI and endpoint reference tables from `lctl --help` and the FastAPI routes.
- Add one real-package smoke test per extra, and pin the extras to tested version ranges. Several tests currently mock away the very failures found here.
