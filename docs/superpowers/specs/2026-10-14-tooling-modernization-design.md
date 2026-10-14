# LCTL Tooling Modernization — Design

- **Date:** 2026-10-14
- **Status:** Awaiting review
- **Branch:** `chore/modernize-tooling`
- **Scope:** Sub-project 1 of 2. Sub-project 2, the Claude Agent SDK integration, gets its own spec.

## 1. Context

LCTL v4.1.0 is a Python library (about 20k lines in `lctl/`, plus about 18k lines of tests) with a VS Code extension in `vscode-lctl/`. An inventory on 2026-10-14 found the following.

**Healthy**
- 1041 tests pass and 14 are skipped, in about 7s.
- Coverage is about 80% line.
- Hatchling build with an SPDX `license`.

**Outdated or missing**
- pip with a hand-made `.venv`. No lockfile.
- Dev tools are an extra. `pytest-asyncio` is undeclared even though 10 test files use async.
- Black and Ruff overlap. Black would reformat 54 of 65 files.
- Ruff reports 286 violations on today's narrow rules, and about 1050 on a modern rule set.
- No type checker and no `py.typed`.
- **No CI, pre-commit or Dependabot.**
- The `License ::` classifier (AGPLv3+) contradicts `license = "AGPL-3.0-only"`.
- The version is defined in two places.
- Scratch files are tracked at the root (`task.md`, `BUG_REPORT.md`, `CODEBASE_REVIEW_REPORT.md`, `debug_sk.py`).
- The extension tracks a generated `vite.config.js` and its `.map`.
- A stray root `node_modules/` exists.
- Doc drift: `CLAUDE.md` says Python 3.9+ and v4.0.0, and leaves out five modules. `docs/api.md` says v4.0.0. The README tells contributors to use `black`.
- Two tests leak un-awaited coroutines.

Current-standard facts this design relies on (verified 2026-10-14):

| Tool | Version | Note |
|---|---|---|
| uv / uv_build | 0.13.1 | `[dependency-groups]` (PEP 735) is the documented dev-dependency mechanism. `--locked` in CI. |
| ruff | 0.17.0 | `ruff format` is documented as a drop-in Black replacement. |
| mypy | 2.4.0 | Stable. |
| ty | 0.0.87 | **Beta.** Diagnostics may change between any two versions, so it isn't used here. |
| pre-commit / prek | 4.7.0 / 0.5.5 | prek is config-compatible. |
| pytest / pytest-asyncio | 9.1.1 / 1.4.0 | Set `asyncio_default_fixture_loop_scope` explicitly. |
| Python | 3.15.0 final 2026-10-09 | 3.10 reached end of life on 2026-10-01. Test 3.11–3.15. |
| actions/checkout · setup-uv · setup-node | v7.0.1 · v10.3.0 · v7.1.0 | GitHub hardening guide: pin to a full SHA and use least-privilege `permissions`. |
| zizmor | 1.30.1 | De facto linter for GitHub Actions security. |
| Dependabot | — | Supports the `uv` ecosystem, `groups` and `cooldown`. |

A probe (`uv lock` on the current `pyproject.toml`) resolved **all nine extras together** (292 packages) with no conflicts.

## 2. Goals and non-goals

**Goals**
1. A single command sets up a reproducible dev environment from a committed lockfile.
2. One toolchain covers lint, format and types, and runs identically in pre-commit and CI.
3. CI runs on every push and PR: Python 3.11–3.15, lint, types, packaging smoke test, and the extension build.
4. Zero lint, format and type findings at merge, with no grandfathered violations.
5. The repo root holds only standard files, and the docs match reality.

**Non-goals**
- **No runtime behavior changes.**
  - The public API, CLI, serialized `.lctl.json` output, runtime dependencies and extras all stay the same.
  - The package version stays at **4.1.0**.
- No PyPI publishing or release pipeline. The project stays a public GitHub repo installed from git.
- No VS Code extension dependency upgrades.
- No new integrations. The Claude Agent SDK integration is sub-project 2.

## 3. Decisions (agreed during brainstorming)

| # | Decision | Rationale |
|---|---|---|
| D1 | Approach A: **uv + Ruff + mypy**, keeping hatchling as the build backend | All stable and conventional. Hatchling already packages the dashboard's static and template files, so switching backends would be churn. ty waits for 1.0. |
| D2 | Modern Ruff rule set, **fixed to zero** | CI is strict from day one, with no baseline debt. |
| D3 | Extension: CI build and lint only, with hygiene fixes | Keeps this change focused. Dependency upgrades are a follow-up. |
| D4 | Root reports go to `docs/archive/`, and `debug_sk.py` is deleted | History stays readable and the root stays clean. |
| D5 | Agent SDK integration is a separate, subsequent sub-project | It lands on the new gates, and each spec is reviewed separately. |
| D6 | DTZ rules excluded | Timezone-aware timestamps would change the serialized output. That's a behavior change, recorded as a follow-up. |

## 4. Design

### 4.1 Packaging & environment

`pyproject.toml`:
- `[project]`
  - `version` becomes `dynamic = ["version"]`, with `[tool.hatch.version] path = "lctl/__init__.py"`. `lctl.__version__` stays the single source.
  - Remove the `License :: OSI Approved :: …` classifier (PEP 639: the SPDX `license` field is authoritative).
  - Add `Programming Language :: Python :: 3.15` and `Typing :: Typed`.
  - `requires-python`, `dependencies` and every runtime extra (`dashboard`, `langchain`, …, `all`) stay unchanged.
- **Remove the `[dev]` extra.** Add a dev dependency group:
  ```toml
  [dependency-groups]
  dev = [
      "pytest>=9.1",
      "pytest-asyncio>=1.4",
      "pytest-cov>=7.0",
      "httpx>=0.28",
      "ruff>=0.17",
      "mypy>=2.4",
      "types-PyYAML",
      "pre-commit>=4.7",
      "lctl[dashboard]",
  ]
  ```
  Black is dropped. The group includes the project's own `dashboard` extra (verified 2026-10-14 with both uv and pip 25.2), so plain `uv sync` gives a complete dev env, and mypy always sees `fastapi` and `slowapi`. pip ≥25.1 users can install it with `pip install -e . --group dev`.

New files:
- `uv.lock`, committed.
- `.python-version` containing `3.13`, matching the existing local env.
- `lctl/py.typed`, empty.

Dev loop:
```bash
uv sync
uv run pytest
uv run pre-commit run --all-files      # ruff, format, mypy, lockfile, zizmor, file hygiene
```

### 4.2 Code quality

**Ruff** (`[tool.ruff]`):
- `line-length = 100`. The target comes from `requires-python`.
- `[tool.ruff.lint] select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "PIE", "RUF", "S", "ASYNC", "PT"]`, `ignore = ["E501"]`.
- Per-file ignores, limited to where a rule is wrong for that context:
  - `tests/**`: `S101`, `S105`, `S106`
  - `benchmarks/**`: `S311`
  - Nothing is ignored file-wide under `lctl/`.
- Scope: `lctl/`, `tests/`, `benchmarks/`, `examples/`.
- The `[tool.black]` table is removed.

**Fix-to-zero procedure.** Separate commits, with the full test suite green after each:
1. `ruff format .` and nothing else. Record this commit's SHA in `.git-blame-ignore-revs`.
2. Safe autofixes only (`ruff check --fix`). Unsafe fixes are reviewed one by one, never bulk-applied.
3. Manual fixes for the remainder. Rules:
   - **No runtime behavior change.** Intentional code a rule flags gets a line-level `# noqa: <RULE>` plus a reason, not a code change.
     - Known case: `S104` on `--host` defaulting to `0.0.0.0` in `lctl/cli/main.py:709`.
     - Anything that looks like a real defect goes to §6 (follow-ups), not into this diff.
   - **Public API is protected.** In `__init__.py` files, re-exports flagged as unused are added to `__all__`, never deleted.
   - `B904` (`raise … from err`) is applied. It only adds exception chaining.

**mypy** (`[tool.mypy]`, basic mode):
```toml
[tool.mypy]
python_version = "3.11"
files = ["lctl"]
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = [
    "langchain.*", "langchain_core.*", "crewai.*", "autogen.*", "autogen_core.*", "ag2.*",
    "agents.*", "pydantic_ai.*", "semantic_kernel.*", "dspy.*", "llama_index.*",
    "slowapi.*", "uvicorn.*", "websockets.*",
]
ignore_missing_imports = true
```
- Unannotated function bodies are not checked (mypy's default). Enabling `check_untyped_defs` is a follow-up ratchet.
- Fixes prefer real annotations, then `Any` for untyped framework objects. Every `# type: ignore[...]` must name its error code, and bare ignores aren't allowed.
- The override list above was measured on 2026-10-14 with mypy 2.4.0. `langchain_community` is left out because nothing imports it, and `warn_unused_configs` would flag it. `websockets` is used lazily by `lctl/streaming/websocket.py` and isn't declared in any extra.

### 4.3 Tests & coverage

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers", "--strict-config", "--cov=lctl", "--cov-report=term-missing"]
filterwarnings = ["error"]
asyncio_mode = "strict"
asyncio_default_fixture_loop_scope = "function"

[tool.coverage.run]
source = ["lctl"]
branch = true

[tool.coverage.report]
show_missing = true
exclude_also = ["if TYPE_CHECKING:", "raise NotImplementedError"]
fail_under = <measured>
```
- **`filterwarnings = ["error"]`.** Any third-party warning that must be tolerated gets a narrow, commented ignore entry. Our own warnings get fixed.
- **`asyncio_mode = "strict"`** is today's implicit default, so test semantics don't change.
- **`fail_under`** is the branch-coverage percentage measured after the suite is green, rounded down to an integer. It's a ratchet, not a target.
- **Fix the two leaky tests:**
  - `tests/test_websocket_comprehensive.py`: `WebSocketServer.start` is never awaited.
  - `tests/test_pydantic_ai_comprehensive.py::TestConcurrentAgents::test_multiple_concurrent_agents`: `make_run` is never awaited.

  Diagnose them with systematic debugging. The fixes belong in the tests unless the diagnosis proves a library bug, in which case it goes to §6.
- **Integration run:** run the suite once locally on 3.13 with `uv sync --all-extras`.
  - **If it's green,** add the CI `integration` job (§4.4) as required.
  - **If it's red,** add no job and record the failures in §6.

### 4.4 Automation

**`.pre-commit-config.yaml`** (works with `pre-commit` or `prek`):
- `pre-commit/pre-commit-hooks` (latest tag at implementation): `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-json` (excluding `tsconfig*.json`, which is JSONC), `check-merge-conflict`, `mixed-line-ending`, and `check-added-large-files` with `uv.lock` and `**/package-lock.json` excluded.
- `astral-sh/ruff-pre-commit` `v0.17.0`: `ruff-check` (`--fix`) and `ruff-format`.
- `astral-sh/uv-pre-commit` (version matching the pinned uv): `uv-lock`.
- Local hook: `mypy` via `uv run --frozen mypy` (`language: system`, `pass_filenames: false`, scoped to `^lctl/`). The dashboard dependencies come from the dev group, so they are always present.
- `zizmorcore/zizmor-pre-commit`: `zizmor`, scoped to `.github/`.

**`.github/workflows/ci.yml`:**
- **Triggers:** `push` to `main`, `pull_request`, `workflow_dispatch`.
- **Concurrency:** `ci-${{ github.ref }}`, with `cancel-in-progress` for PRs.
- **Permissions:** `permissions: {}` at the top, and `contents: read` per job.
- **Actions:** every action is pinned to a full commit SHA with a trailing `# vX.Y.Z` comment.
- **checkout:** `persist-credentials: false`.
- **setup-uv:** an exact `version:` matching the pinned uv, and `enable-cache: true`.

| Job | Runs | Steps |
|---|---|---|
| `lint` | ubuntu, Python from `.python-version` | `uv sync --locked`, `uv run pre-commit run --all-files --show-diff-on-failure` |
| `test` | ubuntu × Python 3.11–3.15 | `uv sync --locked --python <v>`, `uv run pytest` |
| `integration` | ubuntu, 3.13 (**only if §4.3 run is green**) | `uv sync --locked --all-extras`, `uv run pytest` |
| `package` | ubuntu | `uv build`; install the wheel into a fresh venv; `lctl --version` reports 4.1.0; assert the wheel contains `lctl/py.typed`, `lctl/dashboard/static/*` and `lctl/dashboard/templates/*` |
| `extension` | ubuntu, Node 24 | `npm ci` in `vscode-lctl/` and `vscode-lctl/webview-ui/`; `npm run lint`; `npm run compile`; `npm run build:webview` |

There are no path filters. Required checks that are skipped by a path filter stay "pending".

**`.github/dependabot.yml`**: weekly, with a 7-day `cooldown`.
- `uv` at `/`: minor and patch updates grouped into one PR, majors as separate PRs.
- `github-actions` at `/`: grouped. It updates the SHA pins and their version comments.
- `npm` at `/vscode-lctl` and `/vscode-lctl/webview-ui`: `open-pull-requests-limit: 0`, so security updates only (per D3).

**Manual step for the repo owner,** documented in the README: enable Dependabot alerts and security updates in the GitHub repo settings. This isn't done through the API from here.

### 4.5 Repo hygiene & docs

**Hygiene**
- `git mv task.md BUG_REPORT.md CODEBASE_REVIEW_REPORT.md docs/archive/`, and `git rm debug_sk.py`.
- **Extension:**
  - Change `vscode-lctl/webview-ui/tsconfig.node.json` so `tsc` type-checks `vite.config.ts` without emitting, following the current Vite template's pattern. It must stay compatible with the `references` entry in `webview-ui/tsconfig.json`.
  - `git rm` `vite.config.js` and `vite.config.js.map`.
  - Done when `npm run build:webview` leaves `git status` clean.
- **`.gitignore`:** add `.mypy_cache/`, `.coverage.*` and `*.tsbuildinfo`. `.claude` stays ignored, because it holds machine-specific hooks with absolute paths and personal traces.
- **Delete the untracked root `node_modules/`.** It's orphaned, with no root `package.json`, and this is local only.

**Docs**
- **`CLAUDE.md`** — fix drift without rewriting:
  - Python 3.11+, and no hard-coded package version (refer to `lctl/__init__.py`).
  - Add `streaming/`, `metrics/`, `evaluation/`, `core/redaction.py` and `core/schema.py` to the architecture map.
  - Replace the pip/black commands with uv/ruff/mypy/pre-commit.
  - Merge the two copies of the integration-pattern guidance into one.
- **`README.md`:** the development section switches to the uv workflow, `black` references go away, a CI badge is added, and the Dependabot settings note goes in. The user-facing git install commands are unchanged.
- **`CHANGELOG.md`:** add an `[Unreleased]` → `Changed` entry for this work, and correct the 4.1.0 date from `2025-01-19` to `2026-01-19`. Commit history dates the release to January 2026.
- **`docs/api.md`:** update the "generated for LCTL v4.0.0" footer. Any other `4.0` string is changed only if it is the package version, not the LCTL *format* version.
- **`lctl/integrations/claude_code.py` module docstring:** replace the stale `CLAUDE_TOOL_INPUT_*` env-var example with the stdin-JSON mechanism the generated hooks actually use. This is a docstring-only change.

## 5. Acceptance criteria

All of these are verified with command output before the work is called done:

1. `uv sync --locked` works from a fresh clone, and `uv lock --check` passes.
2. `uv run ruff check .` reports 0 findings, `uv run ruff format --check .` reports 0 files to change, and `uv run mypy` reports success.
3. `uv run pytest` passes with at least 1041 tests passing, under `filterwarnings = error`, and coverage at or above `fail_under`.
4. `uv run pre-commit run --all-files` passes.
5. **No behavior change:**
   - A snapshot of `__all__` for `lctl`, `lctl.core`, `lctl.integrations`, `lctl.streaming`, `lctl.metrics` and `lctl.evaluation`, taken before the change, equals the snapshot after it.
   - `lctl stats --json examples/sample-chain.lctl.json` and `lctl trace examples/sample-chain.lctl.json` produce identical output before and after.
6. The wheel from `uv build` contains `py.typed` and the dashboard assets, `lctl --version` prints 4.1.0, and the metadata has `License-Expression: AGPL-3.0-only` and no `License ::` classifier.
7. In `vscode-lctl/`, `npm run lint`, `npm run compile` and `npm run build:webview` succeed, and `git status` stays clean afterwards.
8. `git ls-files` shows none of the four root scratch files at the root, and no `vite.config.js` or `.map`.
9. All CI jobs are green on the PR, with zero zizmor findings. Any zizmor ignore has an inline justification.

## 6. Follow-ups (out of scope, record only)

1. **Claude Agent SDK integration** (`claude-agent-sdk`) as sub-project 2. It's an in-process `BaseTracer`, distinct from the CLI-hook-based `claude_code.py`.
2. **Dashboard binds `0.0.0.0` by default** (`lctl/cli/main.py:709`), which exposes it on the network. Consider defaulting to `127.0.0.1`. This is a behavior change.
3. **Timezone-aware timestamps (DTZ rules).** This needs a check that the serialized format stays compatible with the schema migration.
4. **Extension dependency refresh,** including replacing `@vscode/webview-ui-toolkit`, which Microsoft archived.
5. **mypy ratchet:** `check_untyped_defs`, then extending coverage to `tests/`.
6. Re-evaluate **`ty`** once it reaches 1.0.
7. Integration-test failures found by the all-extras run, if any (§4.3).
8. Anything the `S` rules or the leaky-test diagnosis surface as a real library defect (§4.2, §4.3).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Autofixes silently change behavior. Removing re-exports or rewriting annotations FastAPI evaluates at runtime are the main dangers. | Separate commits, the full suite after each, the `__all__` and CLI-output snapshots (§5.5), and unsafe fixes only when reviewed individually. |
| `filterwarnings = error` turns third-party deprecations into failures, especially on 3.15. | Narrow, commented ignores. Dependabot PRs surface new warnings early, which is the intended effect. |
| Branch coverage reads below today's line coverage. | The floor is set from measurement, never guessed. |
| Third-party dependencies lack wheels for brand-new 3.15. | The core plus the `dashboard` extra is the 3.15 surface. If a dependency blocks it, mark that matrix cell `continue-on-error` with a linked follow-up instead of dropping 3.15. |
| Dependabot `uv` support may lag uv 0.13 (the documented compatibility is uv 0.11). | Watch the first Dependabot run. The lockfile is still enforced by `--locked` in CI either way. |
