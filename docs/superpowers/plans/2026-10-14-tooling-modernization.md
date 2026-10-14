# LCTL Tooling Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move LCTL to uv + Ruff + mypy, with CI, pre-commit and Dependabot, zero lint/type findings, and no runtime behavior change.

**Architecture:**
- Tooling and config only. `pyproject.toml` becomes the single config hub (uv groups, ruff, mypy, pytest, coverage). Pre-commit runs the same checks locally that CI's `lint` job runs.
- Code edits are limited to lint and type fixes that preserve behavior, two test-only leak fixes, and one docstring.

**Tech Stack:** uv 0.13.1, hatchling, ruff 0.17.0, mypy 2.4.0, pytest ≥9.1 with pytest-asyncio ≥1.4 and pytest-cov, pre-commit 4.7, GitHub Actions, Dependabot, Node 24 (extension).

**Spec:** `docs/superpowers/specs/2026-10-14-tooling-modernization-design.md`. Read it first: its §4.2 rules govern every lint and type fix.

## Global Constraints

- **No runtime behavior change.**
  - The public API, CLI output, serialized `.lctl.json`, runtime dependencies and extras are unchanged.
  - The package version stays at `4.1.0`.
- `requires-python = ">=3.11"`, `line-length = 100`.
- **Every commit leaves `uv run pytest --no-cov -q` green.** From Task 1 on that means 1041 passed and 14 skipped, unless a task says otherwise.
- **Intentional code a rule flags** gets a line-level `# noqa: <RULE>  # <reason>` on the reported line. Verified syntax: ruff 0.17 accepts a trailing reason comment.
- **Ignores:**
  - Nothing is ignored file-wide under `lctl/`.
  - `# type: ignore` must name its code(s).
  - Never `--unsafe-fixes` in bulk. The only exception is the reviewed SIM105 pass in Task 3.
- **Public API rule:** in any `__init__.py`, an import flagged unused is a re-export. Add it to `__all__`, never delete it.
- **Behavior baseline** (captured 2026-10-14 from the pre-change code). Directory `$SP/baseline/`, where `SP=/tmp/claude-1000/-home-kim-projects-llm-context-transfer/783deef4-54f3-49cc-81f2-f407fcbaf8ae/scratchpad`:
  - `api.before.json`, produced by `snapshot_api.py`
  - `stats.before.txt`: `lctl stats --json examples/sample-chain.lctl.json`
  - `trace.before.txt`: `lctl trace examples/sample-chain.lctl.json`

  Both CLI outputs are deterministic. If `$SP` is gone, regenerate the baseline from commit `96b8236` (`git worktree add $SP/base 96b8236`) using `snapshot_api.py`, whose source is in Appendix A.
- **Pinned action SHAs** (resolved 2026-10-14):
  - `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1`
  - `astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884  # v10.3.0`
  - `actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1  # v7.1.0`
- **Pre-commit revs:** `pre-commit/pre-commit-hooks v6.0.0`, `astral-sh/ruff-pre-commit v0.17.0`, `astral-sh/uv-pre-commit 0.13.0` (0.13.1 is not tagged yet; the lock format is the same within 0.13), `zizmorcore/zizmor-pre-commit v1.30.1`.
- **Commit trailer:** end every commit message with
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01U7JQRWVBpRadDaH4RESMtZ
  ```
- **Never `git push`.** Integration is decided with the user after Task 12.

## Review Focus

The five failure modes most likely to bite, each pinned by a step in its owning task:

1. **A contributor using pip, not uv.** `pip install -e . --group dev` must install LCTL from the checkout, not from PyPI, along with the dashboard dependencies. Pinned in Task 1, Step 7.
2. **Someone installing from git.** `pip install git+…` builds through hatchling's dynamic version, so the built wheel must report `4.1.0` and contain `py.typed` and the dashboard assets. Pinned in Task 1 (`scripts/check_wheel.py`) and the Task 10 `package` job.
3. **Autofix silently changes behavior.** The danger is deleting a re-export, rewriting a `(str, Enum)` into `StrEnum` (which changes `str()`, and so the serialized output), or altering an annotation FastAPI evaluates at runtime. Pinned by the API and CLI snapshot comparison after Tasks 3, 4 and 5, plus the explicit `UP042` noqa in Task 4.
4. **Warnings-as-errors on the matrix edges.** Python 3.11 and 3.15 may emit warnings that 3.13 doesn't. Pinned in Task 7, Step 7, which runs the suite on 3.11 and 3.15 locally.
5. **CI's `uv run` silently swaps the interpreter.** `.python-version` pins 3.13, so a matrix cell must set the interpreter through setup-uv's `python-version` input (which exports `UV_PYTHON`), or `uv run` would re-create a 3.13 env. Pinned in Task 10's `test` job, plus a step that prints `python --version` in each cell.

---

### Task 1: Migrate packaging to uv

**Files:**
- Create: `scripts/check_wheel.py`, `.python-version`, `lctl/py.typed`, `uv.lock`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `python scripts/check_wheel.py <path-to-wheel>`. It exits 0 when every check passes, otherwise exits 1 and prints one `FAIL: <reason>` line per failed check. Task 10's `package` job calls it.
- Produces: dev group `dev`, installed by plain `uv sync`. It includes `lctl[dashboard]`.

- [ ] **Step 1: Write `scripts/check_wheel.py`**

  It uses the stdlib only (`zipfile`, `re`, `sys`, `pathlib`). It reads the expected version from `lctl/__init__.py` by regex (`^__version__ = "(.+)"`), resolved relative to the script, never by importing. Checks:
  - The wheel contains `lctl/py.typed`, `lctl/dashboard/templates/index.html`, `lctl/dashboard/static/app.js` and `lctl/dashboard/static/styles.css`.
  - `*.dist-info/METADATA` has `Version: <expected>` and `License-Expression: AGPL-3.0-only`.
  - It has no line starting `Classifier: License ::`.
  - It contains `Classifier: Programming Language :: Python :: 3.15` and `Classifier: Typing :: Typed`.

- [ ] **Step 2: Run it against today's packaging to prove it fails**

  Run: `uv build --wheel -o dist/ && python3 scripts/check_wheel.py dist/lctl-4.1.0-py3-none-any.whl`

  Expected: exit 1, with FAIL lines for the missing `lctl/py.typed`, the `License ::` classifier, and the missing 3.15 and `Typing :: Typed` classifiers.

- [ ] **Step 3: Edit `pyproject.toml` and add the new files**
  - **`[project]`:**
    - Replace `version = "4.1.0"` with `dynamic = ["version"]`, and add `[tool.hatch.version]` with `path = "lctl/__init__.py"`.
    - Delete the `"License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"` classifier.
    - Add `"Programming Language :: Python :: 3.15"` and `"Typing :: Typed"`.
  - Delete the `dev = [...]` entry from `[project.optional-dependencies]`. Leave every other extra untouched.
  - Delete the `[tool.black]` table.
  - Add the dev group:
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
  - Create `lctl/py.typed`, empty.
  - Create `.python-version` containing `3.13`.

- [ ] **Step 4: Recreate the environment with uv**

  Run: `rm -rf .venv && uv lock && uv sync && uv lock --check`

  Expected: `uv.lock` is created (about 290 packages), and `uv lock --check` exits 0.

- [ ] **Step 5: Re-run the wheel check**

  Run: `rm -rf dist && uv build --wheel -o dist/ && uv run python scripts/check_wheel.py dist/lctl-4.1.0-py3-none-any.whl`

  Expected: exit 0, no FAIL lines.

- [ ] **Step 6: Run the tests**

  Run: `uv run pytest --no-cov -q`

  Expected: `1041 passed, 14 skipped`. The un-awaited-coroutine warning is still present; Task 7 fixes it.

- [ ] **Step 7: Verify the pip path (Review Focus 1)**

  Run: `python3.14 -m venv $SP/pipcheck && $SP/pipcheck/bin/pip install -q -e . --group dev && $SP/pipcheck/bin/pip show lctl | grep "Editable project location" && $SP/pipcheck/bin/python -c "import lctl, fastapi, slowapi, pytest; print(lctl.__version__)"`

  Expected: the editable location is the repo, and the command prints `4.1.0`.

- [ ] **Step 8: Commit**
  ```bash
  git add pyproject.toml uv.lock .python-version lctl/py.typed scripts/check_wheel.py
  git commit -m "build: Migrate to uv with dependency groups and PEP 639 metadata"
  ```

### Task 2: Replace Black with `ruff format`

**Files:**
- Modify: every `*.py` that `ruff format` changes (formatting only)
- Create: `.git-blame-ignore-revs`

- [ ] **Step 1: Confirm the drift**

  Run: `uv run ruff format --check .`

  Expected: exit 1 and "N files would be reformatted", with N > 50.

- [ ] **Step 2: Format**

  Run: `uv run ruff format .`

- [ ] **Step 3: Run the tests**

  Run: `uv run pytest --no-cov -q`

  Expected: `1041 passed, 14 skipped`.

- [ ] **Step 4: Commit the formatting alone**

  Run `git add -u && git commit -m "style: Reformat with ruff format (replaces black)"`, and confirm `git show --stat HEAD` lists only `.py` files.

- [ ] **Step 5: Record the formatting commit for blame**

  Create `.git-blame-ignore-revs` containing:
  - the line `# ruff format migration (replaces black)`
  - the full 40-character SHA from `git rev-parse HEAD`

- [ ] **Step 6: Commit**
  ```bash
  git add .git-blame-ignore-revs
  git commit -m "chore: Ignore ruff format migration in git blame"
  ```

### Task 3: Modern Ruff rule set, config plus mechanical fixes

**Files:**
- Modify: `pyproject.toml`, plus the `.py` files the fixes touch

**Interfaces:**
- Produces: the final `[tool.ruff.lint]` config. Tasks 4 and 5 drive its findings to zero.

- [ ] **Step 1: Write the config**

  Replace the `[tool.ruff.lint]` table with:
  ```toml
  [tool.ruff.lint]
  select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "PIE", "RUF", "S", "ASYNC", "PT"]
  ignore = ["E501"]

  [tool.ruff.lint.per-file-ignores]
  "tests/**" = ["S101", "S105", "S106"]
  "benchmarks/**" = ["S311"]
  ```
  Keep `[tool.ruff] line-length = 100`.

- [ ] **Step 2: Measure the starting point**

  Run: `uv run ruff check . --statistics | tail -1`

  Expected: about 1100 or more findings, most of them auto-fixable.

- [ ] **Step 3: Apply only the safe autofixes**

  Run: `uv run ruff check . --fix`, then `uv run ruff format .`, since fixes can leave formatting drift.

- [ ] **Step 4: Verify behavior is unchanged (Review Focus 3)**

  Run:
  ```bash
  uv run pytest --no-cov -q
  uv run python $SP/baseline/snapshot_api.py | diff - $SP/baseline/api.before.json
  uv run lctl stats --json examples/sample-chain.lctl.json | diff - $SP/baseline/stats.before.txt
  uv run lctl trace examples/sample-chain.lctl.json | diff - $SP/baseline/trace.before.txt
  ```
  Expected: tests show `1041 passed, 14 skipped`, and all three diffs are empty.

- [ ] **Step 5: Commit**
  ```bash
  git add -u && git commit -m "style: Apply ruff safe autofixes for modern rule set"
  ```

- [ ] **Step 6: Rewrite SIM105 sites to `contextlib.suppress`, with review**

  Run: `uv run ruff check . --select SIM105 --fix --unsafe-fixes`

  Then review `git diff` hunk by hunk. Every hunk must turn `try: <one stmt> / except <E>: pass` into `with contextlib.suppress(<E>): <stmt>` with **the identical exception type(s)**.
  - Restore any comment that sat inside the `except` body as a comment directly above the `with`.
  - Revert any hunk that does anything else.

- [ ] **Step 7: Verify and commit**

  Repeat Step 4's commands, with the same expected results. Then:
  ```bash
  git add -u && git commit -m "refactor: Use contextlib.suppress for intentionally swallowed exceptions"
  ```

### Task 4: Ruff findings in `lctl/` to zero

**Files:**
- Modify: the files under `lctl/` that `uv run ruff check lctl` reports. Measured hot spots: `integrations/autogen.py`, `integrations/claude_code.py`, `integrations/langchain.py`, `core/session.py`, `dashboard/app.py`.

- [ ] **Step 1: List the remaining findings**

  Run: `uv run ruff check lctl --statistics`

  Expected after Task 3: roughly 250. The main rules are `S110` (about 55), `SIM105` (manual leftovers), `RUF015`, `B904`, `C401`, `F841`, `SIM108` and `RUF005`.

- [ ] **Step 2: Fix them by rule**

  The spec (§4.2) decides each case:

  | Rule(s) | Action |
  |---|---|
  | `S110`, `S112` | These are intentional swallows (CLAUDE.md "tracing should never break the main code"). Add `# noqa: S110  # tracing must never break the host application` (or `S112`) on the reported `except` line. `uv run ruff check lctl --select S110,S112 --add-noqa` inserts the codes, and you then append the reason. |
  | `SIM105` leftovers | Rewrite by hand to `contextlib.suppress(<same exception>)`, keeping comments. |
  | `UP042` (`lctl/core/events.py:84` `EventType`, `lctl/streaming/emitter.py:15` `StreamingEventType`) | **Do not convert to StrEnum**, because it changes `str()`/`format()` output, which is serialized. Use `# noqa: UP042  # StrEnum changes str() output used in serialized chains` |
  | `S104` (`lctl/cli/main.py:709`, `lctl/streaming/websocket.py:274`, `:452`) | `# noqa: S104  # bind address is user-configurable; default tracked as spec follow-up 2` |
  | `S603`/`S607` (`lctl/integrations/claude_code.py:895,910,1643,1650,1657`) | Confirm each call uses a list argv with no `shell=True`, then add `# noqa: S603  # fixed argv, no shell` and `# noqa: S607  # resolving git via PATH is intended` |
  | `S311` (`lctl/evaluation/statistical.py:321`) | `# noqa: S311  # statistical sampling, not cryptography` |
  | `ASYNC110` (`lctl/streaming/subscriber.py:398`) | Switching to `asyncio.Event` is a behavior change. Use `# noqa: ASYNC110  # polling loop; Event-based rewrite is a follow-up` and add it to spec §6. |
  | `B024` (`lctl/integrations/base.py:61` `BaseTracer`) | `# noqa: B024  # concrete base with shared defaults; subclasses override hooks` |
  | `B904` | `raise New(...) from err`. If the handler has no bound name, bind it: `except X as err:`. |
  | `RUF015`, `C400`/`C401`, `RUF005`, `SIM102/103/108/110/117`, `RUF010`, `UP*` | Apply the suggested rewrite, which is behavior-equivalent. |
  | `F841`, `B007`, `RUF059` | Drop unused names. Keep any call whose side effect matters, and rename an unused loop variable to `_`. |
  | `F401` outside `__init__.py` | If it's an optional-dependency availability probe (`try: import x`), add `# noqa: F401  # availability probe`. Otherwise delete it. |
  | `F401` inside `__init__.py` | Add the name to that module's `__all__` (Global Constraints, Public API rule). |
  | `RUF022` | Sort `__all__` within its existing comment-delimited groups. Keep the group comments. |
  | `RUF012` | Annotate with `ClassVar[...]`. |
  | `RUF013` | Make it `X \| None`. |
  | `E402` | Move the import to the top if nothing above it is required for it. Otherwise add `# noqa: E402  # <why it must follow>`. |

  If a fix would reveal or require changing actual behavior, don't change it. Add a `noqa` with a reason and append a bullet to spec §6.

- [ ] **Step 3: Verify lint is clean in `lctl/`**

  Run: `uv run ruff check lctl && uv run ruff format --check lctl`

  Expected: `All checks passed!`, and 0 files to reformat.

- [ ] **Step 4: Verify behavior is unchanged**

  Run Task 3 Step 4's four commands. Expected: same results, all diffs empty.

- [ ] **Step 5: Commit**
  ```bash
  git add -u lctl docs/superpowers/specs
  git commit -m "fix: Resolve modern ruff findings in lctl without behavior change"
  ```

### Task 5: Ruff findings in `tests/`, `benchmarks/` and `examples/` to zero

**Files:**
- Modify: the files under `tests/`, `benchmarks/` and `examples/` that ruff reports

- [ ] **Step 1: List the findings**

  Run: `uv run ruff check tests benchmarks examples --statistics`

- [ ] **Step 2: Fix them using Task 4's table, plus these test- and example-specific rules**

  | Rule | Action |
  |---|---|
  | `PT011` | Add `match=` with a stable substring of the expected message. |
  | `E712` (`== True`) | Use `is True` / `is False`. If a test then fails because the value is truthy but not a bool, use plain `assert x` / `assert not x` instead. |
  | `N806` (`tests/test_crewai_async.py:19` `MockCrew`) | `# noqa: N806  # class-like mock name` |
  | `S307` (`examples/langchain_example.py:130`, `examples/openai_agents_example.py:145`) | `# noqa: S307  # demo calculator; see spec follow-up`. Append to spec §6: "examples' calculator tools `eval()` model output; replace with a safe arithmetic evaluator". |
  | `S110` in tests | `# noqa: S110  # test intentionally ignores cleanup errors`, or a more accurate reason if one applies. |

- [ ] **Step 3: Verify lint is clean repo-wide**

  Run: `uv run ruff check . && uv run ruff format --check .`

  Expected: `All checks passed!`, and 0 files to reformat.

- [ ] **Step 4: Run the tests**

  Run: `uv run pytest --no-cov -q`

  Expected: `1041 passed, 14 skipped`.

- [ ] **Step 5: Commit**
  ```bash
  git add -u tests benchmarks examples docs/superpowers/specs
  git commit -m "test: Resolve modern ruff findings in tests, benchmarks and examples"
  ```

### Task 6: mypy (basic mode) to zero

**Files:**
- Modify: `pyproject.toml`, `.gitignore`, and the `lctl/` files mypy reports. Measured 2026-10-14: 50 errors in 12 files, mostly `integrations/autogen.py`, `metrics/aggregators.py`, `integrations/openai_agents.py`, `core/events.py` and `integrations/claude_code.py`.

- [ ] **Step 1: Add the config**
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
  Also append `.mypy_cache/` to `.gitignore`.

- [ ] **Step 2: Measure the starting point**

  Run: `uv run mypy`

  Expected: about 45 errors. There should be no `import-not-found` or `import-untyped` errors, because `types-PyYAML` is in the dev group and the rest are covered by the overrides. If an `import-not-found` remains for an optional dependency, add it to the override list. If `warn_unused_configs` reports an override pattern, remove that pattern.

- [ ] **Step 3: Fix the errors by code**

  | Code | Action |
  |---|---|
  | `unused-ignore` | Delete the stale `# type: ignore`. |
  | `no-redef` / `misc` / `assignment` on optional-import fallbacks (`except ImportError: X = None`) | Use `X = None  # type: ignore[assignment,misc]`, with exactly the codes mypy reports. This is the standard idiom. |
  | `var-annotated` | Add the annotation (e.g. `self._items: dict[str, Any] = {}`). |
  | `arg-type`, `dict-item`, `return-value`, `union-attr`, `index`, `operator`, `attr-defined`, `valid-type` | Fix with correct annotations, using `Any` where the value comes from an untyped framework object. If mypy is pointing at a real latent bug (e.g. an unhandled `None`), don't change the logic: add `# type: ignore[<code>]  # latent bug, spec follow-up` and append a bullet to spec §6. |

- [ ] **Step 4: Verify mypy, lint and tests**

  Run: `uv run mypy && uv run ruff check . && uv run pytest --no-cov -q`

  Expected: `Success: no issues found in N source files`, a clean ruff run, and `1041 passed, 14 skipped`.

- [ ] **Step 5: Commit**
  ```bash
  git add -u pyproject.toml .gitignore lctl docs/superpowers/specs
  git commit -m "fix: Add mypy (basic mode) and resolve type errors"
  ```

### Task 7: pytest config, warnings as errors, branch coverage, leaky tests

**Files:**
- Modify: `pyproject.toml`, `.gitignore`, `tests/test_websocket_comprehensive.py`, `tests/test_pydantic_ai_comprehensive.py`

- [ ] **Step 1: Write the config**

  Replace `[tool.pytest.ini_options]`, and add the coverage tables:
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
  ```
  Also append `.coverage.*` to `.gitignore`.

- [ ] **Step 2: Run the suite and watch it fail**

  Run: `uv run pytest --no-cov -q`

  Expected: FAIL. The leaked coroutines (`WebSocketServer.start` in `tests/test_websocket_comprehensive.py`, and `make_run` in `tests/test_pydantic_ai_comprehensive.py::TestConcurrentAgents::test_multiple_concurrent_agents`) now raise under `filterwarnings = error`. Note any *other* warnings that also surface.

- [ ] **Step 3: Diagnose each leak with superpowers:systematic-debugging**

  Find the allocation site with `uv run python -X tracemalloc=25 -m pytest <nodeid> --no-cov -W error`. The fix goes in the test: await the coroutine, close it, or mock with `AsyncMock` correctly. If the diagnosis shows the library itself leaks a coroutine, leave the library unchanged, add a narrow `filterwarnings` mark on that test with a comment, and add a spec §6 bullet.

- [ ] **Step 4: Handle any third-party warnings from Step 2**

  For each, add a narrow entry after `"error"`, e.g. `"ignore:<message regex>:DeprecationWarning:<module>"`, each with a `# why` comment on the line above. Our own warnings (`lctl.*`) get fixed, never ignored.

- [ ] **Step 5: Measure branch coverage**

  Run: `uv run pytest -q | grep TOTAL`

  Set `[tool.coverage.report] fail_under` to the TOTAL percentage rounded **down** to an integer.

- [ ] **Step 6: Run the full suite**

  Run: `uv run pytest -q`

  Expected: `1041 passed, 14 skipped`, no warnings summary, and coverage at or above `fail_under`.

- [ ] **Step 7: Check the matrix edges (Review Focus 4)**

  Run: `uv run --isolated --python 3.11 pytest -q && uv run --isolated --python 3.15 pytest -q`

  Expected: both green, including the `fail_under` coverage gate, since every CI matrix cell enforces it. Fix any version-specific warning per Step 4. If a version lands below the floor because of version-gated branches, lower `fail_under` to the lowest version's integer floor.

- [ ] **Step 8: Commit**
  ```bash
  git add -u pyproject.toml .gitignore tests docs/superpowers/specs
  git commit -m "test: Enforce warnings as errors and branch coverage; fix leaked coroutines"
  ```

### Task 8: VS Code extension hygiene

**Files:**
- Modify: `vscode-lctl/webview-ui/tsconfig.node.json`, `.gitignore`
- Delete (untrack): `vscode-lctl/webview-ui/vite.config.js`, `vscode-lctl/webview-ui/vite.config.js.map`
- Delete (local, untracked): `/home/kim/projects/llm-context-transfer/node_modules/`

- [ ] **Step 1: Replace `vscode-lctl/webview-ui/tsconfig.node.json`**

  This content was verified 2026-10-14 with TS 5.9.3: both `tsc` and `tsc -b` exit 0, and nothing is emitted into the tree.
  ```json
  {
    "compilerOptions": {
      "composite": true,
      "emitDeclarationOnly": true,
      "outDir": "./node_modules/.tmp/tsconfig.node",
      "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo",
      "skipLibCheck": true,
      "module": "ESNext",
      "moduleResolution": "bundler",
      "allowSyntheticDefaultImports": true,
      "strict": true
    },
    "include": ["vite.config.ts"]
  }
  ```

- [ ] **Step 2: Untrack the generated files and ignore build info**

  Run: `git rm vscode-lctl/webview-ui/vite.config.js vscode-lctl/webview-ui/vite.config.js.map`, and append `*.tsbuildinfo` to `.gitignore`.

- [ ] **Step 3: Verify the extension builds without dirtying the tree**

  Run:
  ```bash
  cd vscode-lctl
  npm ci
  (cd webview-ui && npm ci && npx tsc -b)
  npm run lint
  npm run compile
  npm run build:webview
  cd ..
  git status --porcelain vscode-lctl
  ```
  Expected: every command exits 0, and `git status` shows only the Step 1 and Step 2 changes, with no `vite.config.js` reappearing.

- [ ] **Step 4: Delete the orphan root `node_modules/`**

  First confirm it's untracked with no root `package.json`: `git ls-files node_modules | wc -l` should print `0`, and `ls package.json` should fail. Then run `rm -rf node_modules`.

- [ ] **Step 5: Commit**
  ```bash
  git add -A vscode-lctl .gitignore
  git commit -m "chore(extension): Stop emitting vite.config.js from tsc and untrack it"
  ```

### Task 9: Pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`
- Modify: any files the hygiene hooks fix (whitespace, end-of-file)

- [ ] **Step 1: Write `.pre-commit-config.yaml`**
  ```yaml
  repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v6.0.0
      hooks:
        - id: trailing-whitespace
        - id: end-of-file-fixer
        - id: check-yaml
        - id: check-toml
        - id: check-json
          exclude: tsconfig.*\.json$
        - id: check-merge-conflict
        - id: mixed-line-ending
        - id: check-added-large-files
          exclude: (^uv\.lock|package-lock\.json)$
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.17.0
      hooks:
        - id: ruff-check
          args: [--fix]
        - id: ruff-format
    - repo: https://github.com/astral-sh/uv-pre-commit
      rev: 0.13.0
      hooks:
        - id: uv-lock
    - repo: local
      hooks:
        - id: mypy
          name: mypy
          entry: uv run --frozen mypy
          language: system
          types: [python]
          files: ^lctl/
          pass_filenames: false
    - repo: https://github.com/zizmorcore/zizmor-pre-commit
      rev: v1.30.1
      hooks:
        - id: zizmor
  ```

- [ ] **Step 2: Run every hook**

  Run: `uv run pre-commit run --all-files`

  The first run may fix whitespace and EOF issues in non-Python files (Markdown, JS, CSS, JSON) and report "files were modified".

- [ ] **Step 3: Run again until clean**

  Run: `uv run pre-commit run --all-files`

  Expected: every hook `Passed` or `Skipped`.

- [ ] **Step 4: Commit the hygiene fixes separately, then record that commit in blame-ignore**

  Skip this step if Step 2 changed nothing. Otherwise:
  ```bash
  git add -u && git commit -m "style: Apply pre-commit whitespace and end-of-file fixes"
  ```
  Then append `# pre-commit file hygiene fixes` and `git rev-parse HEAD` to `.git-blame-ignore-revs`.

- [ ] **Step 5: Install the hook and commit**
  ```bash
  uv run pre-commit install
  git add .pre-commit-config.yaml .git-blame-ignore-revs
  git commit -m "ci: Add pre-commit hooks (ruff, mypy, uv-lock, zizmor, file hygiene)"
  ```
  The installed hook runs on this commit and must pass.

### Task 10: GitHub Actions CI and Dependabot

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/dependabot.yml`

**Interfaces:**
- Consumes: `scripts/check_wheel.py` (Task 1), `.pre-commit-config.yaml` (Task 9), and `fail_under` (Task 7).

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

  Global settings, all required:
  - `name: CI`
  - `on`: `push` to `branches: [main]`, `pull_request`, `workflow_dispatch`
  - `permissions: {}` at the top level, and `permissions: { contents: read }` on every job
  - `concurrency: { group: "${{ github.workflow }}-${{ github.ref }}", cancel-in-progress: "${{ github.event_name == 'pull_request' }}" }`
  - Every job runs on `ubuntu-latest`. Every checkout is `actions/checkout@<SHA>  # v7.0.1` with `with: { persist-credentials: false }`.
  - Every `astral-sh/setup-uv@<SHA>  # v10.3.0` gets `with: { version: "0.13.1", enable-cache: true }`, plus `python-version` where noted.

  Jobs:

  | Job | Steps after checkout and setup-uv |
  |---|---|
  | `lint` | `uv sync --locked`, then `uv run pre-commit run --all-files --show-diff-on-failure` |
  | `test` | `strategy: { fail-fast: false, matrix: { python-version: ["3.11", "3.12", "3.13", "3.14", "3.15"] } }`, setup-uv `python-version: ${{ matrix.python-version }}`. Steps: `uv sync --locked`, `uv run python --version` (it must print the matrix version; Review Focus 5), `uv run pytest` |
  | `integration` | **See Step 2.** |
  | `package` | `uv build`, `python3 scripts/check_wheel.py dist/*.whl`, `uv venv "$RUNNER_TEMP/wheeltest"`, `uv pip install --python "$RUNNER_TEMP/wheeltest" dist/*.whl`, `"$RUNNER_TEMP/wheeltest/bin/lctl" --version \| grep -F "4.1.0"`. Use the shell variable `$RUNNER_TEMP`, not `${{ runner.temp }}` inside `run:` (zizmor template-injection hygiene). |
  | `extension` | `actions/setup-node@<SHA>  # v7.1.0` with `node-version: 24`, `cache: npm`, `cache-dependency-path: \| vscode-lctl/package-lock.json vscode-lctl/webview-ui/package-lock.json` (multi-line). Then `npm ci` with `working-directory: vscode-lctl/webview-ui`, then with `working-directory: vscode-lctl`: `npm ci`, `npm run lint`, `npm run compile`, `npm run build:webview` |

- [ ] **Step 2: No `integration` job (decided by the 2026-10-14 probe), and record why in the spec**

  The probe ran the suite on 3.13 with `lctl[all]` installed: `3 failed, 1044 passed, 7 skipped, 1 error`. Per spec §4.3, red means no job.

  Replace spec §6 item 7 with:

  > Integration-test failures with all extras installed (probe 2026-10-14, py3.13):
  > (a) `tests/test_sk_stream.py` collection error: `semantic_kernel` fails to import (`ImportError: cannot import name 'Url' from 'pydantic.networks'`) under the `lctl[all]` resolution, so `[all]` installs a broken semantic-kernel.
  > (b) `tests/test_crewai_llm_trace.py::test_crewai_llm_trace_handler` and `::test_crewai_llm_trace_handler_missing_agent` fail against real CrewAI.
  > (c) `tests/test_claude_code.py::TestLCTLClaudeCodeTracer::test_link_to_git_history` failed, probably because the probe directory was not a git repo. Verify before acting.
  > Add an `integration` CI job once these are fixed.

- [ ] **Step 3: Write `.github/dependabot.yml`**
  ```yaml
  version: 2
  updates:
    - package-ecosystem: uv
      directory: /
      schedule: { interval: weekly }
      cooldown: { default-days: 7 }
      groups:
        python-minor-patch:
          update-types: [minor, patch]
    - package-ecosystem: github-actions
      directory: /
      schedule: { interval: weekly }
      cooldown: { default-days: 7 }
      groups:
        actions:
          patterns: ["*"]
    - package-ecosystem: npm
      directory: /vscode-lctl
      schedule: { interval: weekly }
      cooldown: { default-days: 7 }
      open-pull-requests-limit: 0   # security updates only (spec D3)
    - package-ecosystem: npm
      directory: /vscode-lctl/webview-ui
      schedule: { interval: weekly }
      cooldown: { default-days: 7 }
      open-pull-requests-limit: 0   # security updates only (spec D3)
  ```

- [ ] **Step 4: Verify locally**

  Run: `uv run pre-commit run --all-files`

  Expected: every hook passes, including `check-yaml` and `zizmor`. A zizmor finding is fixed in the workflow, never ignored without an inline justification.

- [ ] **Step 5: Dry-run the `package` job locally**

  Run:
  ```bash
  rm -rf dist
  uv build
  python3 scripts/check_wheel.py dist/*.whl
  uv venv $SP/wheeltest
  uv pip install --python $SP/wheeltest dist/*.whl
  $SP/wheeltest/bin/lctl --version
  ```
  Expected: the check exits 0, and `lctl --version` prints a line containing `4.1.0`.

- [ ] **Step 6: Commit**
  ```bash
  git add .github docs/superpowers/specs
  git commit -m "ci: Add GitHub Actions workflow and Dependabot config"
  ```

### Task 11: Repo hygiene and documentation drift

**Files:**
- Move: `task.md`, `BUG_REPORT.md` and `CODEBASE_REVIEW_REPORT.md` to `docs/archive/`
- Delete: `debug_sk.py`
- Modify: `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `docs/api.md`, and the `lctl/integrations/claude_code.py` module docstring (lines 1–40 only)

- [ ] **Step 1: Archive and delete**
  ```bash
  mkdir -p docs/archive
  git mv task.md BUG_REPORT.md CODEBASE_REVIEW_REPORT.md docs/archive/
  git rm debug_sk.py
  ```

- [ ] **Step 2: Update `CLAUDE.md` (drift only, no rewrite)**
  - **Facts:** "Python: 3.9+" becomes "3.11+". Remove hard-coded package version strings (`4.0.0`, `lctl-4.0.0-py3-none-any.whl`) and refer to `lctl/__init__.py`. Format-version `"4.0"` strings in JSON/YAML examples stay as they are.
  - **Architecture tree:** add `core/redaction.py`, `core/schema.py`, `streaming/` (`emitter.py`, `subscriber.py`, `websocket.py`), `metrics/` (`aggregators.py`, `collectors.py`, `exporters.py`) and `evaluation/` (`comparator.py`, `metrics.py`, `statistical.py`), each with a one-line purpose taken from its module docstring.
  - **Commands:** replace the Setup, Running Tests, Linting and Formatting, Building and Quick Commands sections with the uv workflow: `uv sync`, `uv run pytest`, `uv run pytest --no-cov -x`, `uv run ruff check --fix .`, `uv run ruff format .`, `uv run mypy`, `uv run pre-commit run --all-files`, `uv build`. Remove every `black` and `pip install -e ".[dev]"` reference.
  - **Duplication:** keep the integration-pattern guidance once, under "Integration Guide". Under "Code Patterns → 2. Integration Pattern", replace the duplicate with a one-line pointer to it.

- [ ] **Step 3: Update `README.md`**
  - Under the title, add the badge `[![CI](https://github.com/kimasplund/LLM-Context-Trace-Library/actions/workflows/ci.yml/badge.svg)](https://github.com/kimasplund/LLM-Context-Trace-Library/actions/workflows/ci.yml)`.
  - "Development Installation" (line ~97) and "Contributing → Development Setup" (line ~956): replace them with the uv workflow from Step 2, plus the pip alternative `pip install -e . --group dev` (pip ≥25.1). Remove `black lctl tests` and the `black` contribution step, and add `uv run pre-commit install` as a setup step.
  - Add one sentence to Contributing: "Repository owners: enable Dependabot alerts and security updates under Settings → Code security."
  - The user-facing `pip install git+…` lines are unchanged.

- [ ] **Step 4: Update `CHANGELOG.md`**
  - Insert `## [Unreleased]` above `## [4.1.0]` with a `### Changed` list:
    - uv project management with a committed `uv.lock`
    - ruff format replacing black, plus a modern ruff rule set
    - mypy type checking and `py.typed`
    - pytest warnings as errors and branch coverage
    - GitHub Actions CI and Dependabot
    - pre-commit hooks
  - Change `## [4.1.0] - 2025-01-19` to `## [4.1.0] - 2026-01-19`.

- [ ] **Step 5: Update `docs/api.md`**
  - In the `GET /api/health` example (around line 2023), make the response match `lctl/dashboard/app.py:281`: `{"status": "ok", "version": "4.1.0", "streaming": true}`.
  - Change `*This documentation was generated for LCTL v4.0.0*` to `v4.1.0`.
  - Leave every `"4.0"` format-version string alone.

- [ ] **Step 6: Fix the `claude_code.py` module docstring**

  Replace the stale `CLAUDE_TOOL_INPUT_*` env-var hook example (lines ~7–30) with a short description of the real mechanism, taken from `generate_hooks()` in the same file:
  - `lctl claude init` writes `.claude/hooks/*.sh`.
  - Claude Code passes the hook payload as JSON on stdin.
  - The hook base64-encodes it and hands it to Python, which calls the tracer.

  Change the docstring only. No code changes.

- [ ] **Step 7: Verify**

  Run:
  ```bash
  grep -nE "black|3\.9\+|4\.0\.0" CLAUDE.md README.md docs/api.md
  uv run pre-commit run --all-files
  uv run pytest --no-cov -q
  ```
  Expected: the grep prints nothing, every hook passes, and the tests show `1041 passed, 14 skipped`.

- [ ] **Step 8: Commit**
  ```bash
  git add -A docs CLAUDE.md README.md CHANGELOG.md lctl/integrations/claude_code.py
  git commit -m "docs: Archive scratch reports and fix documentation drift"
  ```

### Task 12: Acceptance verification

No new files. This proves spec §5 criteria 1–8. Criterion 9 (CI green on the PR) needs a push, which is decided with the user afterwards.

- [ ] **Step 1: Fresh clone, locked sync**

  Run:
  ```bash
  rm -rf $SP/fresh
  git clone -q . $SP/fresh
  cd $SP/fresh
  uv sync --locked
  uv lock --check
  ```
  Expected: exit 0.

- [ ] **Step 2: Lint, format and types (criteria 2 and 4)**

  In the fresh clone, run: `uv run pre-commit run --all-files && uv run ruff check . && uv run ruff format --check . && uv run mypy`

  Expected: all pass.

- [ ] **Step 3: Tests (criterion 3)**

  In the fresh clone, run: `uv run pytest -q`

  Expected: at least 1041 passed, no warnings summary, and coverage at or above `fail_under`.

- [ ] **Step 4: No behavior change (criterion 5)**

  Run Task 3 Step 4's three diff commands from the fresh clone. Expected: all empty.

- [ ] **Step 5: Wheel (criterion 6)**

  Run Task 10 Step 5 from the fresh clone. Expected: as stated there.

- [ ] **Step 6: Extension (criterion 7)**

  Run Task 8 Step 3 from the fresh clone. Expected: all exit 0, and `git status --porcelain` is empty.

- [ ] **Step 7: Tree contents (criterion 8)**

  Run: `git ls-files | grep -E '^(task\.md|BUG_REPORT\.md|CODEBASE_REVIEW_REPORT\.md|debug_sk\.py)$|vite\.config\.js'`

  Expected: no output.

- [ ] **Step 8: Report**

  Summarize each criterion's command output for the user. Then invoke superpowers:finishing-a-development-branch to decide whether to push and open a PR, which is also how criterion 9 gets checked.

---

## Appendix A: `snapshot_api.py`

```python
"""Print the public surface of LCTL's packages as sorted JSON (acceptance check 5)."""

import importlib
import json
import sys

MODULES = [
    "lctl",
    "lctl.core",
    "lctl.integrations",
    "lctl.streaming",
    "lctl.metrics",
    "lctl.evaluation",
]
out = {}
for name in MODULES:
    mod = importlib.import_module(name)
    names = getattr(mod, "__all__", None)
    out[name] = (
        sorted(names) if names is not None else sorted(n for n in dir(mod) if not n.startswith("_"))
    )
json.dump(out, sys.stdout, indent=1, sort_keys=True)
print()
```
