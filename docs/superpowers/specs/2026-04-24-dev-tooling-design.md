# Dev Tooling + Contributor Docs — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-24
**Author:** alden + claude
**Sub-project of:** FantasyFootball (cross-cutting; supports all sub-projects)
**Resolves:** TODO #0

---

## 1. Overview

Lock the development experience in place before Plan 2 (Ingest expansion + features) grows the codebase. Three deliverables: pre-commit hooks that enforce lint/format/typecheck on every commit, a `CONTRIBUTING.md` that absorbs all detailed onboarding/workflow/recipe content, and a trim of `CLAUDE.md` to leave it terse and high-signal. Plus a small `README.md` refresh.

### 1.1 Goals

- Catch lint, format, and type regressions at commit time so they never reach a PR.
- Keep `CLAUDE.md` lean — it loads into Claude's context every interaction; every line costs context-window budget.
- Move detailed setup, workflow walkthroughs, and pattern recipes into `CONTRIBUTING.md` where humans can find them and Claude can read on demand.
- Document the `pytest -v before opening a PR` rule, since we deliberately don't run GitHub Actions CI.

### 1.2 Non-goals

- GitHub Actions / CI infrastructure (deliberately deferred).
- Test gates in pre-commit (would slow commits; pytest-as-PR-gate is the chosen alternative).
- Coverage reporting tooling.
- Doc site generation (Sphinx, MkDocs, etc.).
- Per-subsystem `CLAUDE.md` files — premature with one sub-project.
- Architecture / data-layout standalone docs — content is short enough to live in `CONTRIBUTING.md`.

---

## 2. Deliverables

### 2.1 `.pre-commit-config.yaml`

Hooks in order:

1. **`pre-commit-hooks`** (built-in housekeeping):
   - `trailing-whitespace`
   - `end-of-file-fixer`
   - `check-merge-conflict`
   - `check-added-large-files` (default 500 KB max — prevents stray parquet binaries from being committed; data dirs are gitignored anyway)
   - `check-yaml`, `check-toml`, `check-json`
2. **`ruff` (`astral-sh/ruff-pre-commit`)** with two hooks:
   - `ruff` with `--fix` (lint + autofix)
   - `ruff-format`
3. **`mypy` (`pre-commit/mirrors-mypy`)** running `mypy src tests`. Whole-codebase scope (mypy doesn't reason well about isolated files). The `additional_dependencies` field will need to list runtime deps so mypy can find type info — at minimum `pydantic>=2`, `pandas-stubs` (if we add it), `pandera` (already silenced via overrides), `numpy`. Verify what's actually needed at install time.

`pre-commit install` is documented in `CONTRIBUTING.md` as a one-time step. Add `pre-commit` to `[project.optional-dependencies] dev` in `pyproject.toml`.

### 2.2 `[tool.ruff.format]` adoption

Add an empty `[tool.ruff.format]` section to `pyproject.toml` (defaults are sensible — ruff format is opinionated black-compatible, no config needed for v1). Run `ruff format src tests` once against the existing codebase as a single dedicated commit so future format-related diffs are noise-free.

### 2.3 `CONTRIBUTING.md`

Comprehensive contributor doc. Six sections:

1. **Setup** — clone; venv (`python -m venv .venv && . .venv/Scripts/activate` on Windows bash, `.venv\Scripts\Activate.ps1` on PowerShell); `pip install -e ".[dev]"`; `pre-commit install`. Windows-specific notes: `.worktrees/` is gitignored; venv reactivation needed per shell; `gh` CLI install path may not be on PATH after winget install.
2. **Daily commands** — `pytest -v`, focused subsets (`pytest tests/test_schemas/ -v`, single test by `::` selector), `mypy src tests`, `ruff check src tests`, `ruff check --fix src tests`, `ruff format src tests`. Note that pre-commit runs lint/format/typecheck automatically.
3. **Workflow** — full spec → plan → execute walkthrough using superpowers skills. Covers: brainstorming with the user, writing a spec at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, writing a TDD plan at `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`, executing via subagent-driven development with two-stage review for tasks with real logic, branch convention `feat/<short-name>`, worktrees at `.worktrees/<branch-name>`, conventional commits with examples (`feat(scoring):`, `fix(ingest):`, `chore:`, `refactor:`, `docs:`).
4. **PR checklist** — explicit "**Run `pytest -v` before opening a PR**" since we don't have CI; `gh pr create` recipe with PR body template; mypy + ruff are caught by pre-commit so they shouldn't fail in PR review; update `project_management.md` and/or `TODO.md` when the work is foundational.
5. **Adding a new ingest source** — full step-by-step recipe with code skeleton drawn from `weekly_stats.py`. Covers: `_fetch_raw_*` thin wrapper for monkeypatching, `_normalize_one_*` step (rename → dtype coercion → team normalization → position filter → `_PYARROW_STR` cast on string columns), `Schema.validate()` with **reassignment** (`df = SCHEMA.validate(df)`), `write_partition()`, `record_manifest()`. Include the test pattern using a `fake_*_df` fixture in `tests/test_ingest/conftest.py` and `monkeypatch.setattr` against the `_fetch_raw_*` symbol.
6. **Adding a new pandera schema** — full guidance: column `Series[type]` annotations, `pa.Field(...)` constraints (`isin`, `ge`, `le`, `str_matches`), `Series[str]` requires `pd.StringDtype("pyarrow")` upstream, mixed-int+NA needs `pd.Int64Dtype()`, `class Config: strict = "filter"` requires reassignment to actually filter, where to import from (`pandera.pandas` not `pandera` for pandera 0.20+).

### 2.4 `CLAUDE.md` trim pass

Reorganize to keep only what Claude needs in every interaction. Move detail to `CONTRIBUTING.md` with pointers.

**Stays in `CLAUDE.md`:**
- "Correctness over speed" (principle).
- "Tests are the guardrail" (principle, lightly compressed).
- "Project shape" (sub-project names + status pointer; no commands).
- "Stay in sync" (workflow rule — read state files first, update at end of meaningful work).
- "Conventions you must follow" — concise rules only, no examples or rationale paragraphs:
  - `GsisId` is canonical; `validate_gsis_id` is the only sanctioned constructor.
  - `Position` / `Team` / `RosterSlot` enums — never the strings.
  - `normalize_team_code` for any team string from external data.
  - `df = SCHEMA.validate(df)` (reassignment) at every module boundary.
  - `pd.StringDtype("pyarrow")` for nullable string columns; `pd.Int64Dtype()` for nullable ints.
  - Scoring layer is the only place that knows what counts as a fantasy point.
  - `Distribution` is a Protocol; `runtime_checkable` is structural-only.
  - `store.write_partition` / `store.read_partition` are the only sanctioned parquet I/O.
  - Ingest pattern: see `CONTRIBUTING.md`.
- "Reuse before writing" — compressed to principle + pointer.
- "Pointers to deep docs" — explicit links to `CONTRIBUTING.md`, `project_management.md`, `TODO.md`, `docs/superpowers/specs/`, `docs/superpowers/plans/`.
- "Agent Directives: Mechanical Overrides" — kept fully. These are Claude-behavior rules, not human-onboarding content; they belong in `CLAUDE.md`.

**Moves to `CONTRIBUTING.md`:**
- Workflow walkthrough (deeper than the one-line rule that stays in `CLAUDE.md`).
- Commands section (entire block).
- The detailed cross-cutting conventions explanations (rationale, examples, edge-case discussion). The terse rule list above is what stays.

The agent directives' ID hygiene rule (#11) stays in `CLAUDE.md` — it's a Claude-behavior rule, not human-onboarding content, and it's already short.

**Adds to `CLAUDE.md`:**
- One line near the top: "For setup, daily commands, full workflow walkthrough, and pattern recipes, see `CONTRIBUTING.md`."

### 2.5 `README.md` refresh

Currently just `# FantasyFootball` (one line). Expand to a short orientation:

- One-paragraph project description (probabilistic NFL fantasy football toolkit; sub-projects).
- Status snippet (foundations merged; what's working today).
- "If you're a contributor, see `CONTRIBUTING.md`. If you're Claude, see `CLAUDE.md`. Current status and next actions live in `project_management.md`."

Two paragraphs max. Not a tutorial; a signpost.

---

## 3. Order of operations

Single sequenced micro-plan (one branch, one PR):

1. Add `pre-commit` to `[project.optional-dependencies] dev` in `pyproject.toml`.
2. Add `[tool.ruff.format]` empty block to `pyproject.toml`.
3. Run `ruff format src tests` against the existing codebase. Commit as `style: apply ruff format to existing codebase`.
4. Write `.pre-commit-config.yaml`. Run `pre-commit install` and `pre-commit run --all-files` to confirm clean. Commit as `chore: add pre-commit hooks (ruff, ruff-format, mypy, housekeeping)`.
5. Write `CONTRIBUTING.md` (all six sections, full depth). Commit as `docs: add CONTRIBUTING.md with setup, workflow, and pattern recipes`.
6. Trim `CLAUDE.md` per §2.4. Commit as `docs(CLAUDE): trim to terse high-signal; defer detail to CONTRIBUTING.md`.
7. Refresh `README.md` per §2.5. Commit as `docs(readme): expand from placeholder to short orientation`.
8. Update `TODO.md`: remove TODO #0 entry. Update `project_management.md`: add to decision log; remove from next-action; advance to Plan 2 as recommended next step. Commit as `docs(pm): close TODO #0; advance next-action to Plan 2`.

Verify after each commit: `pre-commit run --all-files` clean. Push branch, open PR.

---

## 4. Testing

Mostly verification rather than test-writing:

- `pre-commit run --all-files` exits clean after install.
- `pytest -v` still passes (no behavioral changes; format-only diffs).
- `mypy src tests` clean.
- `ruff check src tests` clean.
- `ruff format --check src tests` clean.
- Manually confirm a deliberate violation (e.g., trailing whitespace) is caught by pre-commit on a test commit.

No new pytest tests are needed — the configuration files are themselves the deliverable, and pre-commit's own `--all-files` is the validation.

---

## 5. Out of scope (explicit)

- GitHub Actions CI (deferred indefinitely per user direction).
- Test gates in pre-commit.
- Coverage reporting.
- Doc site generation.
- Per-subsystem `CLAUDE.md` files (premature with one sub-project).
- Standalone `docs/architecture.md` or `docs/data-layout.md` (content lives in `CONTRIBUTING.md` until it outgrows that).
- `pandas-stubs` adoption (can land separately if/when mypy strictness on DataFrame code becomes valuable).

---

## 6. Open questions

None blocking. Two minor judgment calls during implementation:

- `mypy` pre-commit hook may need `additional_dependencies` tuned per actual environment behavior. Adjust at install time based on what mypy complains about.
- If `ruff format src tests` produces a large diff, the resulting commit will be noisy by definition. That's expected — single-purpose format commits are easy to skip in `git blame` via `--ignore-rev` or `.git-blame-ignore-revs` (consider adding the file as a follow-up, not part of this spec).
