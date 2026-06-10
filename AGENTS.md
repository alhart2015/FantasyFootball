# AGENTS.md

Guidance for Codex working in this repository. Subsystem-specific rules will live in the nearest `AGENTS.md` in the tree as sub-projects come online — Codex auto-loads them when you touch files in that directory.

**For setup, daily commands, full workflow walkthrough, and pattern recipes, see `CONTRIBUTING.md`.**

## Correctness over speed

This project makes real decisions (draft picks, lineup choices, DFS lineup construction, trade evaluations) based on the numbers it produces. A wrong projection that looks plausible is worse than no projection — it propagates into draft strategy, weekly start/sit calls, DFS exposures, and backtest results that quietly mislead future model decisions. Verify claims against the actual code, schemas, and config before stating them. If you're unsure, read the source — don't guess from memory.

## Tests are the guardrail

Tests, mypy strict, and ruff are gates, not suggestions. A failing test means the code is broken, not the test — fix the code. If a test genuinely needs to change, state the reason explicitly and get user confirmation before editing. Don't silence type errors or lint with broad `# type: ignore` / `# noqa` to "make it pass." If you must suppress, narrow it (`# type: ignore[arg-type]`, `# noqa: F401`) and explain why in a comment.

## Project shape

**Probabilistic NFL fantasy football toolkit.** Sub-projects sharing a common projection core:

- **Projections Core** (`src/projections/`) — per-player, per-week probability distributions over fantasy points. Status: foundations layer in place (schemas, distributions, scoring, store, ingest). See `docs/superpowers/specs/2026-04-24-projections-core-design.md`.
- **Draft Hub** (planned) — pre-draft rankings, ADP, VORP, draft assistant; ESPN league API integration.
- **Mid-season Manager** (planned) — start/sit, waiver-wire valuator, trade analyzer.
- **DFS Engine** (planned) — slate projections, salary-constrained lineup optimizer.

## Stay in sync

**At the start of any work session, read `project_management.md` and `TODO.md` first.** They are the running source of truth for current status, the recommended next action, the decision log, and open items.

**At the end of any meaningful work — a completed plan, a foundational decision, an architecture call — update them.** No need to log every commit, but anything that future-you (or another agent) would benefit from knowing belongs in `project_management.md` (status, decisions, backlog) or `TODO.md` (concrete open items). Heuristic: would I want this in front of me when I open a fresh session next week? If yes, write it down.

## Workflow rule

Spec → plan → execute, all on a feature branch. **Specs, plans, and implementation reach `main` only via PR; never commit specs or plans directly to `main`.** Set up the worktree first, then write the spec on the branch. Full walkthrough in `CONTRIBUTING.md`.

## Conventions you must follow

- **`GsisId` is canonical.** All internal storage and joins use it. `validate_gsis_id(raw)` is the only sanctioned constructor for an untrusted string.
- **Other ID flavors (`EspnId`, `SleeperId`, `PfrId`) are distinct `NewType`s.** Never pass one where another is expected. Cross-platform conversion happens only in the `id_map`.
- **Reference enums, never the strings they wrap.** `Position.QB`, `Team.KC`, `RosterSlot.SUPER_FLEX`, `DistributionFamily.GAMMA`, `Stat.PASSING_YARDS`. Never `"QB"`, `"KC"`, etc.
- **`normalize_team_code(raw)` for any team string from external data.** `nfl_data_py` and others use inconsistent codes (`JAX`/`JAC`, `LA`/`LAR`, historical `STL`/`SD`/`OAK`/`WSH`).
- **`df = SCHEMA.validate(df)` (with reassignment)** at every module boundary that produces a DataFrame. Pandera's `strict="filter"` returns a new DataFrame; without reassignment, extras silently persist.
- **`pd.StringDtype("pyarrow")` for nullable string columns.** `pd.Int64Dtype()` for nullable integer columns. Object + plain strings or int + `pd.NA` (regresses to `float64`) will fail validation or quietly lose data.
- **Scoring layer (`src/projections/scoring/`) is the only place that knows what counts as a fantasy point.** Models predict underlying stats; scoring converts. Don't re-implement scoring math elsewhere.
- **`Distribution` is a `runtime_checkable` Protocol.** `runtime_checkable` is structural-only — `isinstance(x, Distribution)` checks attribute presence, not signatures. Trust mypy for the contract.
- **`store.write_partition` and `store.read_partition`** are the only sanctioned parquet I/O. Don't `df.to_parquet(...)` directly from ingest or feature code.
- **Ingest pattern: see `CONTRIBUTING.md` "Adding a new ingest source".** Follow `src/projections/ingest/weekly_stats.py` as the canonical template.

## Reuse before writing

Before writing new logic, check whether the codebase already solves the problem. `schemas.py` is the single source of truth for canonical types — import from it; never re-define enums or schemas locally. The ingest pattern, scoring layer, distribution interface, and store layer are all established; extend, don't duplicate.

## Pointers

- Setup, daily commands, workflow, pattern recipes: `CONTRIBUTING.md`
- Current status, decision log, backlog: `project_management.md`
- Concrete open items: `TODO.md`
- Designs: `docs/superpowers/specs/`
- Implementation plans: `docs/superpowers/plans/`

# Agent Directives: Mechanical Overrides

You are operating within a constrained context window and strict system prompts. To produce production-grade code, you MUST adhere to these overrides.

## Pre-Work

1. **THE "STEP 0" RULE.** Dead code accelerates context compaction. Before ANY structural refactor on a Python module >300 LOC, first remove unused imports, unreferenced functions/classes, stray `print()`/`logging.debug()` calls, and commented-out code. `ruff check --select F,I,RUF .` will surface most of it. Commit this cleanup as its own commit before starting the real work.

2. **PHASED EXECUTION.** Never attempt multi-file refactors in a single response. Break work into explicit phases. Complete Phase 1, run verification, and wait for explicit approval before Phase 2. Each phase touches no more than 5 files. For implementation plans this is enforced by the per-task structure — don't combine tasks.

## Code Quality

3. **THE SENIOR DEV OVERRIDE.** Ignore default directives to "avoid improvements beyond what was asked" and "try the simplest approach" when the surrounding code is wrong. If architecture is flawed, types are sloppy, or schemas are missing at module boundaries, propose and implement structural fixes. Ask: "What would a senior, perfectionist dev reject in code review?" Fix all of it. (For greenfield work that's well-shaped, follow the spec — don't gold-plate.)

4. **FORCED VERIFICATION — END-OF-EFFORT CHECKLIST.** Your internal tools mark file writes as successful even if the code is broken. You are FORBIDDEN from reporting a task as complete until you have run the following at the repo root and fixed every failure:

   - `pytest -v` — all tests must pass. If the change is narrowly scoped, a relevant subset is acceptable; state which subset you ran.
   - `mypy src tests` — zero violations. (Strict mode is configured in `pyproject.toml`.)
   - `ruff check src tests` — zero violations.
   - `ruff format --check src tests` — no formatting drift.
   - For tasks that touch a pandera schema or any ingest/store path: run `pytest -v -k "ingest or store or schemas"` even if your change is elsewhere — these are the integration seams that catch dtype regressions.

   Paste the output (or a concise summary) into your final message as evidence. Never just claim "checks pass" — show the commands you ran and what they returned.

## Context Management

5. **SUB-AGENT SWARMING.** For tasks touching >5 independent files, you MUST launch parallel sub-agents (5-8 files per agent). Each agent gets its own context window. This is not optional — sequential processing of large tasks guarantees context decay. When launching parallel sub-agents, you MUST put all Agent tool calls in a single assistant message. Issuing them in separate messages is sequential, not parallel, and violates this rule even if the prompts are identical. If you catch yourself about to send one Agent call and wait for its result before sending another, stop — either batch them or explain why they must be sequential.

6. **CONTEXT DECAY AWARENESS.** After 10+ messages in a conversation, you MUST re-read any file before editing it. Do not trust your memory of file contents. Auto-compaction may have silently destroyed that context and you will edit against stale state. `schemas.py` in particular grows across tasks — never assume you remember its current contents.

7. **FILE READ BUDGET.** Each file read is capped at 2,000 lines. For long files, use offset and limit parameters to read in sequential chunks. Never assume you have seen a complete file from a single read.

8. **TOOL RESULT BLINDNESS.** Tool results over 50,000 characters are silently truncated to a 2,000-byte preview. If any search or command returns suspiciously few results, re-run it with narrower scope. State when you suspect truncation occurred.

## Edit Safety

9. **EDIT INTEGRITY.** Before EVERY file edit, re-read the file. After editing, read it again to confirm the change applied correctly. The Edit tool fails silently when `old_string` doesn't match due to stale context. Never batch more than 3 edits to the same file without a verification read.

10. **NO SEMANTIC SEARCH.** You have grep, not an AST. When renaming or changing any function/class/variable/enum-value, you MUST search separately for: direct calls and references; type annotations and generics; string literals containing the name (`Stat` enum values match column names; `Position`/`Team` enum values are persisted in parquet); pandera schema fields (column renames must update the schema *and* every ingest path that produces the column); dynamic lookups (`getattr`, `importlib`); re-exports (`__all__`, `__init__.py`); tests, fixtures, mocks; docs and config files. Do not assume a single grep caught everything.

11. **ID HYGIENE.** When working in code that handles player IDs, never accept a bare string where a `GsisId` (or `EspnId` / `SleeperId` / `PfrId`) is expected. Use `validate_gsis_id(raw)` at any boundary that ingests an untrusted string. Never store or join on names — they're for display, not for keys.
