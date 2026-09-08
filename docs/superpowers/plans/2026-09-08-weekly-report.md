# Weekly report — implementation plan

Spec: `docs/superpowers/specs/2026-09-08-weekly-report-design.md` · Branch `feat/weekly-report`

TDD throughout. Phases are barriers: run the gate before starting the next one.

**Gate (every phase end):**
```
pytest -v -k "context or midseason or scripts or web"
mypy src tests && ruff check src tests scripts && ruff format --check src tests scripts
```

**The rule that governs this whole plan: these four tools produce numbers the user acts on.
A silent regression is worse than no wrapper.** So every migration task ends by running the
migrated script against the live league and diffing its output against the pre-migration
output, captured in Task 0. A section whose output changes must explain why before the commit.

---

## Task 0 — capture the baseline BEFORE touching anything

Run all four against the live league, save stdout to
`.worktrees/feat-weekly-report/_baseline/{tool}.txt` (gitignored scratch, not committed):

```
python scripts/projected_standings.py  > _baseline/standings.txt  2>&1
python scripts/start_sit.py            > _baseline/start_sit.txt  2>&1
python scripts/waiver_recommender.py   > _baseline/waivers.txt    2>&1
python scripts/trade_analyzer.py       > _baseline/trades.txt     2>&1
```

Seeds are fixed (`--seed 0` defaults), so these are reproducible and a diff is meaningful.
**Without this the migration is unverifiable** — "it still looks right" is not a check.

---

## Phase 1 — the context

### T1 — `InSeasonContext`, fields only

`src/projections/midseason/context.py`. The frozen dataclass from spec §4, plus
`build_context(args, *, require_team_id=True) -> InSeasonContext` doing what the four scripts
each do today: resolve the target, resolve credentials, fetch the payload, load the config,
load and validate the pool, load the id_map, read weekly_stats, build `MyTeamRun`, derive the
week.

**Tests** (`tests/test_midseason/test_context.py`), driven from payload fixtures with the
network stubbed:
- the pool comes out `_PYARROW_STR` on `gsis_id`, schema-validated, with `is_rookie` attached
- `weekly_stats` missing → empty frame, not an exception (matches today's `try/except`)
- `week` equals `my_team.week`, and an explicit `args.week` overrides it
- `config` comes from `league_config.json`
- **a derived config differing from the file emits a warning naming the difference** — measured
  identical on 2026-09-08, so this exists to catch the day it stops being true

### T2 — memoised accessors

`availability()`, `variance_params()`, `onteam_payload()`, `roster()`.

**Tests:**
- each computes once and is reused (call twice, assert the underlying loader ran once)
- `onteam_payload()` requests `scoring_period=ctx.week` and
  `limit=n_teams * (sum(roster_slots.values()) + 2)` — the copy-pasted line, now in one place
- `roster()` returns only my team, and returns a **copy** (mutating it must not affect the next
  caller — the context is shared by four sections and frozen only at the top level)

**Gate.**

---

## Phase 2 — migrate the two simplest (2 files)

### T3 — `projected_standings.report(ctx, args)`
### T4 — `start_sit.report(ctx, args)`

Each: move the body of `main` after arg-parsing into `report(ctx, args)`, leave `main` as
`args → build_context → report`. No behaviour change.

**Per task:** re-run against the live league and `diff` against the Task 0 baseline. **Expect
zero difference.** Any difference is a regression until explained.

**Tests:** the existing script tests still pass unchanged (that is the point), plus one per
script asserting `report` can be called twice with the same context and produces the same
output — the property `weekly_report` depends on.

**Gate.**

---

## Phase 3 — migrate the two that share the ONTEAM fetch (2 files)

### T5 — `waiver_recommender.report(ctx, args)`

The interesting one: it currently does its own `build_my_team`, its own `rostered_limit`, and
its own ONTEAM fetch. All three come from the context now. Its **free-agent** fetch
(`statuses=("FREEAGENT","WAIVERS")`, `limit=400`) stays local — different request, and only
this tool wants it.

### T6 — `trade_analyzer.report(ctx, args)`

The one with real behaviour change to watch: it moves from `build_league_config(payload)` to
the file, and from an unvalidated pool to the validated one. **Measured identical on
2026-09-08**, so the diff should be empty — but this is the task where a non-empty diff is
plausible, and if it appears it is a finding, not a nuisance.

**Gate**, plus `diff` on all four baselines.

---

## Phase 4 — the report

### T7 — `scripts/weekly_report.py`

Builds one context, calls the four `report`s in spec §6 order, wraps each so a failure prints
under its own header and the rest continue. Flags: the five shared, `--only`, `--skip`,
`--fast`, `--n-sims`, `--seed`.

**Tests:**
- one context is built, and `fetch_league_payload` is called **once** for the whole report
- a section raising does not stop the others, and its failure is visible in the output
- `--only` / `--skip` select correctly, and an unknown name is an error rather than a silent no-op
- register in `tests/test_scripts/test_league_cli_defaults.py`'s `PARSERS`

### T8 — verify and document

- Run it. Confirm every section matches its baseline, and time it against the 31s of running
  the four separately.
- Confirm `fetch_league_payload` really is called once (log or counter), and that the historical
  `weekly_stats` scan happens once rather than up to eight times.
- Update `project_management.md`; open the issue for the `points_to_date={}` TODO that §8 cut.
- **Full gate**, not the subset. Paste real output.

---

## Risks specific to execution

| Risk | Mitigation |
|---|---|
| A migration silently changes a number | Task 0 baseline + a diff at every migration task. This is the whole reason Task 0 is first. |
| `--week` override semantics differ per tool today (waivers/start-sit honour it, standings/trades do not) | Decide **once** in T1 and write it down: `--week` sets `ctx.week` and therefore moves every section together. That is a real behaviour change for standings and trades, so it goes in the PR description, not silently in a diff. |
| Four sections sharing one frozen context | Accessors hand out copies; a test pins that mutating `ctx.roster()` does not leak. |
| `report()` extraction drags I/O along | Each `report` must read only from `ctx` and `args` — no `fetch_*`, no `read_parquet`. A test greps each script's `report` body for those names, the way `tests/test_web/test_app.py` already polices Flask imports in the view layer. |
