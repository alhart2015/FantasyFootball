# depth_charts 2025+ — derivation implementation plan

**Spec:** `docs/superpowers/specs/2026-05-15-depth-charts-2025-derivation-design.md`
**Branch:** `feat/depth-charts-2025`

---

## Phase 1 — Derivation helper + unit tests (1 file + 1 test file)

### Task 1: implement `_derive_weekly_snapshots_from_new_format`

File: `src/projections/ingest/depth_charts.py`

- Add `_NEW_FORMAT_REQUIRED_COLS = {"dt", "team", "gsis_id", "pos_abb", "pos_slot", "pos_rank"}`.
- Add module-level helper `_derive_weekly_snapshots_from_new_format(raw: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame` that returns a DataFrame validated against `DepthChartsSchema`.
- Steps in order: parse `dt` to tz-aware UTC; melt schedules to `(season, week, team, kickoff)` (one row per team-game); for each (season, week, team) find the largest snapshot `dt` strictly before `kickoff` (groupby-apply or a sort-merge); attach that snapshot to the team-week row; pull the snapshot's rows for that team; filter `pos_rank == 1`; filter `pos_abb` to `Position` values; synthesize `depth_team = pos_abb + str(pos_slot)`; set `depth_rank = min(10, max(1, int(pos_slot)))`; emit columns per `_KEEP`; `DepthChartsSchema.validate(df)` and return.
- Log a warning with a count of (team, week) cells that had no closest-prior snapshot.

### Task 2: unit tests in `tests/test_ingest/test_depth_charts.py`

- One synthetic-fixture test per Spec §6 unit-test bullet (6 tests).
- Use small fixtures: 2 teams, 2 weeks, 2-3 snapshots, 5-10 player rows per snapshot. Hand-construct kickoffs so the closest-prior rule is unambiguous.
- Each test asserts: row count, specific (team, week, gsis_id) rows present/absent, depth_team/depth_rank values, position filter.

**Verify Phase 1:** `pytest tests/test_ingest/test_depth_charts.py -v`. Stop and report before Phase 2.

---

## Phase 2 — Dispatch + refresh_depth_charts wiring (1 file + 1 test file)

### Task 3: dispatch in `_normalize_one_season`

- Replace the `raise NotImplementedError` branch with:
  ```python
  if "dt" in raw.columns:
      return _derive_weekly_snapshots_from_new_format(raw, schedules)
  raise NotImplementedError(...)  # genuinely unknown shape
  ```
- Thread `schedules` parameter through `_normalize_one_season`.

### Task 4: wire schedules into `refresh_depth_charts`

- Add optional `schedules: pd.DataFrame | None = None` parameter.
- Inside the per-season loop: if `schedules is None` AND raw payload is new-format, call `read_partition(data_root / "raw", "schedules", season=season)`. If the partition is missing, raise `FileNotFoundError("ingest schedules before depth_charts for season=...")`.
- Pre-2025 path: schedules parameter unused; legacy `_normalize_one_season` ignores it.

### Task 5: extend `tests/test_ingest/test_depth_charts.py`

- Test that `refresh_depth_charts` on a legacy-format fixture takes the legacy path (existing tests cover this; add an explicit "new path is skipped" assertion).
- Test that `refresh_depth_charts` on a new-format fixture reads schedules from disk when not passed.

**Verify Phase 2:** `pytest tests/test_ingest/test_depth_charts.py -v` + `mypy src tests` + `ruff check src tests` + `ruff format --check src tests`. Stop and report before Phase 3.

---

## Phase 3 — Real-data ingest + commit

### Task 6: real-data run

- From the worktree, invoke `refresh_depth_charts(Path('data'), seasons=[2025])`.
- Assert partition written; print row count, week range, sample rows.
- Verify counts are plausible (Spec §6: 5,000-10,000 rows).

### Task 7: update `TODO.md` #34 entry

- Change "**Status.** Captured 2026-05-11..." to add a closure paragraph: spec+plan+impl on `feat/depth-charts-2025`, 2025 partition row count, design choices (closest-prior rule, `pos_rank == 1` filter, `depth_rank = pos_slot`), any plan-vs-execution deviations.

### Task 8: final forced-verification checklist

- `pytest -v` — full suite.
- `mypy src tests` — zero violations.
- `ruff check src tests` — zero violations.
- `ruff format --check src tests` — no drift.
- `pytest -v -k "ingest or store or schemas"` — the dtype-regression seam.

### Task 9: commit

- Per `feedback_branching.md`: spec + plan + impl all land on `feat/depth-charts-2025`, then PR to main.
- Commits: separate spec, plan, impl, TODO update — clean log.
- Mind `project_pre_commit_venv_quirk.md` — prepend `.venv/Scripts` to PATH before `git commit`.
