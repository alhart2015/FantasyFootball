# RB PBP Features Integration — Design

**Status:** approved (brainstorming, 2026-05-01). Ready for implementation plan.
**Date:** 2026-05-01
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** PBP Feature Family Probe (PR #20, merged at `6120ff1`) — the probe shipped 4 PBP team-level compute fns + assembler in `src/projections/features/pbp_team_features.py` and returned `SIGNAL` on `(RB, rushing_yards)` in both augment and swap modes (composite RMSE delta -0.012 to -0.013 fpts ADOPT). This spec promotes that override-parquet path into the production RB feature pipeline.
**Branch:** `feat/rb-pbp-features` cut from `main` at `6120ff1`.

---

## 1. Overview

The PBP Feature Family Probe greenlit the 4-feature family for RB integration. This spec executes that integration: append 4 nullable-float columns to `RbFeaturesSchema`, modify `build_rb_features` to compute and join the 4 PBP team-features onto RB rows, refresh the RB feature cache, and run the full backtest + adoption gate to verify the probe's ADOPT prediction holds.

The work is intentionally narrow: RB only. QB / WR / TE feature schemas and builders are not touched. The probe explicitly returned regression on QB augment-mode (`passing_yards` +0.45 fpts) and net-zero on WR / TE; integrating PBP team-features into those positions is a separate spec, not scoped here.

The shipping decision is binary and bound to the production model class (BaselineModel, "Model A"): if the adoption gate's per-position verdict on `(BaselineModel, RB)` is `ADOPT`, ship. If it is `MARGINAL` or `DO_NOT_ADOPT`, revert per Plan 9 discipline. Other model classes' verdicts (`LightGBMTunedModel`, `LightGBMNbModel`, `EnsembleModel`) are informational — captured in the summary report but not gating.

### 1.1 Goals (in scope)

- Append 4 nullable-float columns to `RbFeaturesSchema` in `src/projections/schemas.py`:
  - `pace_l4: Series[float] = pa.Field(nullable=True)` — team offensive plays per game, trailing 4 prior games.
  - `proe_l4: Series[float] = pa.Field(nullable=True)` — team mean `pass_oe` across non-NaN plays, trailing 4.
  - `team_ayps_l4: Series[float] = pa.Field(ge=0, nullable=True)` — team mean air yards per pass attempt, trailing 4.
  - `team_def_epa_resid_l4: Series[float] = pa.Field(nullable=True)` — opponent's defensive EPA-allowed-per-play residual vs offensive-opponent season-average EPA, trailing 4.
- Extract a shared helper `attach_pbp_family_features(index, pbp) -> pd.DataFrame` in `src/projections/features/pbp_team_features.py` that takes a `(gsis_id, season, week, team, opp)` index and returns it with the 4 columns appended via the existing `compute_team_*` fns + 4 left-merges. Refactor `build_pbp_family_overrides` to consume this helper (the assembler still owns the GSIS-format / dup-key validation).
- Modify `build_rb_features` in `src/projections/features/rb.py` to:
  - Build the player-team-week index it already has (gsis_id, season, week, team) joined with `schedules` for `opp`.
  - Call `attach_pbp_family_features(index, pbp)` to compute the 4 columns.
  - Merge the 4 columns onto the existing RB feature frame.
  - Validate the output against the extended `RbFeaturesSchema`.
- Update `tests/test_features/test_rb.py`:
  - Assert the 4 new columns appear in the output frame on a synthetic-PBP fixture.
  - Add a join-side test: RB on team T facing opp O picks up T's pace/proe/ayps and O's def_epa_resid.
- Refresh the RB feature cache: `python scripts/refresh_features.py rb --seasons 2018-2024`. (Manual; run-time output, not committed.)
- Run the full backtest + adoption gate on RB only, all 4 model classes. Commit the resulting `reports/adoption_gate_rb_pbp_features.{md,csv}`.
- Write `reports/rb_pbp_features_summary.md` consolidating: probe-predicted vs gate-measured magnitudes; per-model-class verdicts; ship/revert decision.
- On `ADOPT` verdict: update `project_management.md` decision log + `TODO.md` #3c. On `MARGINAL` / `DO_NOT_ADOPT`: revert all builder + schema changes (the spec leaves the shared helper extraction in place — it's reusable for the deferred WR/TE follow-up).

### 1.2 Non-goals (deferred)

- **No QB / WR / TE schema changes.** The probe explicitly returned regression on QB augment-mode and net-zero on WR / TE. Integrating PBP team-features into those positions is a separate spec, blocked on a per-position-units re-probe (player aDOT for WR / TE, etc.).
- **No per-feature ablation.** The probe tested all 4 features bundled. Production-Ridge regularization shrinks uninformative coefficients toward 0, so shipping all 4 doesn't degrade prediction quality vs shipping the 1–2 load-bearing ones. A per-feature ablation is a "nice to know" follow-up, not a prerequisite.
- **No widening to other model classes' production routing.** `POSITION_DISPATCH[RB].factories['default']` stays at `BaselineModel`. The new features are added to the schema (which is shared across model classes), so all model classes consume them in the gate run, but the production routing decision is unchanged.
- **No new probe machinery.** The probe code (`src/projections/backtest/feature_probe.py`, `scripts/probe_feature_signal.py`) is not modified. The summary report compares probe-vs-gate calibration but does not re-run the probe.
- **No spec / plan file changes for prior work.** PR #20's spec, plan, and reports stay as historical record.
- **No ingest changes.** The 4 features come from the existing curated `PbpSchema`. If `nfl_data_py` ever changes the columns the computes use (`posteam`, `defteam`, `play_type`, `pass_oe`, `pass_attempt`, `air_yards`, `epa`), the existing `--run-network` smoke at `tests/test_ingest/test_api_drift.py` catches it.
- **No backfill policy change for 2018 weeks 1–4.** The probe accepted 96.6% per-position coverage; the production schema's `nullable=True` on the 4 new columns inherits the same convention. If a future spec ingests 2017 PBP, the NaN rate drops without requiring schema or builder changes.

### 1.3 Success criteria

The spec is complete iff all of:

1. **Schema + builder + tests land cleanly.** `pytest -v` (full suite), `mypy src tests` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check` (no drift).
2. **Refreshed RB feature cache validates against the extended schema** at every `(season, week)` partition.
3. **The full backtest + adoption gate runs successfully on RB** for all 4 model classes (baseline / lightgbm-tuned / lightgbm-nb / ensemble) across the standard `2021-2024` holdout years.
4. **The summary report (`reports/rb_pbp_features_summary.md`) records both:**
   - The probe's predicted composite RMSE delta on `(BaselineModel, RB)`: -0.013 fpts (from PR #20's `feature_probe_pbp_family_augment.csv`).
   - The gate's measured composite RMSE delta on `(BaselineModel, RB)` with 95% CI.
   - The per-(model_class, RB) verdicts for the other 3 model classes.
5. **The shipping decision matches the gate verdict on `(BaselineModel, RB)`:**
   - `ADOPT` → merge PR; update decision logs.
   - `MARGINAL` or `DO_NOT_ADOPT` → revert builder + schema changes (keep the shared helper); document the divergence.

If criterion 1 fails, fix and rerun. If criterion 2 fails, the builder is wrong — fix before running the gate. Criterion 3 is mechanical (the gate either runs or doesn't). Criterion 5 is the binding decision.

---

## 2. Inputs

### 2.1 PBP source

PBP partitions read via `read_partition(raw_root, "pbp", season=s)` for `[seasons.start - 1, seasons.stop)` to provide trailing-4 backfill at week 1–4 of each season. This already happens inside `build_rb_features`'s caller (`scripts/refresh_features.py:88`), which threads `pbp` through the `build_*_features` signature.

### 2.2 Player-team-week index inside `build_rb_features`

`build_rb_features` already produces an internal `(gsis_id, season, week, team, opponent)` frame from `depth_charts` (filtered to RBs in `as_of_week`) inner-joined with `schedules`. The new code reuses this frame, renames `opponent → opp` to match `attach_pbp_family_features`'s contract, and passes it through.

No new ingest source. No new schema-level changes outside `RbFeaturesSchema`.

### 2.3 PBP frame contract

`pbp` must satisfy `PbpSchema` per the existing ingest. Team codes (`posteam`, `defteam`) are canonical post-ingest. The `attach_pbp_family_features` helper (per §3.1) trusts this contract; defensive normalization is not added.

If `pbp` is empty (the existing `_EMPTY_PBP` default in `build_rb_features`'s signature), the helper returns the index with the 4 columns all-NaN. The schema's `nullable=True` accepts this. No special-case branch is needed in `build_rb_features` — the helper handles the empty case as a no-op merge.

---

## 3. Code shape

### 3.1 Shared helper `attach_pbp_family_features`

New public function in `src/projections/features/pbp_team_features.py`:

```python
def attach_pbp_family_features(
    index: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the 4 PBP family features to a player-team-week index.

    Args:
        index: (gsis_id, season, week, team, opp) — one row per player-week.
            Team codes are assumed canonical per the ingest schemas.
        pbp: PBP frame matching ``PbpSchema``, projected to or wider than
            ``_PBP_COLUMNS_USED``.

    Returns:
        A copy of ``index`` with 4 columns appended in order:
        ``pace_l4``, ``proe_l4``, ``team_ayps_l4``, ``team_def_epa_resid_l4``.
        Row count equals ``len(index)``; all 4 columns are float64 nullable
        (NaN where trailing-4 has fewer than 4 prior games).

    The 4 computes are run on a column-projected ``pbp`` (only the columns
    listed in ``_PBP_COLUMNS_USED``). pace/proe/team_ayps join on the
    player's TEAM; team_def_epa_resid joins on the player's OPPONENT.

    Empty ``pbp`` short-circuits to all-NaN columns — same shape as a
    successful call where every row's trailing-4 has fewer than 4 prior
    games. The schema's ``nullable=True`` covers this.
    """
```

Refactor `build_pbp_family_overrides` (the assembler) to consume this helper. The assembler retains responsibility for: GSIS-format validation, duplicate-key validation, the row-count invariant assertion. The new helper takes over the 4 computes + 4 merges + final column projection.

### 3.2 `build_rb_features` integration

In `src/projections/features/rb.py`:

1. After the existing `_filter_to_rostered_rbs` step, the function already produces a `(gsis_id, season, week, team, opponent)` frame. Rename `opponent → opp` for the helper call (or pass through with a column rename inline).
2. Call `attach_pbp_family_features(index_for_pbp, pbp)`.
3. Merge the 4 returned columns onto the existing RB feature frame on `(gsis_id, season, week)`.
4. The final pandera validation against `RbFeaturesSchema` enforces the 4 new columns are present + correctly typed.

The function signature does not change. The unused `pbp` kwarg becomes load-bearing.

### 3.3 Schema change

In `src/projections/schemas.py`, append to `RbFeaturesSchema`:

```python
# PBP team-level features (Plan 10 — RB PBP integration). Trailing 4
# prior games; NaN for early-season weeks where fewer than 4 prior games
# exist (notably 2018 weeks 1-4, the start of the curated PBP window).
pace_l4: Series[float] = pa.Field(nullable=True)
proe_l4: Series[float] = pa.Field(nullable=True)
team_ayps_l4: Series[float] = pa.Field(ge=0, nullable=True)
team_def_epa_resid_l4: Series[float] = pa.Field(nullable=True)
```

Strict mode ("filter") on the schema's `Config` already drops unexpected columns; no other change needed.

### 3.4 Tests

In `tests/test_features/test_rb.py`:

- Extend the existing happy-path test to assert the 4 new columns are present and float-typed.
- Add `test_rb_features_attach_pbp_family_join_sides` — a synthetic-PBP fixture (one team T, one opp O, 5 weeks of plays) where T's offensive features differ from O's defensive features, and asserts:
  - The RB on team T picks up T's pace_l4 / proe_l4 / team_ayps_l4.
  - The RB picks up O's team_def_epa_resid_l4 (joined on opponent).
- The shared helper `attach_pbp_family_features` does not need its own dedicated tests — it's exercised end-to-end by the existing assembler tests (which now route through it) and by the new RB join-side test.

In `tests/test_features/test_pbp_team_features.py`:

- The existing 12 tests continue to cover the 4 computes + assembler. The assembler refactor (consuming `attach_pbp_family_features`) is transparent to the tests since the assembler's contract is unchanged.

---

## 4. Real-data execution sequence (run-once, reports committed)

1. Code changes land + tests pass + lint + typecheck clean (criterion 1.3.1).
2. `python scripts/refresh_features.py rb --seasons 2018-2024` — regenerates RB feature cache. Verify schema validation passes on every (season, week) partition (criterion 1.3.2). Output is not committed (lives under `data/features/rb/...`, gitignored convention).
3. `python scripts/backtest.py --position RB --update-snapshot` — runs the walk-forward backtest on all 4 model classes for RB on holdout years 2021-2024. Captures per-row prediction frames (criterion 1.3.3). Snapshot updates committed.
4. `python scripts/adoption_gate.py --position RB --baseline-run <pre-pr-sha> --candidate-run <branch-sha>` — produces per-(model_class, RB) verdicts. Output: `reports/adoption_gate_rb_pbp_features.{md,csv}`. Commit.
5. Write `reports/rb_pbp_features_summary.md` with: probe-predicted vs gate-measured magnitudes, per-(model_class, RB) verdicts, ship/revert decision (criterion 1.3.4 + 1.3.5). Commit.
6. **If `(BaselineModel, RB)` verdict is `ADOPT`:** update `project_management.md` (top-of-file decision-log entry, format matches PR #20's entry) + `TODO.md` #3c (record shipped, with measured magnitude). Push branch + open PR.
7. **If `(BaselineModel, RB)` verdict is `MARGINAL` or `DO_NOT_ADOPT`:** revert the schema + builder changes in a new commit (keep the shared helper extraction — useful for the WR/TE follow-up). Document the divergence in `project_management.md` + `TODO.md` #3c. Push branch with the revert + summary; open PR labeled "documentation-only" (no code change to ship).

---

## 5. Risks

- **Probe-vs-gate divergence on magnitude.** The probe predicted -0.013 fpts; the real gate could measure something materially different (smaller — e.g., -0.002 fpts ADOPT — or larger). The §1.3.5 rule binds on the verdict label, not magnitude, so this is recorded but not gating. If divergence is dramatic in either direction, the summary report flags it for future probe-tuning.
- **Probe-vs-gate divergence on verdict.** If `(BaselineModel, RB)` returns `MARGINAL` or `DO_NOT_ADOPT`, the spec's revert path fires. This is the Plan 9-style outcome and is documented.
- **Other model classes diverge.** `LightGBMNbModel.RB` returning `DO_NOT_ADOPT` while baseline returns `ADOPT` would echo Plan 9's pattern (lgb-nb is a known skeptic of marginal feature additions). Per §1.3.5, this is informational. The summary report flags it; the production routing decision is still bound to baseline.
- **Helper-extraction regressions.** Refactoring `build_pbp_family_overrides` to consume `attach_pbp_family_features` is a plumbing change. The existing 12 tests in `test_pbp_team_features.py` should pass unchanged after the refactor; if they don't, the helper is wrong. The implementation plan front-loads this verification.
- **Feature cache invalidation.** Adding 4 columns to `RbFeaturesSchema` invalidates the existing RB cache (the schema validate would reject old rows missing the new columns). The spec calls this out in §4 step 2 and runs the refresh explicitly.
- **2018 backfill coverage.** Same 96.6% coverage as the probe; production Ridge handles NaN. If a future ingest brings 2017 PBP online, the NaN rate drops with no schema or builder change required.

---

## 6. Documentation updates on merge

- **`project_management.md`:** Append a top-of-file decision-log entry. Format matches PR #20's entry — title, status, verdict, what shipped or reverted, magnitude.
- **`TODO.md` #3c:** Record the production integration outcome (shipped / reverted, with measured magnitude). Cross-reference the summary report.
- **`docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`:** No changes. The probe spec stays as-is.
- **`CONTRIBUTING.md`:** No changes — the override-regeneration note from PR #20 is still accurate; this spec uses the production path, not the override.

---
