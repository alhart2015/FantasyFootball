# Weather Features RB+WR Integration — Design

**Status:** approved (brainstorming, 2026-05-08). Ready for implementation plan.
**Date:** 2026-05-08
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- Weather Feature Family Probe (PR #28, merged at `39be213`) — shipped the 4 weather compute fns + `compute_weather_features` + `attach_weather_features` joiner + `build_weather_overrides` assembler in `src/projections/features/weather_features.py`, plus the override-generator script `scripts/build_weather_override.py`. Probe verdict was `SIGNAL` via lgb-nb augment composite; the binding cells were two Phase-2 ADOPTs: **(lgb-nb, RB) augment -0.0081 fpts** (CI [-0.0163, -0.0005]) and **(lgb-nb, WR) augment -0.0110 fpts** (CI [-0.0172, -0.0049]). BaselineModel returned 0/120 SIGNAL across both modes for all 4 positions — strong evidence the linear-Ridge class cannot extract the bundle's non-linear thresholds (`is_high_wind` ≥20, surface category) even with explicit boolean encoding.
- TE Trajectory Features Integration (PR #27, merged at `5a9352f`) — set the durable precedent for non-default-binding integrations: ship the schema change for the `LightGBMNbModel` code path, evaluate only `baseline + lgb-nb` in the gate (binding + contingency cells), leave production routing on `BaselineModel`, defer the cross-class flip discussion to a separate post-merge re-eval. This spec inherits the same shape.
- WR Trajectory Features Integration (PR #26, merged at `884d025`) — set the cluster-A test-fixture grep precedent (`opp_allowed_<pos>_fppg_l4` defensive grep to find every site building a synthetic minimal-row feature; commits `1f1f415` / `33eea57` / `807f046` caught 3 leftover sites for WR).
- RB PBP Features Integration (PR #21, merged at `9895dee`) — set the `_<POS>_FEATURE_COLUMNS` extension precedent: lightgbm family derives feature lists from `*FeaturesSchema` dynamically; baseline.py's hardcoded per-position tuple must be edited explicitly or BaselineModel will not see the new features even though the schema validates.

**Branch:** `feat/weather-features-rb-wr` cut from `main` at `39be213`. Worktree at `.worktrees/feat-weather-features-rb-wr/`.

---

## 1. Overview

PR #28 shipped weather features into `data/features_probe/weather.parquet` and ran the family probe; verdict was `SIGNAL` on two cells: `(lgb-nb augment, RB)` and `(lgb-nb augment, WR)`. This spec executes the parallel production integration: append the same 4 nullable-float columns to `RbFeaturesSchema` and `WrFeaturesSchema`, wire `attach_weather_features` into `build_rb_features` and `build_wr_features`, refresh both feature caches, and run the dual-run adoption gate to verify both probe predictions hold on the binding cells.

**Critical shape (matching PR #27 precedent): the binding cells are `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)`, not `(BaselineModel, *)`.** PR #28's probe ADOPT'd both positions only under lgb-nb (-0.0081 fpts for RB, -0.0110 fpts for WR); both positions' BaselineModel cells were DO_NOT_ADOPT (CIs bracketed zero, point estimates near zero — not REGRESSION, just no signal at the linear-Ridge class). Because both positions' production `default_model_class` stays at `baseline` (Plan 8 verdict 2026-04-29), shipping these features does **not** automatically improve RB or WR production output — the feature cols are persisted in the schema for the model classes that demonstrably benefit (`lightgbm-nb`, likely `lightgbm` and `ensemble`), but production routing stays on baseline pending separate cross-class re-evals.

**Why combined RB+WR PR (vs split into two sequential PRs):** the two ADOPT cells share the same model class and binding mode (lgb-nb augment); the two positions' schemas and builders are independent at the edit-locus level, so the changes don't tangle; the backtest+gate run dominates wall-clock time and parallelizes naturally with `--position RB,WR --model baseline,lightgbm-nb`. The §1.3.5 contingency matrix below is structured per-position so partial-revert outcomes ship cleanly within the same PR.

The shipping decision is binary and bound to the **lgb-nb** verdict, **per position independently**: if `(LightGBMNbModel, POS)` returns `ADOPT`, ship that position; if `MARGINAL` or `DO_NOT_ADOPT`, revert that position. Other model classes' verdicts (`LightGBMModel`, `LightGBMTunedModel`, `EnsembleModel`) are explicitly informational, **not run** in the gate per PR #27 precedent (the binding+contingency-only scope completes in ~30-60 min wall-time per position; running all 5 classes risks the same 3-hour `--model all` abort PR #27 hit). This is the **second integration spec in the project to bind on a non-default model class**, and the **first** to bundle two positions into a single PR.

### 1.1 Goals (in scope)

- Append 4 nullable-float columns to **`RbFeaturesSchema`** (`schemas.py:601`) and **`WrFeaturesSchema`** (`schemas.py:484`), identical 4-line block in each:

  ```python
  # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration
  # spec). Sourced from existing SchedulesSchema columns (wind, temp, roof,
  # surface) — no new ingest. Dome / closed-roof games filled with
  # (wind=0, temp=70) per the probe's `compute_weather_features` semantics
  # (controlled environment, not "imputed missing"). Outdoor games with
  # NaN wind/temp upstream propagate NaN; ~8% NaN rate concentrated in
  # 2018-2019 per the probe's audit. lightgbm-family handles NaN natively;
  # BaselineModel imputes with feature mean.
  wind_speed_mph: Series[float] = pa.Field(ge=0, nullable=True)
  is_high_wind: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  temperature_f: Series[float] = pa.Field(nullable=True)
  is_grass_surface: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  ```

  - `wind_speed_mph` lower bound 0 (wind speed is a physical magnitude); no upper bound (extreme-game outliers are real signal, not data quality).
  - `is_high_wind` is a Float64 boolean (1.0 if `wind_speed_mph >= 20`, else 0.0; NaN if `wind_speed_mph` is NaN); bounded `ge=0, le=1`.
  - `temperature_f` is unbounded — cold-weather games are negative on the Fahrenheit scale (single-digit Buffalo December games are real); the upstream `nfl_data_py` schedule data is the source of truth for outliers.
  - `is_grass_surface` is a Float64 boolean (1.0 if `surface == "grass"`, else 0.0); bounded `ge=0, le=1`.

- Modify `build_rb_features` in `src/projections/features/rb.py` to:
  - After the existing assembly that produces `out` with `(gsis_id, season, week, team, opponent, ...)` columns, call `out = attach_weather_features(out, schedules)`. The helper takes a frame with at least `(season, week, team)` and left-merges the 4 weather columns; row count is preserved.
  - The function signature does not change. `schedules` is already a kwarg.
  - The final `RbFeaturesSchema.validate(out)` enforces presence + dtype + bounds.

- Same edit shape in `build_wr_features` (`src/projections/features/wr.py`).

- Update `src/projections/models/baseline.py:_RB_FEATURE_COLUMNS` (line 360) and `_WR_FEATURE_COLUMNS` (line 266) to include the 4 new column names. Same spec gap PR #21 caught at commit `9895dee` (RB), PR #26 caught for WR, PR #27 caught for TE — `baseline.py` hardcodes the per-position feature tuple while the lightgbm family derives from `*FeaturesSchema.to_schema().columns.keys()` filtered through `_NON_FEATURE_COLUMNS` dynamically. Without these updates, `BaselineModel.fit` for RB and WR will not see the new features even though the schema validates. The implementation plan calls each out as its own dedicated edit with a smoke-test verification (assert the 4 names appear in each `_<POS>_FEATURE_COLUMNS` post-edit).

- Update `tests/test_features/test_rb.py` and `tests/test_features/test_wr.py` (parallel additions in each):
  - Extend the existing happy-path test to assert the 4 new columns are present, float-typed, and bounded per the schema.
  - Add `test_<pos>_features_attach_weather_dome_fill` — synthetic schedules row with `roof="dome"`: assert `wind_speed_mph == 0.0`, `temperature_f == 70.0`, `is_high_wind == 0.0`. The probe's `compute_weather_features` already covers this semantics; the test is at the builder boundary (i.e., verifies the integration flows the dome-fill through correctly, not the helper itself).
  - Add `test_<pos>_features_attach_weather_outdoor_high_wind` — synthetic outdoor schedules row with `wind=22, temp=42, roof=NaN`: assert `wind_speed_mph == 22.0`, `temperature_f == 42.0`, `is_high_wind == 1.0`.
  - Add `test_<pos>_features_attach_weather_grass_surface` — two synthetic schedules rows for the same player-week, one with `surface="grass"`, one with `surface="sportturf"`: assert `is_grass_surface` flips between 1.0 and 0.0.
  - Add `test_<pos>_features_attach_weather_bye_week_fallback` — defensive: a player-week where the schedule join is empty (bye week, missing future game). Assert the 4 weather cols are NaN and the schema's `nullable=True` accepts validation. Same edge-case axis as TODO #9a (WR `is_home`/`roof_dome` non-nullable risk on left-merge with bye-week teams).

- Cluster-A grep for synthetic minimal-row feature construction: defensive grep for `opp_allowed_rb_fppg_l4` and `opp_allowed_wr_fppg_l4` (the last existing cols in each `_<POS>_FEATURE_COLUMNS`) to find every site that builds a synthetic minimal RB or WR features row. Likely sites (per PR #26 / PR #27 precedent): `tests/test_features/test_cache.py`, `tests/test_models/test_baseline_rb.py` / `test_baseline_wr.py`, `tests/test_models/test_lightgbm_rb.py` / `test_lightgbm_wr.py`, `tests/test_models/test_ensemble_rb.py` / `test_ensemble_wr.py`, `tests/test_scripts/test_tune_lightgbm.py`. Add `wind_speed_mph=8.0`, `is_high_wind=0.0`, `temperature_f=60.0`, `is_grass_surface=0.0` to each synthetic row. Same defensive-grep pattern as PR #26's commits `1f1f415` / `33eea57` / `807f046`.

- **No `tests/conftest.py` extension.** Weather is a per-game (non-trailing-N) feature — the existing `baseline_weekly_stats_rb` / `baseline_weekly_stats_wr` / `baseline_features_*` fixtures already cover the relevant week ranges. Compare with PR #26 / PR #27, which had to extend to 17/17/4 weeks because trajectory needs 8 prior active games of history. Weather has no such requirement.

- **No caller-script changes.** `schedules` is already loaded and threaded through all 4 caller scripts (`scripts/refresh_features.py`, `scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py`) for all 4 positions. The integration is a closed change inside `build_rb_features` and `build_wr_features`.

- Refresh both feature caches: `python scripts/refresh_features.py rb wr --seasons 2018-2024`. Manual; output not committed (lives under `data/features/{rb,wr}/...`, gitignored convention).

- Run the backtest snapshot regen for RB + WR × `baseline + lgb-nb`: `python scripts/backtest.py --position RB,WR --model baseline,lightgbm-nb --update-snapshot`. 4 cells total. Snapshot updates committed (`tests/backtest/model_metrics.json` rows for these 4 cells; `data/ensemble_weights/ensemble_rb_*.json` and `ensemble_wr_*.json` regen if EnsembleModel is touched — see §5 Risks for the working-tree-deletion caveat).

- Run the adoption gate: `python scripts/adoption_gate.py --position RB,WR --baseline-run <pre-pr-sha> --candidate-run <branch-sha>`. Output: `reports/adoption_gate_weather_features_rb_wr.{md,csv}`. Commit.

  **Note on `--coverage-threshold`:** the flag is a probe-only concern (`scripts/probe_feature_signal.py`). The gate uses row-key matching, not NaN tolerance — see PR #26 spec's note. If the gate exposes an analogous flag and rejects a default of 0.95, match the probe's `0.90` (per PR #28 §3 audit: 8.39% pooled outdoor-NaN rate; per-(position, season) coverage in the 2021-2024 eval window uniformly ≥92%).

- Write `reports/weather_features_rb_wr_summary.md` consolidating: probe-predicted vs gate-measured magnitudes per position; per-(model_class, position) verdicts (only baseline + lgb-nb run; the other 3 classes flagged informational and skipped per spec §1.3.4); 2-position §1.3.5 outcome matrix; ship/revert decision per position; coverage statistics per position at the eval window cross-checked against PR #28's audit; explicit binding-cell-shift note (lgb-nb, not baseline) and the deferred cross-class production-routing question for both RB and WR.

- On any per-position `(lgb-nb, POS)` `ADOPT` verdict: update `project_management.md` decision log + `TODO.md` #25 (record per-position shipped state, with measured magnitude; cross-link to PR #28). On any per-position `MARGINAL` / `DO_NOT_ADOPT`: revert that position's builder + schema + `_<POS>_FEATURE_COLUMNS` changes (the spec leaves cluster-A test fixture defaults in place — cheap to leave; useful for any future weather revisit at the refined-unit level).

### 1.2 Non-goals (deferred)

- **No QB / TE schema changes.** Both returned DO_NOT_ADOPT in PR #28's probe (composite RMSE deltas brackets zero on lgb-nb augment composite; QB augment lgb-nb composite was +0.0077 fpts NOT REGRESSION but distinctly non-negative). Refined-unit candidates remain open under TODO #25: `is_cold_weather` (`temp < 32`, sibling shape to `is_high_wind`), multi-class surface encoding (one bool per surface code), kickoff hour / time-of-day, surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). None queued in this PR.
- **No production routing flips.** `POSITION_DISPATCH[RB|WR].factories['default']` stays at `BaselineModel`. The cross-class flip question (does `lgb-nb-with-weather` beat `baseline-without-weather` for RB or WR at the position level?) is a separate cross-class re-eval — same shape PR #27 deferred for TE trajectory. The summary report flags it explicitly per position; it can run anytime post-merge with the weather cols already in the RB/WR schemas.
- **No new ingest.** Weather sourced from existing `SchedulesSchema` columns (`wind`, `temp`, `roof`, `surface`) at `data/raw/schedules/...`. The opt-in `--run-network` smoke at `tests/test_ingest/test_api_drift.py` already covers `nfl_data_py.import_schedules` column-rename drift.
- **No new helper extraction.** `compute_weather_features`, `attach_weather_features`, and `build_weather_overrides` were all promoted to public in PR #28. The integration consumes `attach_weather_features` directly. No `weather_features.py` edits in this PR.
- **No caller-script changes.** `schedules` already plumbed through all 4 caller scripts for all 4 positions (existing convention since the probe spec landed). Already wired. No script edits in this PR.
- **No `build_weather_overrides` deprecation.** Probe assembler stays as-is for any post-merge probe re-runs at the refined-unit level (PR #21 precedent kept `build_pbp_family_overrides` for the same reason). Weather override script (`scripts/build_weather_override.py`) likewise stays.
- **No per-feature ablation.** The probe tested all 4 features bundled. Production-Ridge regularization shrinks uninformative coefficients toward 0; lgb's tree splits ignore unused features. Shipping all 4 doesn't degrade prediction quality vs shipping the 1-2 load-bearing ones. A per-feature ablation is a "nice to know" follow-up, not a prerequisite.
- **No 5-class gate run** per spec §1.3.3 deviation from the historical "all 5 classes" pattern. PR #27 set the precedent for non-default-binding integrations; ship with `baseline + lgb-nb` only. The 3 skipped classes (`lightgbm`, `lightgbm-tuned`, `ensemble`) are explicitly informational per spec §1.3.4 and back-fillable by a follow-up backtest if the cross-class routing-flip discussion ever needs them. TODO #29 already flags `lightgbm-tuned` as a pruning candidate (dominated 16/16 by lgb-nb on RMSE), so spending wall-clock on it for two more positions doesn't move any decision.
- **No new probe machinery.** The probe code is not modified. The summary report compares probe-vs-gate calibration but does not re-run the probe.
- **No spec / plan file changes for prior work.** PR #28 spec, plan, and reports stay as historical record.
- **No CONTRIBUTING.md changes.** PR #28 already added the "Regenerating the weather override" subsection covering the override-generator path; the production-builder integration is closed inside `build_<pos>_features` and doesn't change any user-facing CLI invocation.

### 1.3 Success criteria

The spec is complete iff all of:

1. **Schema + builder + tests + cluster-A fixture defaults land cleanly.** `pytest -v` (full suite), `mypy src tests` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check` (no drift).

2. **Refreshed RB + WR feature caches validate against the extended schemas** at every `(season, week)` partition.

3. **The full backtest + adoption gate runs successfully on RB + WR × `baseline + lgb-nb`** across the standard `2021-2024` holdout years. 2 positions × 2 model classes = 4 gate cells. The 3 informational classes (`lightgbm`, `lightgbm-tuned`, `ensemble`) are explicitly skipped per §1.3.4 — not gating; if a future reviewer wants those cells, they can be back-filled by a follow-up `--model lightgbm,lightgbm-tuned,ensemble` backtest run without re-touching the schema or builder.

4. **Other model classes are explicitly informational.** The summary report records what cells were *not* run and the rationale (PR #27 precedent + wall-time risk + TODO #29 lightgbm-tuned dominated). A follow-up cross-class re-eval is the right shape for any production-routing-flip discussion.

5. **The summary report (`reports/weather_features_rb_wr_summary.md`) records all of:**
   - The probe's predicted composite RMSE deltas: **(lgb-nb, RB) augment -0.0081 fpts** and **(lgb-nb, WR) augment -0.0110 fpts** (from PR #28's `feature_probe_weather_lgbnb_augment.csv`).
   - The gate's measured composite RMSE deltas on `(lgb-nb, RB)` and `(lgb-nb, WR)` with 95% CIs.
   - The per-(model_class, position) verdict table for the 4 cells run (baseline + lgb-nb × RB + WR).
   - Per-position coverage of the 4 new columns at the eval window (2021-2024) and on the full 2018-2024 history; cross-checked against PR #28's audit (`reports/feature_probe_weather_override_audit.md`).
   - Explicit note that the binding cells are `(LightGBMNbModel, *)` per §1, both production routings remain on `baseline`, and the cross-class flip question is deferred to a separate per-position follow-up for each of RB and WR.
   - The 2-position §1.3.5 contingency matrix outcome (which branch fired for each position; whether ship-as-designed, ship-modified-shape, or full-revert).

6. **The shipping decision is bound per position to the `(LightGBMNbModel, POS)` verdict**, with the §1.3.5 modified-shape contingency for `(BaselineModel, POS)` REGRESSION applied independently to each of RB and WR. Per-position table immediately below.

If criterion 1 fails, fix and rerun. If criterion 2 fails, the builder is wrong — fix before running the gate. Criterion 3 is mechanical (the gate either runs or doesn't). Criterion 6 is the binding decision.

#### 1.3.5 Per-position contingency matrix

Each position's ship/revert decision is independent — no shared "ship together or revert together" coupling. The matrix below applies to each position separately:

| `(lgb-nb, POS)` | `(baseline, POS)` | Action for POS |
|---|---|:---|
| ADOPT | not REGRESSION | **Ship as designed**: `*FeaturesSchema` extension + `build_<pos>_features` integration + `_<POS>_FEATURE_COLUMNS` extension. |
| ADOPT | REGRESSION | **Ship modified-shape**: `*FeaturesSchema` extension + `build_<pos>_features` integration; **revert** `_<POS>_FEATURE_COLUMNS` extension for that position only. Re-run that position's baseline backtest cell to refresh the 4 baseline rows in `tests/backtest/model_metrics.json`. The schema cols still feed the lightgbm family via dynamic derivation; `BaselineModel` doesn't see them for that position. |
| MARGINAL or DO_NOT_ADOPT | (any) | **Revert that position**: undo `*FeaturesSchema` extension + `build_<pos>_features` integration + `_<POS>_FEATURE_COLUMNS` extension. Keep cluster-A test fixture defaults (cheap to leave; useful for any future weather revisit). Document divergence in summary; close that position's branch of TODO #25. |

The probe's evidence on `(BaselineModel, *)` was unusually strong: 0/120 SIGNAL Phase 1 cells across both modes, all 4 positions. The gate is unlikely to flip baseline RB or WR to REGRESSION (which requires CI strictly above zero, not just "non-negative point estimate"); the modified-shape branch is therefore a low-probability contingency, but the spec pre-decides it so the implementation plan doesn't have to.

Worst-case combined outcome: RB ships modified-shape and WR full-reverts (or vice-versa) within the same PR. The implementation plan's Phase 5 (post-gate) is structured around the 4 per-position outcomes (ship-as-designed / ship-modified-shape / revert × RB + WR independently). The summary report's outcome-matrix section makes the per-position decision visible at a glance.

---

## 2. Inputs

### 2.1 Schedules source

`schedules` partitions read via `read_partition(raw_root, "schedules", season=s)` for the eval range. Already loaded by all 4 caller scripts and passed into `build_rb_features` / `build_wr_features`. No new ingest source.

### 2.2 Player-team-week index inside builders

Both `build_rb_features` and `build_wr_features` already produce internal `(gsis_id, season, week, team, opponent, ...)` frames from `depth_charts` (filtered to RBs / WRs in `as_of_week`, deduped per the Plan 3b drift fixes for each builder) inner-joined with `schedules` (bye-week filter from the same Plan 3b drift). The weather integration reuses this frame directly — `attach_weather_features(out, schedules)` joins on `(season, week, team)`, which is already present in `out`. No `opponent → opp` rename needed (weather doesn't use `opp`).

### 2.3 Schedules contract

`schedules` must satisfy `SchedulesSchema` per the existing ingest. The `attach_weather_features` helper (per §3.1) trusts this contract; defensive normalization is not added (existing convention since PR #28).

If `schedules` is empty, the helper's left-merge produces all-NaN rows on the 4 weather cols. The schemas' `nullable=True` accepts this.

---

## 3. Code shape

### 3.1 `attach_weather_features` (already public from PR #28)

Public function in `src/projections/features/weather_features.py`, signature unchanged:

```python
def attach_weather_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge the four weather features onto a player-team-week index."""
```

The helper takes a frame with at least `(season, week, team)` and left-merges the 4 weather columns. Row count is preserved. Dome-fill / outdoor-NaN semantics are owned by `compute_weather_features` (called inside `attach_weather_features`); no edits to either function in this PR.

### 3.2 `build_rb_features` integration

In `src/projections/features/rb.py`, after the existing assembly that produces `out` with `(gsis_id, season, week, team, opponent, ...)` columns (i.e., after the existing `attach_pbp_family_features` line for PBP team-features):

```python
from projections.features.weather_features import attach_weather_features

# ... inside build_rb_features, after the existing attach_pbp_family_features call ...

# Weather features (PR #28 family probe + 2026-05-08 RB+WR integration spec).
# attach_weather_features merges 4 nullable-float cols onto (season, week, team)
# from the existing schedules kwarg. Dome / closed-roof games have wind=0 / temp=70
# per the helper's compute_weather_features semantics.
out = attach_weather_features(out, schedules)
```

The function signature does not change. `schedules` is already a kwarg (existing convention since the original WR builder shipped). The final `RbFeaturesSchema.validate(out)` enforces presence + dtype + bounds of the 4 new cols.

### 3.3 `build_wr_features` integration

Identical edit shape in `src/projections/features/wr.py`, after the existing assembly that produces `out`:

```python
from projections.features.weather_features import attach_weather_features

# ... inside build_wr_features, after the existing trajectory feature attach ...

# Weather features (PR #28 family probe + 2026-05-08 RB+WR integration spec).
out = attach_weather_features(out, schedules)
```

### 3.4 Schema changes

In `src/projections/schemas.py`, append to `RbFeaturesSchema` (after the existing `team_def_epa_resid_l4` line at ~line 651):

```python
# Weather features (PR #28 family probe + 2026-05-08 RB+WR integration
# spec). Sourced from existing SchedulesSchema columns (wind, temp, roof,
# surface) — no new ingest. Dome / closed-roof games filled with
# (wind=0, temp=70) per compute_weather_features semantics. Outdoor NaN
# wind/temp propagates; ~8% NaN rate concentrated in 2018-2019.
wind_speed_mph: Series[float] = pa.Field(ge=0, nullable=True)
is_high_wind: Series[float] = pa.Field(ge=0, le=1, nullable=True)
temperature_f: Series[float] = pa.Field(nullable=True)
is_grass_surface: Series[float] = pa.Field(ge=0, le=1, nullable=True)
```

Append the **identical 4-line block** (with the same 5-line header comment) to `WrFeaturesSchema` after the existing trajectory cols (around line 537).

`wind_speed_mph` lower bound 0 (physical magnitude). `temperature_f` unbounded (cold-weather games are negative; upstream is source of truth for outliers). Both booleans bounded `ge=0, le=1`. Strict mode (`Config.strict = "filter"`) on both schemas already drops unexpected columns; no other change needed.

### 3.5 `baseline.py` hardcoded feature lists

In `src/projections/models/baseline.py`:

- Extend `_RB_FEATURE_COLUMNS` (line 360) to include the 4 new column names. Insert after the existing `team_def_epa_resid_l4` line:

```python
_RB_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # ... existing 24 columns through team_def_epa_resid_l4 ...
    "team_def_epa_resid_l4",
    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration).
    # lightgbm derives feature lists from RbFeaturesSchema dynamically and
    # auto-picks-up; baseline.py is hardcoded so must be updated explicitly.
    "wind_speed_mph",
    "is_high_wind",
    "temperature_f",
    "is_grass_surface",
)
```

- Extend `_WR_FEATURE_COLUMNS` (line 266) with the same 4 names. Insert after the existing `snap_pct_change_l4_vs_prior_l4` (the last trajectory col added in PR #26), with the same 3-line header comment.

The lightgbm family (`models/lightgbm.py:122` and the inheriting `lightgbm_tuned.py` / `lightgbm_nb.py` / `ensemble.py`) derives feature columns dynamically from `RbFeaturesSchema.to_schema().columns.keys()` (and the WR analog) filtered through `_NON_FEATURE_COLUMNS`, so the lightgbm family auto-picks-up the 4 new columns when each schema change lands. **No edit needed to the lightgbm files** — the implementation plan asserts this with a smoke test that reads the dynamically-derived RB and WR feature lists and verifies the 4 new columns appear post-schema-change for each position.

**§1.3.5 modified-shape contingency:** if `(baseline, RB)` returns REGRESSION (CI of RMSE delta strictly above 0) AND `(lgb-nb, RB)` ADOPTs, the modified-shape ship path leaves `_RB_FEATURE_COLUMNS` *unchanged* (no 4-col addition for RB only). The lightgbm family still picks up the cols via the schema; baseline RB doesn't see them. Implementation plan's Phase 5 handles this branch by reverting only the RB extension and re-running the baseline RB backtest cell so `model_metrics.json`'s RB-baseline rows reflect the without-weather baseline. The lightgbm-nb cells stay as-is from Phase 4's snapshot regen. Same matrix applies independently to WR.

### 3.6 Tests

In `tests/test_features/test_rb.py` and `tests/test_features/test_wr.py` (parallel additions in each file):

- **Extend the existing happy-path test** to assert the 4 new columns are present in the output frame, are float-typed, and respect the schema bounds (`wind_speed_mph >= 0`, `is_high_wind` ∈ {0.0, 1.0, NaN}, `temperature_f` unbounded, `is_grass_surface` ∈ {0.0, 1.0, NaN}).

- `test_<pos>_features_attach_weather_dome_fill` — synthetic schedules row with `roof="dome"`. Assert `wind_speed_mph == 0.0`, `temperature_f == 70.0`, `is_high_wind == 0.0`. The probe's `compute_weather_features` already covers this semantics; the test verifies the integration flows the dome-fill through the builder correctly.

- `test_<pos>_features_attach_weather_outdoor_high_wind` — synthetic outdoor schedules row with `wind=22.0, temp=42.0, roof=NaN`: assert `wind_speed_mph == 22.0`, `temperature_f == 42.0`, `is_high_wind == 1.0`.

- `test_<pos>_features_attach_weather_grass_surface` — two synthetic schedules rows for the same player-week, one with `surface="grass"`, one with `surface="sportturf"`. Assert `is_grass_surface == 1.0` in the grass case and `0.0` in the turf case.

- `test_<pos>_features_attach_weather_bye_week_fallback` — defensive: a player-week where the schedule join is empty (bye week, missing future game). Assert the 4 weather cols are NaN and the schema's `nullable=True` accepts validation. Same edge-case axis as TODO #9a (WR `is_home`/`roof_dome` non-nullable risk on left-merge with bye-week teams; since the existing builders already filter rostered teams to those with schedule rows in `as_of_week`, this test confirms the behavior is robust if a future builder change relaxes that filter).

The shared helper `attach_weather_features` does not need new dedicated tests in this PR — PR #28 already exercises it end-to-end through `build_weather_overrides` and the existing 5+ tests in `tests/test_features/test_weather_features.py`. The new tests above cover the builder boundary specifically.

In `tests/test_features/conftest.py`: no changes (no fixture extension required since weather is per-game; cluster-A fixture defaults are added directly at each synthetic-row construction site, not via shared fixtures).

In `tests/conftest.py`: no changes (no fixture extension required; weather has no trailing-N history requirement).

**Cluster-A defensive grep:**

```bash
grep -rn "opp_allowed_rb_fppg_l4" tests/  # last existing col in _RB_FEATURE_COLUMNS
grep -rn "opp_allowed_wr_fppg_l4" tests/  # last existing col in _WR_FEATURE_COLUMNS
```

For each site that builds a synthetic minimal RB or WR features row (likely sites: `tests/test_features/test_cache.py`, `tests/test_models/test_baseline_*.py`, `tests/test_models/test_lightgbm_*.py`, `tests/test_models/test_ensemble_*.py`, `tests/test_scripts/test_tune_lightgbm.py`), add the 4 new col defaults:

```python
"wind_speed_mph": 8.0,
"is_high_wind": 0.0,
"temperature_f": 60.0,
"is_grass_surface": 0.0,
```

Same defensive-grep pattern as PR #26's commits `1f1f415` (cache fixture) / `33eea57` (lightgbm/ensemble synthetic random fixtures) / `807f046` (tune_lightgbm fixture). Implementation plan confirms via post-edit grep that no `opp_allowed_*_fppg_l4` site lacks the 4 weather defaults.

---

## 4. Real-data execution sequence (run-once, reports committed)

1. Code changes land + tests pass + lint + typecheck clean (criterion §1.3.1).
2. Verify `data/raw/schedules/` exists with seasons covering 2018-2024. If not (highly unlikely — the partition has been canonical since the original 2a ingest), `python -c "from projections.ingest.refresh import refresh; refresh(data_root=Path('data'), seasons=range(2018, 2025), only=['schedules'])"`.
3. `python scripts/refresh_features.py rb wr --seasons 2018-2024` — regenerates RB + WR feature caches. Verify schema validation passes on every (season, week) partition for both positions (criterion §1.3.2). Output is not committed (lives under `data/features/{rb,wr}/...`, gitignored convention). **This step invalidates the existing RB + WR caches** (the schema would reject old rows missing the 4 new cols); the refresh is mandatory before any subsequent backtest read.
4. **Coverage cross-check:** per-(position, season) NaN rate on `wind_speed_mph` / `is_high_wind` / `temperature_f` / `is_grass_surface` for the eval window 2021-2024. Should approximate PR #28's per-position coverage at the override level (per `reports/feature_probe_weather_override_audit.md`: 2021-2024 dome rate ~29%, outdoor-NaN ~2-3% per season). If materially different (e.g., dome rate or NaN rate diverges by >5pp), the builder wiring is wrong — investigate before running the gate.
5. `python scripts/backtest.py --position RB,WR --model baseline,lightgbm-nb --update-snapshot` — runs the walk-forward backtest on RB + WR × `baseline + lgb-nb` on holdout years 2021-2024. Captures per-row prediction frames (criterion §1.3.3). 4 cells total. Snapshot updates committed: `tests/backtest/model_metrics.json` rows for these 4 cells; ensemble weight files (`data/ensemble_weights/ensemble_{rb,wr}_*.json`) are *not* expected to regen because §1.3.4 explicitly skips ensemble in the gate scope. **Working-tree caveat:** the session-start `git status` on `main` flagged unstaged deletions of `data/ensemble_weights/*.json` and `data/tuned_params/*.json`. The worktree at `.worktrees/feat-weather-features-rb-wr` starts clean (the deletions are working-tree-only on main, not committed); this PR's snapshot regen is isolated. The deletion question is independent of this spec — typical recovery on main: `git checkout HEAD -- data/ensemble_weights/ data/tuned_params/`.
6. `python scripts/adoption_gate.py --position RB,WR --baseline-run <pre-pr-sha> --candidate-run <branch-sha>` — produces per-(model_class, position) verdicts for the 4 cells. Output: `reports/adoption_gate_weather_features_rb_wr.{md,csv}`. Commit. (If the gate exposes a `--coverage-threshold` flag and rejects the default, match probe's `0.90`.)
7. Write `reports/weather_features_rb_wr_summary.md` with: probe-predicted (-0.0081 / -0.0110) vs gate-measured magnitudes per position; per-(model_class, position) verdicts (4 cells); coverage statistics from step 4; binding-cell shift rationale; ship/revert decision per position per §1.3.5; explicit deferred-follow-up note for the cross-class production routing question per position. Commit.
8. **For each position independently per §1.3.5:**
   - **`(lgb-nb, POS)` ADOPT AND `(baseline, POS)` not REGRESSION** → ship-as-designed: keep schema + builder + `_<POS>_FEATURE_COLUMNS` extension for that position.
   - **`(lgb-nb, POS)` ADOPT AND `(baseline, POS)` REGRESSION** → ship-modified-shape: revert *only* `_<POS>_FEATURE_COLUMNS` extension for that position. Keep schema + builder. Re-run that position's `--model baseline` backtest cell only (snapshot regen for the 4 baseline-position rows in `tests/backtest/model_metrics.json`; lgb-nb rows stay as-is from step 5).
   - **`(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT** → revert that position: undo `*FeaturesSchema` extension + `build_<pos>_features` integration + `_<POS>_FEATURE_COLUMNS` extension. Keep cluster-A test fixture defaults (cheap to leave). Re-run that position × both model classes' backtest cells (4 rows).
9. **Update `project_management.md`** with a top-of-file decision-log entry covering the per-position outcomes (format: PR #28's entry as template, with the 2-position §1.3.5 matrix outcome explicit). **Update `TODO.md` #25** with measured magnitudes per position and the per-position shipped state (shipped / shipped-modified-shape / reverted). Add a per-position cross-class production-routing follow-up note (TODO #25 sub-bullet for each: "POS production routes to baseline. With weather cols now in {Rb|Wr}FeaturesSchema, a cross-class re-eval comparing lightgbm-nb candidate to baseline baseline could justify flipping `_PositionDispatch[POS].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next POS-related work.").
10. Push branch + open PR.

---

## 5. Risks

- **Probe-vs-gate verdict divergence per position.** Weather magnitudes are the smallest-magnitude SIGNAL family probe to date (-0.0081 fpts RB, -0.0110 fpts WR; comparison: PR #28 was the first probe where signal lived only in lgb-nb composite, not BaselineModel). A small calibration error in the wrong direction could flip either binding cell to MARGINAL or DO_NOT_ADOPT. Per-position revert path defined per §1.3.5. Both positions diverging simultaneously is extremely low-probability given they passed the bootstrap CI test independently (CI strictly below 0 for each).
- **Probe-vs-gate magnitude divergence.** Track record of probe → gate calibration: PR #20 → #21 matched to 4 decimal places on RB (-0.0124 → -0.0124); PR #25 → #26 matched within ~0.004 fpts on WR (-0.0414 → -0.0371, within probe CI); PR #25 → #27 matched within ~0.0017 fpts on TE (-0.0107 → -0.0090). For weather at -0.0081 / -0.0110 magnitudes, a ~10% calibration error is ~0.001 fpts — within the per-cell noise floor. The §1.3.5 rule binds on the verdict label, not magnitude.
- **`(baseline, *)` REGRESSION risk.** Probe found 0/120 SIGNAL Phase 1 cells on baseline across both modes, all 4 positions — strongest "no signal on baseline" evidence to date (vs PR #25's TE which had DO_NOT_ADOPT but with point estimates in the bootstrap noise floor). Gate flipping baseline RB or WR to REGRESSION (CI strictly above 0) is low-probability but not zero. The §1.3.5 modified-shape ship path covers it; the implementation plan's Phase 5 is conditional on the per-position gate verdict.
- **`baseline.py:_<POS>_FEATURE_COLUMNS` miss.** PR #21 caught this at commit `9895dee` (RB), PR #26 caught it for WR, PR #27 caught it for TE — same pattern recurs for any per-position feature-list extension. The implementation plan explicitly schedules each (RB and WR) as its own task with a smoke-test verification (assert the 4 names appear in each `_<POS>_FEATURE_COLUMNS` post-edit; assert the lightgbm-family dynamic feature list also includes them).
- **Coverage shift between probe override and production builder.** The probe override at `data/features_probe/weather.parquet` was built from the same `attach_weather_features` helper this spec wires into the production builders. Coverage should match closely; PR #28's 2021-2024 audit showed dome rate ~29%, outdoor-NaN ~2-3% per season, NaN rate ~8% pooled. Step 4 of §4 cross-checks builder-output coverage against the probe audit before running the gate. Materially different coverage (>5pp shift) signals a builder wiring bug — fix the builder, not the threshold.
- **Cluster-A test fixture leftovers.** PR #26 caught 3 leftover sites for WR (cache fixture + lightgbm/ensemble synthetic random fixtures + tune_lightgbm fixture); same defensive-grep pattern applies here for both RB and WR. The implementation plan front-loads the grep and confirms zero leftover sites post-edit.
- **Feature cache invalidation for both RB and WR.** Adding 4 cols to each schema invalidates both existing caches under `data/features/rb/...` and `data/features/wr/...` — the schema validate would reject old rows missing the new cols. §4 step 3 calls this out and runs the refresh explicitly before any backtest invocation that reads the cache. Same pattern as PR #21 (RB), PR #26 (WR), PR #27 (TE) cache invalidations.
- **EnsembleModel weight regen — limited blast radius.** EnsembleModel pulls from C-NB internally per Plan 6. The lgb-nb backtest cell change (RB and WR) means `data/ensemble_weights/ensemble_{rb,wr}_*.json` weight files would regen IF EnsembleModel were re-run as part of the snapshot. But §1.3.4 explicitly skips ensemble in the gate, so this PR does NOT re-run EnsembleModel — the ensemble weight files for RB and WR stay as-is from before the PR (with the schema's lightgbm-nb-derived predictions reflecting pre-weather state for ensemble fitting). **The downstream consequence:** if a future PR runs EnsembleModel for RB or WR, the ensemble fit will pick up the weather cols via dynamic schema derivation and produce different weights. This is acceptable and isolated; no surprise regen in this PR's snapshot.
- **Working-tree noise on main.** The session-start `git status` on `main` flagged unstaged deletions of `data/ensemble_weights/*.json` and `data/tuned_params/*.json`. The worktree at `.worktrees/feat-weather-features-rb-wr` starts clean (the deletions are working-tree-only on main, not committed); this PR's work is isolated. The deletion question is independent of this spec and can be resolved on `main` separately (typical recovery: `git checkout HEAD -- data/ensemble_weights/ data/tuned_params/`).
- **Recurring QB augment regression check.** PR #28 noted the weather probe showed only mild QB augment regression (composite +0.0077 fpts NOT REGRESSION; vs PRs #23/#24/#25 at +0.0268 to +0.0382). QB is not in this PR's scope, but the spec records the check as a guardrail: even if the §1.3.5 modified-shape branch fires for RB or WR, **do not** preemptively extend `_QB_FEATURE_COLUMNS`. The QB rushing_yards 2023 cell from PR #28's audit (+0.0812 fpts CI [+0.0133, +0.1515]) remains a known per-stat regression, but pooled QB is NULL across all 4 modes; refined-unit weather work for QB belongs in TODO #25 follow-ups, not here.
- **First combined-PR shipping two positions on the same model class.** PR #20 → #21 was single-position; PR #25 split into PR #26 (WR) + PR #27 (TE) by position because each binding cell's shape was different (WR baseline-binding, TE lgb-nb-binding). This PR sets the precedent that bundled-positions integration is the right shape when both positions bind on the same model class + mode. Documented prominently in §1, §1.3.5, and the summary report. Future "integrate feature family X into positions P1+P2" specs that follow this pattern will reference this spec as precedent.

---

## 6. Documentation updates on merge

- **`project_management.md`:** Append a top-of-file decision-log entry. Format matches PR #28's entry — title, status, verdict, what shipped or reverted per position, magnitude, probe-vs-gate calibration note per position, **plus** an explicit "first bundled two-position integration" section noting that the per-position contingency matrix was applied independently to RB and WR.
- **`TODO.md` #25:** Record the production integration outcome per position (shipped / shipped-modified-shape / reverted, with measured magnitude). Cross-reference the summary report and PR #28's probe entry. Note that the "broad-cut weather family at the in-builder unit" is now closed for the RB and WR ADOPT cells from PR #28; QB and TE remain DO_NOT_ADOPT at this unit; refined-unit candidates (cold-weather threshold, multi-class surface, kickoff hour, surface × position interactions, per-team weather acclimation, precipitation, wind direction) remain open under the same TODO. None queued.
- **`docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`:** No changes. The probe spec stays as historical record.
- **`CONTRIBUTING.md`:** No changes. PR #28's "Regenerating the weather override" subsection still applies to the override path; the production-builder integration is closed inside `build_<pos>_features` and doesn't change any user-facing CLI invocation.
- **(Cross-class production routing follow-up — per position):** Add a `TODO.md` sub-note under TODO #25 flagging the cross-class question explicitly per position: "RB / WR production routes to `baseline` per Plan 8 (2026-04-29). With weather cols now in `RbFeaturesSchema` / `WrFeaturesSchema`, a cross-class re-eval (`scripts/adoption_gate.py --baseline-run <pre-PR> --candidate-run <post-PR> --position {RB|WR}` comparing `lightgbm-nb` candidate to `baseline` baseline) could justify flipping `_PositionDispatch[{RB|WR}].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next {RB|WR}-related work."

---

## 7. Implementation phasing

The implementation plan should structure work in phases per the CLAUDE.md "PHASED EXECUTION" rule (≤5 files per phase). Suggested phasing:

- **Phase 1 — Schema changes (1 file).** `schemas.py` (4 columns to `RbFeaturesSchema`, identical 4 columns to `WrFeaturesSchema`). Verify: existing tests still pass at the schema-only level; `RbFeaturesSchema.validate(...)` and `WrFeaturesSchema.validate(...)` accept frames that have the 4 new cols and reject frames that don't.

- **Phase 2 — Builder integration (2 files).** `features/rb.py` (add `attach_weather_features` import + call after the existing `attach_pbp_family_features`); `features/wr.py` (add `attach_weather_features` import + call after the existing trajectory attach). Verify: existing RB / WR feature builder tests still pass; some happy-path tests will fail until Phase 3 / Phase 4 add fixture defaults — that's expected and tracked in Phase 4.

- **Phase 3 — `_<POS>_FEATURE_COLUMNS` extensions (1 file).** `models/baseline.py` (extend `_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS` with the 4 new names each). Smoke test: assert each `_<POS>_FEATURE_COLUMNS` tuple contains the 4 new col names; assert the lightgbm-family dynamic feature list (derived from each schema) also contains the 4 new col names. Both checks are 1-line asserts in a new unit test.

- **Phase 4 — Tests + cluster-A leftover fixtures (≤5 files).** `tests/test_features/test_rb.py` (extend happy-path + 4 new tests); `tests/test_features/test_wr.py` (extend happy-path + 4 new tests); cluster-A grep for `opp_allowed_rb_fppg_l4` / `opp_allowed_wr_fppg_l4` and add 4 default cols to every minimal-row construction site (likely 3-5 files: `tests/test_features/test_cache.py`, `tests/test_models/test_*.py` per position, `tests/test_scripts/test_tune_lightgbm.py`). Verify: full pytest suite passes; mypy + ruff clean.

- **Phase 5 — Real-data execution + reports (no code).** §4 steps 2-7. Output: refreshed RB + WR caches, backtest snapshot delta (`tests/backtest/model_metrics.json` rows for the 4 cells), adoption gate report (`reports/adoption_gate_weather_features_rb_wr.{md,csv}`), summary report (`reports/weather_features_rb_wr_summary.md`). The 4-cell gate result determines which §1.3.5 branch fires for each position.

- **Phase 6 — Conditional code adjustments + documentation (1-5 files).** Branches per position independently per §1.3.5. Possible per-position outcomes:
  - **Both positions ship-as-designed** (default branch, expected from probe evidence): no code adjustments. Update `project_management.md` + `TODO.md` per §6 (2 files).
  - **One or both positions ship-modified-shape** (low-probability `(baseline, POS)` REGRESSION): revert the affected position's `_<POS>_FEATURE_COLUMNS` extension in `models/baseline.py` (1 file). Re-run that position's `--model baseline` backtest cell only. Update `project_management.md` + `TODO.md` (+ summary report addendum noting the modified-shape branch fired for that position). 3-4 files total per affected position.
  - **One or both positions full-revert** (lower-probability `(lgb-nb, POS)` MARGINAL/DO_NOT_ADOPT): revert the schema edit in `schemas.py`, the builder edit in `features/<pos>.py`, the `_<POS>_FEATURE_COLUMNS` extension in `baseline.py`, and the cluster-A fixture additions (or keep the cluster-A defaults — cheap to leave). Re-run that position × both model classes' backtest cells. Update `project_management.md` + `TODO.md` documenting the divergence. 4-5 files total per affected position.

This phasing keeps each step ≤5 files. Phase 2 may produce test failures before Phase 4 adds fixture defaults — that's expected, tracked, and the implementation plan resolves it within a single Phase 4 commit. Phase 6's per-position conditional structure keeps the implementation plan's per-task definition crisp — only the actual outcomes execute (typically: both positions ship-as-designed, no Phase 6 code adjustments needed).
