# Weather Features RB+WR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote PR #28's weather family probe RB and WR cells into production. Append 4 nullable-float weather columns to `RbFeaturesSchema` + `WrFeaturesSchema`; wire `attach_weather_features` into both builders; run the dual-run adoption gate; ship per position based on `(LightGBMNbModel, POS)` verdict per spec §1.3.5 (probe predicted -0.0081 fpts on RB and -0.0110 fpts on WR).

**Architecture:** Schema delta + builder integration only — `compute_weather_features`, `attach_weather_features`, `build_weather_overrides` already public from PR #28; `schedules` already plumbed through all 4 caller scripts. Combined RB+WR PR with **per-position contingency matrix** (independent ship/revert decisions). Second integration to bind on a non-default model class (lgb-nb, not baseline) — production routing stays on BaselineModel for both positions; cross-class flip is a deferred per-position follow-up. **First combined-PR shipping two positions on the same model class** (PR #20→#21 was single-position, PR #25 split into PR #26 + PR #27 by position because of differing binding cells).

**Tech Stack:** Python 3.11, pandas (pyarrow strings + nullable Int64/Float64), pandera (`DataFrameModel` + `strict="filter"`), pytest, scikit-learn (`RidgeCV` via `BaselineModel`), LightGBM (Quantile / NB-2 sub-models), pre-commit hooks (ruff, mypy strict).

**Spec:** `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`.

---

## File Structure (decomposition lock-in)

**Modify:**
- `src/projections/schemas.py` — extend `RbFeaturesSchema` (~line 651, after `team_def_epa_resid_l4`) + `WrFeaturesSchema` (~line 537, after the trajectory cols) with 4 nullable-float weather columns each (Phase 1).
- `src/projections/features/rb.py` — add `attach_weather_features` import; call it after the existing `attach_pbp_family_features` merge, before `return RbFeaturesSchema.validate(out)` at line 246 (Phase 2).
- `src/projections/features/wr.py` — add `attach_weather_features` import; call it after the existing trajectory feature merge, before `return WrFeaturesSchema.validate(out)` at line 284 (Phase 2).
- `src/projections/models/baseline.py` — extend `_RB_FEATURE_COLUMNS` (line 360) + `_WR_FEATURE_COLUMNS` (line 266) with the 4 new col names each (Phase 3, per-position conditional revert in Phase 6 per §1.3.5 modified-shape).
- `tests/test_features/test_rb.py` — extend happy-path; add 4 weather behavior tests (Phase 4).
- `tests/test_features/test_wr.py` — same (Phase 4).
- Cluster-A leftover sites discovered via `opp_allowed_rb_fppg_l4` + `opp_allowed_wr_fppg_l4` defensive grep (Phase 4).
- `tests/backtest/model_metrics.json` — backtest snapshot delta for (RB+WR) × (baseline + lgb-nb) = 4 cells (Phase 5).
- `project_management.md`, `TODO.md` — decision log per position + close TODO #25's RB and WR branches (Phase 6, conditional per gate verdict).

**Add (reports, Phase 5 — committed but not code):**
- `reports/adoption_gate_weather_features_rb_wr.{md,csv}` — gate output across baseline + lgb-nb × RB + WR.
- `reports/weather_features_rb_wr_summary.md` — decision log + per-position table + probe-vs-gate calibration + coverage stats + binding-cell-shift rationale + cross-class deferred-follow-up note (per position) + 2-position §1.3.5 outcome matrix.

**Already in place (no changes needed):**
- `src/projections/features/weather_features.py` — public `compute_weather_features`, `attach_weather_features`, `build_weather_overrides` (PR #28).
- `src/projections/features/rb.py` and `wr.py` signatures — `schedules` kwarg already plumbed (existing convention since the original WR builder shipped).
- `scripts/refresh_features.py`, `scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py` — all load + thread `schedules`.
- `data/raw/schedules/` — partition coverage 2018-2024 already canonical since the original 2a ingest.

**No fixture extension required** for trailing-N history (weather is per-game; existing `baseline_weekly_stats_rb` / `baseline_weekly_stats_wr` fixtures already cover the relevant week ranges). No `tests/conftest.py` edits in this plan.

**No caller-script changes** (schedules already plumbed for all 4 positions; no new kwarg).

**No helper extraction** (all 3 weather-features functions public from PR #28).

---

## Phase 1 — Schema changes

Goal: ship the `RbFeaturesSchema` + `WrFeaturesSchema` column deltas in a single commit. After this phase: existing RB / WR feature builder happy-path tests will FAIL (builders don't yet produce the 4 new cols, so `*FeaturesSchema.validate(...)` raises SchemaError) — Phase 2 closes them.

### Task 1: Add 4 weather columns to `RbFeaturesSchema` and `WrFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (two insert sites — RB after `team_def_epa_resid_l4`, WR after the trajectory cols)

- [ ] **Step 1: Re-read the current `RbFeaturesSchema` ending and `WrFeaturesSchema` ending**

Run: `sed -n '601,660p' src/projections/schemas.py`
Expected: shows `RbFeaturesSchema` definition through `team_def_epa_resid_l4: Series[float] = pa.Field(nullable=True)` then a `Config` class.

Run: `sed -n '484,545p' src/projections/schemas.py`
Expected: shows `WrFeaturesSchema` definition through `snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)` (the last trajectory col added in PR #26) then a `Config` class.

- [ ] **Step 2: Append 4 weather columns + comment block to `RbFeaturesSchema` before its `Config` class**

Edit `src/projections/schemas.py`. In `RbFeaturesSchema`, find:

```python
    pace_l4: Series[float] = pa.Field(nullable=True)
    proe_l4: Series[float] = pa.Field(nullable=True)
    team_ayps_l4: Series[float] = pa.Field(ge=0, nullable=True)
    team_def_epa_resid_l4: Series[float] = pa.Field(nullable=True)

    class Config:
```

Replace with:

```python
    pace_l4: Series[float] = pa.Field(nullable=True)
    proe_l4: Series[float] = pa.Field(nullable=True)
    team_ayps_l4: Series[float] = pa.Field(ge=0, nullable=True)
    team_def_epa_resid_l4: Series[float] = pa.Field(nullable=True)

    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration
    # spec). Sourced from existing SchedulesSchema columns (wind, temp, roof,
    # surface) — no new ingest. Dome / closed-roof games filled with
    # (wind=0, temp=70) per compute_weather_features semantics. Outdoor NaN
    # wind/temp propagates; ~8% NaN rate concentrated in 2018-2019.
    wind_speed_mph: Series[float] = pa.Field(ge=0, nullable=True)
    is_high_wind: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    temperature_f: Series[float] = pa.Field(nullable=True)
    is_grass_surface: Series[float] = pa.Field(ge=0, le=1, nullable=True)

    class Config:
```

- [ ] **Step 3: Append the identical 4-col block to `WrFeaturesSchema` before its `Config` class**

In `src/projections/schemas.py`, in `WrFeaturesSchema`, find:

```python
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

    class Config:
```

Replace with:

```python
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration
    # spec). Sourced from existing SchedulesSchema columns (wind, temp, roof,
    # surface) — no new ingest. Dome / closed-roof games filled with
    # (wind=0, temp=70) per compute_weather_features semantics. Outdoor NaN
    # wind/temp propagates; ~8% NaN rate concentrated in 2018-2019.
    wind_speed_mph: Series[float] = pa.Field(ge=0, nullable=True)
    is_high_wind: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    temperature_f: Series[float] = pa.Field(nullable=True)
    is_grass_surface: Series[float] = pa.Field(ge=0, le=1, nullable=True)

    class Config:
```

(The 4 col block is byte-for-byte identical between the two schemas. The comment header is identical.)

- [ ] **Step 4: Confirm both schemas parse and the 4 cols appear in each**

Run: `.venv/Scripts/python.exe -c "
from projections.schemas import RbFeaturesSchema, WrFeaturesSchema
WX = ['wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface']
for name, schema in [('RbFeaturesSchema', RbFeaturesSchema), ('WrFeaturesSchema', WrFeaturesSchema)]:
    cols = list(schema.to_schema().columns.keys())
    print(name, len(cols), 'columns')
    for c in WX:
        print(' ', c, 'OK' if c in cols else 'MISSING')
"`
Expected: `RbFeaturesSchema` shows 28 columns (was 24); `WrFeaturesSchema` shows 28 columns (was 24); all 4 weather cols `OK` for each.

- [ ] **Step 5: Run schema tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/ -v 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 6: Run existing RB + WR builder tests — they should now FAIL (output frame is missing the 4 new columns)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py tests/test_features/test_wr.py -v 2>&1 | tail -20`
Expected: FAIL on the `test_build_<rb|wr>_features_returns_validated_frame` happy-path tests with `SchemaError` mentioning one of `wind_speed_mph` / `is_high_wind` / `temperature_f` / `is_grass_surface` is required but missing. **This is expected** — Phase 2 wires the builders to produce them.

- [ ] **Step 7: Commit (with the RB + WR builder test failures unfixed — Phase 2 closes them)**

```bash
git add src/projections/schemas.py
git commit -m "$(cat <<'EOF'
schema(weather): add 4 weather columns to RbFeaturesSchema + WrFeaturesSchema

Schema-only change. RB + WR builder tests will fail until Phase 2
wires attach_weather_features into both builders. Identical 4-col
block (wind_speed_mph, is_high_wind, temperature_f, is_grass_surface)
in each schema, sourced from existing SchedulesSchema cols — no new
ingest.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Builder integration

Goal: wire `attach_weather_features` into both `build_rb_features` and `build_wr_features` in two separate commits. After this phase: existing happy-path tests pass; cluster-A synthetic minimal-row fixtures still need the 4 new col defaults — Phase 4 closes them.

### Task 2: Wire `attach_weather_features` into `build_rb_features`

**Files:**
- Modify: `src/projections/features/rb.py`

- [ ] **Step 1: Re-read `rb.py` to confirm current state**

Run: `sed -n '1,30p' src/projections/features/rb.py && echo '---' && sed -n '240,250p' src/projections/features/rb.py`
Expected: imports include `attach_pbp_family_features` from `projections.features.pbp_team_features`; the function ends with a final `return RbFeaturesSchema.validate(out)` at line 246. The local var `sch` (set at line ~80) holds the exact-week-mask-filtered schedules.

- [ ] **Step 2: Add `attach_weather_features` import**

Edit `src/projections/features/rb.py`. Find the existing import line:

```python
from projections.features.pbp_team_features import attach_pbp_family_features
```

Replace with:

```python
from projections.features.pbp_team_features import attach_pbp_family_features
from projections.features.weather_features import attach_weather_features
```

- [ ] **Step 3: Add the weather attach call before `return RbFeaturesSchema.validate(out)`**

Edit `src/projections/features/rb.py`. Find:

```python
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return RbFeaturesSchema.validate(out)
```

Replace with:

```python
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    # --- Weather features (PR #28 family probe + 2026-05-08 RB+WR integration) ---
    # attach_weather_features merges 4 nullable-float cols onto (season, week, team)
    # from the exact-week-filtered schedules. Dome / closed-roof games have
    # wind=0 / temp=70 per compute_weather_features semantics.
    out = attach_weather_features(out, sch)

    return RbFeaturesSchema.validate(out)
```

(The local `sch` is the `exact_week_mask`-filtered schedules frame — already restricted to `(season=season, week=as_of_week)`. Weather is per-game, so we want the current week's row only; passing the full `schedules` arg would also work via left-merge but would be semantically unnecessary.)

- [ ] **Step 4: Run the RB happy-path test to confirm weather wiring closes the schema gap**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py::test_build_rb_features_returns_validated_frame -v`
Expected: PASS — the 4 new schema columns are now produced. Some may be NaN (synthetic schedules fixture may not specify roof/wind/temp for every fixture row), but `nullable=True` on all 4 fields accepts NaN.

- [ ] **Step 5: Run the full RB builder test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py -v 2>&1 | tail -10`
Expected: all PASS — weather cols handled correctly.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/rb.py
git commit -m "$(cat <<'EOF'
feat(rb): wire attach_weather_features into build_rb_features

Imports attach_weather_features from features.weather_features and
calls it on the exact-week-filtered schedules frame just before
schema validation. No signature change. Closes the RB happy-path
SchemaError introduced by Phase 1's schema cols. Cluster-A leftover
fixtures will surface in Phase 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire `attach_weather_features` into `build_wr_features`

**Files:**
- Modify: `src/projections/features/wr.py`

- [ ] **Step 1: Re-read `wr.py` import block + tail**

Run: `sed -n '1,30p' src/projections/features/wr.py && echo '---' && sed -n '278,290p' src/projections/features/wr.py`
Expected: imports include `attach_trajectory_features` and `build_draft_lookup` from `trajectory_features`; the function ends with `return WrFeaturesSchema.validate(out)` at line 284 immediately after the trajectory `out.merge(...)` call.

- [ ] **Step 2: Add `attach_weather_features` import**

Edit `src/projections/features/wr.py`. Find the existing import block:

```python
from projections.features.trajectory_features import (
    attach_trajectory_features,
    build_draft_lookup,
)
```

Add a new import line immediately after it:

```python
from projections.features.trajectory_features import (
    attach_trajectory_features,
    build_draft_lookup,
)
from projections.features.weather_features import attach_weather_features
```

- [ ] **Step 3: Add the weather attach call before `return WrFeaturesSchema.validate(out)`**

Edit `src/projections/features/wr.py`. Find:

```python
    out = out.merge(
        traj[
            [
                "gsis_id",
                "season",
                "week",
                "age",
                "is_rookie",
                "volume_trend_l4_minus_prior_l4",
                "snap_pct_change_l4_vs_prior_l4",
            ]
        ],
        on=["gsis_id", "season", "week"],
        how="left",
    )

    return WrFeaturesSchema.validate(out)
```

Replace with:

```python
    out = out.merge(
        traj[
            [
                "gsis_id",
                "season",
                "week",
                "age",
                "is_rookie",
                "volume_trend_l4_minus_prior_l4",
                "snap_pct_change_l4_vs_prior_l4",
            ]
        ],
        on=["gsis_id", "season", "week"],
        how="left",
    )

    # --- Weather features (PR #28 family probe + 2026-05-08 RB+WR integration) ---
    # attach_weather_features merges 4 nullable-float cols onto (season, week, team)
    # from the exact-week-filtered schedules. Dome / closed-roof games have
    # wind=0 / temp=70 per compute_weather_features semantics.
    out = attach_weather_features(out, sch)

    return WrFeaturesSchema.validate(out)
```

- [ ] **Step 4: Run the WR happy-path test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_returns_validated_frame -v`
Expected: PASS.

- [ ] **Step 5: Run the full WR builder test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py -v 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/wr.py
git commit -m "$(cat <<'EOF'
feat(wr): wire attach_weather_features into build_wr_features

Same shape as the RB integration in the prior commit. Imports
attach_weather_features and calls it on the exact-week-filtered
schedules frame just before schema validation. No signature change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — `_<POS>_FEATURE_COLUMNS` extensions

Goal: extend `baseline.py:_RB_FEATURE_COLUMNS` + `_WR_FEATURE_COLUMNS` so `BaselineModel.fit` for RB / WR sees the 4 new features. The lightgbm family derives feature lists dynamically from each schema and auto-picks-up the cols — no edits needed there. After this task: synthetic minimal-row test fixtures will fail because they're missing the 4 new col defaults — Phase 4 closes them.

### Task 4: Extend `_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS`

**Files:**
- Modify: `src/projections/models/baseline.py:266-295` (WR) and `:360-386` (RB)

- [ ] **Step 1: Re-read both feature column tuples**

Run: `sed -n '260,300p' src/projections/models/baseline.py && echo '---' && sed -n '358,390p' src/projections/models/baseline.py`
Expected: shows `_WR_FEATURE_COLUMNS` ending with `"snap_pct_change_l4_vs_prior_l4",` and `_RB_FEATURE_COLUMNS` ending with `"team_def_epa_resid_l4",` (each followed by a closing paren).

- [ ] **Step 2: Append 4 col names + comment block to `_RB_FEATURE_COLUMNS`**

Edit `src/projections/models/baseline.py`. Find:

```python
    # PBP team-level family features (spec 2026-05-01).
    "pace_l4",
    "proe_l4",
    "team_ayps_l4",
    "team_def_epa_resid_l4",
)
```

Replace with:

```python
    # PBP team-level family features (spec 2026-05-01).
    "pace_l4",
    "proe_l4",
    "team_ayps_l4",
    "team_def_epa_resid_l4",
    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration).
    # lightgbm derives feature lists from RbFeaturesSchema dynamically and
    # auto-picks-up; baseline.py is hardcoded so must be updated explicitly.
    # Same spec gap class as PR #21 (RB PBP, 9895dee), PR #26 (WR trajectory),
    # PR #27 (TE trajectory).
    "wind_speed_mph",
    "is_high_wind",
    "temperature_f",
    "is_grass_surface",
)
```

- [ ] **Step 3: Append 4 col names + comment block to `_WR_FEATURE_COLUMNS`**

Edit `src/projections/models/baseline.py`. Find:

```python
    # Trajectory features (PR #25 family probe + 2026-05-03 WR integration).
    # lightgbm derives feature lists from WrFeaturesSchema dynamically and
    # auto-picks-up; baseline.py is hardcoded so must be updated explicitly.
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
)
```

Replace with:

```python
    # Trajectory features (PR #25 family probe + 2026-05-03 WR integration).
    # lightgbm derives feature lists from WrFeaturesSchema dynamically and
    # auto-picks-up; baseline.py is hardcoded so must be updated explicitly.
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration).
    # Same spec-gap class as the trajectory cols above (and PR #21, PR #27).
    "wind_speed_mph",
    "is_high_wind",
    "temperature_f",
    "is_grass_surface",
)
```

- [ ] **Step 4: Smoke-verify both baseline + lightgbm see the 4 weather cols for both positions**

Run: `.venv/Scripts/python.exe -c "
from projections.models.baseline import _RB_FEATURE_COLUMNS, _WR_FEATURE_COLUMNS
from projections.models.lightgbm import _filter_features
from projections.schemas import RbFeaturesSchema, WrFeaturesSchema
WX = ('wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface')
for name, baseline_tuple, schema in [
    ('RB', _RB_FEATURE_COLUMNS, RbFeaturesSchema),
    ('WR', _WR_FEATURE_COLUMNS, WrFeaturesSchema),
]:
    schema_cols = list(schema.to_schema().columns.keys())
    lgbm = _filter_features(tuple(schema_cols))
    for c in WX:
        assert c in baseline_tuple, f'{c} missing from baseline _{name}_FEATURE_COLUMNS'
        assert c in lgbm, f'{c} missing from lightgbm _filter_features({name})'
    print(name, ': both baseline + lightgbm see the 4 weather cols')
print('OK')
"`
Expected: `RB : both baseline + lightgbm see the 4 weather cols`, then `WR : both baseline + lightgbm see the 4 weather cols`, then `OK`. (If `_filter_features` has a different signature, adapt — the important thing is that both lists contain the 4 new col names.)

- [ ] **Step 5: Run baseline + lightgbm RB/WR test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_baseline_rb.py tests/test_models/test_baseline_wr.py tests/test_models/test_lightgbm_rb.py tests/test_models/test_lightgbm_wr.py -v 2>&1 | tail -20`
Expected: tests likely fail because synthetic minimal-RB / minimal-WR feature rows don't have the 4 new columns. **This is expected** — Task 6 closes the cluster-A leftovers.

- [ ] **Step 6: Commit (with the cluster-A failures unfixed — Phase 4 closes them)**

```bash
git add src/projections/models/baseline.py
git commit -m "$(cat <<'EOF'
feat(baseline): extend _RB_FEATURE_COLUMNS + _WR_FEATURE_COLUMNS with 4 weather cols

Same spec-gap class as PR #21 (RB PBP, 9895dee), PR #26 (WR trajectory),
PR #27 (TE trajectory) — baseline.py hardcodes per-position feature
tuples while lightgbm derives from each schema dynamically. Cluster-A
synthetic RB/WR fixtures need the new col defaults; Phase 4 closes the
failures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Tests + cluster-A fixture defaults

Goal: add 4 weather behavior tests per position (8 tests total), then defensive-grep for cluster-A synthetic minimal-row sites and add the 4 col defaults so all fixtures parse against the extended schemas.

### Task 5: Add 4 weather behavior tests for RB

**Files:**
- Modify: `tests/test_features/test_rb.py`

- [ ] **Step 1: Read test_rb.py for fixture structure + imports**

Run: `head -50 tests/test_features/test_rb.py && echo '---' && grep -n "^def test_" tests/test_features/test_rb.py | head -10`
Expected: shows the fixture imports and existing test names. Note the canonical builder kwargs: `weekly_stats=rb_weekly_stats`, `snap_counts=rb_snap_counts`, `depth_charts=rb_depth_charts`, `ngs_rushing=rb_ngs_rushing`, `schedules=rb_schedules`, `pbp=fake_pbp_df`, `season=2024`, `as_of_week=5` (or similar — adjust to match the existing test conventions).

- [ ] **Step 2: Append `test_build_rb_features_attach_weather_dome_fill`**

Add at end of `tests/test_features/test_rb.py`:

```python
def test_build_rb_features_attach_weather_dome_fill(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Dome game: wind_speed_mph=0, temperature_f=70, is_high_wind=0 per
    compute_weather_features semantics. Surface determined separately."""
    # Force every schedules row this week to be a dome game.
    sch = rb_schedules.copy()
    week_mask = sch["week"] == 5
    sch.loc[week_mask, "roof"] = "dome"
    sch.loc[week_mask, "wind"] = pd.NA  # upstream NaN — should be overridden by dome fill
    sch.loc[week_mask, "temp"] = pd.NA

    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
    )
    assert (out["wind_speed_mph"] == 0.0).all(), "dome fill should set wind=0"
    assert (out["temperature_f"] == 70.0).all(), "dome fill should set temp=70"
    assert (out["is_high_wind"] == 0.0).all(), "dome fill ⇒ wind<20 ⇒ is_high_wind=0"
```

(If `rb_schedules` fixture's column types don't accept `pd.NA` directly, use `np.nan` from `numpy as np` — match the fixture's existing column dtypes.)

- [ ] **Step 3: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py::test_build_rb_features_attach_weather_dome_fill -v`
Expected: PASS.

- [ ] **Step 4: Append `test_build_rb_features_attach_weather_outdoor_high_wind`**

Add to `tests/test_features/test_rb.py`:

```python
def test_build_rb_features_attach_weather_outdoor_high_wind(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Outdoor high-wind game: wind=22, temp=42, is_high_wind=1.0."""
    sch = rb_schedules.copy()
    week_mask = sch["week"] == 5
    sch.loc[week_mask, "roof"] = pd.NA  # outdoor
    sch.loc[week_mask, "wind"] = 22.0
    sch.loc[week_mask, "temp"] = 42.0

    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
    )
    assert (out["wind_speed_mph"] == 22.0).all()
    assert (out["temperature_f"] == 42.0).all()
    assert (out["is_high_wind"] == 1.0).all(), "wind=22 >= 20 ⇒ is_high_wind=1"
```

- [ ] **Step 5: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py::test_build_rb_features_attach_weather_outdoor_high_wind -v`
Expected: PASS.

- [ ] **Step 6: Append `test_build_rb_features_attach_weather_grass_surface`**

Add to `tests/test_features/test_rb.py`:

```python
def test_build_rb_features_attach_weather_grass_surface(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Surface code 'grass' ⇒ is_grass_surface=1.0; anything else ⇒ 0.0."""
    sch = rb_schedules.copy()
    week_mask = sch["week"] == 5
    # Make all this week's home games grass surface, opp games turf.
    home_mask = week_mask & sch["home_team"].isin(sch.loc[week_mask, "home_team"].unique())
    sch.loc[home_mask, "surface"] = "grass"
    away_mask = week_mask & ~home_mask
    sch.loc[away_mask, "surface"] = "sportturf"

    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
    )
    # Each player picks up their team's stadium surface (player-team join).
    # Both 0.0 and 1.0 should appear in the output (assuming fixture has
    # both home and away players).
    surface_vals = set(out["is_grass_surface"].dropna().unique())
    assert 1.0 in surface_vals, "grass surface should produce is_grass_surface=1"
    assert 0.0 in surface_vals, "non-grass surface should produce is_grass_surface=0"
```

(If the fixture only has one team or all home/away on the same surface, simplify to "make all rows grass; assert all == 1.0; then make all sportturf; assert all == 0.0" in two builder invocations.)

- [ ] **Step 7: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py::test_build_rb_features_attach_weather_grass_surface -v`
Expected: PASS.

- [ ] **Step 8: Append `test_build_rb_features_attach_weather_bye_week_fallback`**

Add to `tests/test_features/test_rb.py`:

```python
def test_build_rb_features_attach_weather_bye_week_fallback(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Defensive: builder already filters rostered teams to those with
    schedule rows in as_of_week (TODO #9a), so a bye-week row should not
    reach the weather merge in the first place. This test confirms that
    if such a row did reach the merge (e.g., a future builder change
    relaxes the filter), the schema's nullable=True accepts the resulting
    NaN values for the 4 weather cols.

    Approach: pass an empty schedules frame so the helper's left-merge
    produces NaN for every col; verify the output schema validates.
    """
    empty_sch = rb_schedules.iloc[0:0].copy()  # preserve column dtypes, zero rows
    # The bye-week filter inside build_rb_features will reject every depth-chart
    # row when no schedules row matches, so the output frame is empty. We assert
    # the empty-output happy path doesn't raise. (If a future change re-allows
    # bye-week rows to slip through, the schema's nullable=True on the 4 cols
    # still accepts NaN.)
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=empty_sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
    )
    assert len(out) == 0, "empty schedules should drive empty output"
    # All 4 weather cols still present in the output schema (zero-row frames).
    for c in ("wind_speed_mph", "is_high_wind", "temperature_f", "is_grass_surface"):
        assert c in out.columns
```

- [ ] **Step 9: Run all 4 new tests + the existing happy-path**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py -v 2>&1 | tail -20`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add tests/test_features/test_rb.py
git commit -m "$(cat <<'EOF'
test(rb): 4 weather behavior tests at the build_rb_features boundary

Covers dome-fill (wind=0, temp=70), outdoor high-wind (≥20 mph),
grass surface, and the empty-schedules defensive fallback (bye-week
filter robustness). End-to-end through the builder, not the helper
in isolation — PR #28 already covers compute_weather_features and
attach_weather_features unit-level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add 4 weather behavior tests for WR

**Files:**
- Modify: `tests/test_features/test_wr.py`

- [ ] **Step 1: Read test_wr.py for fixture structure + imports**

Run: `head -50 tests/test_features/test_wr.py && echo '---' && grep -n "^def test_" tests/test_features/test_wr.py | head -10`
Expected: shows fixture imports and existing test names. Note the canonical builder kwargs — likely: `weekly_stats=wr_weekly_stats`, `snap_counts=wr_snap_counts`, `depth_charts=wr_depth_charts`, `ngs_receiving=wr_ngs_receiving`, `schedules=wr_schedules`, `pbp=fake_pbp_df`, `draft_picks=wr_draft_picks` (PR #26 added this kwarg), `season=2024`, `as_of_week=5`. Confirm the exact kwarg list against the existing happy-path test in the same file before writing the new tests.

- [ ] **Step 2: Append `test_build_wr_features_attach_weather_dome_fill`**

Add at end of `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_weather_dome_fill(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Dome game: wind_speed_mph=0, temperature_f=70, is_high_wind=0 per
    compute_weather_features semantics."""
    sch = wr_schedules.copy()
    week_mask = sch["week"] == 5
    sch.loc[week_mask, "roof"] = "dome"
    sch.loc[week_mask, "wind"] = pd.NA
    sch.loc[week_mask, "temp"] = pd.NA

    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
    )
    assert (out["wind_speed_mph"] == 0.0).all(), "dome fill should set wind=0"
    assert (out["temperature_f"] == 70.0).all(), "dome fill should set temp=70"
    assert (out["is_high_wind"] == 0.0).all(), "dome fill ⇒ wind<20 ⇒ is_high_wind=0"
```

(If the WR fixture set doesn't have `wr_draft_picks`, drop that kwarg + the corresponding parameter — match the existing happy-path test's canonical kwarg list.)

- [ ] **Step 3: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_attach_weather_dome_fill -v`
Expected: PASS.

- [ ] **Step 4: Append `test_build_wr_features_attach_weather_outdoor_high_wind`**

Add to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_weather_outdoor_high_wind(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Outdoor high-wind game: wind=22, temp=42, is_high_wind=1.0."""
    sch = wr_schedules.copy()
    week_mask = sch["week"] == 5
    sch.loc[week_mask, "roof"] = pd.NA
    sch.loc[week_mask, "wind"] = 22.0
    sch.loc[week_mask, "temp"] = 42.0

    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
    )
    assert (out["wind_speed_mph"] == 22.0).all()
    assert (out["temperature_f"] == 42.0).all()
    assert (out["is_high_wind"] == 1.0).all(), "wind=22 >= 20 ⇒ is_high_wind=1"
```

- [ ] **Step 5: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_attach_weather_outdoor_high_wind -v`
Expected: PASS.

- [ ] **Step 6: Append `test_build_wr_features_attach_weather_grass_surface`**

Add to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_weather_grass_surface(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Surface code 'grass' ⇒ is_grass_surface=1.0; anything else ⇒ 0.0."""
    sch = wr_schedules.copy()
    week_mask = sch["week"] == 5
    home_mask = week_mask & sch["home_team"].isin(sch.loc[week_mask, "home_team"].unique())
    sch.loc[home_mask, "surface"] = "grass"
    away_mask = week_mask & ~home_mask
    sch.loc[away_mask, "surface"] = "sportturf"

    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
    )
    surface_vals = set(out["is_grass_surface"].dropna().unique())
    assert 1.0 in surface_vals, "grass surface should produce is_grass_surface=1"
    assert 0.0 in surface_vals, "non-grass surface should produce is_grass_surface=0"
```

- [ ] **Step 7: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_attach_weather_grass_surface -v`
Expected: PASS.

- [ ] **Step 8: Append `test_build_wr_features_attach_weather_bye_week_fallback`**

Add to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_weather_bye_week_fallback(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Empty schedules: builder's bye-week filter rejects every depth-chart
    row when no schedules row matches, so the output frame is empty.
    Confirms the empty-output happy path doesn't raise even with the new
    weather merge in place. If a future change re-allows bye-week rows
    through the filter, the schema's nullable=True on the 4 weather cols
    would still accept the resulting NaN values."""
    empty_sch = wr_schedules.iloc[0:0].copy()
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=empty_sch,
        season=2024,
        as_of_week=5,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
    )
    assert len(out) == 0, "empty schedules should drive empty output"
    for c in ("wind_speed_mph", "is_high_wind", "temperature_f", "is_grass_surface"):
        assert c in out.columns
```

- [ ] **Step 9: Run all 4 new tests + the existing happy-path**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py -v 2>&1 | tail -20`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add tests/test_features/test_wr.py
git commit -m "$(cat <<'EOF'
test(wr): 4 weather behavior tests at the build_wr_features boundary

Same shape as the RB tests in the prior commit. Covers dome-fill,
outdoor high-wind, grass surface, and the empty-schedules defensive
fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Defensive grep + cluster-A leftover fixture defaults

**Files:**
- Modify: `tests/test_features/test_cache.py` (likely; confirm via grep)
- Modify: `tests/test_models/test_baseline_rb.py` / `test_baseline_wr.py` (likely)
- Modify: `tests/test_models/test_lightgbm_rb.py` / `test_lightgbm_wr.py` (likely)
- Modify: `tests/test_models/test_ensemble_rb.py` / `test_ensemble_wr.py` (likely)
- Modify: `tests/test_scripts/test_tune_lightgbm.py` (likely)
- Modify: any other site discovered

- [ ] **Step 1: Defensive grep for `opp_allowed_rb_fppg_l4` and `opp_allowed_wr_fppg_l4`**

Run: `grep -rn "opp_allowed_rb_fppg_l4" tests/ && echo '---' && grep -rn "opp_allowed_wr_fppg_l4" tests/`
Expected: shows every test file that builds a synthetic RB or WR features row. Compare with PR #26's grep for `opp_allowed_wr_fppg_l4` (3 cluster-A sites: `test_cache.py`, lightgbm/ensemble fixtures, tune_lightgbm test). RB is likely to have an analogous set.

Save the file list — you'll edit each one.

- [ ] **Step 2: For each grep hit, add the 4 weather col defaults to the synthetic minimal-row dict**

For every minimal-features-row dict that ends with something like:
```python
{
    # ... existing RB or WR feature columns ...
    "opp_allowed_rb_fppg_l4": 12.5,  # or "opp_allowed_wr_fppg_l4": 14.2,
}
```

extend to:
```python
{
    # ... existing feature columns ...
    "opp_allowed_rb_fppg_l4": 12.5,
    # ... existing PBP cols (RB) or trajectory cols (WR) — already present from prior PRs ...
    "wind_speed_mph": 8.0,
    "is_high_wind": 0.0,
    "temperature_f": 60.0,
    "is_grass_surface": 0.0,
}
```

(Use sane finite values: `wind_speed_mph` 8.0 (typical NFL gameday), `is_high_wind` 0.0 (under threshold), `temperature_f` 60.0 (typical), `is_grass_surface` 0.0 (most NFL stadiums are turf).)

If the grep finds a helper like `_minimal_rb_features_row()` or `_minimal_wr_features_row()` in `test_cache.py`, modify it once at the helper level — better than per-call-site edits.

For lightgbm / ensemble synthetic random fixtures (PR #26 commit `33eea57` shape — 7 sites for WR), the pattern is typically a `pd.DataFrame({col: rng.normal(...) for col in feature_cols})` builder; add the 4 weather cols with `rng.uniform(0, 30)` for `wind_speed_mph`, `rng.choice([0, 1])` for `is_high_wind` and `is_grass_surface`, and `rng.uniform(20, 80)` for `temperature_f` (or sane finite constants — match the existing fixture's style).

For `tests/test_scripts/test_tune_lightgbm.py`, the pattern (PR #26 commit `807f046`) typically declares a `_RB_FEAT_COLUMNS` / `_WR_FEAT_COLUMNS` tuple at module top — extend each tuple with the 4 weather col names.

- [ ] **Step 3: Re-run baseline + lightgbm + ensemble RB/WR test files + tune_lightgbm**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_baseline_rb.py tests/test_models/test_baseline_wr.py tests/test_models/test_lightgbm_rb.py tests/test_models/test_lightgbm_wr.py tests/test_models/test_ensemble_rb.py tests/test_models/test_ensemble_wr.py tests/test_scripts/test_tune_lightgbm.py tests/test_features/test_cache.py -v 2>&1 | tail -20`
Expected: all PASS — the cluster-A fixture rows now include the 4 weather cols.

- [ ] **Step 4: Confirm no leftover sites by re-grepping for one of the 4 cols across tests**

Run: `grep -rn "wind_speed_mph" tests/ | wc -l`
Expected: at least N hits (where N = number of cluster-A sites discovered + 8 occurrences from Tasks 5 + 6). If suspiciously low (e.g., 0 or 1), some site was missed.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test(weather): special-case 4 weather cols on cluster-A synthetic RB/WR fixtures

Defensive grep on opp_allowed_rb_fppg_l4 + opp_allowed_wr_fppg_l4 found
N cluster-A sites constructing minimal feature rows; each gets
wind_speed_mph / is_high_wind / temperature_f / is_grass_surface at
sane finite defaults (8.0 / 0.0 / 60.0 / 0.0). Same pattern as PR #26's
WR cluster-A edits at commits 1f1f415 / 33eea57 / 807f046.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: End-of-Phase-4 verification — full suite + all gates

**Files:** none (verification only)

- [ ] **Step 1: Full pytest suite**

Run: `.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 2: mypy strict**

Run: `.venv/Scripts/python.exe -m mypy src tests 2>&1 | tail -3`
Expected: `Success: no issues found in N source files`.

- [ ] **Step 3: ruff check**

Run: `.venv/Scripts/python.exe -m ruff check src tests scripts 2>&1 | tail -3`
Expected: `All checks passed!`.

- [ ] **Step 4: ruff format check**

Run: `.venv/Scripts/python.exe -m ruff format --check src tests scripts 2>&1 | tail -3`
Expected: zero formatting drift.

- [ ] **Step 5: Schema-touching integration smoke (CLAUDE.md mechanical override #4)**

Run: `.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas" 2>&1 | tail -10`
Expected: all PASS — confirms the schema delta hasn't introduced dtype regressions in the ingest/store seam.

- [ ] **Step 6: Cross-position parametrized smoke test (catches signature regressions)**

Run: `.venv/Scripts/python.exe -m pytest -k "smoke" -v 2>&1 | tail -10`
Expected: all PASS.

If any step fails, fix and re-run before proceeding to Phase 5.

---

## Phase 5 — Real-data execution + reports

Goal: regenerate the RB + WR feature caches against real data; run the walk-forward backtest for `(RB, WR) × (baseline, lgb-nb)` (4 cells); run the dual-run adoption gate; write the summary report. The gate's per-position verdicts on `(LightGBMNbModel, POS)` AND `(BaselineModel, POS)` determine which Phase 6 branch fires for each position.

### Task 9: Verify `data/raw/schedules/` partition coverage

**Files:** none (data verification only)

- [ ] **Step 1: List partition years**

Run: `ls data/raw/schedules/ 2>/dev/null | head -10`
Expected: `season=2018/`, ..., `season=2024/`.

- [ ] **Step 2: If any season is missing, refresh schedules ingest**

If step 1 shows incomplete coverage:

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; from projections.ingest.refresh import refresh; refresh(data_root=Path('data'), seasons=range(2018, 2025), only=['schedules'])"`
Expected: writes 7 partition files; no error.

- [ ] **Step 3: Confirm**

Run: `ls data/raw/schedules/ | wc -l`
Expected: ≥7 (2018-2024 inclusive).

---

### Task 10: Refresh RB + WR feature caches

**Files:** none (data regeneration only — output gitignored)

- [ ] **Step 1: Refresh both caches**

Run: `.venv/Scripts/python.exe scripts/refresh_features.py rb wr --seasons 2018-2024 2>&1 | tail -20`
Expected: each (season, week) partition validates against the extended `RbFeaturesSchema` / `WrFeaturesSchema`; no SchemaError raised.

- [ ] **Step 2: Spot-check an RB partition**

Run: `.venv/Scripts/python.exe -c "
import pandas as pd
from projections.store import read_partition
from pathlib import Path
df = read_partition(Path('data/features'), 'rb', season=2024, week=5)
WX = ['wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface']
for c in WX:
    assert c in df.columns, f'{c} missing'
print('RB rows:', len(df))
print(df[WX].describe())
"`
Expected: 4 weather cols present; mostly non-NaN values; describe() shows realistic mph / temp ranges.

- [ ] **Step 3: Spot-check a WR partition**

Run: `.venv/Scripts/python.exe -c "
import pandas as pd
from projections.store import read_partition
from pathlib import Path
df = read_partition(Path('data/features'), 'wr', season=2024, week=5)
WX = ['wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface']
for c in WX:
    assert c in df.columns, f'{c} missing'
print('WR rows:', len(df))
print(df[WX].describe())
"`
Expected: same shape as RB.

---

### Task 11: Coverage cross-check vs PR #28's audit

**Files:** none (verification only)

- [ ] **Step 1: Per-(position, season) coverage on the 2021-2024 eval window**

Run: `.venv/Scripts/python.exe -c "
import pandas as pd
from pathlib import Path
from projections.store import read_partition
WX = ['wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface']
for pos in ('rb', 'wr'):
    print(f'=== {pos.upper()} ===')
    for s in range(2021, 2025):
        frames = []
        for w in range(1, 19):
            try:
                frames.append(read_partition(Path('data/features'), pos, season=s, week=w))
            except FileNotFoundError:
                continue
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            continue
        print(f'  {s}: {len(df)} rows')
        for c in WX:
            cov = df[c].notna().mean()
            print(f'    {c}: {cov:.1%} coverage')
"`
Expected:
- For each `(position, season)`, coverage on `wind_speed_mph` / `temperature_f` should be ≥92% (per PR #28's per-(position, season) audit in `reports/feature_probe_weather_override_audit.md`); coverage on `is_high_wind` should match `wind_speed_mph` (NaN-preserving threshold); coverage on `is_grass_surface` should be 100% (computed via `.fillna(False)` in `compute_weather_features`).
- Pooled outdoor-NaN rate ~8% (PR #28 measurement) is concentrated in 2018-2019 — those years are not in the eval window.

If coverage is materially different (e.g., `wind_speed_mph` <80% for 2021-2024), the builder wiring is wrong — investigate before running the gate. **Save these stats** for inclusion in the summary report.

---

### Task 12: Capture pre-PR baseline run sha

**Files:** none (git inspection only)

- [ ] **Step 1: Identify the most recent backtest run on main pre-this-branch**

Run: `git log --all --oneline -- tests/backtest/model_metrics.json | head -5`
Expected: shows the most recent commits that touched the backtest snapshot. The baseline run sha is the most recent one on `main` (pre-`feat/weather-features-rb-wr`).

- [ ] **Step 2: Note the pre-branch baseline sha for use in Task 14**

Capture the commit sha (e.g., `39be213` if main HEAD is the PR #28 merge) — this is the `--baseline-run` argument for `adoption_gate.py`.

---

### Task 13: Run the walk-forward backtest with the new weather features

**Files:**
- Modify: `tests/backtest/model_metrics.json` (snapshot update — 4 cells: RB+WR × baseline+lgb-nb)

- [ ] **Step 1: Run backtest for RB + WR × baseline + lgb-nb**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position RB,WR --model baseline,lightgbm-nb --update-snapshot 2>&1 | tail -20`
Expected: walk-forward over 2021-2024 holdout years × 2 positions × 2 model classes; updates `tests/backtest/model_metrics.json` with the 4 new cells. Runtime: ~30-60 minutes (RB+WR × 2 classes × 4 holdout years × walk-forward; lgb-nb's NB-2 fitting is the heavier path). **Skips ensemble** (and lightgbm, lightgbm-tuned) per spec §1.3.4 — those rows in `model_metrics.json` stay at the pre-PR values.

- [ ] **Step 2: Diff the snapshot to confirm only RB + WR × baseline + lgb-nb rows changed**

Run: `git diff tests/backtest/model_metrics.json | head -80`
Expected: changes confined to rows with `position in ("RB", "WR")` AND `model_class in ("baseline", "lightgbm-nb")`. QB / TE rows unchanged. RB / WR rows for `model_class in ("lightgbm", "lightgbm-tuned", "ensemble")` unchanged (those classes are explicitly skipped).

- [ ] **Step 3: Confirm no surprise ensemble weight regen**

Run: `git status data/ensemble_weights/ 2>&1 | head -10`
Expected: no `ensemble_rb_*.json` or `ensemble_wr_*.json` files modified (we didn't run ensemble in Step 1; the existing pre-PR ensemble weight files for RB / WR remain untouched). If something did change here, we ran ensemble inadvertently — investigate before committing.

- [ ] **Step 4: Commit the snapshot update**

```bash
git add tests/backtest/model_metrics.json
git commit -m "$(cat <<'EOF'
snapshot: backtest update for weather features (RB+WR × baseline+lgb-nb)

Walk-forward over 2021-2024, 4 cells: (RB, WR) × (baseline, lgb-nb).
Other RB/WR model classes (lightgbm, lightgbm-tuned, ensemble) skipped
per spec §1.3.4 — informational and back-fillable. QB/TE rows unchanged.
No ensemble weight regen.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Run the dual-run adoption gate

**Files:**
- Add: `reports/adoption_gate_weather_features_rb_wr.md`
- Add: `reports/adoption_gate_weather_features_rb_wr.csv`

- [ ] **Step 1: Capture the candidate (current branch) sha**

Run: `git rev-parse HEAD`
Expected: 40-char sha.

- [ ] **Step 2: Run the adoption gate**

Run:
```bash
.venv/Scripts/python.exe scripts/adoption_gate.py \
  --position RB,WR \
  --baseline-run <baseline-sha-from-Task-12> \
  --candidate-run <candidate-sha-from-step-1> \
  --output-md reports/adoption_gate_weather_features_rb_wr.md \
  --output-csv reports/adoption_gate_weather_features_rb_wr.csv
```
Expected: produces per-(model_class, position) verdicts for the 4 cells. (If the gate exposes a `--coverage-threshold` flag and rejects the default of 0.95, append `--coverage-threshold 0.90` to match PR #28's probe; per Task 11 step 1 the eval-window coverage should be ≥92% so the default may pass.)

- [ ] **Step 3: Read the gate's `(LightGBMNbModel, RB)` verdict — binding cell #1**

Run: `grep -A 2 "RB.*lightgbm-nb\|lightgbm-nb.*RB" reports/adoption_gate_weather_features_rb_wr.md | head -10`
Expected: a single verdict line — `ADOPT`, `MARGINAL`, or `DO_NOT_ADOPT` — with the composite RMSE delta + 95% CI. **This is the binding decision for RB** (spec §1.3.5).

- [ ] **Step 4: Read the gate's `(LightGBMNbModel, WR)` verdict — binding cell #2**

Run: `grep -A 2 "WR.*lightgbm-nb\|lightgbm-nb.*WR" reports/adoption_gate_weather_features_rb_wr.md | head -10`
Expected: same shape. **Binding decision for WR.**

- [ ] **Step 5: Read the gate's `(BaselineModel, RB)` and `(BaselineModel, WR)` verdicts — needed for modified-shape branch decisions**

Run: `grep -A 2 "RB.*baseline\|WR.*baseline" reports/adoption_gate_weather_features_rb_wr.md | head -20`
Expected: two verdict lines. If either RMSE delta CI is **strictly above zero**, that position's REGRESSION trigger fires the modified-shape branch in Phase 6. Otherwise (DO_NOT_ADOPT or ADOPT), that position ships as designed.

- [ ] **Step 6: Commit the gate reports**

```bash
git add reports/adoption_gate_weather_features_rb_wr.md reports/adoption_gate_weather_features_rb_wr.csv
git commit -m "$(cat <<'EOF'
report(weather-rb-wr): adoption gate output

Dual-run gate for (RB, WR) × (baseline, lgb-nb) on 2021-2024 holdout.
Binding cells: (LightGBMNbModel, RB) and (LightGBMNbModel, WR) per
spec §1.3.5 — second integration to bind on a non-default class,
first to bundle two positions in one PR. (BaselineModel, *) cells
are informational + drive the modified-shape contingency per position.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Write the summary report

**Files:**
- Add: `reports/weather_features_rb_wr_summary.md`

- [ ] **Step 1: Draft the summary report**

Create `reports/weather_features_rb_wr_summary.md` with the following structure (fill in numbers from the gate output + Task 11 coverage stats):

```markdown
# Weather Features RB+WR Integration — Summary Report

**Status:** Per-position outcomes (see §1.3.5 matrix below):
- RB: [ADOPT (ship-as-designed) / ADOPT (ship-modified-shape) / MARGINAL / DO_NOT_ADOPT]
- WR: [same]
**Branch:** `feat/weather-features-rb-wr`
**Spec:** `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`
**Plan:** `docs/superpowers/plans/2026-05-08-weather-features-rb-wr.md`
**Date:** 2026-05-XX

## Decision

[Per-position single-sentence outcomes — ship-as-designed / ship-modified-shape / revert, with the binding magnitude + CI on (lgb-nb, POS) for each.]

## Binding-cell shift — second non-default-class integration; first bundled-position integration

PR #21 (RB PBP) and PR #26 (WR trajectory) bound on `(BaselineModel, position)` because baseline was each position's `default_model_class`. PR #27 (TE trajectory) was the first integration to bind on a non-default class — `(LightGBMNbModel, TE)`. This PR is the **second** non-default-class integration (binds on `(lgb-nb, RB)` and `(lgb-nb, WR)`) AND the **first** to bundle two positions into a single PR. Both binding cells share the same model class + mode (lgb-nb augment) per PR #28's probe; PR #20→#21 and PR #25→#26/#27 had only one binding cell each. RB and WR production routings stay on `baseline`; flipping `_PositionDispatch[POS].default_model_class` is a deferred per-position cross-class follow-up.

## Probe-vs-gate calibration per position

| Position | Source | Composite RMSE Δ on (lgb-nb, POS) augment | 95% CI |
|---|---|---:|---|
| RB | PR #28 probe (predicted) | -0.0081 | [-0.0163, -0.0005] |
| RB | This PR's gate (measured) | [filled] | [filled] |
| WR | PR #28 probe (predicted) | -0.0110 | [-0.0172, -0.0049] |
| WR | This PR's gate (measured) | [filled] | [filled] |

[2-3 sentences per position on whether probe and gate agree. Track record: PR #20→#21 matched to 4 decimals on RB (-0.0124 → -0.0124); PR #25→#26 within ~0.004 fpts on WR; PR #25→#27 within ~0.0017 fpts on TE. For weather at -0.0081 / -0.0110 magnitudes, ~10% calibration error is ~0.001 fpts — within the per-cell noise floor.]

## Per-(model_class, position) verdicts (4 cells)

| Position | Model class | RMSE Δ | 95% CI | Spearman Δ | Verdict |
|---|---|---:|---|---:|:---:|
| RB | baseline | [filled] | [filled] | [filled] | [filled] (informational; drives modified-shape contingency) |
| RB | **lightgbm-nb** | [filled] | [filled] | [filled] | **[filled] (binding)** |
| WR | baseline | [filled] | [filled] | [filled] | [filled] (informational; drives modified-shape contingency) |
| WR | **lightgbm-nb** | [filled] | [filled] | [filled] | **[filled] (binding)** |

**3 informational classes** (lightgbm, lightgbm-tuned, ensemble) **skipped** per spec §1.3.4 + PR #27 precedent. Wall-time risk + TODO #29's lightgbm-tuned pruning candidate framing made the additional ~6 cells low-value. Back-fillable by a follow-up `--model lightgbm,lightgbm-tuned,ensemble` backtest if any cross-class routing-flip discussion needs them.

## Per-position 2-position §1.3.5 contingency matrix outcome

| Position | (lgb-nb, POS) | (baseline, POS) | Branch fired | Action taken |
|---|:---:|:---:|:---:|---|
| RB | [filled] | [filled] | [ship-as-designed / ship-modified-shape / revert] | [filled] |
| WR | [filled] | [filled] | [ship-as-designed / ship-modified-shape / revert] | [filled] |

[Description: did both positions fire the same branch? If divergent, document why.]

## Coverage statistics (2021-2024 eval window)

| Position | Column | Coverage | PR #28 probe coverage | Match (within ~1pp)? |
|---|---|---:|---:|:---:|
| RB | wind_speed_mph | [from Task 11] | [PR #28 audit] | [yes/no] |
| RB | is_high_wind | [from Task 11] | [PR #28 audit] | [yes/no] |
| RB | temperature_f | [from Task 11] | [PR #28 audit] | [yes/no] |
| RB | is_grass_surface | [from Task 11] | 100% | [yes/no] |
| WR | (same 4 rows) | ... | ... | ... |

[If coverage matches the probe within ~1pp, the builder wiring is correct. If divergent, document the cause.]

## Threshold note

[If the gate accepted the default coverage threshold (0.95): "No threshold relaxation needed. Eval-window per-(position, season) coverage uniformly ≥92% per Task 11, and the gate's row-key matching is independent of the probe's pooled coverage check anyway."]

[If the gate rejected the default and required `--coverage-threshold 0.90`: "Same precedent as PR #23 / PR #28."]

## Cross-class deferred follow-up — per position

**RB** production routes to `baseline` per Plan 8 (2026-04-29). With weather cols now in `RbFeaturesSchema`, a separate cross-class re-eval (`scripts/adoption_gate.py --position RB` comparing `lightgbm-nb` candidate to `baseline` baseline at the position level) could justify flipping `_PositionDispatch[RB].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next RB-related work.

**WR** production routes to `baseline`. Same shape as RB above; the `_PositionDispatch[WR].default_model_class` flip question is parallel.

## What this closes

[If RB OR WR ADOPT in any form:] TODO #25's broad-cut weather family at the in-builder unit, on the [RB / WR / both] cells. QB and TE remain DO_NOT_ADOPT at this unit (PR #28 probe verdict, not re-tested in this PR). Refined-unit candidates remain open under the same TODO: cold-weather threshold, multi-class surface encoding, kickoff hour, surface × position interactions, per-team weather acclimation, precipitation, wind direction. None queued.

[If both DO_NOT_ADOPT:] The probe-vs-gate divergence on both RB and WR binding cells is documented; the broad-cut weather family at the in-builder unit is closed across all 4 positions. Refined-unit candidates remain open.

## Next track

[Either: TODO #25 refined-unit weather candidates (cold-weather threshold first), or TODO #23 target decomposition (volume × efficiency), or TODO #29 lightgbm-tuned pruning, or cross-class production-routing flips for RB / WR (deferred above).]
```

- [ ] **Step 2: Fill in numbers from `reports/adoption_gate_weather_features_rb_wr.md`, Task 11 step 1 coverage output, and Task 14 step 5 baseline-cell verdicts**

Open the gate report; copy per-cell numbers into the summary tables.

- [ ] **Step 3: Commit**

```bash
git add reports/weather_features_rb_wr_summary.md
git commit -m "$(cat <<'EOF'
report(weather-rb-wr): family summary — RB [outcome], WR [outcome]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — Conditional code adjustments + documentation

Goal: append the decision-log entry to `project_management.md`, update `TODO.md` #25, and conditionally adjust code based on each position's gate verdict from Task 14. **Per-position branches; each position decided independently per spec §1.3.5.** Possible Phase 6 paths:

- **Both RB and WR ship-as-designed** (default branch — most likely outcome from probe evidence): no code adjustments.
- **One position ships modified-shape** (low-probability `(baseline, POS)` REGRESSION): revert that position's `_<POS>_FEATURE_COLUMNS` extension only.
- **One position full-reverts** (lower-probability `(lgb-nb, POS)` MARGINAL/DO_NOT_ADOPT): revert that position's schema + builder + `_<POS>_FEATURE_COLUMNS`.

The 9 possible (3 × 3) combined outcomes decompose into per-position actions; Tasks 16a / 16b / 16c apply per position.

### Task 16a — IF `(lgb-nb, POS)` ADOPT AND `(baseline, POS)` not REGRESSION (ship-as-designed for that position)

**Applies independently to:** RB and/or WR.

**Files for this position:**
- (no code changes — schema + builder + `_<POS>_FEATURE_COLUMNS` extension all stay as-shipped from Phase 1-3)

If both positions hit this branch, only Task 16d (PM + TODO update) runs. No 16b / 16c branches fire.

---

### Task 16b — IF `(lgb-nb, POS)` ADOPT AND `(baseline, POS)` REGRESSION (ship-modified-shape for that position)

**Files:**
- Modify: `src/projections/models/baseline.py` (revert that position's `_<POS>_FEATURE_COLUMNS` extension only — keep schema + builder)
- Modify: `tests/backtest/model_metrics.json` (re-run that position's `--model baseline` cell only)

For each affected position (RB and/or WR — possibly both):

- [ ] **Step 1: Revert that position's `_<POS>_FEATURE_COLUMNS` extension only**

Edit `src/projections/models/baseline.py` and remove the 4 weather col entries (and the comment block) from `_RB_FEATURE_COLUMNS` (or `_WR_FEATURE_COLUMNS`), restoring it to the pre-Task-4 state. **Do not** revert the schema edit (Task 1) or the builder edit (Task 2 / Task 3) — the lightgbm family auto-picks-up the cols via dynamic schema derivation, and we want them to keep doing that.

- [ ] **Step 2: Re-run baseline-only POS backtest to refresh `model_metrics.json`**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position <POS> --model baseline --update-snapshot 2>&1 | tail -10`
(Where `<POS>` is `RB` or `WR` — or `RB,WR` if both fired this branch.)
Expected: only that position's baseline rows in `tests/backtest/model_metrics.json` change. lgb-nb rows stay as-is from Task 13 (they correctly reflect with-weather-cols since the schema still has them).

- [ ] **Step 3: Verify the lightgbm family still sees the weather cols by re-checking dynamic feature derivation**

Run: `.venv/Scripts/python.exe -c "
from projections.models.lightgbm import _filter_features
from projections.schemas import RbFeaturesSchema, WrFeaturesSchema
WX = ('wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface')
for name, schema in [('RB', RbFeaturesSchema), ('WR', WrFeaturesSchema)]:
    schema_cols = list(schema.to_schema().columns.keys())
    features = _filter_features(tuple(schema_cols))
    for c in WX:
        assert c in features, f'{c} missing from lightgbm derived features for {name}'
    print(name, ': lightgbm family still sees the 4 weather cols (good)')
"`
Expected: confirms the lightgbm family is unaffected by the baseline-side revert.

- [ ] **Step 4: Append modified-shape addendum to summary report**

Edit `reports/weather_features_rb_wr_summary.md`. Add a section at the bottom (per affected position):

```markdown
## Modified-shape ship branch fired for [POS] (spec §1.3.5)

`(BaselineModel, [POS])` returned REGRESSION ([filled] fpts, CI [filled] — strictly above zero). Per spec §1.3.5, the modified-shape ship path fired for [POS]:

- **Schema edit** (4 cols added to `[Pos]FeaturesSchema`) — kept; lightgbm family auto-picks-up via dynamic schema derivation.
- **Builder edit** (`build_[pos]_features` wires `attach_weather_features`) — kept; cols computed at refresh-features time and persisted in cache.
- **`_[POS]_FEATURE_COLUMNS` extension in `baseline.py`** — REVERTED. Baseline [POS] production no longer sees the 4 cols.
- **`tests/backtest/model_metrics.json` baseline [POS] rows** — re-run after the revert. lgb-nb [POS] rows kept from with-weather-cols backtest.

Effect: lightgbm-family [POS] classes consume weather features in production (via `production_model_for(Position.[POS])` if routing ever flips); baseline [POS] production output unchanged from pre-PR.

[If this is the first time the modified-shape branch has fired in the project, note that. PR #27 was the first non-default-class integration but didn't fire modified-shape because TE baseline was DO_NOT_ADOPT, not REGRESSION.]
```

---

### Task 16c — IF `(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT (revert that position)

**Files:**
- Modify: `src/projections/schemas.py` (revert that position's 4 weather cols)
- Modify: `src/projections/features/<rb or wr>.py` (revert that position's builder integration)
- Modify: `src/projections/models/baseline.py` (revert that position's `_<POS>_FEATURE_COLUMNS` extension)
- Modify: `tests/test_features/test_<rb or wr>.py` (revert that position's 4 weather tests + happy-path extension)
- Modify: cluster-A leftover sites for that position (revert weather col additions; or **keep** them — cheap to leave per spec §1.1)
- Modify: `tests/backtest/model_metrics.json` (re-run that position × baseline + lgb-nb post-revert)

For each affected position:

- [ ] **Step 1: Revert that position's changes**

Use `git revert <commit-sha>` for each commit that touched the position's files: `*FeaturesSchema` row of Task 1, the position's builder integration (Task 2 or Task 3), the position's 4 join-side tests (Task 5 or Task 6), the position's row of `_<POS>_FEATURE_COLUMNS` (Task 4), and the position-specific cluster-A leftovers (Task 7). Resolve conflicts from later changes if any.

(Alternative: if you want to keep cluster-A test fixture defaults — the spec allows it — leave the Task 7 commits in place; only revert the schema, builder, and baseline-feature-list edits.)

- [ ] **Step 2: Re-run that position's backtest snapshot post-revert**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position <POS> --model baseline,lightgbm-nb --update-snapshot 2>&1 | tail -10`
Expected: that position's rows in `model_metrics.json` revert to pre-PR values.

- [ ] **Step 3: Re-run full test suite + lint to confirm clean revert**

Run: `.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -10 && .venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests scripts && .venv/Scripts/python.exe -m ruff format --check src tests scripts`
Expected: all PASS, zero violations.

---

### Task 16d — Decision-log + TODO updates + push + PR (always runs)

**Files:**
- Modify: `project_management.md` (top-of-file decision-log entry)
- Modify: `TODO.md` #25 (per-position outcomes)

- [ ] **Step 1: Append decision-log entry to `project_management.md`**

Format matches PR #28's entry. Insert immediately after the line `---` at line 5 (before the existing top-most entry):

```markdown
## Weather Features RB+WR Integration — verdicts: RB [filled], WR [filled] (2026-05-XX, on branch `feat/weather-features-rb-wr`)

**Status:** Production integration of the 4 weather features into `RbFeaturesSchema` + `WrFeaturesSchema` + `build_rb_features` + `build_wr_features` per `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`. Wired `attach_weather_features` (already public from PR #28) into both builders via the existing `schedules` kwarg. Updated `baseline.py:_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS` (same spec gap class as PR #21 / PR #26 / PR #27). No new ingest, no caller-script changes, no fixture extension (weather is per-game, not trailing-N).

**Per-position dual-run gate verdicts:**

| Position | (lgb-nb, POS) | (baseline, POS) | §1.3.5 branch | Action |
|---|:---:|:---:|:---:|---|
| **RB** | [filled] (binding) | [filled] (informational) | [ship-as-designed / ship-modified-shape / revert] | [filled] |
| **WR** | [filled] (binding) | [filled] (informational) | [ship-as-designed / ship-modified-shape / revert] | [filled] |

[For the binding cells: Probe predicted (-0.0081 RB / -0.0110 WR) fpts; gate measured (per-position fills). Calibration commentary.]

**Second integration to bind on a non-default model class (after PR #27 TE trajectory)** and **first integration to bundle two positions into a single PR** with per-position contingency matrix (each position decided independently from the other). Production routings unchanged: RB stays on `baseline`, WR stays on `baseline`.

**3 informational model classes skipped** (lightgbm, lightgbm-tuned, ensemble) per spec §1.3.4 + PR #27 precedent. Back-fillable by a follow-up backtest if cross-class routing-flip discussion needs them.

[If any modified-shape branch fired:]
**Modified-shape branch fired for [POS]:** baseline [POS] returned REGRESSION; `_[POS]_FEATURE_COLUMNS` extension reverted to keep baseline [POS] production unchanged. Schema + builder edits kept so lightgbm family (which derives features dynamically) still consumes the cols. [Note "first time the modified-shape branch fired in the project" if applicable.]

**Coverage statistics (2021-2024 eval window, per Task 11):**
- RB: wind_speed_mph [filled]%, is_high_wind [filled]%, temperature_f [filled]%, is_grass_surface ~100%
- WR: same shape

**What this closes:** TODO #25's broad-cut weather family at the in-builder unit, on the [shipped positions]. QB and TE remain DO_NOT_ADOPT at this unit (PR #28 probe). Refined-unit candidates (cold-weather threshold, multi-class surface, kickoff hour, surface × position, per-team acclimation, precipitation, wind direction) remain open under TODO #25; none queued.

**Cross-class deferred follow-ups:** Per-position routing-flip questions for RB and WR queued under TODO #25 (parallel to PR #27's TE follow-up).

See `reports/weather_features_rb_wr_summary.md` for the full decision log + per-mode table + probe-vs-gate calibration + per-position §1.3.5 outcome matrix.

---
```

- [ ] **Step 2: Update `TODO.md` #25**

Open `TODO.md`, find the section header `### 25. Weather features in per-position builders`. Append at the end of that section:

```markdown
**Update 2026-05-XX (RB+WR weather features integration, branch `feat/weather-features-rb-wr`):** Production integration of the 4 weather features into `RbFeaturesSchema` + `WrFeaturesSchema` + `build_rb_features` + `build_wr_features` per `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`. Per-position dual-run gate verdicts: **(lgb-nb, RB): [filled]** (composite RMSE delta [filled] fpts, CI [filled]); **(lgb-nb, WR): [filled]** (composite RMSE delta [filled] fpts, CI [filled]). Probe predicted -0.0081 RB / -0.0110 WR; gate matched [calibration commentary per position]. **Second integration to bind on a non-default model class** (after PR #27 TE trajectory) and **first to bundle two positions in one PR** with per-position contingency matrix.

[If modified-shape branch fired for any position, document.]

**Cross-class production routing follow-ups:** RB and WR each route to `baseline` per Plan 8. With weather cols now in `RbFeaturesSchema` / `WrFeaturesSchema`, separate cross-class re-evals could justify flipping `_PositionDispatch[{RB|WR}].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next RB- or WR-related work. Same shape as the PR #27 TE follow-up.

**Refined-unit candidates remain unexplored under this TODO:** `is_cold_weather` (`temp < 32`, sibling shape to `is_high_wind`), multi-class surface encoding (one bool per surface code), kickoff hour / time-of-day, surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). Recommended priority order if a refined-unit plan is scoped: cold-weather threshold → multi-class surface → kickoff hour. None queued.

**This closes** the broad-cut weather family at the in-builder unit on the [RB / WR / both] ADOPT cell(s) from PR #28. QB and TE remain DO_NOT_ADOPT at the broad-cut unit. See `reports/weather_features_rb_wr_summary.md`.
```

- [ ] **Step 3: Commit + push + open PR**

```bash
git add project_management.md TODO.md
git commit -m "$(cat <<'EOF'
docs(pm): record weather features RB+WR integration verdicts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push -u origin feat/weather-features-rb-wr
gh pr create --title "feat(weather): RB+WR features integration — RB [verdict], WR [verdict]" --body "$(cat <<'EOF'
## Summary

Promotes PR #28's weather family probe RB and WR cells into production. Adds 4 nullable-float weather columns to `RbFeaturesSchema` + `WrFeaturesSchema`; wires `attach_weather_features` into both builders. **Second integration to bind on a non-default model class** (after PR #27 TE trajectory) and **first integration to bundle two positions in a single PR** with per-position contingency matrix.

- Probe predicted: RB -0.0081 fpts, WR -0.0110 fpts (lgb-nb augment composite).
- Gate measured: RB [filled], WR [filled].
- Per-position §1.3.5 branches: RB [ship-as-designed / modified-shape / revert]; WR [same].
- Production routings unchanged (both stay on `baseline`); cross-class flip questions queued.
- 3 informational classes skipped (lightgbm, lightgbm-tuned, ensemble); back-fillable.

## Test plan

- [x] Full pytest suite passes
- [x] mypy strict zero violations
- [x] ruff check + format clean
- [x] RB + WR feature caches regenerate against extended schemas
- [x] Backtest snapshot diff confined to RB+WR × baseline+lgb-nb cells
- [x] Adoption gate runs on 4 cells; reports committed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist (run before handoff)

Quick scan after Phase 6 completes:

- [ ] All 4 weather columns appear in `RbFeaturesSchema` and `WrFeaturesSchema` (or are reverted on a per-position revert path).
- [ ] `build_rb_features` and `build_wr_features` each call `attach_weather_features` (or are reverted on revert path).
- [ ] `baseline.py:_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS` include the 4 new names ONLY for positions where ship-as-designed branch fired; reverted on ship-modified-shape or revert.
- [ ] Adoption gate report exists at `reports/adoption_gate_weather_features_rb_wr.{md,csv}`.
- [ ] Summary report exists at `reports/weather_features_rb_wr_summary.md` with per-position §1.3.5 outcome matrix + binding-cell-shift rationale + cross-class deferred-follow-up notes (per position) + (if applicable) modified-shape addendum.
- [ ] PM + TODO #25 reflect the per-position verdicts.
- [ ] Coverage check from Task 11 was within ~1pp of PR #28's audit per (position, season) — if divergent, documented in the summary.
- [ ] PR title + body explicitly note both the binding-cell shift from PR #21 / PR #26 pattern AND the bundled-two-position precedent setting.
- [ ] No `data/ensemble_weights/ensemble_{rb,wr}_*.json` files modified (we explicitly skipped ensemble per §1.3.4).
- [ ] No `tests/conftest.py` extension (weather is per-game; no trailing-N history requirement).
- [ ] No caller-script changes (schedules already plumbed).

If all green, the spec's success criteria (§1.3) are satisfied for both positions.
