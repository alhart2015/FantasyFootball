# QB + WR Vegas Team-Context Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote four Vegas team-context features (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) from probe-override status to first-class members of `QbFeaturesSchema` and `WrFeaturesSchema`; wire into `LightGBMNbModel`'s feature list for QB and WR only as a schema-swap; validate via three dual-run adoption gates.

**Architecture:** Schema additions land in `QbFeaturesSchema` + `WrFeaturesSchema`. Builders call `attach_vegas_team_context_features(out, schedules)` before the final `<Schema>.validate(out)` call. `LightGBMNbModel`'s QB and WR factories use explicit swap feature-lists that drop per-game `implied_team_total` + `spread` and add the four new cols. `BaselineModel` and `DecomposedBaselineModel` feature lists are not touched (hardcoded in `baseline.py`) so the Ridge children of `wr_ensemble_decomposed` retain their pre-integration behavior. Other lgb classes (`lightgbm`, `lightgbm-tuned`) derive feature lists from the schema and pick up the AUGMENT treatment automatically — defensible: both are non-production / pruning candidates and probe verdict for them was NULL.

**Tech Stack:** Python 3.x, pandera (DataFrame schemas), pandas, pytest, scikit-learn (Ridge), LightGBM, joblib.

**Spec:** `docs/superpowers/specs/2026-05-17-qb-wr-vegas-team-context-integration-design.md`.

**Branch:** `feat/qb-wr-vegas-team-context-integration` (already created; spec already committed at `dfb4845`).

---

## Phase 0: Schema + builder wire-up (4 files, 4 tasks)

### Task 1: Extend QbFeaturesSchema with 4 Vegas cols

**Files:**
- Modify: `src/projections/schemas.py:572-621` (QbFeaturesSchema class body — insert four `pa.Field` declarations after the existing `implied_team_total` / `spread` block)
- Test: `tests/test_schemas/test_qb_features_schema.py` (file may exist; if not, create — check first via `Glob tests/test_schemas/test_qb_features_schema.py`)

If `tests/test_schemas/test_qb_features_schema.py` does not exist, create it with the standard import header used by other schema tests (search `tests/test_schemas/` for a sibling file as template).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas/test_qb_features_schema.py`:

```python
import numpy as np
import pandas as pd
import pandera.errors
import pytest

from projections.schemas import QbFeaturesSchema, _PYARROW_STR


def _qb_features_minimal_row() -> dict[str, object]:
    """A minimal valid QbFeaturesSchema row. Mirrors the conftest qb_schedules
    week-5-2024 KC-vs-CHI fixture's downstream shape; values chosen to land
    inside every existing pa.Field bound."""
    return {
        "gsis_id": "00-0034857",
        "season": 2024,
        "week": 5,
        "team": "KC",
        "opponent": "CHI",
        "pass_attempts_per_game_l4": 32.0,
        "passing_yards_per_game_l4": 280.0,
        "passing_tds_per_game_l4": 2.0,
        "interceptions_per_game_l4": 0.5,
        "sacks_per_game_l4": 1.5,
        "passing_yards_per_game_std": 270.0,
        "rushing_attempts_per_game_l4": 5.0,
        "rushing_yards_per_game_l4": 20.0,
        "rushing_qb": True,
        "snap_pct_l4": 0.95,
        "depth_rank": 1,
        "aggressiveness_std": 12.5,
        "completion_percentage_above_expectation_std": 1.2,
        "avg_intended_air_yards_std": 8.5,
        "avg_time_to_throw_std": 2.8,
        "implied_team_total": 29.25,
        "spread": -7.5,
        "is_home": False,
        "roof_dome": False,
        "opp_allowed_qb_fppg_l4": 15.0,
    }


def _to_typed_df(row: dict[str, object]) -> pd.DataFrame:
    """Build a 1-row DataFrame with the dtype conventions the schema expects."""
    df = pd.DataFrame([row])
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["gsis_id"] = df["gsis_id"].astype(str)
    df["depth_rank"] = df["depth_rank"].astype(pd.Int64Dtype())
    return df


def test_qb_features_schema_accepts_vegas_team_context_cols() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    df = _to_typed_df(row)
    out = QbFeaturesSchema.validate(df)
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns


def test_qb_features_schema_accepts_nan_on_vegas_cols() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": float("nan"),
            "preseason_spread": float("nan"),
            "season_avg_implied_team_total": float("nan"),  # NaN at week 1 by design
            "season_avg_spread": float("nan"),
        }
    )
    df = _to_typed_df(row)
    QbFeaturesSchema.validate(df)  # must not raise


def test_qb_features_schema_rejects_negative_preseason_implied_total() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": -1.0,  # violates ge=0
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    df = _to_typed_df(row)
    with pytest.raises(pandera.errors.SchemaError):
        QbFeaturesSchema.validate(df)


def test_qb_features_schema_rejects_negative_season_avg_implied_total() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": -1.0,  # violates ge=0
            "season_avg_spread": -5.0,
        }
    )
    df = _to_typed_df(row)
    with pytest.raises(pandera.errors.SchemaError):
        QbFeaturesSchema.validate(df)
```

If the test file directory does not exist, create it: `mkdir -p tests/test_schemas` (PowerShell: `New-Item -ItemType Directory -Path tests/test_schemas -Force`). Add an empty `tests/test_schemas/__init__.py` if other test packages have one (check with `ls tests/`).

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schemas/test_qb_features_schema.py -v
```

Expected: FAIL — the four new cols aren't in the schema yet; `strict="filter"` will silently filter them out OR `validate()` will not raise on the negative-value tests because the columns don't exist as fields.

- [ ] **Step 3: Add cols to QbFeaturesSchema**

In `src/projections/schemas.py`, locate the QbFeaturesSchema class (line ~572) and insert this block after the existing `roof_dome` field (right before the `# Opponent strength proxy` comment block):

```python
    # Vegas team-context (TODO #33c integration). Sourced from
    # SchedulesSchema.spread_line / total_line. preseason_* broadcast from
    # the team's week-1 game; season_avg_* is the expanding mean over
    # weeks 1..N-1 (NaN at week 1 by design).
    preseason_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    preseason_spread: Series[float] = pa.Field(nullable=True)
    season_avg_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    season_avg_spread: Series[float] = pa.Field(nullable=True)
```

(`le=60` mirrors the existing `implied_team_total` ceiling.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_schemas/test_qb_features_schema.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
# (no need to prepend .venv/Scripts on Windows if your shell already has it on PATH;
# user memory notes the mypy hook quirk — if commit blocks, prepend PATH per CLAUDE.md guidance)
git add src/projections/schemas.py tests/test_schemas/test_qb_features_schema.py
git commit -m "feat(33c): QbFeaturesSchema accepts 4 Vegas team-context cols"
```

---

### Task 2: Extend WrFeaturesSchema with 4 Vegas cols

**Files:**
- Modify: `src/projections/schemas.py:496-569` (WrFeaturesSchema class body — insert four `pa.Field` declarations)
- Test: `tests/test_schemas/test_wr_features_schema.py` (create if needed)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas/test_wr_features_schema.py`:

```python
import pandas as pd
import pandera.errors
import pytest

from projections.schemas import WrFeaturesSchema, _PYARROW_STR


def _wr_features_minimal_row() -> dict[str, object]:
    """A minimal valid WrFeaturesSchema row."""
    return {
        "gsis_id": "00-0034857",
        "season": 2024,
        "week": 5,
        "team": "KC",
        "opponent": "CHI",
        "targets_per_game_l4": 8.0,
        "targets_per_game_std": 7.5,
        "target_share_l4": 0.22,
        "air_yards_share_l4": 0.28,
        "receptions_per_game_l4": 5.0,
        "receiving_yards_per_game_l4": 70.0,
        "receiving_tds_per_game_l4": 0.5,
        "rushing_attempts_per_game_l4": 0.5,
        "rushing_yards_per_game_l4": 3.0,
        "designed_rusher": False,
        "snap_pct_l4": 0.85,
        "depth_rank": 1,
        "avg_separation_std": 2.8,
        "avg_intended_air_yards_std": 11.5,
        "percent_share_intended_air_yards_std": 0.28,
        "avg_yac_above_expectation_std": 0.5,
        "implied_team_total": 29.25,
        "spread": -7.5,
        "is_home": False,
        "roof_dome": False,
        "opp_allowed_wr_fppg_l4": 32.0,
        "age": 27.0,
        "is_rookie": 0.0,
        "volume_trend_l4_minus_prior_l4": 0.5,
        "snap_pct_change_l4_vs_prior_l4": 0.0,
        "wind_speed_mph": 8.0,
        "is_high_wind": 0.0,
        "temperature_f": 55.0,
        "is_grass_surface": 1.0,
    }


def _to_typed_df(row: dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["gsis_id"] = df["gsis_id"].astype(str)
    df["depth_rank"] = df["depth_rank"].astype(pd.Int64Dtype())
    return df


def test_wr_features_schema_accepts_vegas_team_context_cols() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    out = WrFeaturesSchema.validate(_to_typed_df(row))
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns


def test_wr_features_schema_accepts_nan_on_vegas_cols() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": float("nan"),
            "preseason_spread": float("nan"),
            "season_avg_implied_team_total": float("nan"),
            "season_avg_spread": float("nan"),
        }
    )
    WrFeaturesSchema.validate(_to_typed_df(row))


def test_wr_features_schema_rejects_negative_preseason_implied_total() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": -1.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        WrFeaturesSchema.validate(_to_typed_df(row))


def test_wr_features_schema_rejects_negative_season_avg_implied_total() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": -1.0,
            "season_avg_spread": -5.0,
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        WrFeaturesSchema.validate(_to_typed_df(row))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schemas/test_wr_features_schema.py -v
```

Expected: FAIL.

- [ ] **Step 3: Add cols to WrFeaturesSchema**

In `src/projections/schemas.py`, locate WrFeaturesSchema (line ~496) and insert this block after the existing `roof_dome` field (right before the `# Opponent strength` comment block):

```python
    # Vegas team-context (TODO #33c integration). Same shape as QbFeaturesSchema —
    # preseason_* broadcast from week 1; season_avg_* is expanding mean over
    # weeks 1..N-1 (NaN at week 1).
    preseason_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    preseason_spread: Series[float] = pa.Field(nullable=True)
    season_avg_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    season_avg_spread: Series[float] = pa.Field(nullable=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_schemas/test_wr_features_schema.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_wr_features_schema.py
git commit -m "feat(33c): WrFeaturesSchema accepts 4 Vegas team-context cols"
```

---

### Task 3: Wire attach_vegas_team_context_features into build_qb_features

**Files:**
- Modify: `src/projections/features/qb.py` (add import + one call at the end of the builder, before `QbFeaturesSchema.validate`)
- Test: `tests/test_features/test_qb.py` (add 1 new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features/test_qb.py`:

```python
def test_build_qb_features_emits_vegas_team_context_cols(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """build_qb_features attaches the four Vegas team-context cols and the
    output validates against the extended QbFeaturesSchema. With a 1-week
    fixture, preseason_* values equal the only-week values; season_avg_*
    is NaN (expanding mean of a single game .shift(1) is NaN)."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns

    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    # KC implied_team_total = (51 - (-7.5))/2 = 29.25; spread = -7.5.
    # preseason_* equals current week (week 5 is the only week in the fixture).
    assert mahomes["preseason_implied_team_total"] == pytest.approx(29.25, abs=1e-6)
    assert mahomes["preseason_spread"] == pytest.approx(-7.5, abs=1e-6)
    # season_avg_* is NaN with a single-game schedule (expanding mean .shift(1)).
    assert pd.isna(mahomes["season_avg_implied_team_total"])
    assert pd.isna(mahomes["season_avg_spread"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features/test_qb.py::test_build_qb_features_emits_vegas_team_context_cols -v
```

Expected: FAIL — builder doesn't produce those cols yet; `QbFeaturesSchema.validate`'s `strict="filter"` silently filters absent cols. The `col in out.columns` assertion fires.

- [ ] **Step 3: Wire into build_qb_features**

In `src/projections/features/qb.py`, add to the imports (alphabetical order, after the existing `_shared` imports):

```python
from projections.features.vegas_team_context_features import attach_vegas_team_context_features
```

At the end of `build_qb_features`, immediately before the existing `return QbFeaturesSchema.validate(out)` line, insert:

```python
    # Vegas team-context (TODO #33c). Pass the FULL `schedules` arg, not the
    # exact-week-masked `sch`, so compute_vegas_team_context_features sees
    # weeks 1..N for preseason broadcast and the expanding-mean season_avg.
    out = attach_vegas_team_context_features(out, schedules)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_features/test_qb.py::test_build_qb_features_emits_vegas_team_context_cols -v
pytest tests/test_features/test_qb.py -v
```

Expected: new test PASS; full QB test module PASS (no regressions in the existing `test_build_qb_features_implied_team_total_from_schedules` etc.).

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/qb.py tests/test_features/test_qb.py
git commit -m "feat(33c): build_qb_features emits 4 Vegas team-context cols"
```

---

### Task 4: Wire attach_vegas_team_context_features into build_wr_features

**Files:**
- Modify: `src/projections/features/wr.py` (add import + one call at the end of the builder, before `WrFeaturesSchema.validate`)
- Test: `tests/test_features/test_wr.py` (add 1 new test)

- [ ] **Step 1: Write the failing test**

Find the analogous WR fixture in `tests/test_features/conftest.py` (search for `def wr_schedules` near line 260). It is a 2-week fixture (one game per team-week). Append to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_emits_vegas_team_context_cols(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """build_wr_features attaches the four Vegas team-context cols and the
    output validates against the extended WrFeaturesSchema."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        draft_picks=wr_draft_picks,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns
    # At least one row has a non-NaN preseason value (every team in the fixture
    # has at least one schedule row, so broadcast must populate).
    assert out["preseason_implied_team_total"].notna().any()
```

If the existing `wr_*` fixture signature names differ (e.g., `wr_draft_picks` is missing), inspect the existing WR test's call signature in `tests/test_features/test_wr.py` and mirror it. Do NOT invent fixture names.

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features/test_wr.py::test_build_wr_features_emits_vegas_team_context_cols -v
```

Expected: FAIL — same reason as Task 3.

- [ ] **Step 3: Wire into build_wr_features**

In `src/projections/features/wr.py`, add to the imports:

```python
from projections.features.vegas_team_context_features import attach_vegas_team_context_features
```

At the end of `build_wr_features`, immediately after the existing `out = attach_weather_features(out, sch)` line and before `return WrFeaturesSchema.validate(out)`, insert:

```python
    # Vegas team-context (TODO #33c). Pass FULL `schedules` for expanding-mean
    # season_avg_* and week-1 preseason broadcast.
    out = attach_vegas_team_context_features(out, schedules)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_features/test_wr.py::test_build_wr_features_emits_vegas_team_context_cols -v
pytest tests/test_features/test_wr.py -v
```

Expected: new test PASS; full WR test module PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/wr.py tests/test_features/test_wr.py
git commit -m "feat(33c): build_wr_features emits 4 Vegas team-context cols"
```

---

### Phase 0 verification

- [ ] **Run the full Phase-0-touching subset:**

```bash
pytest -v tests/test_schemas tests/test_features
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all pass. If anything fails: fix in place, do not move to Phase 1.

---

## Phase 1: lgb-nb feature-list override (2 files, 4 tasks)

### Task 5: Add _swap_for helper + override tuples in lightgbm_nb.py

**Files:**
- Modify: `src/projections/models/lightgbm_nb.py` (add helper + two `Final[tuple[str, ...]]` constants near the top)
- Test: `tests/test_models/test_lightgbm_nb.py` (add 3 new tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models/test_lightgbm_nb.py`:

```python
def test_qb_lightgbm_nb_feature_columns_drop_per_game_vegas_cols() -> None:
    """qb_lightgbm_nb()'s feature_columns must NOT include implied_team_total
    or spread (schema-swap drops the per-game cols)."""
    model = qb_lightgbm_nb()
    cols = set(model._config.feature_columns)
    assert "implied_team_total" not in cols
    assert "spread" not in cols


def test_qb_lightgbm_nb_feature_columns_include_4_vegas_cols() -> None:
    """qb_lightgbm_nb()'s feature_columns must include the four preseason / season_avg cols."""
    model = qb_lightgbm_nb()
    cols = set(model._config.feature_columns)
    for c in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert c in cols, f"missing {c}"


def test_wr_lightgbm_nb_feature_columns_swap_treatment() -> None:
    """Same swap-treatment for WR."""
    model = wr_lightgbm_nb()
    cols = set(model._config.feature_columns)
    assert "implied_team_total" not in cols
    assert "spread" not in cols
    for c in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert c in cols, f"missing {c}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models/test_lightgbm_nb.py::test_qb_lightgbm_nb_feature_columns_drop_per_game_vegas_cols tests/test_models/test_lightgbm_nb.py::test_qb_lightgbm_nb_feature_columns_include_4_vegas_cols tests/test_models/test_lightgbm_nb.py::test_wr_lightgbm_nb_feature_columns_swap_treatment -v
```

Expected: FAIL on all three. Schema-derived `_filter_features(_QB_FEATURE_COLUMNS)` still contains `implied_team_total` + `spread` and does NOT yet contain the 4 new cols (Phase 0 added them to the schema, so they ARE picked up — wait, re-check: schemas were updated in Phase 0, so the schema-derived list now contains all 4 new cols AND the per-game cols. The first test (drop per-game) will fail because per-game cols are still included. The second and third tests' "include 4 Vegas cols" will *pass* because schema-derived already picks them up.

This is fine: the first test (drop) drives the implementation. After Task 5, all three pass.

- [ ] **Step 3: Add helper + override tuples in lightgbm_nb.py**

In `src/projections/models/lightgbm_nb.py`, after the existing import block from `projections.models.lightgbm` (around line 41-57), add:

```python
# TODO #33c schema-swap: drop per-game implied_team_total + spread from the
# lgb-nb feature list for QB + WR; add the four Vegas team-context cols
# (preseason_* + season_avg_*). The schemas keep all cols so other model
# classes (BaselineModel via hardcoded list; lgb / lgb-tuned via schema
# derivation) are unaffected — this override is lgb-nb-specific by design.
_VEGAS_SWAP_REPLACE: Final[frozenset[str]] = frozenset({"implied_team_total", "spread"})
_VEGAS_SWAP_ADD: Final[tuple[str, ...]] = (
    "preseason_implied_team_total",
    "preseason_spread",
    "season_avg_implied_team_total",
    "season_avg_spread",
)


def _swap_for(cols: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the Vegas-swap per-game cols and append the 4 Vegas team-context cols.

    Idempotent on the 'add' side: filters out any cols already in
    `_VEGAS_SWAP_ADD` before appending (so a future schema bump that adds
    the new cols a second time doesn't duplicate them in the lgb-nb list).
    """
    swap_add_set = set(_VEGAS_SWAP_ADD)
    kept = tuple(c for c in cols if c not in _VEGAS_SWAP_REPLACE and c not in swap_add_set)
    return kept + _VEGAS_SWAP_ADD


_QB_FEATURE_COLUMNS_NB: Final[tuple[str, ...]] = _swap_for(_filter_features(_QB_FEATURE_COLUMNS))
_WR_FEATURE_COLUMNS_NB: Final[tuple[str, ...]] = _swap_for(_filter_features(_WR_FEATURE_COLUMNS))
```

(`Final` is already imported at the top of the file — verify line 28.)

- [ ] **Step 4: Run test (still expects failures — factories not yet wired)**

```bash
pytest tests/test_models/test_lightgbm_nb.py::test_qb_lightgbm_nb_feature_columns_drop_per_game_vegas_cols -v
```

Expected: still FAIL — the new constants exist but the factories haven't been updated. Next task wires them.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm_nb.py tests/test_models/test_lightgbm_nb.py
git commit -m "feat(33c): _swap_for helper + Vegas-swap feature-list constants for lgb-nb"
```

---

### Task 6: Update qb_lightgbm_nb and wr_lightgbm_nb factories

**Files:**
- Modify: `src/projections/models/lightgbm_nb.py` (factories at lines ~376 and ~412)

- [ ] **Step 1: Reuse the failing tests from Task 5**

The three feature-list tests from Task 5 are the failing tests for this task. Confirm they currently fail:

```bash
pytest tests/test_models/test_lightgbm_nb.py::test_qb_lightgbm_nb_feature_columns_drop_per_game_vegas_cols -v
```

Expected: FAIL.

- [ ] **Step 2: Update qb_lightgbm_nb()**

In `src/projections/models/lightgbm_nb.py`, replace the existing `qb_lightgbm_nb()` factory body's `feature_columns=` argument:

Change:
```python
def qb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )
```

To:
```python
def qb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_QB_FEATURE_COLUMNS_NB,  # TODO #33c Vegas swap
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )
```

- [ ] **Step 3: Update wr_lightgbm_nb()**

Same shape: change `feature_columns=_filter_features(_WR_FEATURE_COLUMNS)` to `feature_columns=_WR_FEATURE_COLUMNS_NB,  # TODO #33c Vegas swap`.

(`te_lightgbm_nb()` and `rb_lightgbm_nb()` are NOT modified — they retain `_filter_features(_TE_FEATURE_COLUMNS)` / `_filter_features(_RB_FEATURE_COLUMNS)`.)

- [ ] **Step 4: Update `test_yards_stat_predictions_match_tuned_baseline` (broken-premise test)**

After Task 6, `wr_lightgbm_nb()` uses `_WR_FEATURE_COLUMNS_NB` (swap) while `wr_lightgbm_tuned()` continues to derive its feature list from the schema (augment). Their WR feature lists now differ → trained sub-models differ → the existing `_best_iters` equality assertion no longer holds. The mechanism the test was checking (yards-stat training inheritance from parent quantile path) is still intact — it's just no longer observable via identical best_iters for WR specifically.

Open `tests/test_models/test_lightgbm_nb.py`, find `test_yards_stat_predictions_match_tuned_baseline` (around line 186), and replace its body with:

```python
def test_yards_stat_predictions_match_tuned_baseline() -> None:
    """Yards-stat training inheritance from LightGBMTunedModel was originally
    observable as identical `best_iters` between wr_lightgbm_nb and
    wr_lightgbm_tuned on the same fixture. After TODO #33c integration,
    wr_lightgbm_nb uses the Vegas-swap feature list (drops implied_team_total
    + spread, adds 4 preseason_* / season_avg_* cols) while wr_lightgbm_tuned
    keeps the schema-derived (augment) list. The two now have DIFFERENT feature
    columns, so sub-models necessarily diverge — best_iters equality no longer
    holds for WR.

    The inheritance mechanism is preserved: LightGBMNbModel still extends
    LightGBMTunedModel and overrides only count-stat fit logic. Verified here
    by class hierarchy + by confirming the two models' yards-stat config
    blocks are identical *modulo* feature_columns."""
    nb = wr_lightgbm_nb()
    tuned = wr_lightgbm_tuned()
    # Class hierarchy: NB subclasses Tuned, so quantile-yards training path
    # is inherited unchanged.
    assert isinstance(nb, type(tuned))
    # Per-stat target_stats + non_negative_stats configurations are identical
    # (both share _WR_TARGET_STATS and _WR_NON_NEGATIVE).
    assert nb._config.target_stats == tuned._config.target_stats
    assert nb._config.non_negative_stats == tuned._config.non_negative_stats
    # Feature columns INTENTIONALLY differ post-#33c.
    assert set(nb._config.feature_columns) != set(tuned._config.feature_columns)
    swap_added = {
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    }
    swap_removed = {"implied_team_total", "spread"}
    nb_cols = set(nb._config.feature_columns)
    tuned_cols = set(tuned._config.feature_columns)
    assert swap_added.issubset(nb_cols)
    assert swap_added.issubset(tuned_cols)  # tuned auto-picks them up via schema
    assert swap_removed.isdisjoint(nb_cols)  # nb drops them
    assert swap_removed.issubset(tuned_cols)  # tuned keeps them
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_models/test_lightgbm_nb.py -v
```

Expected: all 3 new feature-list tests PASS; the rewritten `test_yards_stat_predictions_match_tuned_baseline` PASSes; all other pre-existing lgb-nb tests still PASS.

The pre-existing `_build_synthetic_wr_features` helper auto-populates any new schema column via its dtype-driven loop (`for col_name, col in schema_cols.items(): ...`) — no manual fixture update needed. If a test does fail with "missing column" or "feature columns differ from training", that's a real ordering bug — stop and investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/lightgbm_nb.py tests/test_models/test_lightgbm_nb.py
git commit -m "feat(33c): qb_lightgbm_nb + wr_lightgbm_nb adopt Vegas-swap feature list"
```

---

### Task 7: Extend _code_hash_files_nb with vegas_team_context_features.py

**Files:**
- Modify: `src/projections/models/lightgbm_nb.py:97-128` (`_code_hash_files_nb` function)
- Test: `tests/test_models/test_lightgbm_nb.py` (add 1 new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models/test_lightgbm_nb.py`:

```python
def test_code_hash_files_nb_includes_vegas_team_context_features() -> None:
    """vegas_team_context_features.py is a transitive dep of QB + WR builders
    post-#33c integration. Without it in the hash set, a fix-only edit to
    that module would fail to invalidate cached model_ids."""
    from projections.models.lightgbm_nb import _code_hash_files_nb
    from projections.schemas import Position

    for pos in (Position.QB, Position.WR, Position.TE, Position.RB):
        files = _code_hash_files_nb(pos)
        vegas_in_set = any(p.name == "vegas_team_context_features.py" for p in files)
        assert vegas_in_set, (
            f"vegas_team_context_features.py missing from _code_hash_files_nb({pos})"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models/test_lightgbm_nb.py::test_code_hash_files_nb_includes_vegas_team_context_features -v
```

Expected: FAIL — `vegas_team_context_features.py` is not yet in the hash set.

- [ ] **Step 3: Add the file to the hash set**

In `src/projections/models/lightgbm_nb.py`, modify `_code_hash_files_nb` to include the new path. After the existing `src / "features" / feat_module,` line, add:

```python
        src / "features" / "vegas_team_context_features.py",
```

Final shape of the return tuple:
```python
    return (
        src / "models" / "lightgbm_nb.py",
        src / "models" / "lightgbm_tuned.py",
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "vegas_team_context_features.py",
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
        _TUNED_PARAMS_PATH,
    )
```

(Add unconditionally for all 4 positions — the docstring already explains the QB+WR mechanism via `feat_module`; the unconditional addition has a one-time model_id rebuild cost for TE + RB lgb-nb but no behavioral change.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_models/test_lightgbm_nb.py::test_code_hash_files_nb_includes_vegas_team_context_features -v
pytest tests/test_models/test_lightgbm_nb.py -v
```

Expected: new test PASS; existing `test_code_hash_differs_from_tuned` still PASS (NB's hash list is a strict superset of Tuned's).

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm_nb.py tests/test_models/test_lightgbm_nb.py
git commit -m "feat(33c): _code_hash_files_nb tracks vegas_team_context_features.py"
```

---

### Task 8: Pin lgb / lgb-tuned augment treatment + TE/RB lgb-nb unchanged

**Files:**
- Test only: `tests/test_models/test_lightgbm_nb.py` + `tests/test_models/test_lightgbm.py` (one assertion each)

These tests pin the asymmetry that the spec encodes — they don't drive new implementation, they document the design intent.

- [ ] **Step 1: Write the pinning tests**

Append to `tests/test_models/test_lightgbm_nb.py`:

```python
def test_te_lightgbm_nb_feature_columns_unchanged_by_vegas_integration() -> None:
    """TE was NULL in the probe — not adopted; feature list must not carry
    the four Vegas cols and must still include per-game implied_team_total +
    spread (whatever the TE schema produces)."""
    from projections.models.lightgbm import _filter_features, _TE_FEATURE_COLUMNS
    model = te_lightgbm_nb()
    expected = _filter_features(_TE_FEATURE_COLUMNS)
    assert model._config.feature_columns == expected


def test_rb_lightgbm_nb_feature_columns_unchanged_by_vegas_integration() -> None:
    """RB just-missed-ADOPT in the probe and is deferred to a separate
    preseason_*-only follow-up. Feature list must be the same as pre-#33c."""
    from projections.models.lightgbm import _filter_features, _RB_FEATURE_COLUMNS
    model = rb_lightgbm_nb()
    expected = _filter_features(_RB_FEATURE_COLUMNS)
    assert model._config.feature_columns == expected
```

Append to `tests/test_models/test_lightgbm.py` (if the file does not exist, locate the analogous file via `Glob tests/test_models/test_lightgbm*.py`):

```python
def test_qb_lightgbm_feature_columns_keep_per_game_vegas_cols() -> None:
    """lgb (untuned) is a non-production class with schema-derived feature
    list — augment treatment by default for the four new Vegas cols. Probe
    verdict for lgb augment composite was NULL; not a regression. Pinned
    here to surface a deliberate decision if anyone tries to swap it later."""
    from projections.models.lightgbm import qb_lightgbm
    model = qb_lightgbm()
    cols = set(model._config.feature_columns)
    assert "implied_team_total" in cols
    assert "spread" in cols
    # And the four new cols are picked up via schema-derivation.
    for c in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert c in cols


def test_qb_lightgbm_tuned_feature_columns_keep_per_game_vegas_cols() -> None:
    from projections.models.lightgbm_tuned import qb_lightgbm_tuned
    model = qb_lightgbm_tuned()
    cols = set(model._config.feature_columns)
    assert "implied_team_total" in cols
    assert "spread" in cols
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_models/test_lightgbm.py tests/test_models/test_lightgbm_nb.py -v
```

Expected: all PASS (no implementation needed — these pin existing state).

- [ ] **Step 3: Commit**

```bash
git add tests/test_models/test_lightgbm.py tests/test_models/test_lightgbm_nb.py
git commit -m "test(33c): pin lgb/lgb-tuned augment + TE/RB lgb-nb unchanged"
```

---

### Phase 1 verification

- [ ] **Run the Phase-1-touching subset:**

```bash
pytest -v tests/test_models tests/test_schemas tests/test_features
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all pass. Phase 1 is the load-bearing surface for Phase 2's gates.

- [ ] **Optional smoke: instantiate the WR ensemble-decomposed.** This is not gated but surfaces any factory-construction regression early:

```bash
python -c "from projections.models import wr_ensemble_decomposed; m = wr_ensemble_decomposed(); print('OK', m._config.position, m._config.child_b_factory().__class__.__name__)"
```

Expected output: `OK Position.WR LightGBMNbModel`.

---

## Phase 2: Adoption gates + report (no source changes, 6 tasks)

Phase 2 has no source changes; it produces backtest snapshots, runs gates, and writes the verdict report. **Each backtest run can take 5–30 minutes** depending on hardware — these are not 2-minute tasks, but each sub-step is a single command.

### Task 9: Pre-integration backtest runs on `main` HEAD

The "pre" side of the dual-run gate. PR #49 did not touch schemas or builders, so `main` HEAD reproduces the pre-integration feature builder + lgb-nb behavior.

**WARNING:** This task switches branches. Stash any uncommitted work on the integration branch first.

- [ ] **Step 1: Switch to main (PR #49 merge)**

```bash
git stash push -u -m "wip before pre-integration backtest" || echo "nothing to stash"
git checkout main
git pull --ff-only
git log -1 --oneline
```

Expected: HEAD shows the PR #49 merge commit.

- [ ] **Step 2: Refresh feature parquets**

```bash
python scripts/refresh_features.py
```

Expected: `data/features/{qb,wr,rb,te}/season=YYYY/week=WW/part.parquet` files exist. No errors. (If `refresh_features.py` takes flags for seasons, use the defaults — the gate expects the same coverage as `tests/backtest/`'s default backtest window.)

- [ ] **Step 3: Run pre-integration backtest for lgb-nb (all positions)**

```bash
python scripts/backtest.py --report --model lightgbm-nb
```

Expected output: writes `data/backtest/run_<timestamp>/results.parquet` (and `season_results.parquet`). Capture the timestamp.

- [ ] **Step 4: Rename the run directory to a stable label**

PowerShell:
```powershell
Move-Item "data/backtest/run_<timestamp>" "data/backtest/run_pre_vegas_lgbnb"
```

Bash:
```bash
mv "data/backtest/run_<timestamp>" "data/backtest/run_pre_vegas_lgbnb"
```

Verify: `ls data/backtest/run_pre_vegas_lgbnb/results.parquet`.

- [ ] **Step 5: Run pre-integration backtest for ensemble-decomposed (WR only)**

```bash
python scripts/backtest.py --report --model ensemble-decomposed
```

Expected: writes a new `data/backtest/run_<timestamp>/`. Rename to `data/backtest/run_pre_vegas_ensdec`.

- [ ] **Step 6: Switch back to the integration branch**

```bash
git checkout feat/qb-wr-vegas-team-context-integration
git stash pop || echo "nothing to pop"
```

(No commit — `data/backtest/` is gitignored per `_write_diagnostic_outputs` docstring.)

---

### Task 10: Post-integration backtest runs on the integration branch

- [ ] **Step 1: Refresh feature parquets on the integration branch**

```bash
python scripts/refresh_features.py
```

Expected: feature parquets are regenerated; QB + WR partitions now carry the four new cols (verified by `python -c "import pandas as pd; df = pd.read_parquet('data/features/qb/season=2024/week=10/part.parquet'); print(sorted(df.columns))"` showing the new cols).

- [ ] **Step 2: Run post-integration backtest for lgb-nb**

```bash
python scripts/backtest.py --report --model lightgbm-nb
```

Rename to `data/backtest/run_post_vegas_lgbnb`.

- [ ] **Step 3: Run post-integration backtest for ensemble-decomposed**

```bash
python scripts/backtest.py --report --model ensemble-decomposed
```

Rename to `data/backtest/run_post_vegas_ensdec`.

---

### Task 11: Run three dual-run adoption gates

- [ ] **Step 1: Gate 1 — `(lgb-nb, QB)`**

```bash
python scripts/adoption_gate.py \
    --baseline-run data/backtest/run_pre_vegas_lgbnb \
    --candidate-run data/backtest/run_post_vegas_lgbnb \
    --position QB \
    --csv-out reports/gate_33c_lgbnb_qb.csv
```

Expected: stdout markdown table with composite_rmse delta + CI. Inspect: ΔRMSE point estimate should be ≈ −0.0587 fpts (probe prediction); CI should be strictly below 0 → ADOPT.

If CI brackets 0 or point estimate is > −0.030 fpts (i.e., ≥50% miss of probe), STOP and debug the feature-builder. The most likely cause is a sign-convention or join-key issue in the integrated builder vs. the probe's override parquet. Cross-check by loading `data/features/qb/season=2024/week=10/part.parquet` and the probe's override parquet (regenerable via `python scripts/build_vegas_team_context_override.py`) and asserting the four Vegas cols match.

- [ ] **Step 2: Gate 2 — `(lgb-nb, WR)`**

```bash
python scripts/adoption_gate.py \
    --baseline-run data/backtest/run_pre_vegas_lgbnb \
    --candidate-run data/backtest/run_post_vegas_lgbnb \
    --position WR \
    --csv-out reports/gate_33c_lgbnb_wr.csv
```

Expected: ΔRMSE ≈ −0.0130 fpts (probe prediction); CI strictly below 0 → ADOPT.

- [ ] **Step 3: Gate 3 — `(ensemble-decomposed, WR)` (production-route gate)**

```bash
python scripts/adoption_gate.py \
    --baseline-run data/backtest/run_pre_vegas_ensdec \
    --candidate-run data/backtest/run_post_vegas_ensdec \
    --position WR \
    --csv-out reports/gate_33c_ensdec_wr.csv
```

Expected: ΔRMSE direction uncertain (pinball-weight re-fit). ADOPT if CI < 0; MARGINAL acceptable if CI brackets 0 with point ≤ 0; STOP if CI > 0.

If REGRESSION: document in the integration summary report (Task 12) and pause for user decision — ship lgb-nb integration anyway (gates 1+2 are clean) vs. revert this branch.

---

### Task 12: Write integration summary report

**Files:**
- Create: `reports/qb_wr_vegas_team_context_integration_summary.md`

- [ ] **Step 1: Draft the report**

```markdown
# TODO #33c integration summary — QB + WR Vegas team-context

**Branch:** `feat/qb-wr-vegas-team-context-integration`.
**Spec:** `docs/superpowers/specs/2026-05-17-qb-wr-vegas-team-context-integration-design.md`.
**Predecessor probe:** PR #49 — `reports/feature_probe_vegas_team_context_summary.md`.
**Gate runs:** `data/backtest/run_{pre,post}_vegas_{lgbnb,ensdec}`.

## Gate verdicts

| Gate # | (model, position) | Predicted ΔRMSE (probe) | Observed ΔRMSE | 95% CI | Verdict |
|---|---|---|---|---|---|
| 1 | (lgb-nb, QB) | −0.0587 | <fill from `reports/gate_33c_lgbnb_qb.csv`> | <fill> | <ADOPT / MARGINAL / REGRESSION> |
| 2 | (lgb-nb, WR) | −0.0130 | <fill> | <fill> | <fill> |
| 3 | (ensemble-decomposed, WR) | n/a — production route | <fill> | <fill> | <fill> |

## Probe-vs-integration replication

Per probe spec convention: gate ΔRMSE within ±50% of probe point estimate counts as replication. Outside that band signals a builder bug.

| Position | Probe point | Gate point | Within ±50% band? |
|---|---|---|---|
| QB | −0.0587 | <fill> | <yes / no> |
| WR | −0.0130 | <fill> | <yes / no> |

If "no": diagnostic notes here (which test failed? what was the discrepancy in cell values? did the override audit show the same coverage?).

## Ship decision

- If gates 1 + 2 ADOPT and gate 3 ADOPT or MARGINAL: SHIP.
- If gate 3 REGRESSION: pause; document the per-stat pinball weight drift below.

## Per-stat pinball weights (ensemble-decomposed WR)

Read from `data/ensemble_weights/ensemble_wr_<hash>_<train_span>.json` for pre and post:

| Stat | Pre weight (A vs B) | Post weight (A vs B) | Δ |
|---|---|---|---|
| receptions | <fill> | <fill> | <fill> |
| receiving_yards | <fill> | <fill> | <fill> |
| receiving_tds | <fill> | <fill> | <fill> |
| rushing_yards | <fill> | <fill> | <fill> |
| rushing_tds | <fill> | <fill> | <fill> |
| fumbles_lost | <fill> | <fill> | <fill> |

A = decomposed-baseline (Ridge); B = lgb-nb. Higher post-B weights mean the gate over-credits the now-augmented lgb-nb child.

## Caveats (carried from probe verdict)

- QB ΔRMSE ≈ −0.06 fpts ≈ 1–2% per-week; **Chase 250→403 elite-magnitude gap is not closed.** Necessary but not sufficient.
- Next direction if gates ADOPT but elite-magnitude persists: external preseason Vegas data (May win totals, OC/HC tenure, FA flags) — separate spec.
- RB just-missed-ADOPT — follow-up probe queued (`preseason_*`-only override).
- TE NULL — closed.
```

- [ ] **Step 2: Populate the report from the three gate CSVs**

The three `reports/gate_33c_*.csv` files contain the per-position rows (`position`, `composite_rmse_delta`, `composite_rmse_ci_low`, `composite_rmse_ci_high`, `verdict`). Open each, copy the relevant numbers into the table.

For the per-stat pinball weights, find the most recent `data/ensemble_weights/ensemble_wr_<hash>_<train_span>.json` from the pre- and post-integration runs and copy the `weights` blob.

- [ ] **Step 3: Commit the report**

```bash
git add reports/qb_wr_vegas_team_context_integration_summary.md reports/gate_33c_lgbnb_qb.csv reports/gate_33c_lgbnb_wr.csv reports/gate_33c_ensdec_wr.csv
git commit -m "report(33c): QB+WR Vegas team-context integration gate verdicts"
```

---

### Task 13: Update TODO + project_management.md

- [ ] **Step 1: Update TODO.md #33c entry**

Open `TODO.md`. Locate the "33c — Phase 1 family probe complete, 2026-05-17: SIGNAL at lgb-nb swap" entry. Append an integration-verdict paragraph after the existing "Next step:" paragraph:

```markdown
**33c — integration shipped, <date>: <ADOPT / MIXED / REGRESSION>.** Integrated on branch `feat/qb-wr-vegas-team-context-integration`. Schema-swap on lgb-nb only (QbFeaturesSchema + WrFeaturesSchema gained 4 Vegas cols; lgb-nb's feature_columns drops per-game implied_team_total + spread and adds preseason_* + season_avg_*). BaselineModel + DecomposedBaselineModel feature lists untouched — Ridge children of wr_ensemble_decomposed unchanged. Gate verdicts:

- (lgb-nb, QB) ΔRMSE <observed> CI <CI> → <verdict>.
- (lgb-nb, WR) ΔRMSE <observed> CI <CI> → <verdict>.
- (ensemble-decomposed, WR) ΔRMSE <observed> CI <CI> → <verdict>.

<If REGRESSION on gate 3: ship decision rationale.>

**Next direction:** <[1] external preseason Vegas data spec — May win totals, OC/HC tenure, FA flag; [2] RB preseason_*-only follow-up probe; [3] follow-up retrospective on 2024 actuals to confirm elite-magnitude still gaps.> Pick one based on the post-integration retrospective.

See `reports/qb_wr_vegas_team_context_integration_summary.md`.
```

- [ ] **Step 2: Update project_management.md**

Append a similar (more decision-log-style) entry to project_management.md's most recent "Next action" section, encoding the verdict and the chosen next direction.

- [ ] **Step 3: Commit**

```bash
git add TODO.md project_management.md
git commit -m "docs(33c): integration verdict + next-direction recommendation"
```

---

### Task 14: Final verification

- [ ] **Step 1: Full pre-PR check**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

Per CLAUDE.md "Forced Verification" rule: paste this output into the PR description as evidence. If full pytest is impractical (>30 min), run the narrow subset:

```bash
pytest -v -k "vegas or lightgbm_nb or qb_features or wr_features or schemas or ensemble_decomposed"
```

- [ ] **Step 2: Confirm branch state and open PR**

```bash
git log main..HEAD --oneline
git push -u origin feat/qb-wr-vegas-team-context-integration
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "feat(33c): QB+WR Vegas team-context integration — <verdict line>" --body "$(cat <<'EOF'
## Summary

- Phase 2 of TODO #33c (follow-up to PR #49's family probe). Schema-swap on lgb-nb only for QB + WR: QbFeaturesSchema + WrFeaturesSchema gain `preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`. lgb-nb's QB + WR feature lists drop per-game `implied_team_total` + `spread` and adopt the four new cols. BaselineModel + DecomposedBaselineModel feature lists untouched so Ridge children of `wr_ensemble_decomposed` remain on pre-integration behavior.

## Verdict

<table from `reports/qb_wr_vegas_team_context_integration_summary.md`>

## Test plan

- [x] <count> new tests pass.
- [x] Targeted pytest subset clean.
- [x] `mypy src tests` — 0 violations.
- [x] `ruff check + format --check src tests` — clean.

## Files

**New:** <list>.
**Modified:** <list>.

Spec: `docs/superpowers/specs/2026-05-17-qb-wr-vegas-team-context-integration-design.md`.
Plan: `docs/superpowers/plans/2026-05-17-qb-wr-vegas-team-context-integration.md`.
Summary: `reports/qb_wr_vegas_team_context_integration_summary.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Done criteria

- All Phase-0, Phase-1, and Phase-2 commits land on `feat/qb-wr-vegas-team-context-integration`.
- PR open against `main`.
- Gates 1 + 2 ADOPT (or, with explicit ship decision, MARGINAL); gate 3 ADOPT or MARGINAL (or REGRESSION + documented ship decision).
- `reports/qb_wr_vegas_team_context_integration_summary.md` filled in with observed numbers.
- TODO #33c entry updated.
- `pytest`, `mypy strict`, `ruff` all clean.
