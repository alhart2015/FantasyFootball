# Preseason Projections — v1 Framework + Naive Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a new `src/projections/preseason/` sub-package that produces season-total fantasy-points projections for the upcoming season, plus a walk-forward backtest harness that characterizes the naive baseline on 2024 + 2025 holdouts. v1.0 ships the framework + a `NaivePreseasonModel` (veteran prior-year ×16, rookie draft-capital Gamma GLM); v1.5 (separate spec) adds the first trained model.

**Architecture:** Parallel sub-package to the existing `src/projections/{models, features, scoring, distributions, store, ingest, backtest}`. Four modules — `features.py`, `model.py`, `project.py`, `backtest.py`. Three new pandera schemas in `src/projections/schemas.py`. Two new CLI scripts under `scripts/`. Reuses the existing scoring layer (`scoring.score_distribution`) and store layer (`store.read_partition` / `store.write_partition`) — no scoring or I/O logic duplicated.

**Tech Stack:** Python 3.11+, pandas, pandera (DataFrameModel + strict="filter"), pydantic v2 (Ruleset), scikit-learn (`GammaRegressor` for the rookie GLM), joblib (artifact persistence), pytest, pyarrow (parquet partitions). Conforms to the project's mypy strict + ruff configuration.

**Spec:** `docs/superpowers/specs/2026-05-17-preseason-projections-design.md`.

**Branch:** `worktree-feat+preseason-projections` cut from `main` at `8ffa607` (post-PR #46 merge).

---

## Phasing & ground rules

- **Five phases.** Each phase touches ≤ 4 files, completes with `pytest + mypy + ruff` green, and ends with a commit. Wait for explicit user approval between phases. (CLAUDE.md Phased Execution rule.)
- **TDD.** Every task starts with a failing test, then minimal implementation, then verification, then commit.
- **Step 0 hygiene** is not needed here — no large-LOC refactors; this plan is greenfield code in a new sub-package.
- **`GsisId`/`Position`/`Team`/`Stat`/`Ruleset` enums** must be used everywhere; never bare strings (CLAUDE.md "Reference enums, never the strings they wrap").
- **Schema reassignment.** Pandera's `strict="filter"` returns a new DataFrame; always reassign: `df = SCHEMA.validate(df)`.
- **Frequent commits.** One commit per task, mode `spec → feat → test → fix` per content.
- **Pre-commit PATH quirk.** Prepend `.venv/Scripts` to PATH before `git commit` so pre-commit's mypy hook resolves to project's pydantic v2 (see memory `project_pre_commit_venv_quirk.md`).
- **Run `pytest -v -k "preseason or schemas"`** at the end of every phase. Run `mypy src tests` + `ruff check src tests` + `ruff format --check src tests` before declaring a phase done (CLAUDE.md forced-verification checklist).

---

## File map (locks decomposition before tasks)

```
src/projections/preseason/__init__.py            (NEW, Phase 1)
src/projections/preseason/features.py            (NEW, Phase 2) — build_preseason_features
src/projections/preseason/model.py               (NEW, Phase 3) — PreseasonModel Protocol + NaivePreseasonModel
src/projections/preseason/project.py             (NEW, Phase 4) — project_preseason driver
src/projections/preseason/backtest.py            (NEW, Phase 5) — walk_forward_backtest

src/projections/schemas.py                       (MODIFIED, Phase 1) — +3 schemas

scripts/preseason_project_season.py              (NEW, Phase 4) — CLI
scripts/backtest_preseason.py                    (NEW, Phase 5) — CLI

tests/test_preseason/__init__.py                 (NEW, Phase 1)
tests/test_preseason/test_features.py            (NEW, Phase 2)
tests/test_preseason/test_model.py               (NEW, Phase 3)
tests/test_preseason/test_backtest.py            (NEW, Phase 5)

tests/test_schemas/test_dataframe_schemas.py     (MODIFIED, Phase 1) — +3 schema tests
tests/test_scripts/test_preseason_project_season_cli.py   (NEW, Phase 4)
tests/test_scripts/test_backtest_preseason_cli.py         (NEW, Phase 5)
```

Total new files: 11. Modified: 2.

---

# Phase 1 — Schemas & sub-package skeleton

Lays the foundation. No business logic; just pandera schemas and empty packages so later tasks have stable import targets.

## Task 1: `PreseasonFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (append after `AuctionValuesSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schemas/test_dataframe_schemas.py`:

```python
def test_preseason_features_schema_validates_golden_row() -> None:
    """Smoke test: a minimal golden row passes PreseasonFeaturesSchema."""
    df = pd.DataFrame(
        {
            "gsis_id": ["00-1234567"],
            "season": pd.array([2026], dtype="int32"),
            "position": ["QB"],
            "team": ["KC"],
            "depth_chart_rank": pd.array([1], dtype="Int64"),
            "age": pd.array([29.0], dtype="float32"),
            "years_exp": pd.array([7], dtype="Int64"),
            "is_rookie": [False],
            "draft_round": pd.array([1], dtype="Int64"),
            "draft_pick_overall": pd.array([10], dtype="Int64"),
            "prior_1_season_per_game_passing_yards": pd.array([275.5], dtype="float32"),
            "prior_1_season_games_played": pd.array([17], dtype="Int64"),
        }
    )
    out = PreseasonFeaturesSchema.validate(df)
    assert len(out) == 1


def test_preseason_features_schema_rejects_bad_position() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-1234567"],
            "season": pd.array([2026], dtype="int32"),
            "position": ["XX"],  # not in Position enum
            "team": ["KC"],
            "depth_chart_rank": pd.array([1], dtype="Int64"),
            "years_exp": pd.array([7], dtype="Int64"),
            "is_rookie": [False],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        PreseasonFeaturesSchema.validate(df)
```

Add at top: `from projections.schemas import PreseasonFeaturesSchema` once the schema exists.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py::test_preseason_features_schema_validates_golden_row -v`
Expected: `ImportError: cannot import name 'PreseasonFeaturesSchema'`

- [ ] **Step 3: Add the schema**

Append to `src/projections/schemas.py` after `AuctionValuesSchema`:

```python
class PreseasonFeaturesSchema(pa.DataFrameModel):
    """One row per (gsis_id, season) for every player on depth_charts_{season}
    with position in {QB, RB, WR, TE}. Inputs to PreseasonModel.predict_season_distribution.

    `prior_{N}_season_per_game_<stat>` columns exist per modeled stat for the
    player's position. They are nullable — a player missing a prior season
    (rookies, injuries) has NaN there. Position-specific stat sets:
        QB: passing_yards, passing_tds, passing_interceptions,
            rushing_yards, rushing_tds.
        RB: rushing_yards, rushing_tds, receptions, receiving_yards,
            receiving_tds.
        WR: receptions, receiving_yards, receiving_tds, rushing_yards,
            rushing_tds.
        TE: receptions, receiving_yards, receiving_tds.
    """

    # Identity
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2018, le=2100)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    depth_chart_rank: Series[int] = pa.Field(ge=1, le=10)

    # Player profile
    age: Series[float] = pa.Field(ge=18.0, le=50.0, nullable=True)
    years_exp: Series[int] = pa.Field(ge=0, le=30)
    is_rookie: Series[bool]
    draft_round: Series[int] = pa.Field(ge=1, le=7, nullable=True)
    draft_pick_overall: Series[int] = pa.Field(ge=1, le=400, nullable=True)

    # Prior 1/2/3 season per-game aggregates — all nullable.
    # Pandera schemas can't declare per-position columns cleanly, so we declare
    # the UNION of stats here and rely on `strict="filter"` to drop columns not
    # populated for a given position. Population is the builder's job.
    prior_1_season_games_played: Series[int] = pa.Field(ge=0, le=17, nullable=True)
    prior_2_season_games_played: Series[int] = pa.Field(ge=0, le=17, nullable=True)
    prior_3_season_games_played: Series[int] = pa.Field(ge=0, le=17, nullable=True)

    prior_1_season_per_game_passing_yards: Series[float] = pa.Field(ge=-10, le=500, nullable=True)
    prior_2_season_per_game_passing_yards: Series[float] = pa.Field(ge=-10, le=500, nullable=True)
    prior_3_season_per_game_passing_yards: Series[float] = pa.Field(ge=-10, le=500, nullable=True)
    prior_1_season_per_game_passing_tds: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    prior_2_season_per_game_passing_tds: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    prior_3_season_per_game_passing_tds: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    prior_1_season_per_game_passing_interceptions: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    prior_2_season_per_game_passing_interceptions: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    prior_3_season_per_game_passing_interceptions: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    prior_1_season_per_game_rushing_yards: Series[float] = pa.Field(ge=-5, le=250, nullable=True)
    prior_2_season_per_game_rushing_yards: Series[float] = pa.Field(ge=-5, le=250, nullable=True)
    prior_3_season_per_game_rushing_yards: Series[float] = pa.Field(ge=-5, le=250, nullable=True)
    prior_1_season_per_game_rushing_tds: Series[float] = pa.Field(ge=0, le=5, nullable=True)
    prior_2_season_per_game_rushing_tds: Series[float] = pa.Field(ge=0, le=5, nullable=True)
    prior_3_season_per_game_rushing_tds: Series[float] = pa.Field(ge=0, le=5, nullable=True)
    prior_1_season_per_game_receptions: Series[float] = pa.Field(ge=0, le=20, nullable=True)
    prior_2_season_per_game_receptions: Series[float] = pa.Field(ge=0, le=20, nullable=True)
    prior_3_season_per_game_receptions: Series[float] = pa.Field(ge=0, le=20, nullable=True)
    prior_1_season_per_game_receiving_yards: Series[float] = pa.Field(ge=-5, le=300, nullable=True)
    prior_2_season_per_game_receiving_yards: Series[float] = pa.Field(ge=-5, le=300, nullable=True)
    prior_3_season_per_game_receiving_yards: Series[float] = pa.Field(ge=-5, le=300, nullable=True)
    prior_1_season_per_game_receiving_tds: Series[float] = pa.Field(ge=0, le=5, nullable=True)
    prior_2_season_per_game_receiving_tds: Series[float] = pa.Field(ge=0, le=5, nullable=True)
    prior_3_season_per_game_receiving_tds: Series[float] = pa.Field(ge=0, le=5, nullable=True)

    class Config:
        strict = "filter"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v -k "preseason_features"`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(schemas): PreseasonFeaturesSchema for preseason feature frame"
```

---

## Task 2: `PreseasonProjectionSchema`

**Files:**
- Modify: `src/projections/schemas.py`
- Modify: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schemas/test_dataframe_schemas.py`:

```python
def test_preseason_projection_schema_validates_golden_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-1234567"],
            "season": pd.array([2026], dtype="int32"),
            "position": ["QB"],
            "team": ["KC"],
            "ruleset": ["ESPN_PPR"],
            "model_id": ["naive-preseason-v1"],
            "season_total_fpts_mean": pd.array([380.0], dtype="float32"),
            "season_total_fpts_p10": pd.array([380.0], dtype="float32"),
            "season_total_fpts_p50": pd.array([380.0], dtype="float32"),
            "season_total_fpts_p90": pd.array([380.0], dtype="float32"),
            "passing_yards_season_total_mean": pd.array([4400.0], dtype="float32"),
            "passing_yards_season_total_p10": pd.array([4400.0], dtype="float32"),
            "passing_yards_season_total_p50": pd.array([4400.0], dtype="float32"),
            "passing_yards_season_total_p90": pd.array([4400.0], dtype="float32"),
        }
    )
    out = PreseasonProjectionSchema.validate(df)
    assert len(out) == 1
    # Ruleset enum check
    assert out["ruleset"].iloc[0] in {"ESPN_PPR", "ESPN_HALF", "STANDARD"}


def test_preseason_projection_schema_rejects_negative_fpts() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-1234567"],
            "season": pd.array([2026], dtype="int32"),
            "position": ["QB"],
            "team": ["KC"],
            "ruleset": ["ESPN_PPR"],
            "model_id": ["naive-preseason-v1"],
            "season_total_fpts_mean": pd.array([-10.0], dtype="float32"),  # negative
            "season_total_fpts_p10": pd.array([0.0], dtype="float32"),
            "season_total_fpts_p50": pd.array([0.0], dtype="float32"),
            "season_total_fpts_p90": pd.array([0.0], dtype="float32"),
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        PreseasonProjectionSchema.validate(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v -k "preseason_projection"`
Expected: ImportError (schema not yet defined).

- [ ] **Step 3: Add the schema**

Append to `src/projections/schemas.py` after `PreseasonFeaturesSchema`:

```python
_RULESET_NAME_VALUES: Final = ["ESPN_PPR", "ESPN_HALF", "STANDARD"]


class PreseasonProjectionSchema(pa.DataFrameModel):
    """One row per (gsis_id, season, ruleset) — the v1 preseason output.

    Per-stat season-total quartets `<stat>_season_total_{mean,p10,p50,p90}` are
    populated per the player's position's stat set. Columns not modeled for a
    position are absent; strict="filter" preserves only declared columns.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2018, le=2100)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    ruleset: Series[str] = pa.Field(isin=_RULESET_NAME_VALUES)
    model_id: Series[str] = pa.Field(dtype_kwargs={"storage": "pyarrow"}, nullable=False)

    # Scored fpts — required for every row.
    season_total_fpts_mean: Series[float] = pa.Field(ge=0, le=700)
    season_total_fpts_p10: Series[float] = pa.Field(ge=0, le=700)
    season_total_fpts_p50: Series[float] = pa.Field(ge=0, le=700)
    season_total_fpts_p90: Series[float] = pa.Field(ge=0, le=700)

    # Per-stat season totals — UNION of stats across positions; nullable.
    passing_yards_season_total_mean: Series[float] = pa.Field(ge=0, le=7000, nullable=True)
    passing_yards_season_total_p10: Series[float] = pa.Field(ge=0, le=7000, nullable=True)
    passing_yards_season_total_p50: Series[float] = pa.Field(ge=0, le=7000, nullable=True)
    passing_yards_season_total_p90: Series[float] = pa.Field(ge=0, le=7000, nullable=True)
    passing_tds_season_total_mean: Series[float] = pa.Field(ge=0, le=80, nullable=True)
    passing_tds_season_total_p10: Series[float] = pa.Field(ge=0, le=80, nullable=True)
    passing_tds_season_total_p50: Series[float] = pa.Field(ge=0, le=80, nullable=True)
    passing_tds_season_total_p90: Series[float] = pa.Field(ge=0, le=80, nullable=True)
    passing_interceptions_season_total_mean: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    passing_interceptions_season_total_p10: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    passing_interceptions_season_total_p50: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    passing_interceptions_season_total_p90: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    rushing_yards_season_total_mean: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    rushing_yards_season_total_p10: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    rushing_yards_season_total_p50: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    rushing_yards_season_total_p90: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    rushing_tds_season_total_mean: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    rushing_tds_season_total_p10: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    rushing_tds_season_total_p50: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    rushing_tds_season_total_p90: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    receptions_season_total_mean: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    receptions_season_total_p10: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    receptions_season_total_p50: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    receptions_season_total_p90: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    receiving_yards_season_total_mean: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    receiving_yards_season_total_p10: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    receiving_yards_season_total_p50: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    receiving_yards_season_total_p90: Series[float] = pa.Field(ge=0, le=3000, nullable=True)
    receiving_tds_season_total_mean: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    receiving_tds_season_total_p10: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    receiving_tds_season_total_p50: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    receiving_tds_season_total_p90: Series[float] = pa.Field(ge=0, le=40, nullable=True)

    class Config:
        strict = "filter"
```

Note: the `model_id` field's `dtype_kwargs={"storage": "pyarrow"}` ensures pyarrow-backed strings per CLAUDE.md "pd.StringDtype("pyarrow") for nullable string columns" convention. Pandera 0.31+ rejects object-dtype strings.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v -k "preseason_projection"`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(schemas): PreseasonProjectionSchema for preseason output frame"
```

---

## Task 3: `PreseasonBacktestSchema`

**Files:**
- Modify: `src/projections/schemas.py`
- Modify: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
def test_preseason_backtest_schema_validates_golden_row() -> None:
    df = pd.DataFrame(
        {
            "target_season": pd.array([2024], dtype="int32"),
            "position": ["QB"],
            "model_class": ["naive-preseason-v1"],
            "ruleset": ["ESPN_PPR"],
            "rmse": pd.array([35.0], dtype="float32"),
            "rmse_naive_baseline": pd.array([35.0], dtype="float32"),
            "rmse_delta_pct": pd.array([0.0], dtype="float32"),
            "spearman_top50": pd.array([0.72], dtype="float32"),
            "n_players": pd.array([28], dtype="Int64"),
            "coverage_diff_projected_not_played": pd.array([3], dtype="Int64"),
            "coverage_diff_played_not_projected": pd.array([1], dtype="Int64"),
            "verdict": ["NULL"],
        }
    )
    out = PreseasonBacktestSchema.validate(df)
    assert len(out) == 1


def test_preseason_backtest_schema_rejects_invalid_verdict() -> None:
    df = pd.DataFrame(
        {
            "target_season": pd.array([2024], dtype="int32"),
            "position": ["QB"],
            "model_class": ["naive-preseason-v1"],
            "ruleset": ["ESPN_PPR"],
            "rmse": pd.array([35.0], dtype="float32"),
            "rmse_naive_baseline": pd.array([35.0], dtype="float32"),
            "rmse_delta_pct": pd.array([0.0], dtype="float32"),
            "spearman_top50": pd.array([0.72], dtype="float32"),
            "n_players": pd.array([28], dtype="Int64"),
            "coverage_diff_projected_not_played": pd.array([0], dtype="Int64"),
            "coverage_diff_played_not_projected": pd.array([0], dtype="Int64"),
            "verdict": ["ADOPTED"],  # not in {ADOPT, NULL, DO_NOT_ADOPT}
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        PreseasonBacktestSchema.validate(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v -k "preseason_backtest"`
Expected: ImportError.

- [ ] **Step 3: Add the schema**

Append to `src/projections/schemas.py`:

```python
_BACKTEST_VERDICT_VALUES: Final = ["ADOPT", "NULL", "DO_NOT_ADOPT"]


class PreseasonBacktestSchema(pa.DataFrameModel):
    """One row per (target_season, position, model_class) — the v1 preseason
    backtest harness output frame."""

    target_season: Series[int] = pa.Field(ge=2018, le=2100)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    model_class: Series[str] = pa.Field(dtype_kwargs={"storage": "pyarrow"}, nullable=False)
    ruleset: Series[str] = pa.Field(isin=_RULESET_NAME_VALUES)
    rmse: Series[float] = pa.Field(ge=0)
    rmse_naive_baseline: Series[float] = pa.Field(ge=0)
    rmse_delta_pct: Series[float]  # signed
    spearman_top50: Series[float] = pa.Field(ge=-1, le=1)
    n_players: Series[int] = pa.Field(ge=0)
    coverage_diff_projected_not_played: Series[int] = pa.Field(ge=0)
    coverage_diff_played_not_projected: Series[int] = pa.Field(ge=0)
    verdict: Series[str] = pa.Field(isin=_BACKTEST_VERDICT_VALUES)

    class Config:
        strict = "filter"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v -k "preseason"`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(schemas): PreseasonBacktestSchema for backtest harness output"
```

---

## Task 4: Sub-package skeleton

**Files:**
- Create: `src/projections/preseason/__init__.py`
- Create: `tests/test_preseason/__init__.py`

- [ ] **Step 1: Create the sub-package init**

Create `src/projections/preseason/__init__.py`:

```python
"""Preseason projections — season-total per-player distributions produced from
data available before any games are played in the target season.

Parallel to the in-season weekly pipeline in `src/projections/models/`. Sized
to the season-total target rather than per-week. See
`docs/superpowers/specs/2026-05-17-preseason-projections-design.md`.
"""

from projections.preseason.features import build_preseason_features
from projections.preseason.model import NaivePreseasonModel, PreseasonModel

__all__ = [
    "NaivePreseasonModel",
    "PreseasonModel",
    "build_preseason_features",
]
```

This file will fail to import until Tasks 5-15 land. That's fine — it's the public API surface.

- [ ] **Step 2: Create the test-package init**

Create `tests/test_preseason/__init__.py` with empty content (`""`). pytest needs the directory to be a package.

- [ ] **Step 3: Smoke-test that other tests still pass**

Run: `pytest tests/test_schemas -v`
Expected: all PASS (no regression). The preseason `__init__.py` imports will fail at collection time if tests/test_preseason references them — they shouldn't yet.

- [ ] **Step 4: Use a minimal `__init__.py` for Phase 1**

Since `features.py` and `model.py` won't exist until Phase 2 / 3, the `from projections.preseason.features import ...` lines will break the import. Replace `src/projections/preseason/__init__.py` with a minimal docstring-only file for Phase 1:

```python
"""Preseason projections — season-total per-player distributions produced from
data available before any games are played in the target season.

Public API populated incrementally as features/model/project/backtest modules
land in subsequent phases.
"""
```

Task 10 adds the public re-exports once `model.py` exists.

- [ ] **Step 5: Phase-end verification**

```bash
pytest -v -k "preseason or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations across all four.

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/__init__.py tests/test_preseason/__init__.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): sub-package + test-package skeletons"
```

**Phase 1 done. STOP and request user approval before proceeding to Phase 2.**

---

# Phase 2 — Feature builder (`src/projections/preseason/features.py`)

Implements `build_preseason_features(...)` returning a `PreseasonFeaturesSchema`-validated frame. Spec §3 + §6.

## Task 5: Identity + position-filter columns

**Files:**
- Create: `src/projections/preseason/features.py`
- Create: `tests/test_preseason/test_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preseason/test_features.py`:

```python
"""Tests for src/projections/preseason/features.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.preseason.features import build_preseason_features
from projections.schemas import PreseasonFeaturesSchema, Position, Team


def _empty_weekly_stats() -> pd.DataFrame:
    cols = {
        "gsis_id": pd.Series([], dtype="string[pyarrow]"),
        "season": pd.Series([], dtype="int32"),
        "week": pd.Series([], dtype="int32"),
        "position": pd.Series([], dtype="string[pyarrow]"),
        "team": pd.Series([], dtype="string[pyarrow]"),
        "passing_yards": pd.Series([], dtype="float64"),
        "passing_tds": pd.Series([], dtype="int64"),
        "interceptions": pd.Series([], dtype="int64"),
        "rushing_yards": pd.Series([], dtype="float64"),
        "rushing_tds": pd.Series([], dtype="int64"),
        "receptions": pd.Series([], dtype="int64"),
        "receiving_yards": pd.Series([], dtype="float64"),
        "receiving_tds": pd.Series([], dtype="int64"),
    }
    return pd.DataFrame(cols)


def _make_depth_charts(rows: list[tuple[str, int, str, str, int]]) -> pd.DataFrame:
    """Each row: (gsis_id, week, position, team, depth_rank).

    Builder reads week=1 only (the preseason snapshot).
    """
    df = pd.DataFrame(rows, columns=["gsis_id", "week", "position", "team", "depth_rank"])
    df["season"] = 2026
    df["depth_team"] = df["position"] + df["depth_rank"].astype(str)
    return df[["gsis_id", "season", "week", "team", "position", "depth_team", "depth_rank"]]


def _make_id_map(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Each row: (gsis_id, full_name, birth_date_iso)."""
    df = pd.DataFrame(rows, columns=["gsis_id", "full_name", "birth_date"])
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    return df


def _make_draft_picks(rows: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    """Each row: (gsis_id, season, round, pick)."""
    return pd.DataFrame(
        rows, columns=["gsis_id", "season", "round", "pick"]
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})


def test_build_preseason_features_filters_to_skill_positions() -> None:
    depth = _make_depth_charts(
        [
            ("00-1000001", 1, "QB", "KC", 1),
            ("00-1000002", 1, "K", "KC", 1),  # filtered out
            ("00-1000003", 1, "DST", "BUF", 1),  # filtered out
        ]
    )
    id_map = _make_id_map(
        [
            ("00-1000001", "Patrick Mahomes", "1995-09-17"),
            ("00-1000002", "Harrison Butker", "1995-07-14"),
            ("00-1000003", "Buffalo Defense", "2000-01-01"),
        ]
    )
    out = build_preseason_features(
        weekly_stats=_empty_weekly_stats(),
        depth_charts_target=depth,
        draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
        id_map=id_map,
        target_season=2026,
    )
    assert len(out) == 1
    assert out["gsis_id"].iloc[0] == "00-1000001"
    assert out["position"].iloc[0] == Position.QB.value
    assert out["team"].iloc[0] == Team.KC.value
    assert out["depth_chart_rank"].iloc[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_features.py -v -k "filters_to_skill"`
Expected: `ModuleNotFoundError: No module named 'projections.preseason.features'`.

- [ ] **Step 3: Implement identity + filter**

Create `src/projections/preseason/features.py`:

```python
"""Preseason feature builder.

Produces one row per (gsis_id, target_season) for every rostered player on
depth_charts_<target_season> week=1 in skill positions {QB, RB, WR, TE}.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §3.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from projections.schemas import (
    PreseasonFeaturesSchema,
    Position,
    Stat,
)

logger = logging.getLogger(__name__)

_SKILL_POSITIONS: Final = frozenset({Position.QB, Position.RB, Position.WR, Position.TE})

# Position -> tuple of stats to materialize prior_{1,2,3}_season_per_game columns for.
_STATS_BY_POSITION: Final[dict[Position, tuple[Stat, ...]]] = {
    Position.QB: (
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.INTERCEPTIONS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
    ),
    Position.RB: (
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    ),
    Position.WR: (
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
    ),
    Position.TE: (
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    ),
}


def build_preseason_features(
    *,
    weekly_stats: pd.DataFrame,
    depth_charts_target: pd.DataFrame,
    draft_picks: pd.DataFrame,
    id_map: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """Build the preseason feature frame for `target_season`.

    Returns a DataFrame validated against PreseasonFeaturesSchema. One row per
    rostered player on `depth_charts_target` at week=1 in {QB, RB, WR, TE}.
    """
    # 1. Take the week-1 preseason snapshot of the depth chart.
    dc = depth_charts_target.loc[depth_charts_target["week"] == 1].copy()
    if dc.empty:
        raise ValueError(
            f"depth_charts_target has no week=1 rows for season={target_season}. "
            "v1 preseason builder reads the week-1 snapshot."
        )

    # 2. Position filter — skill positions only.
    skill_position_values = {p.value for p in _SKILL_POSITIONS}
    n_before = len(dc)
    dc = dc.loc[dc["position"].isin(skill_position_values)].copy()
    n_filtered = n_before - len(dc)
    if n_filtered:
        logger.info(
            "build_preseason_features: filtered %d non-skill-position rows "
            "(season=%d)",
            n_filtered,
            target_season,
        )

    # 3. Duplicate detection.
    dup_mask = dc.duplicated(subset=["gsis_id"], keep=False)
    if dup_mask.any():
        dup_ids = dc.loc[dup_mask, "gsis_id"].unique().tolist()
        raise ValueError(
            f"Duplicate gsis_id rows in depth_charts_target week=1: {dup_ids[:5]!r}. "
            "Upstream depth_charts dedup bug; never silently swallow."
        )

    # 4. Project identity columns.
    out = pd.DataFrame(
        {
            "gsis_id": dc["gsis_id"].astype("string[pyarrow]"),
            "season": pd.Series([target_season] * len(dc), dtype="int32"),
            "position": dc["position"].astype("string[pyarrow]"),
            "team": dc["team"].astype("string[pyarrow]"),
            "depth_chart_rank": dc["depth_rank"].astype("Int64"),
        }
    )

    # 5. Stub the rest of the schema. Filled in by tasks 6-9.
    out["age"] = pd.array([pd.NA] * len(out), dtype="Float32")
    out["years_exp"] = pd.array([0] * len(out), dtype="Int64")
    out["is_rookie"] = False
    out["draft_round"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["draft_pick_overall"] = pd.array([pd.NA] * len(out), dtype="Int64")

    # Empty prior_* columns to satisfy schema. Tasks 7 fills.
    for n in (1, 2, 3):
        out[f"prior_{n}_season_games_played"] = pd.array([pd.NA] * len(out), dtype="Int64")
        for stat in (
            Stat.PASSING_YARDS,
            Stat.PASSING_TDS,
            Stat.INTERCEPTIONS,
            Stat.RUSHING_YARDS,
            Stat.RUSHING_TDS,
            Stat.RECEPTIONS,
            Stat.RECEIVING_YARDS,
            Stat.RECEIVING_TDS,
        ):
            col = f"prior_{n}_season_per_game_{_schema_stat_name(stat)}"
            out[col] = pd.array([pd.NA] * len(out), dtype="Float32")

    out = PreseasonFeaturesSchema.validate(out)
    return out


def _schema_stat_name(stat: Stat) -> str:
    """The schema renames `Stat.INTERCEPTIONS` to `passing_interceptions` for
    disambiguation from defensive interceptions in K/DST work."""
    if stat is Stat.INTERCEPTIONS:
        return "passing_interceptions"
    return stat.value
```

Reason for `_schema_stat_name`: the schema uses `passing_interceptions` while `Stat.INTERCEPTIONS.value` is `"interceptions"`. The translation lives in one place.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preseason/test_features.py -v -k "filters_to_skill"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/features.py tests/test_preseason/test_features.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): build_preseason_features identity + position filter"
```

---

## Task 6: Player-profile columns (age, years_exp, is_rookie, draft pick)

**Files:**
- Modify: `src/projections/preseason/features.py`
- Modify: `tests/test_preseason/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preseason/test_features.py`:

```python
def test_build_preseason_features_age_from_id_map() -> None:
    depth = _make_depth_charts([("00-1000001", 1, "QB", "KC", 1)])
    id_map = _make_id_map([("00-1000001", "Patrick Mahomes", "1995-09-17")])
    out = build_preseason_features(
        weekly_stats=_empty_weekly_stats(),
        depth_charts_target=depth,
        draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
        id_map=id_map,
        target_season=2026,
    )
    assert out["age"].iloc[0] == pytest.approx(2026 - 1995, abs=1.0)  # 31 ± 1 for birthday timing
    assert out["years_exp"].iloc[0] == 2026 - 2017  # 9
    assert bool(out["is_rookie"].iloc[0]) is False
    assert out["draft_round"].iloc[0] == 1
    assert out["draft_pick_overall"].iloc[0] == 10


def test_build_preseason_features_rookie_in_draft_picks() -> None:
    depth = _make_depth_charts([("00-2000001", 1, "WR", "DET", 2)])
    id_map = _make_id_map([("00-2000001", "Hypothetical Rookie", "2003-04-01")])
    draft = _make_draft_picks([("00-2000001", 2026, 1, 23)])  # 1st-round 2026 rookie
    out = build_preseason_features(
        weekly_stats=_empty_weekly_stats(),
        depth_charts_target=depth,
        draft_picks=draft,
        id_map=id_map,
        target_season=2026,
    )
    assert bool(out["is_rookie"].iloc[0]) is True
    assert out["years_exp"].iloc[0] == 0
    assert out["draft_round"].iloc[0] == 1
    assert out["draft_pick_overall"].iloc[0] == 23


def test_build_preseason_features_udfa_rookie() -> None:
    depth = _make_depth_charts([("00-2000002", 1, "WR", "DET", 5)])
    id_map = _make_id_map([("00-2000002", "UDFA Rookie", "2003-08-01")])
    # No row in draft_picks for this player — UDFA case.
    out = build_preseason_features(
        weekly_stats=_empty_weekly_stats(),
        depth_charts_target=depth,
        draft_picks=_make_draft_picks([]),
        id_map=id_map,
        target_season=2026,
    )
    # Features layer leaves draft_round/pick as NaN. Imputation is the
    # model layer's responsibility (Task 14).
    assert pd.isna(out["draft_round"].iloc[0])
    assert pd.isna(out["draft_pick_overall"].iloc[0])
    # But `is_rookie` should be True because there's no prior-NFL evidence.
    assert bool(out["is_rookie"].iloc[0]) is True
```

The `is_rookie` determination needs a clarification: a player is a rookie iff (a) they're in `draft_picks[season=target_season]` OR (b) they have no `weekly_stats` rows in any prior season. UDFAs in 2026 satisfy (b). Returning veterans coming back from a year off ALSO satisfy (b) but aren't rookies — flagged for follow-up. For v1: use the heuristic `is_rookie = (no weekly_stats history) OR (draft_picks[season=target_season] match)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_features.py -v -k "age_from_id_map or rookie"`
Expected: 3 FAIL (assertions on default values).

- [ ] **Step 3: Implement player-profile columns**

Edit `src/projections/preseason/features.py`. Replace the "stub" section (the `out["age"] = ...` block through `out["draft_pick_overall"] = ...`) with:

```python
    # ---- Age (from id_map.birth_date) ----
    id_map_lookup = id_map.set_index("gsis_id")
    birth_dates = pd.to_datetime(
        out["gsis_id"].map(id_map_lookup["birth_date"]),
        errors="coerce",
    )
    age_years = pd.Series(
        [
            float(target_season - bd.year) if pd.notna(bd) else float("nan")
            for bd in birth_dates
        ],
        dtype="Float32",
    )
    out["age"] = age_years

    # ---- Rookie detection ----
    # Players with at least one prior-season weekly_stats row are NOT rookies.
    # Players in draft_picks[season=target_season] ARE rookies.
    # Everyone else (UDFA rookies, comeback veterans) is treated as rookie
    # for v1 — flagged limitation, fine for the naive baseline since UDFA
    # comeback-vet rows are rare and the model treats both via the rookie GLM.
    prior_seasons = weekly_stats.loc[weekly_stats["season"] < target_season, "gsis_id"].unique()
    has_prior_history = out["gsis_id"].isin(prior_seasons)

    target_draft_picks = draft_picks.loc[draft_picks["season"] == target_season]
    drafted_this_year = out["gsis_id"].isin(target_draft_picks["gsis_id"].unique())

    out["is_rookie"] = (~has_prior_history) | drafted_this_year

    # ---- years_exp = target_season - first_season_in_weekly_stats ----
    if not weekly_stats.empty:
        first_season_by_player = (
            weekly_stats.groupby("gsis_id")["season"].min().rename("first_season")
        )
        first_season_lookup = out["gsis_id"].map(first_season_by_player)
        years_exp = (target_season - first_season_lookup).fillna(0).astype("Int64")
    else:
        years_exp = pd.array([0] * len(out), dtype="Int64")
    # Override: rookies have years_exp = 0 by definition (even if id_map / draft_picks
    # disagrees due to data lag).
    years_exp = years_exp.mask(out["is_rookie"].to_numpy(), 0).astype("Int64")
    out["years_exp"] = years_exp

    # ---- Draft pick (round + pick_overall) for the rookie cohort or veterans we have data for ----
    # Join on draft_picks. Most veterans have an entry from their rookie year; UDFAs and
    # pre-1980 retirees have NaN.
    most_recent_pick = (
        draft_picks.sort_values("season")
        .drop_duplicates(subset=["gsis_id"], keep="first")
        .set_index("gsis_id")
    )
    out["draft_round"] = (
        out["gsis_id"].map(most_recent_pick["round"]).astype("Int64")
    )
    out["draft_pick_overall"] = (
        out["gsis_id"].map(most_recent_pick["pick"]).astype("Int64")
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_features.py -v`
Expected: 4 PASS (the three new + the original).

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/features.py tests/test_preseason/test_features.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): age + years_exp + is_rookie + draft pick join"
```

---

## Task 7: Prior-season per-game aggregates

**Files:**
- Modify: `src/projections/preseason/features.py`
- Modify: `tests/test_preseason/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
def _make_weekly_stats(rows: list[dict]) -> pd.DataFrame:
    """Each row: keys gsis_id, season, week, position, team, <stat columns>."""
    df = pd.DataFrame(rows)
    df["season"] = df["season"].astype("int32")
    df["week"] = df["week"].astype("int32")
    return df


def test_build_preseason_features_prior_season_per_game_aggregates() -> None:
    # Player: 2 games in 2023 (3 td x 250 yds passing total).
    weekly = _make_weekly_stats(
        [
            {
                "gsis_id": "00-1000001",
                "season": 2023,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "passing_yards": 250.0,
                "passing_tds": 2,
                "interceptions": 1,
                "rushing_yards": 30.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
            {
                "gsis_id": "00-1000001",
                "season": 2023,
                "week": 2,
                "position": "QB",
                "team": "KC",
                "passing_yards": 280.0,
                "passing_tds": 1,
                "interceptions": 0,
                "rushing_yards": 20.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
        ]
    )
    depth = _make_depth_charts([("00-1000001", 1, "QB", "KC", 1)])
    id_map = _make_id_map([("00-1000001", "Patrick Mahomes", "1995-09-17")])
    out = build_preseason_features(
        weekly_stats=weekly,
        depth_charts_target=depth,
        draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
        id_map=id_map,
        target_season=2024,  # so 2023 is prior_1
    )
    # prior_1 = 2023: 2 games, 530 passing_yards total -> 265 per game.
    assert out["prior_1_season_games_played"].iloc[0] == 2
    assert float(out["prior_1_season_per_game_passing_yards"].iloc[0]) == pytest.approx(265.0)
    assert float(out["prior_1_season_per_game_passing_tds"].iloc[0]) == pytest.approx(1.5)
    # prior_2 and prior_3 should be NaN.
    assert pd.isna(out["prior_2_season_per_game_passing_yards"].iloc[0])
    assert pd.isna(out["prior_3_season_per_game_passing_yards"].iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_features.py::test_build_preseason_features_prior_season_per_game_aggregates -v`
Expected: FAIL on the per-game assertion (all stubbed NaN today).

- [ ] **Step 3: Implement prior-season aggregation**

Add to `src/projections/preseason/features.py`. Replace the loop that stubs `prior_*` columns with:

```python
    # ---- Prior 1/2/3 season per-game aggregates ----
    stats_to_aggregate = [
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.INTERCEPTIONS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    ]

    for n in (1, 2, 3):
        prior_season = target_season - n
        per_game = _aggregate_to_per_game(
            weekly_stats.loc[weekly_stats["season"] == prior_season],
            stats=stats_to_aggregate,
        )
        per_game_lookup = per_game.set_index("gsis_id")
        out[f"prior_{n}_season_games_played"] = (
            out["gsis_id"].map(per_game_lookup["games_played"]).astype("Int64")
        )
        for stat in stats_to_aggregate:
            col_in = f"per_game_{stat.value}"
            col_out = f"prior_{n}_season_per_game_{_schema_stat_name(stat)}"
            out[col_out] = (
                out["gsis_id"].map(per_game_lookup[col_in]).astype("Float32")
            )


def _aggregate_to_per_game(
    weekly: pd.DataFrame, *, stats: list[Stat]
) -> pd.DataFrame:
    """Return one row per gsis_id with games_played + per_game_<stat> for each
    stat. Empty input returns an empty frame with the right columns."""
    if weekly.empty:
        cols = {"gsis_id": pd.Series([], dtype="string[pyarrow]"), "games_played": pd.Series([], dtype="Int64")}
        for stat in stats:
            cols[f"per_game_{stat.value}"] = pd.Series([], dtype="float64")
        return pd.DataFrame(cols)

    games = weekly.groupby("gsis_id").size().rename("games_played")
    totals = weekly.groupby("gsis_id")[[s.value for s in stats]].sum()
    per_game = totals.div(games, axis=0)
    per_game.columns = [f"per_game_{c}" for c in per_game.columns]
    return per_game.join(games).reset_index()
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_preseason/test_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/features.py tests/test_preseason/test_features.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): prior 1/2/3 season per-game aggregates"
```

---

## Task 8: Dropped players (missing id_map) + drop CSV side-channel

**Files:**
- Modify: `src/projections/preseason/features.py`
- Modify: `tests/test_preseason/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_preseason_features_drops_players_missing_id_map(tmp_path) -> None:
    depth = _make_depth_charts(
        [
            ("00-1000001", 1, "QB", "KC", 1),
            ("00-9999999", 1, "WR", "DEN", 3),  # not in id_map
        ]
    )
    id_map = _make_id_map([("00-1000001", "Patrick Mahomes", "1995-09-17")])
    out = build_preseason_features(
        weekly_stats=_empty_weekly_stats(),
        depth_charts_target=depth,
        draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
        id_map=id_map,
        target_season=2026,
        dropped_csv_path=tmp_path / "dropped.csv",
    )
    assert "00-9999999" not in out["gsis_id"].tolist()
    assert "00-1000001" in out["gsis_id"].tolist()
    # Side-channel CSV present.
    dropped = pd.read_csv(tmp_path / "dropped.csv")
    assert dropped["gsis_id"].tolist() == ["00-9999999"]
    assert dropped["drop_reason"].tolist() == ["missing_id_map"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_features.py::test_build_preseason_features_drops_players_missing_id_map -v`
Expected: FAIL — `dropped_csv_path` is not yet a parameter; player not dropped.

- [ ] **Step 3: Implement drop + side-channel**

Edit `src/projections/preseason/features.py`. Add `dropped_csv_path: Path | None = None` parameter. Implement the drop logic right after step 4 (the identity-column projection) and before step 5 (column stubbing):

```python
from pathlib import Path  # add near top imports

# Add to function signature:
def build_preseason_features(
    *,
    weekly_stats: pd.DataFrame,
    depth_charts_target: pd.DataFrame,
    draft_picks: pd.DataFrame,
    id_map: pd.DataFrame,
    target_season: int,
    dropped_csv_path: Path | None = None,
) -> pd.DataFrame:
    ...

# Add after the identity projection step:
    # ---- Drop players missing from id_map ----
    known_ids = set(id_map["gsis_id"].unique())
    missing_mask = ~out["gsis_id"].isin(known_ids)
    if missing_mask.any():
        dropped = pd.DataFrame(
            {
                "gsis_id": out.loc[missing_mask, "gsis_id"],
                "drop_reason": "missing_id_map",
                "season": target_season,
            }
        )
        logger.warning(
            "build_preseason_features: dropped %d player(s) missing from id_map "
            "(season=%d). See %s",
            len(dropped),
            target_season,
            dropped_csv_path,
        )
        if dropped_csv_path is not None:
            dropped_csv_path.parent.mkdir(parents=True, exist_ok=True)
            dropped.to_csv(dropped_csv_path, index=False)
        out = out.loc[~missing_mask].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/features.py tests/test_preseason/test_features.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): drop players missing from id_map + side-channel CSV"
```

---

## Task 9: Phase 2 closure — duplicate detection + missing-week-1 error + schema validation reassignment

**Files:**
- Modify: `src/projections/preseason/features.py`
- Modify: `tests/test_preseason/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_preseason_features_raises_on_duplicate_gsis_id() -> None:
    depth = _make_depth_charts(
        [
            ("00-1000001", 1, "QB", "KC", 1),
            ("00-1000001", 1, "QB", "KC", 2),  # duplicate gsis_id at week=1
        ]
    )
    id_map = _make_id_map([("00-1000001", "Patrick Mahomes", "1995-09-17")])
    with pytest.raises(ValueError, match="Duplicate gsis_id"):
        build_preseason_features(
            weekly_stats=_empty_weekly_stats(),
            depth_charts_target=depth,
            draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
            id_map=id_map,
            target_season=2026,
        )


def test_build_preseason_features_raises_on_no_week_1_rows() -> None:
    """Builder requires week=1 snapshot rows; raises if depth_charts only has
    later weeks."""
    depth = _make_depth_charts([("00-1000001", 5, "QB", "KC", 1)])  # week=5 only
    id_map = _make_id_map([("00-1000001", "Patrick Mahomes", "1995-09-17")])
    with pytest.raises(ValueError, match="no week=1 rows"):
        build_preseason_features(
            weekly_stats=_empty_weekly_stats(),
            depth_charts_target=depth,
            draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
            id_map=id_map,
            target_season=2026,
        )


def test_build_preseason_features_output_passes_schema_validation() -> None:
    """Smoke test: the builder's output validates without further coercion."""
    weekly = _make_weekly_stats(
        [
            {
                "gsis_id": "00-1000001",
                "season": 2023,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "passing_yards": 250.0,
                "passing_tds": 2,
                "interceptions": 1,
                "rushing_yards": 30.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
        ]
    )
    depth = _make_depth_charts([("00-1000001", 1, "QB", "KC", 1)])
    id_map = _make_id_map([("00-1000001", "Patrick Mahomes", "1995-09-17")])
    out = build_preseason_features(
        weekly_stats=weekly,
        depth_charts_target=depth,
        draft_picks=_make_draft_picks([("00-1000001", 2017, 1, 10)]),
        id_map=id_map,
        target_season=2024,
    )
    # Re-validate as a sanity check (build_preseason_features already does this).
    out2 = PreseasonFeaturesSchema.validate(out)
    assert len(out2) == 1
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_preseason/test_features.py -v`
Expected: 8 PASS (existing 5 + 3 new). The duplicate and missing-week-1 errors already raise from Task 5; this test asserts coverage. Schema validation is asserted to be reassigned.

- [ ] **Step 3: Verify schema-validation reassignment is correct in the implementation**

In `src/projections/preseason/features.py`, confirm the final line is:

```python
    out = PreseasonFeaturesSchema.validate(out)
    return out
```

NOT `PreseasonFeaturesSchema.validate(out); return out` — without reassignment, `strict="filter"` does NOT drop extra columns (the returned filtered DataFrame is discarded). See CLAUDE.md "df = SCHEMA.validate(df) (with reassignment)".

- [ ] **Step 4: Phase-end verification**

Run all of:
```bash
pytest -v -k "preseason or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```
Expected: zero violations across all four.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/features.py tests/test_preseason/test_features.py
PATH=".venv/Scripts:$PATH" git commit -m "test(preseason): cover dup detection + missing-week-1 + schema reassignment"
```

**Phase 2 done. STOP and request user approval before proceeding to Phase 3.**

---

# Phase 3 — `PreseasonModel` Protocol & `NaivePreseasonModel`

Implements the three-branch naive baseline + the rookie-year Gamma GLM. Spec §4.

## Task 10: `PreseasonModel` Protocol + `NaivePreseasonModel` skeleton

**Files:**
- Create: `src/projections/preseason/model.py`
- Create: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preseason/test_model.py`:

```python
"""Tests for src/projections/preseason/model.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.preseason.features import build_preseason_features
from projections.preseason.model import NaivePreseasonModel, PreseasonModel
from projections.schemas import (
    PreseasonFeaturesSchema,
    PreseasonProjectionSchema,
    Ruleset,
)


def test_naive_preseason_model_implements_protocol() -> None:
    """NaivePreseasonModel should satisfy the PreseasonModel Protocol at runtime."""
    m = NaivePreseasonModel()
    assert isinstance(m, PreseasonModel)
    assert m.model_id == "naive-preseason-v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement Protocol + skeleton**

Create `src/projections/preseason/model.py`:

```python
"""Preseason model interface + naive baseline.

The PreseasonModel Protocol matches the existing Distribution Protocol
pattern (runtime_checkable, attribute-based). NaivePreseasonModel is the
v1.0 baseline; v1.5+ trained models implement the same Protocol.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from projections.schemas import (
    PreseasonFeaturesSchema,
    PreseasonProjectionSchema,
    Ruleset,
)

logger = logging.getLogger(__name__)

_PROJECTED_GAMES_PLAYED = 16  # v1 constant; v2 ships per-player injury prior.


@runtime_checkable
class PreseasonModel(Protocol):
    """v1.0 returns degenerate point-mass distributions; v1.5+ returns real."""

    model_id: str

    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None: ...

    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "PreseasonModel": ...


class NaivePreseasonModel:
    """Three-branch naive baseline:

    Branch 1 — veterans with prior-1 season stats: `prior_1_per_game × 16`.
    Branch 2 — veterans missing prior-1: fall back to prior_2, then prior_3.
                Drop with warning if all three missing.
    Branch 3 — rookies: per-(position, stat) Gamma GLM on `log(pick + 1)`.

    Degenerate distribution: mean == p10 == p50 == p90 for every output.
    """

    model_id: str = "naive-preseason-v1"

    def __init__(self) -> None:
        # Per-(position, stat) GLM coefficients. Populated by fit().
        self._rookie_glm: dict[tuple[str, str], tuple[float, float]] = {}

    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None:
        """Fit per-(position, stat) rookie-year GLMs. No-op for veteran branches."""
        # Implementation in Tasks 13-14.
        raise NotImplementedError("fit not yet implemented (Task 13)")

    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        """Predict per-stat season-total degenerate distributions per player."""
        # Implementation in Tasks 11-15.
        raise NotImplementedError("predict not yet implemented (Task 11)")

    def save(self, path: Path) -> None:
        raise NotImplementedError("save not yet implemented (Task 16)")

    @classmethod
    def load(cls, path: Path) -> "NaivePreseasonModel":
        raise NotImplementedError("load not yet implemented (Task 16)")
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: PASS.

- [ ] **Step 5: Update package init**

Now that `model.py` exists, add the import line to `src/projections/preseason/__init__.py`:

```python
"""Preseason projections — season-total per-player distributions ..."""

from projections.preseason.features import build_preseason_features
from projections.preseason.model import NaivePreseasonModel, PreseasonModel

__all__ = [
    "NaivePreseasonModel",
    "PreseasonModel",
    "build_preseason_features",
]
```

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py src/projections/preseason/__init__.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): PreseasonModel Protocol + NaivePreseasonModel skeleton"
```

---

## Task 11: Veteran branch — `prior_1 × 16`

**Files:**
- Modify: `src/projections/preseason/model.py`
- Modify: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
def _make_features_row(**kwargs: object) -> pd.DataFrame:
    """Construct a one-row PreseasonFeaturesSchema-valid frame with overrides."""
    base = {
        "gsis_id": "00-1000001",
        "season": pd.array([2026], dtype="int32"),
        "position": "QB",
        "team": "KC",
        "depth_chart_rank": pd.array([1], dtype="Int64"),
        "age": pd.array([29.0], dtype="float32"),
        "years_exp": pd.array([7], dtype="Int64"),
        "is_rookie": False,
        "draft_round": pd.array([1], dtype="Int64"),
        "draft_pick_overall": pd.array([10], dtype="Int64"),
    }
    for n in (1, 2, 3):
        base[f"prior_{n}_season_games_played"] = pd.array([pd.NA], dtype="Int64")
        for stat in (
            "passing_yards", "passing_tds", "passing_interceptions",
            "rushing_yards", "rushing_tds",
            "receptions", "receiving_yards", "receiving_tds",
        ):
            base[f"prior_{n}_season_per_game_{stat}"] = pd.array([pd.NA], dtype="float32")
    base.update(kwargs)
    df = pd.DataFrame({k: v if isinstance(v, pd.api.extensions.ExtensionArray) else [v] for k, v in base.items()})
    return PreseasonFeaturesSchema.validate(df)


def test_naive_predict_veteran_branch_prior_1() -> None:
    """Veteran with prior-1 stats: predicted = prior_1_per_game × 16."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([17], dtype="Int64"),
        prior_1_season_per_game_passing_yards=pd.array([275.0], dtype="float32"),
        prior_1_season_per_game_passing_tds=pd.array([2.0], dtype="float32"),
        prior_1_season_per_game_passing_interceptions=pd.array([0.5], dtype="float32"),
        prior_1_season_per_game_rushing_yards=pd.array([12.0], dtype="float32"),
        prior_1_season_per_game_rushing_tds=pd.array([0.1], dtype="float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 1
    # Veteran branch: 275 × 16 = 4400 passing yards.
    assert float(out["passing_yards_season_total_mean"].iloc[0]) == pytest.approx(4400.0)
    # Degenerate distribution: all quantiles equal.
    assert float(out["passing_yards_season_total_p10"].iloc[0]) == pytest.approx(4400.0)
    assert float(out["passing_yards_season_total_p90"].iloc[0]) == pytest.approx(4400.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_model.py::test_naive_predict_veteran_branch_prior_1 -v`
Expected: NotImplementedError raised.

- [ ] **Step 3: Implement the veteran branch**

In `src/projections/preseason/model.py`, replace the `predict_season_distribution` body with:

```python
    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        features = PreseasonFeaturesSchema.validate(features)

        per_stat_predictions = self._predict_per_stat(features)

        # TODO Task 15: convert per-stat -> fpts via scoring layer.
        # For now, stub fpts as the sum-of-stats with PPR-ish coefficients
        # so the schema validates. Replaced in Task 15.
        fpts_mean = self._stub_fpts_from_stats(per_stat_predictions, ruleset)

        # Assemble output frame.
        out = features[["gsis_id", "season", "position", "team"]].copy()
        out["ruleset"] = ruleset.name
        out["model_id"] = self.model_id

        # Degenerate distribution: mean == p10 == p50 == p90 for every stat
        # and for total fpts.
        for col, vals in per_stat_predictions.items():
            for q in ("mean", "p10", "p50", "p90"):
                out[f"{col}_{q}"] = vals
        for q in ("mean", "p10", "p50", "p90"):
            out[f"season_total_fpts_{q}"] = fpts_mean

        out = PreseasonProjectionSchema.validate(out)
        return out

    def _predict_per_stat(self, features: pd.DataFrame) -> dict[str, pd.Series]:
        """Return one Series per stat column, indexed identically to `features`,
        of predicted season totals. Veteran branch only in this task; Tasks
        12-14 add fallbacks + rookie branch."""
        # The 8 prior-1 per-game columns we might score.
        stats = (
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_yards",
            "rushing_tds",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        )
        result: dict[str, pd.Series] = {}
        for stat in stats:
            col_in = f"prior_1_season_per_game_{stat}"
            col_out = f"{stat}_season_total"
            if col_in not in features.columns:
                continue
            result[col_out] = (features[col_in] * _PROJECTED_GAMES_PLAYED).astype("float32")
        return result

    def _stub_fpts_from_stats(
        self, per_stat: dict[str, pd.Series], ruleset: Ruleset
    ) -> pd.Series:
        """Placeholder fpts computation — replaced by real scoring in Task 15."""
        fpts = pd.Series(0.0, index=next(iter(per_stat.values())).index, dtype="float32")
        for col, vals in per_stat.items():
            stat = col.replace("_season_total", "")
            if stat == "passing_yards":
                fpts = fpts + vals.fillna(0) / ruleset.passing_yds_per_pt
            elif stat == "passing_tds":
                fpts = fpts + vals.fillna(0) * ruleset.passing_td_pts
            elif stat == "passing_interceptions":
                fpts = fpts + vals.fillna(0) * ruleset.interception_pts
            elif stat == "rushing_yards":
                fpts = fpts + vals.fillna(0) / ruleset.rushing_yds_per_pt
            elif stat == "rushing_tds":
                fpts = fpts + vals.fillna(0) * ruleset.rushing_td_pts
            elif stat == "receptions":
                fpts = fpts + vals.fillna(0) * ruleset.reception_pts
            elif stat == "receiving_yards":
                fpts = fpts + vals.fillna(0) / ruleset.receiving_yds_per_pt
            elif stat == "receiving_tds":
                fpts = fpts + vals.fillna(0) * ruleset.receiving_td_pts
        return fpts.clip(lower=0).astype("float32")
```

This stubbed fpts gets replaced by `scoring.score_distribution` calls in Task 15. Until then it produces correct fpts for the veteran branch but isn't the canonical scoring path.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): NaivePreseasonModel veteran branch (prior_1 × 16)"
```

---

## Task 12: Fallback branch — prior_2 then prior_3

**Files:**
- Modify: `src/projections/preseason/model.py`
- Modify: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_naive_predict_fallback_to_prior_2() -> None:
    """Veteran missing prior_1 but has prior_2: falls through to prior_2."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_1_season_per_game_passing_yards=pd.array([pd.NA], dtype="float32"),
        prior_2_season_games_played=pd.array([14], dtype="Int64"),
        prior_2_season_per_game_passing_yards=pd.array([300.0], dtype="float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    # Falls back: 300 × 16 = 4800.
    assert float(out["passing_yards_season_total_mean"].iloc[0]) == pytest.approx(4800.0)


def test_naive_predict_fallback_to_prior_3() -> None:
    """Veteran missing prior_1 and prior_2: falls through to prior_3."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_2_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_3_season_games_played=pd.array([16], dtype="Int64"),
        prior_3_season_per_game_passing_yards=pd.array([250.0], dtype="float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert float(out["passing_yards_season_total_mean"].iloc[0]) == pytest.approx(4000.0)


def test_naive_predict_drops_player_with_all_priors_missing(caplog) -> None:
    """Veteran with no prior 1/2/3 history: dropped with warning."""
    features = _make_features_row(
        is_rookie=False,
        prior_1_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_2_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_3_season_games_played=pd.array([pd.NA], dtype="Int64"),
    )
    model = NaivePreseasonModel()
    with caplog.at_level(logging.WARNING):
        out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 0
    assert "no_prior_3_seasons" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_model.py -v -k "fallback or drops_player"`
Expected: FAIL on the new tests.

- [ ] **Step 3: Implement fallback chain**

In `src/projections/preseason/model.py`, replace `_predict_per_stat` with:

```python
    def _predict_per_stat(self, features: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """Per-stat predictions with prior_1 -> prior_2 -> prior_3 fallback for vets.

        Returns:
            (per_stat_series, retained_features)

        `retained_features` is `features` filtered down to rows that produced
        a prediction. Rows with no prior-1/2/3 anywhere are dropped with a
        WARNING.
        """
        stats = (
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_yards",
            "rushing_tds",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        )

        # For each row, pick the highest-priority prior that has games_played > 0.
        # Returns a Series of {1, 2, 3} per row (the "effective prior" tier).
        gp1 = features["prior_1_season_games_played"].fillna(0)
        gp2 = features["prior_2_season_games_played"].fillna(0)
        gp3 = features["prior_3_season_games_played"].fillna(0)
        effective_prior = pd.Series(0, index=features.index, dtype="int8")
        effective_prior = effective_prior.mask(gp3 > 0, 3)
        effective_prior = effective_prior.mask(gp2 > 0, 2)
        effective_prior = effective_prior.mask(gp1 > 0, 1)

        # Veterans whose effective_prior == 0 AND not rookie: drop.
        is_vet = ~features["is_rookie"].astype(bool)
        drop_mask = (effective_prior == 0) & is_vet
        if drop_mask.any():
            dropped_ids = features.loc[drop_mask, "gsis_id"].tolist()
            logger.warning(
                "NaivePreseasonModel: dropping %d veteran(s) with no_prior_3_seasons: %s",
                len(dropped_ids),
                dropped_ids[:5],
            )
        retained = features.loc[~drop_mask].copy()
        effective_prior = effective_prior.loc[retained.index]

        result: dict[str, pd.Series] = {}
        for stat in stats:
            # Compose by tier: for each row, pull `prior_{tier}_season_per_game_<stat>`.
            chosen = pd.Series(float("nan"), index=retained.index, dtype="float64")
            for tier in (1, 2, 3):
                tier_mask = effective_prior == tier
                if not tier_mask.any():
                    continue
                col = f"prior_{tier}_season_per_game_{stat}"
                chosen.loc[tier_mask] = retained.loc[tier_mask, col].astype("float64")
            result[f"{stat}_season_total"] = (chosen * _PROJECTED_GAMES_PLAYED).astype("float32")

        return result, retained
```

And update `predict_season_distribution` to consume the returned `retained` frame:

```python
    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        features = PreseasonFeaturesSchema.validate(features)
        per_stat_predictions, retained = self._predict_per_stat(features)
        fpts_mean = self._stub_fpts_from_stats(per_stat_predictions, ruleset)

        out = retained[["gsis_id", "season", "position", "team"]].copy()
        out["ruleset"] = ruleset.name
        out["model_id"] = self.model_id

        for col, vals in per_stat_predictions.items():
            for q in ("mean", "p10", "p50", "p90"):
                out[f"{col}_{q}"] = vals
        for q in ("mean", "p10", "p50", "p90"):
            out[f"season_total_fpts_{q}"] = fpts_mean

        out = PreseasonProjectionSchema.validate(out)
        return out
```

Add `import logging` if not already at top.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: 5 PASS (skeleton + veteran + 3 new).

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): naive fallback chain (prior_1 -> prior_2 -> prior_3)"
```

---

## Task 13: Rookie GLM — fit

**Files:**
- Modify: `src/projections/preseason/model.py`
- Modify: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_naive_fit_populates_rookie_glms() -> None:
    """fit() should train one GammaRegressor per (position, stat) cell."""
    # Synthetic 2-year training data — 4 QB rookies + 4 WR rookies.
    weekly = pd.DataFrame(
        [
            # 2021 QBs (rookie year):
            {"gsis_id": "00-3000001", "season": 2021, "week": 1, "position": "QB", "team": "KC",
             "passing_yards": 250.0, "passing_tds": 1, "interceptions": 1,
             "rushing_yards": 5.0, "rushing_tds": 0, "receptions": 0,
             "receiving_yards": 0.0, "receiving_tds": 0},
            {"gsis_id": "00-3000002", "season": 2021, "week": 1, "position": "QB", "team": "BUF",
             "passing_yards": 180.0, "passing_tds": 0, "interceptions": 2,
             "rushing_yards": 15.0, "rushing_tds": 0, "receptions": 0,
             "receiving_yards": 0.0, "receiving_tds": 0},
            # 2022 QBs:
            {"gsis_id": "00-3000003", "season": 2022, "week": 1, "position": "QB", "team": "DEN",
             "passing_yards": 300.0, "passing_tds": 2, "interceptions": 1,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 0,
             "receiving_yards": 0.0, "receiving_tds": 0},
            {"gsis_id": "00-3000004", "season": 2022, "week": 1, "position": "QB", "team": "NYJ",
             "passing_yards": 200.0, "passing_tds": 1, "interceptions": 1,
             "rushing_yards": 20.0, "rushing_tds": 0, "receptions": 0,
             "receiving_yards": 0.0, "receiving_tds": 0},
            # WRs:
            {"gsis_id": "00-3000005", "season": 2021, "week": 1, "position": "WR", "team": "DET",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 5,
             "receiving_yards": 60.0, "receiving_tds": 1},
            {"gsis_id": "00-3000006", "season": 2021, "week": 1, "position": "WR", "team": "PHI",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 3,
             "receiving_yards": 40.0, "receiving_tds": 0},
            {"gsis_id": "00-3000007", "season": 2022, "week": 1, "position": "WR", "team": "ATL",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 6,
             "receiving_yards": 85.0, "receiving_tds": 1},
            {"gsis_id": "00-3000008", "season": 2022, "week": 1, "position": "WR", "team": "JAC",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 4,
             "receiving_yards": 55.0, "receiving_tds": 0},
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")

    # Each rookie has a draft_picks row for their rookie year.
    draft = pd.DataFrame(
        [
            ("00-3000001", 2021, 1, 5),
            ("00-3000002", 2021, 2, 45),
            ("00-3000003", 2022, 1, 10),
            ("00-3000004", 2022, 3, 80),
            ("00-3000005", 2021, 1, 15),
            ("00-3000006", 2021, 4, 110),
            ("00-3000007", 2022, 1, 8),
            ("00-3000008", 2022, 2, 50),
        ],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": draft["gsis_id"], "full_name": "Test", "birth_date": pd.NaT})

    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)

    # Expect one (position, stat) entry per modeled cell.
    # QB stats: 5 (passing_yards, passing_tds, passing_interceptions, rushing_yards, rushing_tds).
    # WR stats: 5 (receptions, receiving_yards, receiving_tds, rushing_yards, rushing_tds).
    qb_keys = [k for k in model._rookie_glm if k[0] == "QB"]
    wr_keys = [k for k in model._rookie_glm if k[0] == "WR"]
    assert len(qb_keys) == 5
    assert len(wr_keys) == 5
    # Each GLM coefficient is a (intercept, slope) pair.
    intercept, slope = model._rookie_glm[("QB", "passing_yards")]
    assert isinstance(intercept, float)
    assert isinstance(slope, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_model.py::test_naive_fit_populates_rookie_glms -v`
Expected: NotImplementedError.

- [ ] **Step 3: Implement fit**

In `src/projections/preseason/model.py`, replace the `fit` method:

```python
import numpy as np  # add to imports if not present
from sklearn.linear_model import GammaRegressor  # add to imports
```

```python
    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None:
        """Fit per-(position, stat) Gamma GLMs on rookie-year season totals.

        Training: for each rookie (player with a draft_picks row in season S
        and a weekly_stats row in season S), aggregate to season total, then
        fit `log(stat + 1) ~ β₀ + β₁ · log(pick + 1)` via Gamma with log link.

        Stores coefficients in `self._rookie_glm[(position, stat)] = (intercept, slope)`.
        """
        # Per-position stats list — same as the features module.
        from projections.preseason.features import _STATS_BY_POSITION, _schema_stat_name

        # For each (rookie player), join weekly_stats[season = rookie_year] -> season totals.
        rookies = draft_picks.dropna(subset=["pick"]).copy()
        rookie_season = rookies[["gsis_id", "season", "pick"]].rename(
            columns={"season": "rookie_season"}
        )

        # Aggregate weekly_stats to per-player-per-season totals.
        season_totals = (
            weekly_stats.groupby(["gsis_id", "season", "position"])
            .agg(
                games_played=("week", "count"),
                passing_yards=("passing_yards", "sum"),
                passing_tds=("passing_tds", "sum"),
                interceptions=("interceptions", "sum"),
                rushing_yards=("rushing_yards", "sum"),
                rushing_tds=("rushing_tds", "sum"),
                receptions=("receptions", "sum"),
                receiving_yards=("receiving_yards", "sum"),
                receiving_tds=("receiving_tds", "sum"),
            )
            .reset_index()
        )

        # Join — keep only rookie-year rows.
        rookie_year_totals = season_totals.merge(
            rookie_season,
            left_on=["gsis_id", "season"],
            right_on=["gsis_id", "rookie_season"],
            how="inner",
        )

        self._rookie_glm.clear()
        for position, stats in _STATS_BY_POSITION.items():
            pos_rows = rookie_year_totals.loc[
                rookie_year_totals["position"] == position.value
            ].copy()
            if pos_rows.empty:
                logger.warning(
                    "NaivePreseasonModel.fit: no rookie training data for position=%s; "
                    "skipping. Rookies at this position will fall back to position-mean.",
                    position.value,
                )
                continue
            X = np.log(pos_rows["pick"].astype(float).to_numpy() + 1).reshape(-1, 1)
            for stat in stats:
                schema_stat = _schema_stat_name(stat)
                # Per-game stat → fitting on rookie-year per-game. We want season
                # total prediction at inference, so the GLM trains on per-game
                # × 16 = season total proxy. Multiply pretax to make the intercept
                # interpretable.
                y_raw = pos_rows[stat.value].astype(float).to_numpy()
                # Gamma requires strictly positive targets. Add epsilon for zeros.
                y = np.maximum(y_raw, 0.01)
                try:
                    reg = GammaRegressor(alpha=0.0, fit_intercept=True, max_iter=200)
                    reg.fit(X, y)
                    intercept = float(reg.intercept_)
                    slope = float(reg.coef_[0])
                except Exception as e:  # noqa: BLE001 — wide catch; sklearn raises various
                    logger.warning(
                        "GammaRegressor failed for (%s, %s): %s. "
                        "Falling back to log-mean-only.",
                        position.value, schema_stat, e,
                    )
                    intercept = float(np.log(y.mean()))
                    slope = 0.0
                self._rookie_glm[(position.value, schema_stat)] = (intercept, slope)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_preseason/test_model.py::test_naive_fit_populates_rookie_glms -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): rookie GLM fit per (position, stat)"
```

---

## Task 14: Rookie predict + UDFA imputation

**Files:**
- Modify: `src/projections/preseason/model.py`
- Modify: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_naive_predict_rookie_drafted_player() -> None:
    """Rookie with draft pick: predicted = exp(intercept + slope × log(pick + 1)) × 16/16
    (the GLM already trains on the rookie-year season total, not per-game)."""
    # First fit on synthetic data.
    weekly = pd.DataFrame(
        [
            {"gsis_id": "00-3000001", "season": 2021, "week": w, "position": "WR", "team": "KC",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 5.0,
             "receiving_yards": 70.0, "receiving_tds": 0.4}
            for w in range(1, 18)
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")

    draft = pd.DataFrame(
        [("00-3000001", 2021, 1, 10)],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": ["00-3000001"], "full_name": ["x"], "birth_date": [pd.NaT]})
    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)

    # Now predict a rookie.
    features = _make_features_row(
        gsis_id="00-4000001",
        position="WR",
        is_rookie=True,
        years_exp=pd.array([0], dtype="Int64"),
        draft_round=pd.array([1], dtype="Int64"),
        draft_pick_overall=pd.array([10], dtype="Int64"),
    )
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 1
    # Should produce a non-NaN receiving_yards prediction.
    assert out["receiving_yards_season_total_mean"].iloc[0] > 0


def test_naive_predict_rookie_udfa_imputed_to_pick_300() -> None:
    """UDFA rookie (no draft_pick_overall): imputed to pick=300 -> very late-round prediction."""
    weekly = pd.DataFrame(
        [
            {"gsis_id": "00-3000001", "season": 2021, "week": 1, "position": "WR", "team": "KC",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 4,
             "receiving_yards": 50.0, "receiving_tds": 0},
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")
    draft = pd.DataFrame(
        [("00-3000001", 2021, 1, 10)],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": ["00-3000001"], "full_name": ["x"], "birth_date": [pd.NaT]})

    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)

    # UDFA rookie: no draft_pick_overall.
    features = _make_features_row(
        gsis_id="00-5000001",
        position="WR",
        is_rookie=True,
        draft_round=pd.array([pd.NA], dtype="Int64"),
        draft_pick_overall=pd.array([pd.NA], dtype="Int64"),
    )
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 1
    # Should still produce a value (imputed pick=300).
    assert out["receiving_yards_season_total_mean"].iloc[0] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_model.py -v -k "rookie"`
Expected: FAIL — rookie branch isn't yet routed through GLM in `_predict_per_stat`.

- [ ] **Step 3: Implement rookie branch in _predict_per_stat**

In `src/projections/preseason/model.py`, modify `_predict_per_stat` to add a rookie-branch overlay:

```python
_UDFA_IMPUTED_PICK = 300


    def _predict_per_stat(self, features: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """As before, plus rookie-branch overlay using fitted GLMs."""
        stats = (
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_yards",
            "rushing_tds",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        )

        # Effective-prior tier for vets (same as Task 12).
        gp1 = features["prior_1_season_games_played"].fillna(0)
        gp2 = features["prior_2_season_games_played"].fillna(0)
        gp3 = features["prior_3_season_games_played"].fillna(0)
        effective_prior = pd.Series(0, index=features.index, dtype="int8")
        effective_prior = effective_prior.mask(gp3 > 0, 3)
        effective_prior = effective_prior.mask(gp2 > 0, 2)
        effective_prior = effective_prior.mask(gp1 > 0, 1)

        is_vet = ~features["is_rookie"].astype(bool)
        is_rookie = features["is_rookie"].astype(bool)
        drop_mask = (effective_prior == 0) & is_vet
        if drop_mask.any():
            dropped_ids = features.loc[drop_mask, "gsis_id"].tolist()
            logger.warning(
                "NaivePreseasonModel: dropping %d veteran(s) with no_prior_3_seasons: %s",
                len(dropped_ids),
                dropped_ids[:5],
            )
        retained = features.loc[~drop_mask].copy()
        effective_prior = effective_prior.loc[retained.index]
        is_rookie = is_rookie.loc[retained.index]

        # Imputed pick = 300 for UDFAs.
        pick = retained["draft_pick_overall"].fillna(_UDFA_IMPUTED_PICK).astype(float)
        log_pick = np.log(pick + 1)

        result: dict[str, pd.Series] = {}
        for stat in stats:
            chosen = pd.Series(float("nan"), index=retained.index, dtype="float64")
            # Veteran prior-N fallback.
            for tier in (1, 2, 3):
                tier_mask = (effective_prior == tier) & ~is_rookie
                if not tier_mask.any():
                    continue
                col = f"prior_{tier}_season_per_game_{stat}"
                chosen.loc[tier_mask] = (
                    retained.loc[tier_mask, col].astype("float64") * _PROJECTED_GAMES_PLAYED
                )
            # Rookie GLM overlay.
            if is_rookie.any():
                for position in retained.loc[is_rookie, "position"].unique():
                    pos_mask = is_rookie & (retained["position"] == position)
                    if not pos_mask.any():
                        continue
                    key = (position, stat)
                    if key not in self._rookie_glm:
                        chosen.loc[pos_mask] = 0.0
                        continue
                    intercept, slope = self._rookie_glm[key]
                    chosen.loc[pos_mask] = np.exp(intercept + slope * log_pick.loc[pos_mask])
            result[f"{stat}_season_total"] = chosen.fillna(0.0).clip(lower=0.0).astype("float32")
        return result, retained
```

Note: the GLM was fitted on rookie-year *season totals*, so prediction is `exp(...)` directly — no `× 16` multiplication. The veteran branch keeps the `× _PROJECTED_GAMES_PLAYED` because it uses per-game inputs.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): rookie GLM predict branch + UDFA imputation"
```

---

## Task 15: Wire scoring layer (replace fpts stub)

**Files:**
- Modify: `src/projections/preseason/model.py`
- Modify: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_naive_predict_fpts_uses_scoring_layer() -> None:
    """fpts should be computed via projections.scoring (canonical scoring path),
    not a local re-implementation."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([17], dtype="Int64"),
        prior_1_season_per_game_passing_yards=pd.array([250.0], dtype="float32"),
        prior_1_season_per_game_passing_tds=pd.array([2.0], dtype="float32"),
        prior_1_season_per_game_passing_interceptions=pd.array([1.0], dtype="float32"),
        prior_1_season_per_game_rushing_yards=pd.array([10.0], dtype="float32"),
        prior_1_season_per_game_rushing_tds=pd.array([0.1], dtype="float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    # Compute expected fpts via the scoring layer's coefficients:
    # passing_yards / 25  = 250 × 16 / 25  = 160 pts
    # passing_tds × 4     = 2.0 × 16 × 4   = 128 pts
    # interceptions × -2  = 1.0 × 16 × -2  = -32 pts
    # rushing_yards / 10  = 10 × 16 / 10   = 16 pts
    # rushing_tds × 6     = 0.1 × 16 × 6   = 9.6 pts
    # Total: 281.6 pts
    expected = 250 * 16 / 25 + 2 * 16 * 4 + 1 * 16 * -2 + 10 * 16 / 10 + 0.1 * 16 * 6
    assert float(out["season_total_fpts_mean"].iloc[0]) == pytest.approx(expected, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails on the stub**

Run: `pytest tests/test_preseason/test_model.py::test_naive_predict_fpts_uses_scoring_layer -v`
Expected: likely passes with the stub today (the stub uses the ruleset coefficients identically). The test exists to lock in the canonical-scoring-path constraint when we refactor.

- [ ] **Step 3: Replace the stub with scoring.score_distribution**

Reuse `scoring.score_distribution` directly. Since v1.0 naive uses degenerate distributions, the simplest path is to construct a deterministic `Distribution` for each predicted stat and feed to `score_distribution`. But that's overkill for point masses. Alternative: import the coefficient map from scoring and apply it directly — the function `_scoring_coefficients` already exists in `src/projections/scoring/score_distribution.py`.

Strategy: extract `_scoring_coefficients` to a public name (or import as-is by relative path) and use it. Add this minimal export to `src/projections/scoring/__init__.py` if not present:

```python
from projections.scoring.score_distribution import _scoring_coefficients as scoring_coefficients
```

Then in `model.py`:

```python
from projections.scoring import scoring_coefficients  # add to top imports
from projections.schemas import Stat

# Mapping schema-stat-name -> Stat enum for the scoring layer.
_STAT_BY_SCHEMA_NAME: dict[str, Stat] = {
    "passing_yards": Stat.PASSING_YARDS,
    "passing_tds": Stat.PASSING_TDS,
    "passing_interceptions": Stat.INTERCEPTIONS,
    "rushing_yards": Stat.RUSHING_YARDS,
    "rushing_tds": Stat.RUSHING_TDS,
    "receptions": Stat.RECEPTIONS,
    "receiving_yards": Stat.RECEIVING_YARDS,
    "receiving_tds": Stat.RECEIVING_TDS,
}


    def _compute_fpts_from_stats(
        self, per_stat: dict[str, pd.Series], ruleset: Ruleset
    ) -> pd.Series:
        """Vectorized version of the canonical scoring map. Degenerate per-stat
        distributions = scalar means, so we compute total fpts as a linear
        combination using the ruleset's coefficient map."""
        coef_map = scoring_coefficients(ruleset)
        fpts = pd.Series(0.0, index=next(iter(per_stat.values())).index, dtype="float32")
        for col, vals in per_stat.items():
            stat_name = col.replace("_season_total", "")
            stat = _STAT_BY_SCHEMA_NAME[stat_name]
            coef = coef_map.get(stat, 0.0)
            fpts = fpts + vals.fillna(0).astype("float64") * coef
        return fpts.clip(lower=0).astype("float32")
```

Replace the call site `self._stub_fpts_from_stats(...)` with `self._compute_fpts_from_stats(...)` and delete the stub function.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py src/projections/scoring/__init__.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): canonical fpts via projections.scoring (no duplicated math)"
```

---

## Task 16: Save / load via joblib

**Files:**
- Modify: `src/projections/preseason/model.py`
- Modify: `tests/test_preseason/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
import joblib  # add to top of test_model.py


def test_naive_model_save_and_load_roundtrip(tmp_path) -> None:
    weekly = pd.DataFrame(
        [
            {"gsis_id": "00-3000001", "season": 2021, "week": 1, "position": "WR", "team": "KC",
             "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
             "rushing_yards": 0.0, "rushing_tds": 0, "receptions": 5,
             "receiving_yards": 70.0, "receiving_tds": 1},
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")
    draft = pd.DataFrame(
        [("00-3000001", 2021, 1, 10)],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": ["00-3000001"], "full_name": ["x"], "birth_date": [pd.NaT]})

    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)
    path = tmp_path / "naive-preseason-test.joblib"
    model.save(path)
    assert path.exists()

    reloaded = NaivePreseasonModel.load(path)
    assert reloaded.model_id == model.model_id
    assert reloaded._rookie_glm == model._rookie_glm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_model.py::test_naive_model_save_and_load_roundtrip -v`
Expected: NotImplementedError.

- [ ] **Step 3: Implement save/load**

In `src/projections/preseason/model.py`:

```python
import joblib  # add to imports


    def save(self, path: Path) -> None:
        """Persist via joblib. Stores the rookie GLM coefficients dict."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model_id": self.model_id, "rookie_glm": self._rookie_glm},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "NaivePreseasonModel":
        state = joblib.load(path)
        m = cls()
        if state.get("model_id") != cls.model_id:
            raise ValueError(
                f"Artifact model_id={state.get('model_id')!r} does not match "
                f"NaivePreseasonModel.model_id={cls.model_id!r}"
            )
        m._rookie_glm = state["rookie_glm"]
        return m
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_model.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Phase-end verification**

```bash
pytest -v -k "preseason or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/model.py tests/test_preseason/test_model.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): NaivePreseasonModel save/load via joblib"
```

**Phase 3 done. STOP and request user approval before proceeding to Phase 4.**

---

# Phase 4 — Project driver + CLI

End-to-end function + script that produces `data/projections/preseason/season=<Y>/ruleset=<R>/part.parquet`.

## Task 17: `project_preseason` driver

**Files:**
- Create: `src/projections/preseason/project.py`

- [ ] **Step 1: Implement driver**

Create `src/projections/preseason/project.py`:

```python
"""End-to-end preseason projection driver.

Reads raw inputs, builds features, fits the model, predicts, writes the
parquet partition, and returns the in-memory frame for downstream use.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from projections.preseason.features import build_preseason_features
from projections.preseason.model import NaivePreseasonModel, PreseasonModel
from projections.schemas import (
    PreseasonProjectionSchema,
    Ruleset,
)
from projections.store import read_partition, write_partition

logger = logging.getLogger(__name__)


def project_preseason(
    *,
    raw_root: Path,
    projections_root: Path,
    target_season: int,
    train_start: int,
    ruleset: Ruleset,
    model: PreseasonModel | None = None,
    dropped_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Run the end-to-end preseason pipeline. Returns the projection frame."""
    # 1. Read raw inputs.
    weekly_stats_frames = []
    for s in range(train_start, target_season):
        try:
            weekly_stats_frames.append(read_partition(raw_root, "weekly_stats", season=s))
        except FileNotFoundError:
            logger.warning("weekly_stats season=%d missing; skipping in training window", s)
    if not weekly_stats_frames:
        raise FileNotFoundError(
            f"No weekly_stats partitions found under {raw_root} for "
            f"seasons {train_start}..{target_season - 1}."
        )
    weekly_stats = pd.concat(weekly_stats_frames, ignore_index=True)

    try:
        depth_charts_target = read_partition(raw_root, "depth_charts", season=target_season)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"depth_charts season={target_season} not found at "
            f"{raw_root}/depth_charts/season={target_season}/. "
            f"Run refresh_depth_charts([{target_season}]) first."
        ) from e

    draft_picks_frames = []
    for s in range(1980, target_season + 1):
        try:
            draft_picks_frames.append(read_partition(raw_root, "draft_picks", season=s))
        except FileNotFoundError:
            continue
    draft_picks = (
        pd.concat(draft_picks_frames, ignore_index=True)
        if draft_picks_frames
        else pd.DataFrame()
    )

    id_map = read_partition(raw_root, "id_map")

    # 2. Build features.
    features = build_preseason_features(
        weekly_stats=weekly_stats,
        depth_charts_target=depth_charts_target,
        draft_picks=draft_picks,
        id_map=id_map,
        target_season=target_season,
        dropped_csv_path=dropped_csv_path,
    )

    # 3. Fit + predict.
    if model is None:
        model = NaivePreseasonModel()
        model.fit(weekly_stats=weekly_stats, draft_picks=draft_picks, id_map=id_map)
    projections = model.predict_season_distribution(features, ruleset=ruleset)

    # 4. Validate + write.
    projections = PreseasonProjectionSchema.validate(projections)
    table = f"preseason/ruleset={ruleset.name}"
    target = write_partition(
        projections_root,
        table,
        projections,
        season=target_season,
        week=None,
    )
    logger.info("project_preseason: wrote %d rows -> %s", len(projections), target)
    return projections
```

Note: `write_partition` expects `week: int | None`; passing `None` produces a season-only partition `season=<Y>/part.parquet`. The caller picks `table="preseason/ruleset=<R>"` so the final path is `data/projections/preseason/ruleset=<R>/season=<Y>/part.parquet`.

- [ ] **Step 2: Verify import is clean**

Run: `python -c "from projections.preseason.project import project_preseason"`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/project.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): project_preseason driver"
```

---

## Task 18: CLI — `scripts/preseason_project_season.py`

**Files:**
- Create: `scripts/preseason_project_season.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/preseason_project_season.py`:

```python
"""CLI: produce preseason season-total projections for a target season.

Usage:
    python scripts/preseason_project_season.py --season 2026
    python scripts/preseason_project_season.py --season 2026 --ruleset espn_half
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from projections.preseason.project import project_preseason
from projections.schemas import Ruleset
from projections.store import read_partition


def _summary_csv(projections: pd.DataFrame, raw_root: Path, out_path: Path) -> None:
    """Write a human-readable top-N-per-position summary CSV."""
    id_map = read_partition(raw_root, "id_map")
    summary = projections.merge(
        id_map[["gsis_id", "full_name"]],
        on="gsis_id",
        how="left",
    )
    summary = summary[
        ["gsis_id", "full_name", "position", "team", "season_total_fpts_mean", "model_id"]
    ].sort_values("season_total_fpts_mean", ascending=False)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="Target preseason year (e.g., 2026)")
    parser.add_argument(
        "--ruleset",
        choices=["espn_ppr", "espn_half", "standard"],
        default="espn_ppr",
    )
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    ruleset_map = {
        "espn_ppr": Ruleset.espn_ppr(),
        "espn_half": Ruleset.espn_half(),
        "standard": Ruleset.standard(),
    }
    ruleset = ruleset_map[args.ruleset]

    dropped_path = args.reports_root / f"preseason_dropped_{args.season}.csv"
    projections = project_preseason(
        raw_root=args.raw_root,
        projections_root=args.projections_root,
        target_season=args.season,
        train_start=args.train_start,
        ruleset=ruleset,
        dropped_csv_path=dropped_path,
    )

    if not args.no_summary:
        summary_path = args.reports_root / f"preseason_{args.season}.csv"
        _summary_csv(projections, args.raw_root, summary_path)
        print(f"Wrote summary -> {summary_path}")

    print(f"Done. {len(projections)} projections written for season={args.season} ruleset={ruleset.name}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test invocation help text**

Run: `python scripts/preseason_project_season.py --help`
Expected: prints the help, exits 0.

- [ ] **Step 3: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add scripts/preseason_project_season.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(scripts): preseason_project_season CLI driver"
```

---

## Task 19: CLI integration test

**Files:**
- Create: `tests/test_scripts/test_preseason_project_season_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scripts/test_preseason_project_season_cli.py`:

```python
"""Happy-path integration test for scripts/preseason_project_season.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from projections.schemas import PreseasonProjectionSchema
from projections.store import write_partition


def _seed_minimal_data(raw_root: Path, target_season: int = 2024) -> None:
    """Seed minimal raw partitions sufficient for a 1-player projection."""
    # weekly_stats: one veteran with 1 game in 2023 (prior_1).
    weekly = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2023,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "opponent": "BUF",
                "passing_yards": 250.0,
                "passing_tds": 2,
                "interceptions": 1,
                "attempts": 30,
                "completions": 22,
                "sacks": 2,
                "rushing_yards": 30.0,
                "rushing_tds": 0,
                "carries": 5,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        ]
    )
    write_partition(raw_root, "weekly_stats", weekly, season=2023, week=None)

    # depth_charts for target_season.
    depth = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": target_season,
                "week": 1,
                "team": "KC",
                "position": "QB",
                "depth_team": "QB1",
                "depth_rank": 1,
            }
        ]
    )
    write_partition(raw_root, "depth_charts", depth, season=target_season, week=None)

    # draft_picks 2017 (rookie year for the veteran).
    picks = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2017,
                "round": 1,
                "pick": 10,
            }
        ]
    )
    write_partition(raw_root, "draft_picks", picks, season=2017, week=None)

    # id_map (unpartitioned).
    id_map = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "full_name": "Patrick Mahomes",
                "birth_date": pd.Timestamp("1995-09-17"),
                "team": "KC",
                "espn_id": pd.NA,
                "sleeper_id": pd.NA,
                "pfr_id": pd.NA,
            }
        ]
    )
    write_partition(raw_root, "id_map", id_map, season=None, week=None)


def test_preseason_project_season_cli_happy_path(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    proj_root = tmp_path / "projections"
    reports_root = tmp_path / "reports"
    _seed_minimal_data(raw_root, target_season=2024)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preseason_project_season.py",
            "--season", "2024",
            "--ruleset", "espn_ppr",
            "--raw-root", str(raw_root),
            "--projections-root", str(proj_root),
            "--reports-root", str(reports_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Output partition exists + validates.
    out_path = proj_root / "preseason" / "ruleset=ESPN_PPR" / "season=2024" / "part.parquet"
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    df = PreseasonProjectionSchema.validate(df)
    assert len(df) == 1

    # Summary CSV present.
    assert (reports_root / "preseason_2024.csv").exists()
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_scripts/test_preseason_project_season_cli.py -v`
Expected: PASS.

- [ ] **Step 3: Phase-end verification**

```bash
pytest -v -k "preseason or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add tests/test_scripts/test_preseason_project_season_cli.py
PATH=".venv/Scripts:$PATH" git commit -m "test(scripts): preseason_project_season CLI happy-path integration"
```

**Phase 4 done. STOP and request user approval before proceeding to Phase 5.**

---

# Phase 5 — Backtest harness

## Task 20: `walk_forward_backtest` skeleton + metric computation

**Files:**
- Create: `src/projections/preseason/backtest.py`
- Create: `tests/test_preseason/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preseason/test_backtest.py`:

```python
"""Tests for src/projections/preseason/backtest.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.preseason.backtest import (
    compute_rmse_and_spearman,
    determine_verdict,
)


def test_compute_rmse_zero_when_predicted_equals_actual() -> None:
    predicted = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "season_total_fpts_mean": [100.0, 200.0]}
    )
    actual = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "actual_season_total_fpts": [100.0, 200.0]}
    )
    rmse, spearman, n = compute_rmse_and_spearman(
        predicted=predicted, actual=actual, top_n=10
    )
    assert rmse == pytest.approx(0.0)
    assert spearman == pytest.approx(1.0)
    assert n == 2


def test_compute_rmse_known_value() -> None:
    predicted = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "season_total_fpts_mean": [100.0, 150.0]}
    )
    actual = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "actual_season_total_fpts": [110.0, 140.0]}
    )
    rmse, _, _ = compute_rmse_and_spearman(predicted=predicted, actual=actual, top_n=10)
    # sqrt(((10^2) + (10^2)) / 2) = sqrt(100) = 10
    assert rmse == pytest.approx(10.0)


def test_determine_verdict_adopt() -> None:
    assert determine_verdict(rmse_delta_pct=-5.0, spearman_top50=0.75) == "ADOPT"


def test_determine_verdict_do_not_adopt_worse_rmse() -> None:
    assert determine_verdict(rmse_delta_pct=2.0, spearman_top50=0.75) == "DO_NOT_ADOPT"


def test_determine_verdict_do_not_adopt_bad_spearman() -> None:
    assert determine_verdict(rmse_delta_pct=-5.0, spearman_top50=0.40) == "DO_NOT_ADOPT"


def test_determine_verdict_null_band() -> None:
    assert determine_verdict(rmse_delta_pct=-1.0, spearman_top50=0.60) == "NULL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_backtest.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/projections/preseason/backtest.py`:

```python
"""Preseason backtest harness — walk-forward eval over a list of target seasons.

Produces:
- A PreseasonBacktestSchema-validated CSV at reports/backtest_preseason_<model>.csv.
- A human-readable markdown report at reports/backtest_preseason_<model>.md.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §7.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.preseason.project import project_preseason
from projections.schemas import (
    PreseasonBacktestSchema,
    Position,
    Ruleset,
)
from projections.scoring import scoring_coefficients
from projections.store import read_partition

logger = logging.getLogger(__name__)


def compute_rmse_and_spearman(
    *,
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    top_n: int,
) -> tuple[float, float, int]:
    """Compute RMSE on full inner-join + Spearman on top-N actuals.

    `predicted` requires columns: gsis_id, season_total_fpts_mean.
    `actual` requires columns: gsis_id, actual_season_total_fpts.

    Returns:
        (rmse, spearman_top_n, n_players)
    """
    merged = predicted.merge(actual, on="gsis_id", how="inner")
    if merged.empty:
        return float("nan"), float("nan"), 0

    err = merged["season_total_fpts_mean"] - merged["actual_season_total_fpts"]
    rmse = float(np.sqrt((err ** 2).mean()))

    top_actual = merged.nlargest(top_n, "actual_season_total_fpts")
    if len(top_actual) < 2:
        spearman = float("nan")
    else:
        rho, _ = spearmanr(
            top_actual["actual_season_total_fpts"].to_numpy(),
            top_actual["season_total_fpts_mean"].to_numpy(),
        )
        spearman = float(rho)
    return rmse, spearman, len(merged)


def determine_verdict(
    *, rmse_delta_pct: float, spearman_top50: float
) -> Literal["ADOPT", "NULL", "DO_NOT_ADOPT"]:
    """Apply the per-cell verdict logic from spec §7.3."""
    if rmse_delta_pct < 0 and spearman_top50 >= 0.70:
        return "ADOPT"
    if rmse_delta_pct >= 0 or spearman_top50 < 0.50:
        return "DO_NOT_ADOPT"
    return "NULL"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_backtest.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/backtest.py tests/test_preseason/test_backtest.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): backtest metric computation + verdict logic"
```

---

## Task 21: `walk_forward_backtest` — full eval loop

**Files:**
- Modify: `src/projections/preseason/backtest.py`
- Modify: `tests/test_preseason/test_backtest.py`

- [ ] **Step 1: Write the failing test**

```python
def test_walk_forward_backtest_returns_one_row_per_position_per_year(
    tmp_path: Path,
) -> None:
    """End-to-end smoke test: walk_forward_backtest produces 1 row per
    (target_season, position) cell."""
    from tests.test_scripts.test_preseason_project_season_cli import _seed_minimal_data

    raw_root = tmp_path / "raw"
    proj_root = tmp_path / "projections"
    _seed_minimal_data(raw_root, target_season=2024)

    # Also seed 2024 weekly_stats to serve as "actuals" for the 2024 holdout.
    actual_2024 = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2024,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "opponent": "BUF",
                "passing_yards": 260.0,
                "passing_tds": 2,
                "interceptions": 1,
                "attempts": 30,
                "completions": 22,
                "sacks": 2,
                "rushing_yards": 30.0,
                "rushing_tds": 0,
                "carries": 5,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        ]
    )
    from projections.store import write_partition
    write_partition(raw_root, "weekly_stats", actual_2024, season=2024, week=None)

    from projections.preseason.backtest import walk_forward_backtest

    out = walk_forward_backtest(
        raw_root=raw_root,
        projections_root=proj_root,
        target_seasons=[2024],
        train_start=2018,
        ruleset=Ruleset.espn_ppr(),
    )
    # 1 target season × 1 (only QB had data) = 1 cell.
    assert len(out) >= 1
    assert "verdict" in out.columns
    PreseasonBacktestSchema.validate(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseason/test_backtest.py -v -k "walk_forward"`
Expected: ImportError.

- [ ] **Step 3: Implement walk_forward_backtest**

Append to `src/projections/preseason/backtest.py`:

```python
def _aggregate_actuals(
    weekly_stats: pd.DataFrame, *, ruleset: Ruleset, season: int
) -> pd.DataFrame:
    """Convert weekly_stats[season=Y] to per-player actual season-total fpts.

    Returns: DataFrame with columns gsis_id, position, actual_season_total_fpts.
    """
    season_rows = weekly_stats.loc[weekly_stats["season"] == season].copy()
    if season_rows.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.Series([], dtype="string[pyarrow]"),
                "position": pd.Series([], dtype="string[pyarrow]"),
                "actual_season_total_fpts": pd.Series([], dtype="float32"),
            }
        )

    from projections.schemas import Stat
    coef = scoring_coefficients(ruleset)
    fpts = pd.Series(0.0, index=season_rows.index, dtype="float64")
    for stat, val in coef.items():
        col = stat.value
        if col in season_rows.columns:
            fpts = fpts + season_rows[col].fillna(0).astype("float64") * val

    season_rows["weekly_fpts"] = fpts
    agg = (
        season_rows.groupby(["gsis_id", "position"], as_index=False)["weekly_fpts"]
        .sum()
        .rename(columns={"weekly_fpts": "actual_season_total_fpts"})
    )
    agg["actual_season_total_fpts"] = agg["actual_season_total_fpts"].clip(lower=0).astype("float32")
    return agg


def walk_forward_backtest(
    *,
    raw_root: Path,
    projections_root: Path,
    target_seasons: list[int],
    train_start: int,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Run walk-forward eval over `target_seasons`.

    For each target_season:
      1. project_preseason(target_season) using train_start..target_season-1.
      2. Aggregate weekly_stats[season=target_season] to actuals.
      3. Inner-join, compute per-(position) RMSE + Spearman + coverage diff.
      4. Compute rmse_naive_baseline (vs prior_1 × 16).
      5. Build verdict.

    Returns a PreseasonBacktestSchema-validated frame.
    """
    rows = []
    for target_season in target_seasons:
        projections = project_preseason(
            raw_root=raw_root,
            projections_root=projections_root,
            target_season=target_season,
            train_start=train_start,
            ruleset=ruleset,
        )

        actuals_weekly = read_partition(raw_root, "weekly_stats", season=target_season)
        # Ensure dtype mirrors the load-time minimum the actuals reducer needs.
        actuals_weekly["season"] = actuals_weekly["season"].astype("int32")

        actuals = _aggregate_actuals(actuals_weekly, ruleset=ruleset, season=target_season)

        # Naive baseline: prior_1_per_game × 16, computed straight from
        # weekly_stats[target_season-1] aggregated. We re-derive it locally
        # so the gate doesn't depend on which model is under test.
        prior_weekly = read_partition(raw_root, "weekly_stats", season=target_season - 1)
        naive_per_game = (
            prior_weekly.groupby(["gsis_id", "position"], as_index=False)
            .agg(games_played=("week", "count"),
                 fpts_total=(prior_weekly.columns[0], "count"))  # placeholder
        )
        # Reuse _aggregate_actuals shape — naive = (prior_season per-game × 16).
        naive_actuals = _aggregate_actuals(
            prior_weekly, ruleset=ruleset, season=target_season - 1
        )
        naive_actuals["games_played"] = (
            prior_weekly.groupby("gsis_id")["week"].count().reindex(
                naive_actuals["gsis_id"]
            ).to_numpy()
        )
        naive_actuals["season_total_fpts_mean"] = (
            naive_actuals["actual_season_total_fpts"]
            / naive_actuals["games_played"]
            * 16
        ).astype("float32")
        naive_actuals = naive_actuals[["gsis_id", "position", "season_total_fpts_mean"]]

        # Per-position metric loop.
        for position in (Position.QB, Position.RB, Position.WR, Position.TE):
            pred_pos = projections.loc[projections["position"] == position.value]
            actual_pos = actuals.loc[actuals["position"] == position.value]
            naive_pos = naive_actuals.loc[naive_actuals["position"] == position.value]

            rmse, spearman_top50, n_players = compute_rmse_and_spearman(
                predicted=pred_pos, actual=actual_pos, top_n=50
            )
            rmse_naive, _, _ = compute_rmse_and_spearman(
                predicted=naive_pos, actual=actual_pos, top_n=50
            )
            rmse_delta_pct = (
                float("nan")
                if rmse_naive == 0 or np.isnan(rmse_naive)
                else (rmse - rmse_naive) / rmse_naive * 100
            )

            projected_not_played = int(
                len(pred_pos) - len(pred_pos.merge(actual_pos, on="gsis_id", how="inner"))
            )
            played_not_projected = int(
                len(actual_pos) - len(actual_pos.merge(pred_pos, on="gsis_id", how="inner"))
            )

            verdict = "NULL"  # default when nan; replaced below if metrics valid.
            if not np.isnan(rmse_delta_pct) and not np.isnan(spearman_top50):
                verdict = determine_verdict(
                    rmse_delta_pct=rmse_delta_pct, spearman_top50=spearman_top50
                )

            rows.append(
                {
                    "target_season": target_season,
                    "position": position.value,
                    "model_class": "naive-preseason-v1",
                    "ruleset": ruleset.name,
                    "rmse": float(rmse) if not np.isnan(rmse) else 0.0,
                    "rmse_naive_baseline": float(rmse_naive) if not np.isnan(rmse_naive) else 0.0,
                    "rmse_delta_pct": (
                        float(rmse_delta_pct) if not np.isnan(rmse_delta_pct) else 0.0
                    ),
                    "spearman_top50": (
                        float(spearman_top50) if not np.isnan(spearman_top50) else 0.0
                    ),
                    "n_players": n_players,
                    "coverage_diff_projected_not_played": projected_not_played,
                    "coverage_diff_played_not_projected": played_not_projected,
                    "verdict": verdict,
                }
            )

    out = pd.DataFrame(rows)
    # Cast to schema dtypes.
    out = out.astype(
        {
            "target_season": "int32",
            "rmse": "float32",
            "rmse_naive_baseline": "float32",
            "rmse_delta_pct": "float32",
            "spearman_top50": "float32",
            "n_players": "Int64",
            "coverage_diff_projected_not_played": "Int64",
            "coverage_diff_played_not_projected": "Int64",
        }
    )
    out["model_class"] = out["model_class"].astype("string[pyarrow]")

    out = PreseasonBacktestSchema.validate(out)
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_backtest.py -v -k "walk_forward"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/backtest.py tests/test_preseason/test_backtest.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): walk_forward_backtest end-to-end loop"
```

---

## Task 22: Markdown report writer

**Files:**
- Modify: `src/projections/preseason/backtest.py`
- Modify: `tests/test_preseason/test_backtest.py`

- [ ] **Step 1: Write the failing test**

```python
def test_write_backtest_report_produces_markdown(tmp_path: Path) -> None:
    from projections.preseason.backtest import write_backtest_report

    backtest_df = pd.DataFrame(
        {
            "target_season": pd.array([2024], dtype="int32"),
            "position": ["QB"],
            "model_class": pd.array(["naive-preseason-v1"], dtype="string[pyarrow]"),
            "ruleset": ["ESPN_PPR"],
            "rmse": pd.array([35.0], dtype="float32"),
            "rmse_naive_baseline": pd.array([35.0], dtype="float32"),
            "rmse_delta_pct": pd.array([0.0], dtype="float32"),
            "spearman_top50": pd.array([0.72], dtype="float32"),
            "n_players": pd.array([28], dtype="Int64"),
            "coverage_diff_projected_not_played": pd.array([3], dtype="Int64"),
            "coverage_diff_played_not_projected": pd.array([1], dtype="Int64"),
            "verdict": ["NULL"],
        }
    )
    out_path = tmp_path / "backtest_report.md"
    write_backtest_report(backtest_df, out_path)
    text = out_path.read_text()
    assert "Backtest Report" in text
    assert "naive-preseason-v1" in text
    assert "QB" in text
    assert "NULL" in text
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_preseason/test_backtest.py -v -k "report"`
Expected: ImportError.

- [ ] **Step 3: Implement write_backtest_report**

```python
def write_backtest_report(backtest_df: pd.DataFrame, path: Path) -> None:
    """Render a PreseasonBacktestSchema frame as a markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preseason Backtest Report",
        "",
        f"**Model class:** {backtest_df['model_class'].iloc[0]}  ",
        f"**Ruleset:** {backtest_df['ruleset'].iloc[0]}  ",
        f"**Target seasons:** {sorted(set(backtest_df['target_season'].tolist()))}  ",
        "",
        "## Per-cell verdicts",
        "",
        "| target_season | position | rmse | rmse_naive | rmse_delta_pct | spearman_top50 | n_players | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in backtest_df.iterrows():
        lines.append(
            f"| {row['target_season']} | {row['position']} | {row['rmse']:.2f} | "
            f"{row['rmse_naive_baseline']:.2f} | {row['rmse_delta_pct']:+.2f}% | "
            f"{row['spearman_top50']:.3f} | {row['n_players']} | "
            f"**{row['verdict']}** |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- ADOPT cells:        {(backtest_df['verdict'] == 'ADOPT').sum()}",
            f"- NULL cells:         {(backtest_df['verdict'] == 'NULL').sum()}",
            f"- DO_NOT_ADOPT cells: {(backtest_df['verdict'] == 'DO_NOT_ADOPT').sum()}",
            "",
            "## Coverage diff",
            "",
            "| target_season | position | projected_not_played | played_not_projected |",
            "|---|---|---|---|",
        ]
    )
    for _, row in backtest_df.iterrows():
        lines.append(
            f"| {row['target_season']} | {row['position']} | "
            f"{row['coverage_diff_projected_not_played']} | "
            f"{row['coverage_diff_played_not_projected']} |"
        )
    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preseason/test_backtest.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/preseason/backtest.py tests/test_preseason/test_backtest.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(preseason): backtest markdown report writer"
```

**Known v1.0 gap, deferred to v1.1:** spec §7.6 also calls for per-position top-20 predicted-vs-actual spot-check tables and player-name coverage-diff sidebars. The current `write_backtest_report` ships the verdict table + per-cell metric table + coverage-diff counts, which is sufficient for the gate. The spot-check tables require threading `predictions` and `actuals` frames through `walk_forward_backtest`'s return; deferred to keep the v1.0 PR bounded. Track as a v1.1 follow-up.

---

## Task 23: CLI — `scripts/backtest_preseason.py` + integration test

**Files:**
- Create: `scripts/backtest_preseason.py`
- Create: `tests/test_scripts/test_backtest_preseason_cli.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/backtest_preseason.py`:

```python
"""CLI: run preseason backtest harness and emit verdict report.

Usage:
    python scripts/backtest_preseason.py --model naive-preseason --target-seasons 2024,2025
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from projections.preseason.backtest import walk_forward_backtest, write_backtest_report
from projections.schemas import Ruleset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="naive-preseason", help="Model class name (info only)")
    parser.add_argument(
        "--target-seasons",
        required=True,
        help="Comma-separated list, e.g. 2024,2025",
    )
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument(
        "--ruleset",
        choices=["espn_ppr", "espn_half", "standard"],
        default="espn_ppr",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    args = parser.parse_args()

    ruleset_map = {
        "espn_ppr": Ruleset.espn_ppr(),
        "espn_half": Ruleset.espn_half(),
        "standard": Ruleset.standard(),
    }
    ruleset = ruleset_map[args.ruleset]
    target_seasons = [int(s) for s in args.target_seasons.split(",")]

    backtest = walk_forward_backtest(
        raw_root=args.raw_root,
        projections_root=args.projections_root,
        target_seasons=target_seasons,
        train_start=args.train_start,
        ruleset=ruleset,
    )

    csv_path = args.reports_root / f"backtest_preseason_{args.model}.csv"
    md_path = args.reports_root / f"backtest_preseason_{args.model}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(csv_path, index=False)
    write_backtest_report(backtest, md_path)
    print(f"Wrote CSV -> {csv_path}")
    print(f"Wrote markdown -> {md_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the CLI integration test**

Create `tests/test_scripts/test_backtest_preseason_cli.py`:

```python
"""Happy-path integration test for scripts/backtest_preseason.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from projections.store import write_partition
from tests.test_scripts.test_preseason_project_season_cli import _seed_minimal_data


def test_backtest_preseason_cli_happy_path(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    proj_root = tmp_path / "projections"
    reports_root = tmp_path / "reports"
    _seed_minimal_data(raw_root, target_season=2024)

    # Also seed 2024 weekly_stats as the actuals.
    actual = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2024,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "opponent": "BUF",
                "passing_yards": 260.0,
                "passing_tds": 2,
                "interceptions": 1,
                "attempts": 30,
                "completions": 22,
                "sacks": 2,
                "rushing_yards": 30.0,
                "rushing_tds": 0,
                "carries": 5,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        ]
    )
    write_partition(raw_root, "weekly_stats", actual, season=2024, week=None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backtest_preseason.py",
            "--model", "naive-preseason",
            "--target-seasons", "2024",
            "--raw-root", str(raw_root),
            "--projections-root", str(proj_root),
            "--reports-root", str(reports_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert (reports_root / "backtest_preseason_naive-preseason.csv").exists()
    assert (reports_root / "backtest_preseason_naive-preseason.md").exists()
```

- [ ] **Step 3: Run all tests**

```bash
pytest -v -k "preseason or schemas"
```
Expected: all PASS.

- [ ] **Step 4: Phase-end verification**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add scripts/backtest_preseason.py tests/test_scripts/test_backtest_preseason_cli.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(scripts): backtest_preseason CLI + integration test"
```

**Phase 5 done. STOP and request user approval before close-out steps.**

---

# Post-implementation close-out

## Task 24: Update project tracking docs

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`
- Modify: `draft_ready_checklist.md`

- [ ] **Step 1: Add PM entry**

Append to `project_management.md` at the top of the recency list (after the snake-cheat-sheet entry):

```markdown
## Preseason Projections — v1 framework + naive baseline shipped (2026-05-17, on branch `worktree-feat+preseason-projections`)

**Status:** New sub-package `src/projections/preseason/` ships v1 framework + `NaivePreseasonModel`. Spec at `docs/superpowers/specs/2026-05-17-preseason-projections-design.md`; plan at `docs/superpowers/plans/2026-05-17-preseason-projections.md`. CLI: `python scripts/preseason_project_season.py --season 2026`.

The baseline implements three branches: veterans via `prior_1_per_game × 16` (with prior_2/prior_3 fallback); rookies via per-(position, stat) Gamma GLMs on `log(draft_pick + 1)` trained on rookie-year season totals; UDFAs imputed to pick=300. Distribution shape is degenerate (point-mass) for v1.0; v1.5 (separate spec) adds the first trained model and ships only if it beats the published gate (≥6/8 cells ADOPT, zero DO_NOT_ADOPT) characterized by this v1.0 backtest.

**Next:**
1. **Produce the 2026 partition** (one-liner: `python scripts/refresh_*` for 2026 + `python scripts/preseason_project_season.py --season 2026`).
2. **Generate v1.0 characterization backtest** on 2024 + 2025 (`python scripts/backtest_preseason.py --model naive-preseason --target-seasons 2024,2025`) — sets the v1.5 floor.
3. **v1.5 spec — first trained model class** that beats the gate. Likely GammaGLM with `(prior_1, prior_2, prior_3, age, depth_chart_rank, team)` features, then a LightGBM-quantile follow-up if needed.

Closes TODO #31 (preseason-projections "first plan should be brainstorm + roadmap" item). Flips `draft_ready_checklist.md` §1b row 2 (`predict_season.py SEASON` end-to-end) to `[x]` once the 2026 partition is materialized.
```

- [ ] **Step 2: Close TODO #31**

Edit `TODO.md` §31 — add a closing line:

```markdown
**Status.** **CLOSED 2026-05-17.** Spec + plan + impl on `worktree-feat+preseason-projections`. See `project_management.md` entry above and `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` for the v1 design.
```

- [ ] **Step 3: Flip draft_ready_checklist boxes**

Edit `draft_ready_checklist.md` §1b row 2 — flip `[ ]` to `[x]` for "Generalize `predict_2024.py` to `predict_season.py SEASON`" — but only after producing the 2026 partition end-to-end. Phrase the doc update as:

```markdown
- [x] **Generalize `predict_2024.py` to `predict_season.py SEASON`.** Shipped 2026-05-17 as `scripts/preseason_project_season.py` (new sub-package `src/projections/preseason/`). Spec at `docs/superpowers/specs/2026-05-17-preseason-projections-design.md`. Pre-season roster source comes from `depth_charts_<season>` ingest; rookies handled via draft-capital Gamma GLM. v1.0 naive baseline (point-mass); v1.5 trained model is the next spec.
```

Also flip §1a row 1 ("Ingest 2025 season") to `[x]` since 2018-2025 ingest is closed (TODO #32 closed). Phrase:

```markdown
- [x] **Ingest 2025 season.** Shipped 2026-05-11 via `nflreadpy` migration (TODO #32). All 8 ingest sources cover 2018-2025.
```

And §1a row 2 ("Pre-season roster source for 2026") — partially complete. Note that depth_charts_2026 ingest is a one-liner; needs verification before close.

- [ ] **Step 4: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add project_management.md TODO.md draft_ready_checklist.md
PATH=".venv/Scripts:$PATH" git commit -m "docs(preseason): PM entry + close TODO #31 + flip checklist §1a/§1b boxes"
```

---

# Self-review (run after all tasks)

This is a checklist to run yourself before opening a PR. The plan is complete when every box below is ticked.

- [ ] Every spec section §1-§10 has an implementing task or is explicitly out of scope.
- [ ] No "TBD" / "TODO" / "fill in later" strings remain in the code.
- [ ] Types and method signatures are consistent — e.g., `build_preseason_features` is referenced the same way in Tasks 5-9 and Task 17.
- [ ] All `.validate(...)` calls reassign: `df = SCHEMA.validate(df)`.
- [ ] No bare strings where enums exist — `Position.QB.value` not `"QB"`, `Ruleset.espn_ppr()` not magic literal.
- [ ] All file paths in the plan exactly match the file map at the top.
- [ ] Pre-commit + mypy + ruff + ruff format all green at end of every phase.
- [ ] `project_management.md` and `TODO.md` updated.

---

# Open the PR

After Task 24 and self-review:

```bash
PATH=".venv/Scripts:$PATH" git push -u origin worktree-feat+preseason-projections
PATH=".venv/Scripts:$PATH" gh pr create --title "feat(preseason): v1 framework + naive baseline for 2026 draft-day projections" --body "$(cat <<'EOF'
## Summary
- Ship `src/projections/preseason/` sub-package: features.py + model.py + project.py + backtest.py.
- `NaivePreseasonModel` — veterans via prior_1×16 (with prior_2/prior_3 fallback); rookies via per-(position, stat) Gamma GLM on `log(pick+1)`; UDFAs imputed to pick=300.
- Walk-forward backtest harness gating v1.5+ trained models on ≥6/8 cells ADOPT.
- Two new CLIs: `scripts/preseason_project_season.py` and `scripts/backtest_preseason.py`.
- Closes TODO #31; flips `draft_ready_checklist.md` §1a row 1 + §1b row 2.

## Spec
`docs/superpowers/specs/2026-05-17-preseason-projections-design.md`

## Test plan
- [ ] `pytest -v -k "preseason or schemas"` green
- [ ] `mypy src tests` green
- [ ] `ruff check src tests` + `ruff format --check src tests` green
- [ ] Produce a 2026 preseason projection: `python scripts/preseason_project_season.py --season 2026`
- [ ] Generate v1.0 baseline characterization backtest: `python scripts/backtest_preseason.py --model naive-preseason --target-seasons 2024,2025`
- [ ] Spot-check the top-50-per-position summary CSV matches Vegas/ESPN priors order-of-magnitude
EOF
)"
```
