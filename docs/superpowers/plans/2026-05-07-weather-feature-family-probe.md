# Weather Feature Family Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe four weather-related per-game features (wind_speed_mph, is_high_wind, temperature_f, is_grass_surface) as a family at BaselineModel + lgb-nb composite, returning a `SIGNAL`/`NULL` family verdict that decides whether to scope a follow-up production-builder plan.

**Architecture:** Sibling module to `pbp_team_features.py` (PR #20), `pbp_redzone_features.py` (PR #23), `pbp_pressure_features.py` (PR #24), and `trajectory_features.py` (PR #25), with one pure compute fn + attach helper + public assembler. Override-builder script reuses the established `_read_concat` / `_build_player_team_week_index` / `_parse_season_range` pattern from PR #20–#25 override scripts. Probe CLI (PR #18) and `family_verdict_from_reports` helper (PR #20) are reused unchanged. The `--force-composite` flag (PR #19) is mandatory on lgb-nb runs to avoid the bare-lgb-nb-tautology PR #22 surfaced.

**Tech Stack:** pandas (pure-pandas computes), pyarrow (parquet I/O via `projections.store.read_partition` + `df.to_parquet`), pytest, mypy strict, ruff. No new schema, no new ingest — `SchedulesSchema` already covers the source columns.

**Spec:** `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`.

---

## File Structure

**Create:**
- `src/projections/features/weather_features.py` — `compute_weather_features` + `attach_weather_features` + `build_weather_overrides`. Single file, ~80 LOC implementation.
- `scripts/build_weather_override.py` — argparse + I/O glue; loads schedules + depth_charts per-season, builds player-team-week index, calls assembler, writes parquet. Prints audit numbers (dome rate, outdoor-NaN rate, is_high_wind rate, is_grass_surface rate) to stdout.
- `tests/test_features/test_weather_features.py` — synthetic-fixture tests for compute + attach + assembler (~14 tests).
- `tests/test_scripts/test_build_weather_override_cli.py` — CLI tests (mirrors PR #25's CLI-test pattern; 4 tests).
- `reports/feature_probe_weather_baseline_{augment,swap}.{md,csv}` — 4 baseline probe outputs (1 CSV + 1 stdout-redirected markdown per mode).
- `reports/feature_probe_weather_lgbnb_{augment,swap}.{md,csv}` — 4 lgb-nb probe outputs, same shape.
- `reports/feature_probe_weather_override_audit.md` — hand-written audit report from Task 6 stdout output.
- `reports/feature_probe_weather_summary.md` — hand-written family summary with per-mode table + family verdict + mechanism annotation.

**Modify:**
- `CONTRIBUTING.md` — add "Regenerating the weather override" subsection sibling to existing override-regeneration entries.
- `TODO.md` — append a paragraph under #25 with the family verdict and date.
- `project_management.md` — add a "Weather Feature Family Probe" decision-log entry at the top.

**Untouched (deliberately):**
- `src/projections/schemas.py` — `SchedulesSchema` already declares `wind`, `temp`, `roof`, `surface`. No additions.
- `src/projections/features/_shared.py` — `build_game_environment` already collapses `roof in {dome, closed}` into `roof_dome=True`; weather features will reuse the same predicate but in a separate module to avoid scope creep on a probe-only spec.
- `src/projections/backtest/feature_probe.py` — `family_verdict_from_reports` already handles this spec's verdict rule.
- `scripts/probe_feature_signal.py` — reused as-is.
- All per-position `*FeaturesSchema` and `BaselineModel._<POS>_FEATURE_COLUMNS` — this spec is probe-only; no production wiring.

---

## Task 1: `compute_weather_features` — outdoor + dome + NaN-propagation behaviors

**Files:**
- Create: `src/projections/features/weather_features.py`
- Create: `tests/test_features/test_weather_features.py`

The compute fn produces a per-team-game frame from the schedules table. One schedule row → two output rows (home + away), each carrying the four weather feature columns. Dome / closed-roof rows are filled per spec §3.5: `wind_speed_mph=0.0`, `temperature_f=70.0`, `is_high_wind=0.0`. Outdoor rows with NaN `wind` or `temp` propagate NaN into all four cells (NaN-preserving `is_high_wind`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_weather_features.py`:

```python
"""Weather feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_schedule_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic schedules frame with sane defaults for unspecified columns.

    Mirrors `SchedulesSchema`'s shape: nullable Int64 for wind/temp, pyarrow
    string for surface/roof, etc. Tests fill in only the columns the function
    under test reads, defaults the rest.
    """
    defaults: dict[str, object] = {
        "season": 2024,
        "week": 1,
        "game_id": "2024_01_KC_BAL",
        "home_team": "KC",
        "away_team": "BAL",
        "kickoff": pd.Timestamp("2024-09-08 17:00:00", tz="UTC"),
        "spread_line": 0.0,
        "total_line": 50.0,
        "home_moneyline": -110,
        "away_moneyline": -110,
        "surface": "grass",
        "roof": "outdoors",
        "temp": 70,
        "wind": 5,
    }
    out = []
    for r in rows:
        merged = {**defaults, **r}
        out.append(merged)
    df = pd.DataFrame(out)
    # Match SchedulesSchema dtypes.
    df["temp"] = df["temp"].astype(pd.Int64Dtype())
    df["wind"] = df["wind"].astype(pd.Int64Dtype())
    df["surface"] = df["surface"].astype(pd.StringDtype("pyarrow"))
    df["roof"] = df["roof"].astype(pd.StringDtype("pyarrow"))
    return df


def test_outdoor_basic_returns_two_rows_per_game() -> None:
    """One schedule row → two output rows (home + away), both carry the same
    weather features (game-level attribute applies to both teams)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 10, "temp": 50, "roof": "outdoors", "surface": "grass"},
    ])
    out = compute_weather_features(sch)

    assert len(out) == 2
    assert set(out["team"]) == {"KC", "BAL"}
    for team in ("KC", "BAL"):
        row = out.query("team == @team").iloc[0]
        assert row["wind_speed_mph"] == 10.0
        assert row["is_high_wind"] == 0.0
        assert row["temperature_f"] == 50.0
        assert row["is_grass_surface"] == 1.0


def test_high_wind_threshold_at_20_inclusive() -> None:
    """is_high_wind = 1.0 if wind_speed_mph >= 20.0 — boundary is inclusive."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 20, "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["is_high_wind"].tolist() == [1.0, 1.0]


def test_high_wind_threshold_below_20() -> None:
    """wind=19 → is_high_wind = 0.0 (strict < threshold)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 19, "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["is_high_wind"].tolist() == [0.0, 0.0]


def test_high_wind_threshold_well_above_20() -> None:
    """wind=35 → is_high_wind = 1.0."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 35, "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["is_high_wind"].tolist() == [1.0, 1.0]


def test_dome_roof_fills_wind_zero_temp_70() -> None:
    """roof=dome → wind_speed_mph=0.0, temperature_f=70.0, is_high_wind=0.0
    even when source wind/temp are NaN. Spec §3.5."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "MIN", "away_team": "DET",
         "wind": pd.NA, "temp": pd.NA, "roof": "dome", "surface": "fieldturf"},
    ])
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].tolist() == [0.0, 0.0]
    assert out["temperature_f"].tolist() == [70.0, 70.0]
    assert out["is_high_wind"].tolist() == [0.0, 0.0]
    assert out["is_grass_surface"].tolist() == [0.0, 0.0]


def test_closed_roof_treated_as_dome() -> None:
    """roof=closed → same fill as dome (spec §3.5: closed retracted = dome
    for game-time conditions)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 5, "home_team": "DAL", "away_team": "NYG",
         "wind": pd.NA, "temp": pd.NA, "roof": "closed", "surface": "matrixturf"},
    ])
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].tolist() == [0.0, 0.0]
    assert out["temperature_f"].tolist() == [70.0, 70.0]
    assert out["is_high_wind"].tolist() == [0.0, 0.0]


def test_outdoor_nan_wind_propagates_nan() -> None:
    """Outdoor game with NaN wind → wind_speed_mph=NaN, is_high_wind=NaN.
    The naive `(wind >= 20).astype(float)` would map NaN→False→0.0, masking
    the data-quality issue. Spec §3.2 requires NaN-preserving."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "BUF", "away_team": "NYJ",
         "wind": pd.NA, "temp": 45, "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].isna().all()
    assert out["is_high_wind"].isna().all()
    # temp is fine, only wind is NaN
    assert out["temperature_f"].tolist() == [45.0, 45.0]


def test_outdoor_nan_temp_propagates_nan() -> None:
    """Outdoor game with NaN temp → temperature_f=NaN; wind features unaffected."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "GB", "away_team": "DET",
         "wind": 10, "temp": pd.NA, "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["temperature_f"].isna().all()
    assert out["wind_speed_mph"].tolist() == [10.0, 10.0]
    assert out["is_high_wind"].tolist() == [0.0, 0.0]


def test_surface_grass_returns_one() -> None:
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "GB", "away_team": "DET",
         "surface": "grass", "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["is_grass_surface"].tolist() == [1.0, 1.0]


def test_surface_fieldturf_returns_zero() -> None:
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "DET", "away_team": "GB",
         "surface": "fieldturf", "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["is_grass_surface"].tolist() == [0.0, 0.0]


def test_surface_nan_returns_zero() -> None:
    """Null surface → is_grass_surface = 0.0 (informational fallback per spec §3.4)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "GB", "away_team": "DET",
         "surface": pd.NA, "roof": "outdoors"},
    ])
    out = compute_weather_features(sch)
    assert out["is_grass_surface"].tolist() == [0.0, 0.0]


def test_unknown_roof_treated_as_outdoor() -> None:
    """roof='open' (not in {dome, closed}) → no fill, NaN wind/temp propagate."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "GB", "away_team": "DET",
         "wind": pd.NA, "temp": pd.NA, "roof": "open"},
    ])
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].isna().all()
    assert out["temperature_f"].isna().all()
    assert out["is_high_wind"].isna().all()


def test_null_roof_treated_as_outdoor() -> None:
    """roof=NaN → no fill (matches build_game_environment's
    `.isin([dome, closed]).fillna(False)` predicate)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "GB", "away_team": "DET",
         "wind": pd.NA, "temp": pd.NA, "roof": pd.NA},
    ])
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].isna().all()
    assert out["temperature_f"].isna().all()


def test_output_columns_and_dtypes() -> None:
    """Output schema: (season, week, team, wind_speed_mph, is_high_wind,
    temperature_f, is_grass_surface). All four feature cols are Float64."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL"},
    ])
    out = compute_weather_features(sch)

    expected_cols = {
        "season", "week", "team",
        "wind_speed_mph", "is_high_wind", "temperature_f", "is_grass_surface",
    }
    assert set(out.columns) == expected_cols
    for col in ("wind_speed_mph", "is_high_wind", "temperature_f", "is_grass_surface"):
        assert str(out[col].dtype) == "Float64", f"{col} dtype: {out[col].dtype}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v`
Expected: 14 errors, all `ModuleNotFoundError: No module named 'projections.features.weather_features'`.

- [ ] **Step 3: Write `weather_features.py` skeleton + compute_weather_features**

Create `src/projections/features/weather_features.py`:

```python
"""Weather feature computes for the weather family probe.

Sourced from `SchedulesSchema` columns (`wind`, `temp`, `roof`, `surface`)
already in `data/raw/schedules`. Dome / closed-roof games are filled per
spec §3.5: a controlled environment has no weather, so wind=0 / temp=70
is semantically correct, not "imputed missing."

Probe-only — features land in the override parquet, not in
`*FeaturesSchema`. Integration follow-up is conditional on the family-probe
verdict per `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`.
"""

from __future__ import annotations

import pandas as pd

_HIGH_WIND_MPH = 20.0
_DOME_FILL_TEMP_F = 70.0
_DOME_FILL_WIND_MPH = 0.0


def compute_weather_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team-game frame with four weather features.

    One row per (game, team) — each schedules row produces two output rows
    (home + away). Weather is a game-level attribute, so both teams in a
    matchup carry identical wind/temp/surface values.

    Dome / closed-roof handling (spec §3.5):
        wind_speed_mph = 0.0
        temperature_f = 70.0
        is_high_wind = 0.0 (falls out of wind_speed_mph = 0.0)
        is_grass_surface = surface == 'grass' (no override)

    Outdoor handling: NaN wind / temp propagate; is_high_wind preserves NaN.

    Args:
        schedules: frame validated against `SchedulesSchema` (must carry
            season, week, home_team, away_team, wind, temp, roof, surface).

    Returns:
        DataFrame with columns:
            season, week, team,
            wind_speed_mph, is_high_wind, temperature_f, is_grass_surface
        All four feature columns are nullable Float64. season / week are
        Int64; team is StringDtype("pyarrow") (inherited from inputs).
    """
    cols = ["season", "week", "wind", "temp", "roof", "surface"]
    home = schedules[cols + ["home_team"]].rename(columns={"home_team": "team"}).copy()
    away = schedules[cols + ["away_team"]].rename(columns={"away_team": "team"}).copy()
    games = pd.concat([home, away], ignore_index=True)

    # Dome / closed-roof predicate matches `_shared.build_game_environment`'s
    # logic exactly: any roof not in {dome, closed} (including NaN) is treated
    # as outdoor.
    is_indoor = games["roof"].isin(["dome", "closed"]).fillna(False)

    wind_f = games["wind"].astype("Float64")
    games["wind_speed_mph"] = wind_f.where(~is_indoor, _DOME_FILL_WIND_MPH)

    temp_f = games["temp"].astype("Float64")
    games["temperature_f"] = temp_f.where(~is_indoor, _DOME_FILL_TEMP_F)

    # NaN-preserving threshold (spec §3.2). pandas Float64 + NA propagates
    # the comparison: (NA >= 20.0) -> NA -> NaN in resulting Float64.
    wind_speed = games["wind_speed_mph"]
    games["is_high_wind"] = (wind_speed >= _HIGH_WIND_MPH).astype("Float64")

    games["is_grass_surface"] = (
        (games["surface"] == "grass").fillna(False).astype("Float64")
    )

    return games[
        [
            "season",
            "week",
            "team",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v`
Expected: 14 passed.

- [ ] **Step 5: Run mypy + ruff + format on the new files**

```
PYTHONPATH=src mypy src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: all green. Fix any violations inline before committing.

- [ ] **Step 6: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "feat(weather): add compute_weather_features with dome-fill + NaN-preserving is_high_wind"
```

---

## Task 2: `attach_weather_features` joiner

**Files:**
- Modify: `src/projections/features/weather_features.py`
- Modify: `tests/test_features/test_weather_features.py`

The joiner takes a player-team-week index and a schedules frame, computes weather features per team-game, then left-merges onto the index on `(season, week, team)`. Rows in the index without a matching schedule entry retain NaN in all four weather cols (the assembler logs the unmatched-row rate).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_weather_features.py`:

```python
def _make_index_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic player-team-week index frame matching
    `_build_player_team_week_index`'s output shape."""
    defaults: dict[str, object] = {
        "gsis_id": "00-0011111",
        "season": 2024,
        "week": 1,
        "team": "KC",
        "opp": "BAL",
        "position": "WR",
    }
    out = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(out)
    return df.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "week": pd.Int64Dtype(),
            "team": pd.StringDtype("pyarrow"),
            "opp": pd.StringDtype("pyarrow"),
            "position": pd.StringDtype("pyarrow"),
        }
    )


def test_attach_weather_features_basic() -> None:
    """Joins compute output onto index on (season, week, team); preserves
    every input row."""
    from projections.features.weather_features import attach_weather_features

    idx = _make_index_rows([
        {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        {"gsis_id": "00-0022222", "season": 2024, "week": 1, "team": "BAL", "opp": "KC"},
    ])
    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 12, "temp": 65, "roof": "outdoors", "surface": "grass"},
    ])

    out = attach_weather_features(idx, sch)

    assert len(out) == 2
    # Both teams in the matchup pick up the same weather (game-level)
    for team in ("KC", "BAL"):
        row = out.query("team == @team").iloc[0]
        assert row["wind_speed_mph"] == 12.0
        assert row["temperature_f"] == 65.0
        assert row["is_high_wind"] == 0.0
        assert row["is_grass_surface"] == 1.0
    # Original index columns preserved
    assert set(out.columns) == {
        "gsis_id", "season", "week", "team", "opp", "position",
        "wind_speed_mph", "is_high_wind", "temperature_f", "is_grass_surface",
    }


def test_attach_weather_features_unmatched_index_row_propagates_nan() -> None:
    """Index row with no matching schedule entry → NaN in all four weather cols."""
    from projections.features.weather_features import attach_weather_features

    idx = _make_index_rows([
        {"gsis_id": "00-0011111", "season": 2024, "week": 99, "team": "KC", "opp": "BAL"},
    ])
    sch = _make_schedule_rows([
        # Different week — won't match.
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 12, "temp": 65, "roof": "outdoors", "surface": "grass"},
    ])

    out = attach_weather_features(idx, sch)

    assert len(out) == 1
    row = out.iloc[0]
    assert pd.isna(row["wind_speed_mph"])
    assert pd.isna(row["is_high_wind"])
    assert pd.isna(row["temperature_f"])
    assert pd.isna(row["is_grass_surface"])


def test_attach_weather_features_preserves_index_row_count() -> None:
    """Multiple players on the same team-game produce multiple output rows
    (one per index row), all carrying identical weather."""
    from projections.features.weather_features import attach_weather_features

    idx = _make_index_rows([
        {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL", "position": "QB"},
        {"gsis_id": "00-0022222", "season": 2024, "week": 1, "team": "KC", "opp": "BAL", "position": "WR"},
        {"gsis_id": "00-0033333", "season": 2024, "week": 1, "team": "KC", "opp": "BAL", "position": "TE"},
    ])
    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 25, "temp": 30, "roof": "outdoors", "surface": "grass"},
    ])

    out = attach_weather_features(idx, sch)

    assert len(out) == 3
    assert (out["wind_speed_mph"] == 25.0).all()
    assert (out["is_high_wind"] == 1.0).all()
    assert (out["temperature_f"] == 30.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_features/test_weather_features.py::test_attach_weather_features_basic -v`
Expected: ImportError on `attach_weather_features`.

- [ ] **Step 3: Implement `attach_weather_features`**

Append to `src/projections/features/weather_features.py`:

```python
def attach_weather_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge the four weather features onto a player-team-week index.

    Args:
        index: frame with at least (season, week, team) columns. Typically
            the player-team-week index from
            `scripts.build_weather_override._build_player_team_week_index`,
            carrying (gsis_id, season, week, team, opp, position).
        schedules: frame validated against `SchedulesSchema`.

    Returns:
        Copy of index with four nullable Float64 cols appended:
        wind_speed_mph, is_high_wind, temperature_f, is_grass_surface.
        Index rows without a matching (season, week, team) in schedules
        retain NaN in all four cols.
    """
    weather = compute_weather_features(schedules)
    return index.merge(weather, on=["season", "week", "team"], how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v`
Expected: 17 passed (14 prior + 3 new).

- [ ] **Step 5: Run mypy + ruff + format**

```
PYTHONPATH=src mypy src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "feat(weather): add attach_weather_features joiner"
```

---

## Task 3: `build_weather_overrides` public assembler

**Files:**
- Modify: `src/projections/features/weather_features.py`
- Modify: `tests/test_features/test_weather_features.py`

Top-level entry that the override-building script calls. Currently a thin wrapper over `attach_weather_features`, but the spec keeps the three-function shape (compute / attach / build_overrides) for parallelism with PR #25's `trajectory_features.py`. The build entrypoint owns input validation (key uniqueness, required columns) so the script stays as plumbing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_weather_features.py`:

```python
def test_build_weather_overrides_returns_one_row_per_index_row() -> None:
    from projections.features.weather_features import build_weather_overrides

    idx = _make_index_rows([
        {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        {"gsis_id": "00-0022222", "season": 2024, "week": 2, "team": "KC", "opp": "CIN"},
    ])
    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 10, "temp": 55, "roof": "outdoors"},
        {"season": 2024, "week": 2, "home_team": "KC", "away_team": "CIN",
         "wind": 5, "temp": 75, "roof": "outdoors"},
    ])

    out = build_weather_overrides(sch, idx)
    assert len(out) == 2
    assert out.query("week == 1")["wind_speed_mph"].iloc[0] == 10.0
    assert out.query("week == 2")["wind_speed_mph"].iloc[0] == 5.0


def test_build_weather_overrides_raises_on_duplicate_index_keys() -> None:
    """Probe override must have unique (gsis_id, season, week) keys to merge
    cleanly into the probe runner. Duplicates raise immediately."""
    from projections.features.weather_features import build_weather_overrides

    idx = _make_index_rows([
        {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
    ])
    sch = _make_schedule_rows([
        {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
         "wind": 10, "roof": "outdoors"},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        build_weather_overrides(sch, idx)


def test_build_weather_overrides_raises_on_missing_required_index_columns() -> None:
    from projections.features.weather_features import build_weather_overrides

    bad = pd.DataFrame({"gsis_id": ["00-0011111"], "season": [2024]})
    sch = _make_schedule_rows([{}])

    with pytest.raises(ValueError, match="required column"):
        build_weather_overrides(sch, bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_features/test_weather_features.py::test_build_weather_overrides_returns_one_row_per_index_row -v`
Expected: ImportError on `build_weather_overrides`.

- [ ] **Step 3: Implement `build_weather_overrides`**

Append to `src/projections/features/weather_features.py`:

```python
_REQUIRED_INDEX_COLS = ("gsis_id", "season", "week", "team", "opp", "position")


def build_weather_overrides(
    schedules: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Build the weather override frame from a schedules table + a
    player-team-week index.

    Args:
        schedules: validated against `SchedulesSchema`.
        player_team_week_index: frame from `_build_player_team_week_index`
            with columns (gsis_id, season, week, team, opp, position).
            Must have unique (gsis_id, season, week) keys.

    Returns:
        Frame with columns
            (gsis_id, season, week, position,
             wind_speed_mph, is_high_wind, temperature_f, is_grass_surface)
        — one row per index input row. Designed to feed
        `scripts.probe_feature_signal --override`.

    Raises:
        ValueError: index missing a required column or carrying duplicate
            (gsis_id, season, week) keys.
    """
    missing = [c for c in _REQUIRED_INDEX_COLS if c not in player_team_week_index.columns]
    if missing:
        raise ValueError(f"player_team_week_index missing required column(s): {missing}")

    key_cols = ["gsis_id", "season", "week"]
    dups = player_team_week_index.duplicated(subset=key_cols)
    if dups.any():
        n = int(dups.sum())
        raise ValueError(
            f"player_team_week_index has {n} duplicate (gsis_id, season, week) keys"
        )

    attached = attach_weather_features(player_team_week_index, schedules)
    return attached[
        [
            "gsis_id",
            "season",
            "week",
            "position",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v`
Expected: 20 passed (17 prior + 3 new).

- [ ] **Step 5: Run mypy + ruff + format**

```
PYTHONPATH=src mypy src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "feat(weather): add build_weather_overrides public assembler"
```

---

## Task 4: `scripts/build_weather_override.py` + CLI tests

**Files:**
- Create: `scripts/build_weather_override.py`
- Create: `tests/test_scripts/test_build_weather_override_cli.py`

The CLI mirrors PR #25's `build_trajectory_override.py` but is simpler — no draft-picks ingest dependency, no trailing-N history-season backfill (weather is per-game, not trailing). The audit numbers (dome rate, outdoor-NaN rate, is_high_wind rate, is_grass_surface rate) are printed to stdout so Task 6 can capture them into the audit report.

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_scripts/test_build_weather_override_cli.py`:

```python
"""build_weather_override CLI smokes."""

from __future__ import annotations

import pandas as pd
import pytest


def test_parse_season_range_single_year() -> None:
    from scripts.build_weather_override import _parse_season_range

    r = _parse_season_range("2024")
    assert list(r) == [2024]


def test_parse_season_range_inclusive_range() -> None:
    from scripts.build_weather_override import _parse_season_range

    r = _parse_season_range("2018-2024")
    assert list(r) == [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def test_parse_args_defaults(tmp_path) -> None:
    from scripts.build_weather_override import parse_args

    out = tmp_path / "weather.parquet"
    args = parse_args(["--output", str(out)])
    assert args.output == out
    assert list(args.seasons) == list(range(2018, 2025))
    assert args.data_root.name == "data"


def test_parse_args_refuses_overwrite_without_force(tmp_path) -> None:
    """If output exists and --force not set, parser exits."""
    from scripts.build_weather_override import parse_args

    out = tmp_path / "weather.parquet"
    out.write_text("")  # touch
    with pytest.raises(SystemExit):
        parse_args(["--output", str(out)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_scripts/test_build_weather_override_cli.py -v`
Expected: 4 ImportErrors on missing `scripts.build_weather_override`.

- [ ] **Step 3: Write `build_weather_override.py`**

Create `scripts/build_weather_override.py`:

```python
"""Build the weather override parquet for the weather family probe.

One-shot CLI. Loads schedules + depth_charts across the requested season
range, builds the player-team-week index, calls build_weather_overrides,
writes the resulting frame to a parquet. Prints audit numbers (dome rate,
outdoor-NaN rate, is_high_wind rate, is_grass_surface rate) so a follow-up
step can capture them into reports/feature_probe_weather_override_audit.md.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_weather_override --seasons 2018-2024
    python -m scripts.build_weather_override --seasons 2018-2024 --force
    python -m scripts.build_weather_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.weather_features import build_weather_overrides
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/weather.parquet")


def _parse_season_range(s: str) -> range:
    """`'2018-2024'` -> `range(2018, 2025)`; `'2024'` -> `range(2024, 2025)`."""
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return range(lo, hi + 1)
    n = int(s)
    return range(n, n + 1)


def _read_concat(raw_root: Path, table: str, seasons: Sequence[int]) -> pd.DataFrame:
    """Read one partition per season and concat. Skip seasons without a partition."""
    frames: list[pd.DataFrame] = []
    for s in seasons:
        try:
            frames.append(read_partition(raw_root, table, season=s))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError(
            f"no partitions found for table={table!r} in seasons={list(seasons)}"
        )
    return pd.concat(frames, ignore_index=True)


_FANTASY_POSITIONS: tuple[str, ...] = tuple(
    p.value for p in (Position.QB, Position.RB, Position.WR, Position.TE)
)


def _build_player_team_week_index(
    depth_charts: pd.DataFrame, schedules: pd.DataFrame, seasons: range
) -> pd.DataFrame:
    """Inner-join depth_charts (filtered to fantasy positions) with schedules
    to produce ``(gsis_id, season, week, team, opp, position)``.

    Mirrors PR #25's helper. Pins canonical dtypes on the output:
    ``gsis_id`` / ``team`` / ``opp`` / ``position`` -> StringDtype("pyarrow"),
    ``season`` / ``week`` -> Int64Dtype.
    """
    dc = depth_charts[
        depth_charts["season"].isin(seasons)
        & depth_charts["position"].isin(_FANTASY_POSITIONS)
    ][["gsis_id", "season", "week", "team", "position"]].drop_duplicates(
        subset=["gsis_id", "season", "week"]
    )
    sch = schedules[schedules["season"].isin(seasons)][
        ["season", "week", "home_team", "away_team"]
    ]
    home = sch.rename(columns={"home_team": "team", "away_team": "opp"})
    away = sch.rename(columns={"away_team": "team", "home_team": "opp"})
    team_opp = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "opp"]
    ]
    result = dc.merge(team_opp, on=["season", "week", "team"], how="inner")
    return result.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "week": pd.Int64Dtype(),
            "team": pd.StringDtype("pyarrow"),
            "opp": pd.StringDtype("pyarrow"),
            "position": pd.StringDtype("pyarrow"),
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args. Extracted for testability — same pattern as
    `scripts.build_trajectory_override.main`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--seasons",
        type=_parse_season_range,
        default=range(2018, 2025),
        help="Season range, e.g. '2018-2024' or '2024'. Default: 2018-2024.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root for raw partitions. Default: data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Override output parquet path. Default: {_DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output if it already exists.",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite.")

    return args


def _print_audit(overrides: pd.DataFrame, schedules: pd.DataFrame) -> None:
    """Print audit numbers for `reports/feature_probe_weather_override_audit.md`.

    Numbers reported:
        - Pooled dome / closed-roof game share (% of games).
        - Outdoor-NaN rate per weather feature (% of override rows).
        - Pooled is_high_wind rate (% of override rows where True).
        - Pooled is_grass_surface rate (% of override rows where True).
    """
    n = len(overrides)
    is_indoor = schedules["roof"].isin(["dome", "closed"]).fillna(False)
    n_indoor_games = int(is_indoor.sum())
    n_total_games = len(schedules)
    indoor_pct = (n_indoor_games / n_total_games * 100.0) if n_total_games else 0.0

    nan_rates = {
        col: overrides[col].isna().mean() * 100.0
        for col in (
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_grass_surface",
        )
    }
    high_wind_rate = overrides["is_high_wind"].fillna(0.0).mean() * 100.0
    grass_rate = overrides["is_grass_surface"].fillna(0.0).mean() * 100.0

    print(f"weather override audit ({n} rows):")
    print(
        f"  indoor games (dome+closed): "
        f"{n_indoor_games}/{n_total_games} = {indoor_pct:.1f}%"
    )
    for col, pct in nan_rates.items():
        print(f"  {col} NaN rate: {pct:.2f}%")
    print(f"  is_high_wind=1.0 rate (incl. dome): {high_wind_rate:.2f}%")
    print(f"  is_grass_surface=1.0 rate: {grass_rate:.2f}%")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seasons: range = args.seasons
    raw_root = args.data_root / "raw"

    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))

    idx = _build_player_team_week_index(depth_charts, schedules, seasons)
    overrides = build_weather_overrides(schedules, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    _print_audit(overrides, schedules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_scripts/test_build_weather_override_cli.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run mypy + ruff + format**

```
PYTHONPATH=src mypy scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
ruff check scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
ruff format --check scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "feat(weather): add build_weather_override CLI + tests"
```

---

## Task 5: `CONTRIBUTING.md` regenerate-override subsection

**Files:**
- Modify: `CONTRIBUTING.md`

A short procedural note in the existing override-regeneration section — the "Regenerating the trajectory override" / "Regenerating the PBP pressure override" precedents.

- [ ] **Step 1: Read the existing override-regeneration section**

Read `CONTRIBUTING.md` and find the section that lists the existing override-regeneration commands (typically just below the "Daily commands" section). Confirm the heading format (e.g., `### Regenerating the trajectory override`).

- [ ] **Step 2: Append the weather subsection**

Add a sibling subsection following the established pattern. Example content (adapt phrasing to match the immediately preceding subsection's style):

```markdown
### Regenerating the weather override

The weather feature family probe (PR #28) reads from `data/raw/schedules/`
only; no new ingest required (`SchedulesSchema` already covers `wind`,
`temp`, `roof`, `surface`).

```bash
python -m scripts.build_weather_override --seasons 2018-2024 --force
```

Writes `data/features_probe/weather.parquet` and prints audit numbers (dome
rate, outdoor-NaN rate, is_high_wind / is_grass_surface rates) to stdout.
The audit output is captured into
`reports/feature_probe_weather_override_audit.md` per spec §6.7.

Spec: `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`.
```

- [ ] **Step 3: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add CONTRIBUTING.md
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "docs(contributing): add weather override regeneration subsection"
```

---

## Task 6: Generate override + write audit report

**Files:**
- Create (regenerable, NOT committed): `data/features_probe/weather.parquet`
- Create: `reports/feature_probe_weather_override_audit.md`

This task is operational — runs the build script against the live `data/raw/schedules` partitions, captures the audit stdout into a markdown file, and commits the audit report (the parquet itself is not committed; it's regenerable per spec §4 / `CONTRIBUTING.md`).

- [ ] **Step 1: Verify the schedules partitions exist**

```
ls data/raw/schedules/season=*/part.parquet | head -5
```

Expected: at least seasons 2018-2024 present. If missing, run the schedule ingest first:
```
python -c "from projections.ingest.schedules import refresh_schedules; from pathlib import Path; refresh_schedules(Path('data'), seasons=range(2018, 2025))"
```

- [ ] **Step 2: Generate the override**

```
PYTHONPATH=src python -m scripts.build_weather_override --seasons 2018-2024
```

Expected output (rates approximate; actual numbers go into the audit report):
```
wrote N rows to data/features_probe/weather.parquet
weather override audit (N rows):
  indoor games (dome+closed): ~30% of games
  wind_speed_mph NaN rate: <2%
  is_high_wind NaN rate: <2%
  temperature_f NaN rate: <2%
  is_grass_surface NaN rate: 0%
  is_high_wind=1.0 rate (incl. dome): a few %
  is_grass_surface=1.0 rate: ~40-50%
```

If any NaN rate exceeds 5%, that's worth a flag in the audit report (probably real data-quality and we lower the probe's `--coverage-threshold` to 0.90).

If row count is 0 or much smaller than the trajectory override (~100k rows), the index build is wrong — likely the depth_charts position filter or the schedules merge is dropping more than expected. Investigate before continuing.

- [ ] **Step 3: Capture audit numbers into a markdown report**

Create `reports/feature_probe_weather_override_audit.md` using the stdout from Step 2. Template:

```markdown
# Weather Override Audit

**Generated:** YYYY-MM-DD via `python -m scripts.build_weather_override --seasons 2018-2024`
**Output:** `data/features_probe/weather.parquet` (N rows, regenerable)
**Spec:** `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md` §6.7
**Branch:** `feat/probe-weather`

## Pooled (2018-2024)

| Metric | Value |
|---|---:|
| Total override rows | N |
| Total schedule rows | M |
| Indoor games (dome + closed) | X / M (Y%) |
| `wind_speed_mph` NaN rate | Z% |
| `is_high_wind` NaN rate | Z% |
| `temperature_f` NaN rate | Z% |
| `is_grass_surface` NaN rate | Z% |
| `is_high_wind=1.0` rate (incl. dome) | A% |
| `is_grass_surface=1.0` rate | B% |

## Notes

- Indoor share consistent with the league makeup (10 dome teams / 32 = 31.25%
  + retractable closures).
- Outdoor-NaN rates are data-quality (dome rows are deliberately filled per
  spec §3.5, not counted as missing).
- `is_high_wind` rate is a single-digit percentage — outdoor games with sustained
  wind ≥ 20 mph are rare; the threshold is intentionally strict per spec §3.2.

[Add any anomalies observed during generation — unexpected NaN clusters in a
particular season, surface-code values not previously seen, etc.]
```

- [ ] **Step 4: Commit the audit report (NOT the parquet)**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add reports/feature_probe_weather_override_audit.md
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "report(weather-probe): override audit (N rows, dome rate Y%, outdoor-NaN <Z%)"
```

The parquet at `data/features_probe/weather.parquet` is intentionally NOT committed — `data/` is ignored in `.gitignore` per project convention; the file is regenerable from raw partitions per `CONTRIBUTING.md`.

---

## Task 7: Run the 4 probes + write summary report

**Files:**
- Create: `reports/feature_probe_weather_baseline_{augment,swap}.{md,csv}` (4 files)
- Create: `reports/feature_probe_weather_lgbnb_{augment,swap}.{md,csv}` (4 files)
- Create: `reports/feature_probe_weather_summary.md`

This task is the executable validation phase. The override at `data/features_probe/weather.parquet` must already exist from Task 6.

- [ ] **Step 1: Run the 4 probes**

The probe CLI takes `--override` (parquet path) + `--drop` (column list, only for swap mode) + `--csv-out` (CSV output path). The markdown report is written to stdout — redirect with `>` to capture it. Augment mode = override only; swap mode = override + drop with the same column names. Per spec §5.2, weather features have no v1 column counterparts — swap mode effectively *adds* the override cols just like augment, since `--drop` with weather-feature names finds no existing baseline columns to remove. (Same swap-mode-on-brand-new-features semantics as PR #25 trajectory.)

The `--candidate-name` becomes the report header; use distinct names per (model, mode) for clarity.

```bash
COLS="wind_speed_mph,is_high_wind,temperature_f,is_grass_surface"

# baseline × augment
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_baseline_augment \
    --override data/features_probe/weather.parquet \
    --model baseline \
    --csv-out reports/feature_probe_weather_baseline_augment.csv \
    > reports/feature_probe_weather_baseline_augment.md

# baseline × swap
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_baseline_swap \
    --override data/features_probe/weather.parquet \
    --drop "$COLS" \
    --model baseline \
    --csv-out reports/feature_probe_weather_baseline_swap.csv \
    > reports/feature_probe_weather_baseline_swap.md

# lgb-nb × augment (force-composite — Phase 1 is RidgeCV-only, so without
# --force-composite, Phase 2 never fires and the run is tautological with baseline)
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_lgbnb_augment \
    --override data/features_probe/weather.parquet \
    --model lightgbm-nb \
    --force-composite \
    --csv-out reports/feature_probe_weather_lgbnb_augment.csv \
    > reports/feature_probe_weather_lgbnb_augment.md

# lgb-nb × swap
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_lgbnb_swap \
    --override data/features_probe/weather.parquet \
    --drop "$COLS" \
    --model lightgbm-nb \
    --force-composite \
    --csv-out reports/feature_probe_weather_lgbnb_swap.csv \
    > reports/feature_probe_weather_lgbnb_swap.md
```

Expected: each baseline run completes in ~1-2 minutes; each lgb-nb run ~10-15 minutes. If any run fails with an `OverrideCoverageError` (coverage below 0.95), retry that run with `--coverage-threshold 0.90` per spec §1.3 fallback. Document the relaxation in the summary report. Spec expects ≥98% pooled coverage given dome-fill, so the relaxation should not be needed.

`--position` defaults to all four (QB / RB / WR / TE) when omitted, which is what we want — no need to pass it explicitly.

- [ ] **Step 2: Inspect each report's verdict**

For each of the 4 markdown reports, scan the per-stat verdict table (Phase 1) and the composite verdict block (Phase 2). Note:
- Phase 1 SIGNAL count per (position, target_stat) cell.
- Phase 2 ADOPT / MARGINAL / DO_NOT_ADOPT verdict per position.
- Any cell with `REGRESSION` (CI strictly worse than zero).

Pay attention to the QB augment cells specifically — PRs #23, #24, #25 each saw QB augment regress on context / team / trajectory adds. If the same pattern recurs here, document it.

- [ ] **Step 3: Write `feature_probe_weather_summary.md`**

Read `reports/feature_probe_trajectory_summary.md` first as a template. Create `reports/feature_probe_weather_summary.md` mirroring its structure:

```markdown
# Weather Feature Family Probe — Summary

**Date:** YYYY-MM-DD
**Branch:** `feat/probe-weather`
**Spec:** `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-07-weather-feature-family-probe.md`
**Override:** `data/features_probe/weather.parquet` (N rows; audit:
`reports/feature_probe_weather_override_audit.md`)

## Family verdict: <SIGNAL | NULL (durable) | etc.>

[One-paragraph headline framing the verdict + the binding cells.]

## Per-mode verdict table

| Mode | Model | Phase 1 SIGNAL cells | Phase 2 verdicts (per position) |
|---|---|---:|---|
| augment | baseline | X / 16 | QB: V, RB: V, WR: V, TE: V |
| swap | baseline | X / 16 | QB: V, RB: V, WR: V, TE: V |
| augment | lgb-nb composite | X / 16 | QB: V, RB: V, WR: V, TE: V |
| swap | lgb-nb composite | X / 16 | QB: V, RB: V, WR: V, TE: V |

(`V` ∈ {ADOPT, MARGINAL, DO_NOT_ADOPT, REGRESSION}.)

## Mechanism annotation

[Did wind features fire on QB / WR / TE pass-volume cells? Did temperature
fire? Did surface fire on yards-after-catch-leaning cells (RB receiving,
WR / TE)? Cite specific (position, target_stat) cells from the per-mode
reports.]

## Coverage note

Pooled coverage: X% (default `--coverage-threshold 0.95` `passed` / `relaxed
to 0.90`). Per-(position, season) coverage breakdown — flag any cell below
0.90.

## Recurring QB augment regression check

[PRs #23 / #24 / #25 each saw QB augment regress on context-adjacent
features. Did the pattern recur here? Cite the QB augment Phase 2 verdict
on each of {baseline, lgb-nb}.]

## Refined-unit candidates left unexplored

- Precipitation (would require new ingest source — out of scope per spec §1.4).
- Kickoff hour / time-of-day.
- Cold-weather threshold (`is_cold_weather = temp < 32`).
- Multi-class surface encoding (one bool per surface code).
- Surface × position interaction terms.
- Per-team weather acclimation effects.
- Wind direction encoding.

[Note: refined units are revisit-only-on-SIGNAL territory per spec §1.4.]

## Decision log

| Date | Decision | Reason |
|---|---|---|
| YYYY-MM-DD | Probe shipped at `feat/probe-weather` HEAD `<sha>` | All 4 reports complete; verdict <V>. |
| YYYY-MM-DD | [Coverage relaxation, if applied] | [Why] |

## Recommended next direction

[If verdict is SIGNAL: greenlight a per-position integration plan analogous
to PR #21 / PR #26 / PR #27. Identify the binding (model, position) cell.

If verdict is NULL durable: close the family at this unit. The next slot in
the modeling-improvement queue is target decomposition (TODO #23) or a
refined-unit weather candidate if independent evidence emerges.]
```

- [ ] **Step 4: Commit reports**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add reports/feature_probe_weather_*.md reports/feature_probe_weather_*.csv
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "report(weather-probe): family summary — verdict <verdict>"
```

(Replace `<verdict>` with the actual outcome — `SIGNAL`, `NULL (durable)`, etc.)

---

## Task 8: PM/TODO updates + final verification gates

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Add a `Weather Feature Family Probe` decision-log entry to `project_management.md`**

At the top of `project_management.md` (just below the `---` separator on line 5), insert a new entry following the structure of the existing "TE Trajectory Features Integration" / "Trajectory Family Probe" entries. Sample structure:

```markdown
## Weather Feature Family Probe — verdict <V> (YYYY-MM-DD, on branch `feat/probe-weather`)

**Status:** Probe-only family check at `(BaselineModel + lgb-nb composite, augment + swap)` × 4 positions per `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`. Bundled four weather features (`wind_speed_mph`, `is_high_wind` ≥20mph, `temperature_f`, `is_grass_surface`) sourced from existing `SchedulesSchema` columns (no new ingest). Dome / closed-roof games filled with (wind=0, temp=70) per spec §3.5 — semantically correct, preserves coverage on the ~30% of games played indoors.

**Family verdict:** **<V>** [Concise framing: which mode/model cells fired, the binding cell, the runner-up cells, and any regressions. If SIGNAL, give the headline composite RMSE delta + CI for the binding cell. If NULL durable, summarize the closest cells.]

**Coverage:** pooled X%, per-(position, season) ≥Y%. `--coverage-threshold` <default 0.95 / relaxed to 0.90>.

**Recurring QB augment regression check:** [Did the PR #23 / PR #24 / PR #25 pattern recur? If yes, cite the new instance.]

**Mechanism annotation:** [Which feature drove signal, or which `target_stat` cells regressed.]

**Refined-unit candidates remain unexplored:** precipitation (would require new ingest source — see TODO #25), kickoff hour, cold-weather threshold, multi-class surface, per-team weather acclimation, wind-direction encoding. None queued; revisit only with independent evidence the unit choice was the binding constraint.

**Reports:** `reports/feature_probe_weather_summary.md`, `reports/feature_probe_weather_override_audit.md`, 4 per-mode reports.

[If SIGNAL: **Recommended follow-up:** per-position integration plan analogous to PR #21 / PR #26 / PR #27. Binding cell is `(model, position)`.]
[If NULL durable: closes TODO #25's broad-cut weather family branch. Track 2 next slot is target decomposition (TODO #23) or a refined-unit weather candidate if independent evidence emerges.]
```

- [ ] **Step 2: Update the `Current status` and `Next action` sections in `project_management.md`**

The "Current status (as of 2026-05-03)" section needs updating to reflect:
- PR #26 (WR trajectory) and PR #27 (TE trajectory) integrations are now shipped.
- This new weather family probe is the latest entry (verdict V).
- Update the multi-track scoreboard with the 6th family probe entry.

The "Next action" section needs replacing the old recommendations (1: TE trajectory integration, 2: weather probe, 3: trajectory refined units) with new options based on the weather verdict:
- If weather SIGNAL: option 1 is the per-position weather integration plan.
- If weather NULL: option 1 is target decomposition (TODO #23) or trajectory refined units.

Re-read the current "Current status" + "Next action" blocks before editing to preserve the structure and tone.

- [ ] **Step 3: Append a paragraph to TODO #25 in `TODO.md`**

Find `### 25. Weather features in per-position builders` and append an `**Update YYYY-MM-DD (weather family probe, branch `feat/probe-weather`):**` paragraph following the structure of #24's trajectory updates. State the verdict, the binding cells (or closest cells if NULL), the coverage threshold used, and what remains open in #25. If the verdict is NULL durable, update the heading status to reflect that the broad-cut weather family is closed at this unit; refined units remain unexplored.

- [ ] **Step 4: Commit project_management.md + TODO.md updates**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather add project_management.md TODO.md
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather commit -m "docs(pm): record weather family probe verdict"
```

- [ ] **Step 5: Final verification gate**

Per CLAUDE.md "End-of-effort checklist":

```
PYTHONPATH=src pytest -v -k "weather or schedules or schemas"
PYTHONPATH=src mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Expected: all green. If any fail, fix before declaring the task complete and creating the PR.

If a narrowly-scoped subset is acceptable per CLAUDE.md "Forced verification — end-of-effort checklist," state which subset was run.

- [ ] **Step 6: Run the full pytest sweep one last time**

```
PYTHONPATH=src pytest -v
```

Expected: ~840+ passed, 17 skipped (network smokes). Confirm no cross-module regressions before opening the PR.

- [ ] **Step 7: Push branch + open PR**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-weather push -u origin feat/probe-weather
gh pr create --base main --head feat/probe-weather \
  --title "Weather feature family probe — verdict <V>" \
  --body "$(cat <<'EOF'
## Summary
- Probe-only family check on 4 weather features (wind_speed_mph, is_high_wind, temperature_f, is_grass_surface) sourced from existing SchedulesSchema columns. No new ingest, no schema changes.
- Verdict: <V>. [Headline framing.]
- Spec: `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`.
- Plan: `docs/superpowers/plans/2026-05-07-weather-feature-family-probe.md`.
- Reports: `reports/feature_probe_weather_summary.md`, `reports/feature_probe_weather_override_audit.md`.

## Test plan
- [x] `pytest -v -k "weather or schedules or schemas"` — green
- [x] `mypy src tests scripts` strict — zero violations
- [x] `ruff check src tests scripts` — zero violations
- [x] `ruff format --check src tests scripts` — no drift
- [x] Full `pytest -v` — no cross-module regressions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(Update `<V>` and the headline framing once the verdict is in. The PR title and body are filled in at PR-creation time.)

---

## Self-review checklist (post-plan, before handoff)

**Spec coverage:**
- §1.3 success criteria 1 (coverage ≥95%): Task 7 step 1 (run probes; relax to 0.90 if needed).
- §1.3 success criteria 2 (4 reports): Task 7 step 1.
- §1.3 success criteria 3 (lgb-nb composite via --force-composite): Task 7 step 1 lgb-nb runs.
- §1.3 success criteria 4 (mypy / ruff / pytest clean): Task 8 step 5 + 6.
- §1.4 out-of-scope (production integration, precipitation ingest, refined units): documented as deferred in Task 7 summary template + Task 8 PM/TODO entries; no plan task for them.
- §2 source data — already ingested: confirmed in plan File Structure section ("Untouched (deliberately): src/projections/schemas.py").
- §3.1-§3.5 feature definitions: Task 1 (compute_weather_features tests cover outdoor / dome / closed / NaN propagation / surface variants).
- §3.6 schema integration deferred: documented in Task 7 summary template + plan File Structure.
- §4.1 new files: Tasks 1-4 + Tasks 6, 7 reports.
- §4.2 modified files: Tasks 5 (CONTRIBUTING.md) + 8 (TODO.md, project_management.md).
- §4.3 interface (compute / attach / build_overrides): Tasks 1, 2, 3.
- §4.4 CLI: Task 4.
- §5 probe protocol: Task 7.
- §6.1 per-feature unit tests (~12): Task 1 has 14 tests covering all 4 features × outdoor / dome / NaN / boundary cases.
- §6.2 override assembler tests (~3): Task 3 has 3 tests covering happy path + duplicate keys + missing columns.
- §6.3 CLI tests (4): Task 4 has 4 tests.
- §6.4 / §6.5 (no new ingest, no new network smoke): N/A — no plan task.
- §6.6 verification gate: Task 8 step 5 + 6.
- §6.7 audit report: Task 6.

**Type consistency:**
- Compute fn signature: `compute_weather_features(schedules) -> DataFrame` returns `(season, week, team, wind_speed_mph, is_high_wind, temperature_f, is_grass_surface)`. Consistent across spec §4.3, Task 1 implementation, Task 2 attach call site, Task 3 build_weather_overrides call site.
- Attach fn signature: `attach_weather_features(index, schedules) -> DataFrame`. Consistent across spec §4.3, Task 2 implementation, Task 3 call site.
- Build_overrides signature: `build_weather_overrides(schedules, player_team_week_index) -> DataFrame`. Consistent across spec §4.3, Task 3 implementation, Task 4 CLI call site.
- Feature column names (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`): identical across Tasks 1, 2, 3, 4, 6, 7.
- Required index columns (`gsis_id`, `season`, `week`, `team`, `opp`, `position`): consistent between Task 3's `_REQUIRED_INDEX_COLS` and Task 4's `_build_player_team_week_index` output.

**Placeholder scan:** none. Audit / summary / PM templates contain explicit placeholders (`X`, `Y`, `<V>`) that are filled in at execution time with actual numbers / verdicts; these are documented as fillable inputs, not "TBD" placeholders.
