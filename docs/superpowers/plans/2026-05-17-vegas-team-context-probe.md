# Vegas Team-Context Feature Family Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe whether a forward-looking Vegas team-context feature bundle (preseason + season-to-date encodings of `spread_line` / `total_line`) carries orthogonal signal beyond v1 + every Track 2 family already shipped.

**Architecture:** New `compute_vegas_team_context_features` module wraps the existing `_shared.build_game_environment` helper to broadcast each team-season's week-1 line values (`preseason_*`) and compute leakage-safe expanding-mean-shift-1 of `(spread, implied_team_total)` (`season_avg_*`). A CLI override generator mirrors `scripts/build_weather_override.py`. The existing `scripts/probe_feature_signal.py` runner consumes the override unchanged — 4 probe invocations across (Baseline × lgb-nb) × (augment × swap) produce per-position verdicts.

**Tech Stack:** Python 3.11, pandas, pandera (schema validation), Ridge / LightGBM-NB (probe runner), pytest.

**Spec:** `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`

---

## File map

**New files:**
- `src/projections/features/vegas_team_context_features.py` — compute_fn + attach + build_overrides
- `scripts/build_vegas_team_context_override.py` — override generator CLI
- `tests/test_features/test_vegas_team_context_features.py` — ~15 feature unit tests
- `tests/test_scripts/test_build_vegas_team_context_override_cli.py` — 5 CLI tests
- `reports/feature_probe_vegas_team_context_override_audit.md` — generated audit (Task 8)
- `reports/feature_probe_vegas_team_context_baseline_augment.{md,csv}` — generated probe output (Task 9)
- `reports/feature_probe_vegas_team_context_baseline_swap.{md,csv}` — generated probe output (Task 9)
- `reports/feature_probe_vegas_team_context_lgbnb_augment.{md,csv}` — generated probe output (Task 10)
- `reports/feature_probe_vegas_team_context_lgbnb_swap.{md,csv}` — generated probe output (Task 10)
- `reports/feature_probe_vegas_team_context_summary.md` — synthesis (Task 11)

**Modified files (verification + docs only):**
- `project_management.md` — append "Vegas Team-Context Probe" entry (Task 12)
- `TODO.md` — strike 33c, flip recommended next direction (Task 12)

---

## Task 1: Compute fn skeleton + preseason broadcast tests

**Files:**
- Create: `src/projections/features/vegas_team_context_features.py`
- Create: `tests/test_features/test_vegas_team_context_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_vegas_team_context_features.py`:

```python
"""Vegas team-context feature computes — tests."""

from __future__ import annotations

import pandas as pd


def _make_schedule_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic schedules frame with sane defaults for unspecified columns.

    Mirrors `SchedulesSchema`'s shape. Tests fill in only the columns the function
    under test reads, defaults the rest.
    """
    defaults: dict[str, object] = {
        "season": 2024,
        "week": 1,
        "game_id": "2024_01_HOME_AWAY",
        "home_team": "HOME",
        "away_team": "AWAY",
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
        out.append({**defaults, **r})
    df = pd.DataFrame(out)
    df["temp"] = df["temp"].astype(pd.Int64Dtype())
    df["wind"] = df["wind"].astype(pd.Int64Dtype())
    df["surface"] = df["surface"].astype(pd.StringDtype("pyarrow"))
    df["roof"] = df["roof"].astype(pd.StringDtype("pyarrow"))
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True).astype("datetime64[us, UTC]")
    return df


def test_compute_returns_two_rows_per_game_and_four_feature_cols() -> None:
    """One schedule row → two output rows (home + away). Both carry
    preseason_implied_team_total / preseason_spread; both have NaN
    season_avg_* at week 1 (cold-start)."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,  # KC favored by 3 (home favored → +spread_line)
                "total_line": 48.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)

    assert len(out) == 2
    assert set(out["team"]) == {"KC", "BAL"}
    for team, expected_spread, expected_itt in (
        ("KC", -3.0, 25.5),  # favorite: spread = -3, ITT = (48+3)/2
        ("BAL", 3.0, 22.5),  # dog: spread = +3, ITT = (48-3)/2
    ):
        row = out.loc[out["team"] == team].iloc[0]
        assert row["preseason_spread"] == expected_spread
        assert row["preseason_implied_team_total"] == expected_itt
        # season_avg_* NaN at week 1 (no prior games)
        assert pd.isna(row["season_avg_spread"])
        assert pd.isna(row["season_avg_implied_team_total"])


def test_compute_preseason_broadcasts_across_all_weeks() -> None:
    """A team's week-1 spread + ITT values are broadcast across all weeks of
    that team-season — same values appear in week 4, week 10, etc."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # Week 1: KC home vs BAL, KC favored by 3
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            # Week 2: KC away vs PHI, KC favored by 1 (away favored → -spread_line)
            {"season": 2024, "week": 2, "home_team": "PHI", "away_team": "KC",
             "spread_line": -1.0, "total_line": 46.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)

    kc_rows = out.loc[out["team"] == "KC"].sort_values("week")
    # preseason_* broadcasts week-1 values to all weeks of KC's season
    assert (kc_rows["preseason_spread"] == -3.0).all()
    assert (kc_rows["preseason_implied_team_total"] == 25.5).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_vegas_team_context_features.py -v`
Expected: FAIL with "No module named projections.features.vegas_team_context_features"

- [ ] **Step 3: Implement compute_vegas_team_context_features (preseason broadcast only)**

Create `src/projections/features/vegas_team_context_features.py`:

```python
"""Vegas team-context features for the 33c family probe.

Sourced from `SchedulesSchema` columns (`spread_line`, `total_line`) already
in `data/raw/schedules`. Wraps `_shared.build_game_environment` to produce
per-team-game `(spread, implied_team_total)` rows in the canonical
sign convention (team-perspective: favorite negative, dog positive).

Two mechanism axes per spec §3:
  1. `preseason_*` — team's week-1 game values, broadcast across all weeks of
     the season. Vegas's preseason team-strength view.
  2. `season_avg_*` — leakage-safe expanding mean of weeks 1..N-1. As-of-time
     market view of team strength.

Probe-only — features land in the override parquet, not in `*FeaturesSchema`.
Integration follow-up is conditional on the family-probe verdict per
`docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.features._shared import build_game_environment
from projections.schemas import GSIS_ID_PATTERN

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")

_FEATURE_COLS: Final[tuple[str, ...]] = (
    "preseason_implied_team_total",
    "preseason_spread",
    "season_avg_implied_team_total",
    "season_avg_spread",
)


def compute_vegas_team_context_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team-game frame with four Vegas team-context features.

    One row per (game, team) — each schedules row produces two output rows
    (home + away). Both teams in a matchup carry their own
    team-perspective values (the home team's spread is the negative of the
    away team's spread, etc.).

    Spec §3:
      - `preseason_*`: broadcast week-1-game values across all weeks of the
        season for that team.
      - `season_avg_*`: expanding mean of weeks 1..N-1 (leakage-safe via
        .shift(1)). NaN at week 1.

    Args:
        schedules: frame validated against `SchedulesSchema` (must carry
            season, week, home_team, away_team, spread_line, total_line, roof).

    Returns:
        DataFrame sorted by (season, week, team) with columns:
            season, week, team,
            preseason_implied_team_total, preseason_spread,
            season_avg_implied_team_total, season_avg_spread.
        All four feature columns are Float64. season / week are Int64;
        team is StringDtype("pyarrow") (inherited from build_game_environment).
    """
    game_env = build_game_environment(schedules)
    # Keep only the columns we need; cast spread / implied_team_total to Float64
    # for downstream-Float64 consistency (build_game_environment returns float).
    games = game_env[["season", "week", "team", "spread", "implied_team_total"]].copy()
    games["spread"] = games["spread"].astype("Float64")
    games["implied_team_total"] = games["implied_team_total"].astype("Float64")

    # Preseason broadcast: for each (season, team), look up the min-week row,
    # broadcast its spread + implied_team_total across all weeks.
    sorted_games = games.sort_values(["season", "team", "week"])
    first_week_idx = sorted_games.groupby(["season", "team"], as_index=False).head(1)
    preseason = first_week_idx[["season", "team", "spread", "implied_team_total"]].rename(
        columns={
            "spread": "preseason_spread",
            "implied_team_total": "preseason_implied_team_total",
        }
    )
    out = games.merge(preseason, on=["season", "team"], how="left")

    # season_avg_*: expanding mean shifted by 1 (so week-N row sees only
    # weeks 1..N-1). NaN at week 1.
    out = out.sort_values(["season", "team", "week"]).reset_index(drop=True)
    grouped = out.groupby(["season", "team"], group_keys=False)
    out["season_avg_spread"] = grouped["spread"].apply(
        lambda s: s.expanding().mean().shift(1)
    )
    out["season_avg_implied_team_total"] = grouped["implied_team_total"].apply(
        lambda s: s.expanding().mean().shift(1)
    )

    return out[
        [
            "season",
            "week",
            "team",
            "preseason_implied_team_total",
            "preseason_spread",
            "season_avg_implied_team_total",
            "season_avg_spread",
        ]
    ].sort_values(["season", "week", "team"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_vegas_team_context_features.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/vegas_team_context_features.py tests/test_features/test_vegas_team_context_features.py
git commit -m "feat(33c): compute_vegas_team_context_features preseason broadcast"
```

---

## Task 2: season_avg_* expanding-mean tests

**Files:**
- Modify: `tests/test_features/test_vegas_team_context_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_vegas_team_context_features.py`:

```python
def test_compute_season_avg_is_expanding_mean_shifted_by_one() -> None:
    """At week N, season_avg_* = mean of (spread, implied_team_total) over
    weeks 1..N-1. Leakage-safe: week N does NOT include its own value."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # KC vs BAL, week 1: KC favored 3, total 48 -> KC ITT=25.5, BAL ITT=22.5
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            # KC vs PHI, week 2: KC favored 1 (away favored), total 46 -> KC ITT=23.5
            {"season": 2024, "week": 2, "home_team": "PHI", "away_team": "KC",
             "spread_line": -1.0, "total_line": 46.0},
            # KC vs NYJ, week 3: KC home favored 5, total 50 -> KC ITT=27.5
            {"season": 2024, "week": 3, "home_team": "KC", "away_team": "NYJ",
             "spread_line": 5.0, "total_line": 50.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].sort_values("week")
    # Week 1: NaN (no prior games)
    assert pd.isna(kc.iloc[0]["season_avg_spread"])
    assert pd.isna(kc.iloc[0]["season_avg_implied_team_total"])
    # Week 2: mean of week 1 only -> KC spread = -3, ITT = 25.5
    assert kc.iloc[1]["season_avg_spread"] == -3.0
    assert kc.iloc[1]["season_avg_implied_team_total"] == 25.5
    # Week 3: mean of weeks 1+2 -> KC spread = (-3 + -1) / 2 = -2, ITT = (25.5 + 23.5) / 2 = 24.5
    assert kc.iloc[2]["season_avg_spread"] == -2.0
    assert kc.iloc[2]["season_avg_implied_team_total"] == 24.5


def test_compute_season_avg_skips_bye_weeks_correctly() -> None:
    """A team's bye week produces no schedule row; the expanding mean updates
    only on weeks with actual games."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # KC week 1: home favored 3, total 48 -> KC ITT=25.5
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            # KC bye in week 2 (no row)
            # KC week 3: away favored 1, total 46 -> KC ITT=23.5
            {"season": 2024, "week": 3, "home_team": "PHI", "away_team": "KC",
             "spread_line": -1.0, "total_line": 46.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].sort_values("week")
    # Two rows: week 1 + week 3 (bye week 2 produces no row)
    assert list(kc["week"]) == [1, 3]
    # Week 3 sees week 1's value (only prior game)
    assert kc.iloc[1]["season_avg_spread"] == -3.0
    assert kc.iloc[1]["season_avg_implied_team_total"] == 25.5


def test_compute_independent_seasons_do_not_leak() -> None:
    """season_avg_* resets at the start of each season. Week 1 of 2024 is
    NaN regardless of 2023's values."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # 2023 KC games
            {"season": 2023, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            {"season": 2023, "week": 2, "home_team": "KC", "away_team": "PHI",
             "spread_line": 7.0, "total_line": 52.0},
            # 2024 KC week 1
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "NYJ",
             "spread_line": 5.0, "total_line": 50.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc_2024_wk1 = out.loc[(out["team"] == "KC") & (out["season"] == 2024) & (out["week"] == 1)].iloc[0]
    assert pd.isna(kc_2024_wk1["season_avg_spread"])
    assert pd.isna(kc_2024_wk1["season_avg_implied_team_total"])
    # And 2024 KC preseason_* uses 2024-week-1 line, not 2023's
    assert kc_2024_wk1["preseason_spread"] == -5.0
    assert kc_2024_wk1["preseason_implied_team_total"] == 27.5


def test_compute_sign_convention_favorite_negative_dog_positive() -> None:
    """Sanity-check: a team that was the favorite in its week-1 game has
    negative preseason_spread; a dog has positive. Matches
    _shared.build_game_environment's team-perspective convention."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # 2023 KC vs DET week 1: KC home favored ~6 (real-world value, sign-flipped for spread_line=+)
            {"season": 2023, "week": 1, "home_team": "KC", "away_team": "DET",
             "spread_line": 6.5, "total_line": 53.0},
            # 2023 ARI vs WAS week 1: ARI home dog ~6.5 (negative spread_line means away favored)
            {"season": 2023, "week": 1, "home_team": "ARI", "away_team": "WAS",
             "spread_line": -6.5, "total_line": 41.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].iloc[0]
    ari = out.loc[out["team"] == "ARI"].iloc[0]
    assert kc["preseason_spread"] < 0  # favored
    assert ari["preseason_spread"] > 0  # dog


def test_compute_handles_nan_spread_line_propagates_nan() -> None:
    """A schedule row with NaN spread_line or total_line produces NaN in
    derived spread / implied_team_total / season_avg_* (NaN propagation
    via build_game_environment)."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": float("nan"), "total_line": 48.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].iloc[0]
    assert pd.isna(kc["preseason_spread"])
    assert pd.isna(kc["preseason_implied_team_total"])


def test_compute_returns_sorted_by_season_week_team() -> None:
    """Output row order is (season, week, team) ascending — caller convenience."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 3, "home_team": "BAL", "away_team": "NYJ",
             "spread_line": 4.0, "total_line": 45.0},
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
        ]
    )
    out = compute_vegas_team_context_features(sch)
    # Lexicographic on (season, week, team)
    assert list(out["week"]) == [1, 1, 3, 3]
```

- [ ] **Step 2: Run tests to verify they pass**

The compute_vegas_team_context_features from Task 1 already implements all this logic. Run:
`pytest tests/test_features/test_vegas_team_context_features.py -v`
Expected: 8 PASS (2 from Task 1, 6 from Task 2).

If any FAIL: debug by reading the failure and adjusting the compute fn. Most likely failure mode is the `groupby().apply()` for expanding mean returning the wrong shape — check that `group_keys=False` is set.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_vegas_team_context_features.py
git commit -m "test(33c): season_avg expanding-mean + sign-convention + NaN tests"
```

---

## Task 3: attach_vegas_team_context_features

**Files:**
- Modify: `src/projections/features/vegas_team_context_features.py`
- Modify: `tests/test_features/test_vegas_team_context_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_vegas_team_context_features.py`:

```python
def _make_index_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic player-team-week index with canonical dtypes."""
    defaults: dict[str, object] = {
        "gsis_id": "00-0033873",
        "season": 2024,
        "week": 1,
        "team": "KC",
        "opp": "BAL",
        "position": "QB",
    }
    out = []
    for r in rows:
        out.append({**defaults, **r})
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


def test_attach_appends_four_feature_cols_via_left_merge() -> None:
    """attach_vegas_team_context_features adds the 4 feature cols to the
    index by left-merge on (season, week, team)."""
    from projections.features.vegas_team_context_features import (
        attach_vegas_team_context_features,
    )

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC"},
            {"gsis_id": "00-0033873", "season": 2024, "week": 2, "team": "KC"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            {"season": 2024, "week": 2, "home_team": "PHI", "away_team": "KC",
             "spread_line": -1.0, "total_line": 46.0},
        ]
    )
    out = attach_vegas_team_context_features(idx, sch)

    assert len(out) == 2
    assert set(out.columns) >= {
        "gsis_id", "season", "week", "team", "opp", "position",
        "preseason_implied_team_total", "preseason_spread",
        "season_avg_implied_team_total", "season_avg_spread",
    }
    wk1 = out.loc[out["week"] == 1].iloc[0]
    wk2 = out.loc[out["week"] == 2].iloc[0]
    # Preseason values identical across weeks (broadcast from week 1)
    assert wk1["preseason_spread"] == wk2["preseason_spread"] == -3.0
    # Week 2 season_avg sees week 1 only
    assert wk2["season_avg_spread"] == -3.0


def test_attach_index_row_without_matching_schedule_gets_nan() -> None:
    """Index row whose (season, week, team) doesn't match any schedule
    retains NaN in all 4 feature cols."""
    from projections.features.vegas_team_context_features import (
        attach_vegas_team_context_features,
    )

    idx = _make_index_rows(
        [
            # Index has KC week 1 + week 3; schedule has only week 1.
            {"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC"},
            {"gsis_id": "00-0033873", "season": 2024, "week": 3, "team": "KC"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
        ]
    )
    out = attach_vegas_team_context_features(idx, sch)

    wk3 = out.loc[out["week"] == 3].iloc[0]
    assert pd.isna(wk3["preseason_spread"])
    assert pd.isna(wk3["preseason_implied_team_total"])
    assert pd.isna(wk3["season_avg_spread"])
    assert pd.isna(wk3["season_avg_implied_team_total"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_vegas_team_context_features.py::test_attach_appends_four_feature_cols_via_left_merge -v`
Expected: FAIL with "cannot import name 'attach_vegas_team_context_features'".

- [ ] **Step 3: Add attach_vegas_team_context_features**

Append to `src/projections/features/vegas_team_context_features.py` (after `compute_vegas_team_context_features`):

```python
def attach_vegas_team_context_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge the four Vegas team-context features onto a player-team-week index.

    Args:
        index: frame with at least (season, week, team) columns. Typically
            the player-team-week index from
            `scripts.build_vegas_team_context_override._build_player_team_week_index`,
            carrying (gsis_id, season, week, team, opp, position).
        schedules: frame validated against `SchedulesSchema`.

    Returns:
        Copy of index with four nullable Float64 cols appended:
        preseason_implied_team_total, preseason_spread,
        season_avg_implied_team_total, season_avg_spread.
        Index rows without a matching (season, week, team) in schedules
        retain NaN in all four cols.
    """
    feats = compute_vegas_team_context_features(schedules)
    return index.merge(feats, on=["season", "week", "team"], how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_vegas_team_context_features.py -v`
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/vegas_team_context_features.py tests/test_features/test_vegas_team_context_features.py
git commit -m "feat(33c): attach_vegas_team_context_features left-merge"
```

---

## Task 4: build_vegas_team_context_overrides

**Files:**
- Modify: `src/projections/features/vegas_team_context_features.py`
- Modify: `tests/test_features/test_vegas_team_context_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_vegas_team_context_features.py`:

```python
import pytest


def test_build_overrides_returns_canonical_columns() -> None:
    """build_vegas_team_context_overrides returns the exact override-parquet
    shape: (gsis_id, season, week, position, 4 feature cols), one row per
    input index row."""
    from projections.features.vegas_team_context_features import (
        build_vegas_team_context_overrides,
    )

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC",
             "position": "QB"},
            {"gsis_id": "00-0033873", "season": 2024, "week": 2, "team": "KC",
             "position": "QB"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            {"season": 2024, "week": 2, "home_team": "PHI", "away_team": "KC",
             "spread_line": -1.0, "total_line": 46.0},
        ]
    )
    out = build_vegas_team_context_overrides(sch, idx)
    assert list(out.columns) == [
        "gsis_id", "season", "week", "position",
        "preseason_implied_team_total", "preseason_spread",
        "season_avg_implied_team_total", "season_avg_spread",
    ]
    assert len(out) == 2


def test_build_overrides_rejects_missing_required_column() -> None:
    """Missing any of (gsis_id, season, week, team, opp, position) in the
    index raises ValueError."""
    from projections.features.vegas_team_context_features import (
        build_vegas_team_context_overrides,
    )

    idx = _make_index_rows(
        [{"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC"}]
    ).drop(columns=["opp"])
    sch = _make_schedule_rows([])
    with pytest.raises(ValueError, match="missing required column"):
        build_vegas_team_context_overrides(sch, idx)


def test_build_overrides_rejects_malformed_gsis_id() -> None:
    """An index row with a gsis_id that doesn't match GSIS_ID_PATTERN raises."""
    from projections.features.vegas_team_context_features import (
        build_vegas_team_context_overrides,
    )

    idx = _make_index_rows(
        [{"gsis_id": "not-a-real-gsis-id", "season": 2024, "week": 1, "team": "KC"}]
    )
    sch = _make_schedule_rows([])
    with pytest.raises(ValueError, match="invalid gsis_id format"):
        build_vegas_team_context_overrides(sch, idx)


def test_build_overrides_rejects_duplicate_keys() -> None:
    """Duplicate (gsis_id, season, week) keys in the index raise."""
    from projections.features.vegas_team_context_features import (
        build_vegas_team_context_overrides,
    )

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC"},
            {"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC"},
        ]
    )
    sch = _make_schedule_rows([])
    with pytest.raises(ValueError, match="duplicate"):
        build_vegas_team_context_overrides(sch, idx)


def test_build_overrides_row_count_invariant() -> None:
    """Output row count equals input index row count (left-merge property)."""
    from projections.features.vegas_team_context_features import (
        build_vegas_team_context_overrides,
    )

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0033873", "season": 2024, "week": 1, "team": "KC"},
            {"gsis_id": "00-0036971", "season": 2024, "week": 1, "team": "BAL"},
            {"gsis_id": "00-0033873", "season": 2024, "week": 2, "team": "KC"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "spread_line": 3.0, "total_line": 48.0},
            {"season": 2024, "week": 2, "home_team": "PHI", "away_team": "KC",
             "spread_line": -1.0, "total_line": 46.0},
        ]
    )
    out = build_vegas_team_context_overrides(sch, idx)
    assert len(out) == len(idx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_vegas_team_context_features.py::test_build_overrides_returns_canonical_columns -v`
Expected: FAIL with "cannot import name 'build_vegas_team_context_overrides'".

- [ ] **Step 3: Add build_vegas_team_context_overrides**

Append to `src/projections/features/vegas_team_context_features.py`:

```python
_REQUIRED_INDEX_COLS: Final[tuple[str, ...]] = (
    "gsis_id", "season", "week", "team", "opp", "position",
)


def build_vegas_team_context_overrides(
    schedules: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Vegas team-context override frame.

    Args:
        schedules: validated against `SchedulesSchema`.
        player_team_week_index: frame with columns (gsis_id, season, week,
            team, opp, position). Must have unique (gsis_id, season, week)
            keys.

    Returns:
        Frame with columns (gsis_id, season, week, position,
        preseason_implied_team_total, preseason_spread,
        season_avg_implied_team_total, season_avg_spread) — one row per
        index input row. Feeds `scripts.probe_feature_signal --override`.

    Raises:
        ValueError: index missing a required column, carrying a malformed
            gsis_id, or carrying duplicate (gsis_id, season, week) keys.
        AssertionError: row-count mismatch after the feature merge
            (internal-invariant violation; signals a regression introducing
            duplicate (season, week, team) keys in compute).
    """
    missing = [c for c in _REQUIRED_INDEX_COLS if c not in player_team_week_index.columns]
    if missing:
        raise ValueError(f"player_team_week_index missing required column(s): {missing}")

    bad_ids = [
        g for g in player_team_week_index["gsis_id"].dropna() if not _GSIS_RE.match(str(g))
    ]
    if bad_ids:
        raise ValueError(
            f"invalid gsis_id format(s): {bad_ids[:3]} "
            f"(and {max(0, len(bad_ids) - 3)} more)"
        )

    key_cols = ["gsis_id", "season", "week"]
    dups = player_team_week_index.duplicated(subset=key_cols)
    if dups.any():
        n = int(dups.sum())
        raise ValueError(
            f"player_team_week_index has {n} duplicate (gsis_id, season, week) keys"
        )

    attached = attach_vegas_team_context_features(player_team_week_index, schedules)
    if len(attached) != len(player_team_week_index):
        raise AssertionError(
            f"row count mismatch: input had {len(player_team_week_index)} rows, "
            f"output has {len(attached)}; suggests a many-to-many merge regression"
        )

    return attached[
        ["gsis_id", "season", "week", "position", *_FEATURE_COLS]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_vegas_team_context_features.py -v`
Expected: 15 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/vegas_team_context_features.py tests/test_features/test_vegas_team_context_features.py
git commit -m "feat(33c): build_vegas_team_context_overrides with input validation"
```

---

## Task 5: CLI scaffold + parse_args tests

**Files:**
- Create: `scripts/build_vegas_team_context_override.py`
- Create: `tests/test_scripts/test_build_vegas_team_context_override_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scripts/test_build_vegas_team_context_override_cli.py`:

```python
"""build_vegas_team_context_override CLI smokes."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_season_range_single_year() -> None:
    from scripts.build_vegas_team_context_override import _parse_season_range

    r = _parse_season_range("2024")
    assert list(r) == [2024]


def test_parse_season_range_inclusive_range() -> None:
    from scripts.build_vegas_team_context_override import _parse_season_range

    r = _parse_season_range("2018-2024")
    assert list(r) == [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def test_parse_args_defaults(tmp_path: Path) -> None:
    from scripts.build_vegas_team_context_override import parse_args

    out = tmp_path / "vegas_team_context.parquet"
    args = parse_args(["--output", str(out)])
    assert args.output == out
    assert list(args.seasons) == list(range(2018, 2025))
    assert args.data_root == Path("data")


def test_parse_args_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """If output exists and --force not set, parser exits."""
    from scripts.build_vegas_team_context_override import parse_args

    out = tmp_path / "vegas_team_context.parquet"
    out.write_text("")  # touch
    with pytest.raises(SystemExit):
        parse_args(["--output", str(out)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scripts/test_build_vegas_team_context_override_cli.py -v`
Expected: 4 FAIL with "No module named scripts.build_vegas_team_context_override".

- [ ] **Step 3: Create the CLI module**

Create `scripts/build_vegas_team_context_override.py`:

```python
"""Build the Vegas team-context override parquet for the 33c family probe.

One-shot CLI. Loads schedules + depth_charts across the requested season
range, builds the player-team-week index, calls
build_vegas_team_context_overrides, writes the resulting frame to a parquet.
Prints audit numbers (per-column coverage, week-1 NaN rate, unique
team-season count, histogram bounds) so a follow-up step can capture them
into reports/feature_probe_vegas_team_context_override_audit.md.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_vegas_team_context_override --seasons 2018-2024
    python -m scripts.build_vegas_team_context_override --seasons 2018-2024 --force
    python -m scripts.build_vegas_team_context_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.vegas_team_context_features import (
    build_vegas_team_context_overrides,
)
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/vegas_team_context.parquet")


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

    Mirrors PR #25 / PR #28's helper. Pins canonical dtypes on the output:
    ``gsis_id`` / ``team`` / ``opp`` / ``position`` -> StringDtype("pyarrow"),
    ``season`` / ``week`` -> Int64Dtype.
    """
    dc = depth_charts[
        depth_charts["season"].isin(seasons) & depth_charts["position"].isin(_FANTASY_POSITIONS)
    ][["gsis_id", "season", "week", "team", "position"]].drop_duplicates(
        subset=["gsis_id", "season", "week"]
    )
    sch = schedules[schedules["season"].isin(seasons)][
        ["season", "week", "home_team", "away_team"]
    ]
    home = sch.rename(columns={"home_team": "team", "away_team": "opp"})
    away = sch.rename(columns={"away_team": "team", "home_team": "opp"})
    team_opp = pd.concat([home, away], ignore_index=True)[["season", "week", "team", "opp"]]
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
    `scripts.build_weather_override.parse_args`."""
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


def _print_audit(overrides: pd.DataFrame) -> None:
    """Print audit numbers for `reports/feature_probe_vegas_team_context_override_audit.md`.

    Numbers reported per spec §6.7:
        - Per-column coverage rate (% non-NaN).
        - Week-1 NaN rate on season_avg_* (expected ~6%).
        - Unique team-season count per season (expected 32).
        - Histogram bounds: min/max/mean of each feature col.
    """
    n = len(overrides)
    print(f"vegas_team_context override audit ({n} rows):")

    feature_cols = (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    )
    for col in feature_cols:
        coverage = overrides[col].notna().mean() * 100.0
        print(f"  {col} coverage: {coverage:.2f}%")

    # week-1 NaN rate on season_avg_*
    wk1 = overrides[overrides["week"] == 1]
    n_wk1 = len(wk1)
    if n_wk1 > 0:
        for col in ("season_avg_spread", "season_avg_implied_team_total"):
            wk1_nan_pct = wk1[col].isna().mean() * 100.0
            print(f"  {col} week-1 NaN rate: {wk1_nan_pct:.2f}% (expected ~100%)")

    # Unique team-season count
    for season, group in overrides.groupby("season"):
        n_unique = group[["preseason_spread"]].drop_duplicates().shape[0]
        print(f"  season {season}: {n_unique} unique preseason_spread values (expected 32)")

    # Histogram bounds
    for col in feature_cols:
        s = overrides[col].dropna()
        if len(s) > 0:
            print(
                f"  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seasons: range = args.seasons
    raw_root = args.data_root / "raw"

    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))

    idx = _build_player_team_week_index(depth_charts, schedules, seasons)
    overrides = build_vegas_team_context_overrides(schedules, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    _print_audit(overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scripts/test_build_vegas_team_context_override_cli.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_vegas_team_context_override.py tests/test_scripts/test_build_vegas_team_context_override_cli.py
git commit -m "feat(33c): build_vegas_team_context_override CLI scaffold"
```

---

## Task 6: _print_audit test

**Files:**
- Modify: `tests/test_scripts/test_build_vegas_team_context_override_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scripts/test_build_vegas_team_context_override_cli.py`:

```python
def test_print_audit_includes_coverage_and_week1_rates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`_print_audit` prints per-column coverage rates, week-1 NaN rates,
    unique team-season counts, and histogram bounds. Direct unit test on a
    small synthetic frame — avoids the wall-time of a main() integration test."""
    import pandas as pd
    from scripts.build_vegas_team_context_override import _print_audit

    overrides = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "preseason_implied_team_total": [25.5, 25.5, 22.5, 22.5],
            "preseason_spread": [-3.0, -3.0, 3.0, 3.0],
            "season_avg_implied_team_total": [
                float("nan"), 25.5, float("nan"), 22.5,
            ],
            "season_avg_spread": [float("nan"), -3.0, float("nan"), 3.0],
        }
    )

    _print_audit(overrides)
    captured = capsys.readouterr()
    # Coverage rates printed for each feature col
    assert "preseason_spread coverage:" in captured.out
    assert "season_avg_spread coverage:" in captured.out
    # Week-1 NaN rate printed
    assert "season_avg_spread week-1 NaN rate:" in captured.out
    # Unique-team-season count per season
    assert "season 2024:" in captured.out
    # Histogram bounds
    assert "preseason_spread: min=" in captured.out
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_build_vegas_team_context_override_cli.py::test_print_audit_includes_coverage_and_week1_rates -v`
Expected: PASS (the audit fn is already in Task 5's CLI module).

- [ ] **Step 3: Commit**

```bash
git add tests/test_scripts/test_build_vegas_team_context_override_cli.py
git commit -m "test(33c): _print_audit covers coverage rates + week-1 NaN + unique counts"
```

---

## Task 7: Verification gates — feature module + CLI

**Files:**
- (verification only)

- [ ] **Step 1: Run pytest subset**

Run: `pytest tests/test_features/test_vegas_team_context_features.py tests/test_scripts/test_build_vegas_team_context_override_cli.py -v`
Expected: 20 PASS, 0 FAIL.

- [ ] **Step 2: Run mypy strict**

Run: `mypy src tests scripts`
Expected: 0 errors. If `mypy` reports issues:
- Most likely culprit: `groupby().apply()` return type. Annotate the lambda's return with an explicit `pd.Series`.
- Fix and re-run.

- [ ] **Step 3: Run ruff check + format**

Run: `ruff check src tests scripts`
Expected: 0 violations.

Run: `ruff format --check src tests scripts`
Expected: 0 reformatting needed.

If ruff format wants to reformat, run `ruff format src tests scripts` and commit the formatting.

- [ ] **Step 4: Run broader subset to catch cross-module regressions**

Run: `pytest -v -k "vegas_team_context or schedules or schemas or _shared"`
Expected: all PASS.

- [ ] **Step 5: If formatting was applied, commit**

```bash
git add -u
git commit -m "style(33c): ruff format vegas team-context module"
```

(Skip this step if ruff format was already clean.)

---

## Task 8: Generate override + audit report (real data)

**Files:**
- Create: `reports/feature_probe_vegas_team_context_override_audit.md`
- Generate: `data/features_probe/vegas_team_context.parquet` (NOT committed)

- [ ] **Step 1: Generate the override parquet**

Run:
```bash
$env:PATH = ".venv\Scripts;" + $env:PATH
python -m scripts.build_vegas_team_context_override --seasons 2018-2024 > /tmp/vegas_audit_raw.txt 2>&1
cat /tmp/vegas_audit_raw.txt
```

(On Windows PowerShell the `2>&1` redirect captures stderr; the script's stdout includes "wrote N rows to ..." + audit numbers.)

Expected output:
- File `data/features_probe/vegas_team_context.parquet` created.
- Console shows `wrote ~38000-50000 rows to data/features_probe/vegas_team_context.parquet` (rows depend on depth-chart density across 2018-2024).
- Audit lists per-column coverage (preseason_* > 99%, season_avg_* ~93-94%), week-1 NaN rate on season_avg_* (~100%), unique team-season counts (32 per season), histogram bounds.

If coverage on `preseason_*` is below 99% or unique-team-season count is not 32 per season:
- Investigate: read `data/raw/schedules/season=*/part.parquet` for NaN `spread_line` / `total_line` rows.
- Most likely cause: a particular season had a postponed week-1 game (rare). Document in the audit report.

- [ ] **Step 2: Write the audit report**

Create `reports/feature_probe_vegas_team_context_override_audit.md` using the captured audit output. Template:

```markdown
# Vegas team-context override — audit

Generated by `python -m scripts.build_vegas_team_context_override --seasons 2018-2024`
on YYYY-MM-DD.

Output: `data/features_probe/vegas_team_context.parquet` (NOT committed; regenerable).

## Per-column coverage

| Column | Coverage |
|---|---|
| `preseason_implied_team_total` | XX.XX% |
| `preseason_spread` | XX.XX% |
| `season_avg_implied_team_total` | XX.XX% |
| `season_avg_spread` | XX.XX% |

Expected: `preseason_*` ≥99% (a team-season with all 4 cols non-null requires
the week-1 game's `spread_line` + `total_line` non-null in pbp). `season_avg_*`
~93-94% (week-1 rows are NaN by construction — ~6% of rows; the remainder is
the small `<1%` of subsequent weeks with missing pbp lines).

## Week-1 NaN rate on season_avg_*

`season_avg_spread` week-1 NaN: ~100% (expected — cold-start by construction).
`season_avg_implied_team_total` week-1 NaN: ~100%.

## Unique team-season counts

| Season | Unique preseason_spread values |
|---|---|
| 2018 | 32 |
| 2019 | 32 |
| 2020 | 32 |
| 2021 | 32 |
| 2022 | 32 |
| 2023 | 32 |
| 2024 | 32 |

(Expected 32 per season — one preseason value per team.)

## Histogram bounds

| Column | min | max | mean |
|---|---:|---:|---:|
| `preseason_implied_team_total` | X.XX | X.XX | X.XX |
| `preseason_spread` | X.XX | X.XX | X.XX |
| `season_avg_implied_team_total` | X.XX | X.XX | X.XX |
| `season_avg_spread` | X.XX | X.XX | X.XX |

## Sign-convention sanity (spec §3.6 verification)

Filter `data/features_probe/vegas_team_context.parquet` to a known
team-season:

- 2023 KC (favored vs DET week 1 by ~6.5): `preseason_spread` is negative ✓
- 2023 ARI (dog vs WAS week 1 by ~6.5): `preseason_spread` is positive ✓

(Run `python -c "import pandas as pd; df = pd.read_parquet('data/features_probe/vegas_team_context.parquet'); print(df[df['season']==2023][['preseason_spread']].drop_duplicates())"` to verify.)

## Notes / anomalies

(Fill in any postponed weeks, missing data, or unusual histogram bounds here.)
```

Fill in the actual numbers from the audit captured in Step 1. Run the sign-convention verification command and confirm.

- [ ] **Step 3: Commit the audit report**

```bash
git add reports/feature_probe_vegas_team_context_override_audit.md
git commit -m "report(33c): override audit — coverage, week-1 cold-start, sign-convention sanity"
```

---

## Task 9: Run BaselineModel probes (augment + swap)

**Files:**
- Create: `reports/feature_probe_vegas_team_context_baseline_augment.md`
- Create: `reports/feature_probe_vegas_team_context_baseline_augment.csv`
- Create: `reports/feature_probe_vegas_team_context_baseline_swap.md`
- Create: `reports/feature_probe_vegas_team_context_baseline_swap.csv`

- [ ] **Step 1: Run BaselineModel × augment**

Run:
```bash
$env:PATH = ".venv\Scripts;" + $env:PATH
python -m scripts.probe_feature_signal `
    --candidate-name "vegas_team_context_baseline_augment" `
    --override data/features_probe/vegas_team_context.parquet `
    --model baseline `
    --seasons 2018-2024 `
    --holdout-years 2021-2024 `
    --coverage-threshold 0.90 `
    --csv-out reports/feature_probe_vegas_team_context_baseline_augment.csv `
    > reports/feature_probe_vegas_team_context_baseline_augment.md
```

Expected:
- Phase 1 runs across QB / RB / WR / TE with RidgeCV per stat.
- Phase 2 runs only if Phase 1 finds a pooled SIGNAL cell.
- Coverage check passes at 0.90 (cold-start week-1 NaN accounted for).
- Total runtime: 5-15 minutes.

If coverage check fails:
- Lower threshold to actual observed coverage (e.g. 0.88) and re-run, documenting the relaxation in the summary report.
- Do NOT silently widen — record the actual rate per spec §5.4.

- [ ] **Step 2: Run BaselineModel × swap**

Run:
```bash
python -m scripts.probe_feature_signal `
    --candidate-name "vegas_team_context_baseline_swap" `
    --override data/features_probe/vegas_team_context.parquet `
    --drop implied_team_total,spread `
    --model baseline `
    --seasons 2018-2024 `
    --holdout-years 2021-2024 `
    --coverage-threshold 0.90 `
    --csv-out reports/feature_probe_vegas_team_context_baseline_swap.csv `
    > reports/feature_probe_vegas_team_context_baseline_swap.md
```

Expected: same shape as augment; Phase 2 gates on a Phase 1 pooled SIGNAL.

- [ ] **Step 3: Inspect outputs**

Read the two `.md` files. Check:
- Phase 1 verdict per position (count of SIGNAL / NULL / REGRESSION cells).
- Phase 2 verdict if fired (ADOPT / MARGINAL / DO_NOT_ADOPT per position).
- Any "REGRESSION" cells — flag for the summary report.

Note any deviations from the pre-registered mechanism prediction (spec §1.2):
- Predicted: RB swap most likely SIGNAL.
- Actual: ???

- [ ] **Step 4: Commit the BaselineModel reports**

```bash
git add reports/feature_probe_vegas_team_context_baseline_augment.md `
        reports/feature_probe_vegas_team_context_baseline_augment.csv `
        reports/feature_probe_vegas_team_context_baseline_swap.md `
        reports/feature_probe_vegas_team_context_baseline_swap.csv
git commit -m "report(33c): BaselineModel augment + swap probe outputs"
```

---

## Task 10: Run lgb-nb probes (augment + swap, force-composite)

**Files:**
- Create: `reports/feature_probe_vegas_team_context_lgbnb_augment.md`
- Create: `reports/feature_probe_vegas_team_context_lgbnb_augment.csv`
- Create: `reports/feature_probe_vegas_team_context_lgbnb_swap.md`
- Create: `reports/feature_probe_vegas_team_context_lgbnb_swap.csv`

- [ ] **Step 1: Run lgb-nb × augment with --force-composite**

Run:
```bash
$env:PATH = ".venv\Scripts;" + $env:PATH
python -m scripts.probe_feature_signal `
    --candidate-name "vegas_team_context_lgbnb_augment" `
    --override data/features_probe/vegas_team_context.parquet `
    --model lightgbm-nb `
    --seasons 2018-2024 `
    --holdout-years 2021-2024 `
    --coverage-threshold 0.90 `
    --force-composite `
    --csv-out reports/feature_probe_vegas_team_context_lgbnb_augment.csv `
    > reports/feature_probe_vegas_team_context_lgbnb_augment.md
```

Expected:
- Phase 1 is tautological with baseline augment (RidgeCV regardless of `--model`).
- Phase 2 runs unconditionally (`--force-composite`); this is the informative cell for lgb-nb.
- Runtime: 1-2 hours (NB tree fits are slow). Run in background or expect to wait.

If the run errors mid-way:
- Most likely: out-of-memory on a large position-season cell. Reduce `--seasons` (e.g., `2020-2024`) and re-run; this changes the training span but keeps the holdout window intact.
- Document the deviation in the summary report.

- [ ] **Step 2: Run lgb-nb × swap with --force-composite**

Run:
```bash
python -m scripts.probe_feature_signal `
    --candidate-name "vegas_team_context_lgbnb_swap" `
    --override data/features_probe/vegas_team_context.parquet `
    --drop implied_team_total,spread `
    --model lightgbm-nb `
    --seasons 2018-2024 `
    --holdout-years 2021-2024 `
    --coverage-threshold 0.90 `
    --force-composite `
    --csv-out reports/feature_probe_vegas_team_context_lgbnb_swap.csv `
    > reports/feature_probe_vegas_team_context_lgbnb_swap.md
```

Expected: same shape; ~1-2 hour runtime.

- [ ] **Step 3: Inspect outputs**

Read the two `.md` files. Same checks as Task 9 Step 3 — Phase 2 ADOPT count per position, any REGRESSION cells. Compare lgb-nb composite verdict to baseline composite verdict (does lgb-nb extract signal Ridge missed, or is it the same story?).

- [ ] **Step 4: Commit the lgb-nb reports**

```bash
git add reports/feature_probe_vegas_team_context_lgbnb_augment.md `
        reports/feature_probe_vegas_team_context_lgbnb_augment.csv `
        reports/feature_probe_vegas_team_context_lgbnb_swap.md `
        reports/feature_probe_vegas_team_context_lgbnb_swap.csv
git commit -m "report(33c): lgb-nb force-composite augment + swap probe outputs"
```

---

## Task 11: Synthesis summary report

**Files:**
- Create: `reports/feature_probe_vegas_team_context_summary.md`

- [ ] **Step 1: Write the summary report**

Create `reports/feature_probe_vegas_team_context_summary.md`:

```markdown
# Vegas team-context family probe — summary

**Branch:** `feat/probe-vegas-team-context`
**Date:** YYYY-MM-DD
**Spec:** `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`

## Decision log

| Date | Commit | Decision |
|---|---|---|
| YYYY-MM-DD | <hash> | Spec approved + plan started |
| YYYY-MM-DD | <hash> | Override generated, audit passed |
| YYYY-MM-DD | <hash> | BaselineModel probes complete |
| YYYY-MM-DD | <hash> | lgb-nb probes complete |

## Per-(model, mode) verdict table

| Model | Mode | Phase 1 SIGNAL cells | Phase 2 verdict per position | Notes |
|---|---|---|---|---|
| BaselineModel | augment | X/Y | QB:?, RB:?, WR:?, TE:? | |
| BaselineModel | swap | X/Y | QB:?, RB:?, WR:?, TE:? | |
| lgb-nb composite | augment (forced) | (tautological) | QB:?, RB:?, WR:?, TE:? | |
| lgb-nb composite | swap (forced) | (tautological) | QB:?, RB:?, WR:?, TE:? | |

(Fill in actual counts and verdicts from Task 9 + 10 reports.)

## Mechanism annotation

**Pre-registered prediction (spec §1.2):**
- Most likely SIGNAL: RB swap mode (preseason team-strength as forward-looking RB-rushing-volume proxy).
- Possible SIGNAL at Phase 1: passing_yards / receiving_yards at QB / WR.
- Predicted NULL at TE.
- Predicted tautological at lgb-nb augment (trees get signal from existing continuous cols).

**Observed:**
- (Describe per-position outcomes vs prediction.)
- Did `preseason_*` cols carry the signal, or `season_avg_*`? Check the per-stat Phase 1 cells.
- Did the per-stat SIGNAL cells (if any) decompose to a composite SIGNAL?

## Coverage relaxation

- Default threshold: 0.95 (PR #28 weather precedent).
- Used: 0.90 (cold-start week-1 NaN on `season_avg_*` accounted for).
- Actual observed coverage:
  - `preseason_implied_team_total`: XX.XX%
  - `preseason_spread`: XX.XX%
  - `season_avg_implied_team_total`: XX.XX%
  - `season_avg_spread`: XX.XX%

## Refined-unit candidates left unexplored

Per spec §1.4 + §8:
- External Vegas data (season win totals, preseason O/U) from a non-pbp source.
- Non-linear encodings of existing Vegas cols (`is_favored`, `is_heavy_favorite`, `is_high_total`, spread×ITT interaction).
- Opponent-side rollups (opp's `season_avg_*`).
- Line-movement features (open-to-close spread movement).
- Position-specific encodings (`preseason_pass_volume_proxy = preseason_implied_team_total × pass_rate_prior`).
- 2025 eval extension (requires `refresh_features --seasons 2025` first).

## Recurring QB augment regression check

PRs #23 / #24 / #25 / #28 each saw QB augment regress on context / team / trajectory / weather adds. Does this pattern recur on Vegas team-context?

- BaselineModel × augment, QB Phase 2 verdict: ???
- lgb-nb × augment, QB Phase 2 verdict: ???

## Mechanism reflection on the 33c hypothesis

- **If NULL:** the pbp-derivable preseason proxy (week-1 closing line) is not enough to capture the forward-looking signal TODO #33c hypothesizes. Next step is external preseason data (true May win totals, OC/HC tenure, FA flags) or a pivot to a different forward-looking class.
- **If SIGNAL:** greenlight a per-position integration plan extending `_shared.build_game_environment` to emit the 4 cols, refreshing feature caches, running the dual-run adoption gate. The 33c hypothesis is partially validated at this approximation level.

## Predicted vs actual mechanism

(Describe whether the mechanism prediction held. If it didn't, what does the actual pattern suggest about the next probe?)
```

Fill in actual numbers from Task 9 + 10 reports.

- [ ] **Step 2: Commit the summary**

```bash
git add reports/feature_probe_vegas_team_context_summary.md
git commit -m "report(33c): family-probe summary — <verdict in 1 sentence>"
```

(Verdict in commit message: e.g., "NO SIGNAL durable across all 8 cells", "SIGNAL on RB swap Phase 2", etc.)

---

## Task 12: Documentation updates

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Update project_management.md**

Append a new entry at the top of `project_management.md` (the "Append new entries at the top" convention):

```markdown
## Vegas Team-Context Feature Family Probe — <verdict> (YYYY-MM-DD, on branch `feat/probe-vegas-team-context`)

**Status:** Spec + plan + impl on `feat/probe-vegas-team-context`. Phase 1 of TODO #33c. New compute module `src/projections/features/vegas_team_context_features.py` produces 4 candidate cols (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) from `SchedulesSchema`'s already-ingested `spread_line` / `total_line`. Override generator CLI `scripts/build_vegas_team_context_override.py`. Probe runs via existing `scripts/probe_feature_signal.py` (no changes). Spec at `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`; plan at `docs/superpowers/plans/2026-05-17-vegas-team-context-probe.md`.

**Verdict: <ONE-LINE SUMMARY>.** (Fill in based on Task 11 summary report.)

**Shipped surface:**
- `src/projections/features/vegas_team_context_features.py` — `compute_vegas_team_context_features`, `attach_vegas_team_context_features`, `build_vegas_team_context_overrides`.
- `scripts/build_vegas_team_context_override.py` — override generator CLI.
- 20 new tests (15 feature + 5 CLI).
- 6 report artifacts: override audit + 4 probe outputs + summary.

**Decision log:**
- Hybrid 4-col bundle (preseason × 2 + season-to-date × 2) per spec §3.
- Coverage threshold relaxed to 0.90 for cold-start week-1 NaN on `season_avg_*`.
- Eval window 2021-2024 holdout; 2025 deferred to a follow-up.
- Schema integration deferred — probe-only.

**Recommended next direction:**
- (If SIGNAL): integration plan extending `_shared.build_game_environment` + per-position schema updates + adoption gate.
- (If NULL): pivot to non-pbp forward-looking signals (TODO #33c residual) — external preseason ADP / win totals / coaching tenure / FA flags.

See `reports/feature_probe_vegas_team_context_summary.md` for the full per-(model, mode) verdict.
```

Replace `<verdict>`, `<ONE-LINE SUMMARY>`, etc. with actuals from Task 11.

- [ ] **Step 2: Update TODO.md**

In TODO.md, find the 33c entry (line ~620-630 in the post-PR-47 TODO; search for `33c — forward-looking Vegas`) and append a status block:

```markdown
**33c — Phase 1 family probe complete, YYYY-MM-DD: <VERDICT>.** Probe shipped on branch `feat/probe-vegas-team-context`. 4-col bundle (preseason × 2 + season-to-date × 2) from already-ingested `spread_line` / `total_line`. Probe matrix: BaselineModel × {augment, swap} + lgb-nb × {augment, swap}, all with `--force-composite` where applicable. Result: <SUMMARY OF VERDICT TABLE>. Next step: <SIGNAL → integration plan | NULL → external Vegas / forward-looking signal pivot>.

See `reports/feature_probe_vegas_team_context_summary.md`.
```

- [ ] **Step 3: Commit documentation updates**

```bash
git add project_management.md TODO.md
git commit -m "docs(33c): probe verdict + recommended next direction"
```

---

## Task 13: Final verification gate

**Files:**
- (verification only)

- [ ] **Step 1: Full pytest**

Run: `pytest -v`
Expected: all PASS (modulo the pre-existing `test_dispatch_default_model_class_for_wr_is_unchanged` failure — documented in project_management.md PR #45 section).

If any other tests fail: debug and fix. Do not ship a probe with regressions.

- [ ] **Step 2: mypy + ruff final pass**

Run: `mypy src tests scripts`
Run: `ruff check src tests scripts`
Run: `ruff format --check src tests scripts`
Expected: all 0 violations.

- [ ] **Step 3: Push branch + open PR**

```bash
git push -u origin feat/probe-vegas-team-context
```

Open PR via `gh pr create`. PR body should reference the summary report and quote the headline verdict.

```bash
gh pr create --title "feat(33c): Vegas team-context feature family probe" --body "$(cat <<'EOF'
## Summary

- Phase 1 of TODO #33c. 4-col bundle (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) derived from already-ingested `spread_line` / `total_line` in `SchedulesSchema`. No new ingest.
- Probe ran across BaselineModel × {augment, swap} + lgb-nb × {augment, swap} with `--force-composite`. Verdict: <ONE-LINE FROM SUMMARY>.
- 20 new tests; mypy + ruff clean; full pytest clean.

## Verdict

See `reports/feature_probe_vegas_team_context_summary.md` for the full per-(model, mode) verdict table + mechanism annotation. Headline: <verdict>.

## Test plan

- [x] Full `pytest -v` clean (modulo pre-existing PR #45-tracked failure).
- [x] `mypy src tests scripts` — 0 violations.
- [x] `ruff check + format --check` — 0 violations.
- [x] Override audit (`reports/feature_probe_vegas_team_context_override_audit.md`) — coverage + sign-convention sanity verified.

Spec: `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`
Plan: `docs/superpowers/plans/2026-05-17-vegas-team-context-probe.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

**Spec coverage check (run after writing the plan):**
- §1 goal + success criteria → Tasks 1-7 (build + verify) + Tasks 8-11 (run + report).
- §3 feature definitions → Tasks 1-2 (compute_fn implementation maps directly).
- §4 module structure → Tasks 1, 3, 4, 5 (files match).
- §5 probe protocol — 4 modes × 4 reports → Tasks 9 + 10.
- §6 testing strategy — 15 feature tests + 5 CLI tests → Tasks 1, 2, 3, 4, 5, 6.
- §6.7 audit report → Task 8.
- §7 decision log → Captured in Task 11 summary report.
- §8 risks → Documented in Task 11 summary's coverage relaxation + mechanism reflection sections.
- §9 predecessor pointers → Used as templates for Tasks 1, 5.

**Identifier consistency check:**
- `compute_vegas_team_context_features` — defined Task 1, used Tasks 3, 4. ✓
- `attach_vegas_team_context_features` — defined Task 3, used Task 4. ✓
- `build_vegas_team_context_overrides` — defined Task 4, used Task 5 CLI's `main`. ✓
- `_FEATURE_COLS` constant — defined Task 1, used Task 4 build_overrides. ✓
- `_REQUIRED_INDEX_COLS` — defined Task 4. ✓
- `_GSIS_RE` — defined Task 1 (imported pattern from schemas). ✓
- CLI args (`--seasons`, `--data-root`, `--output`, `--force`) — defined Task 5, used Tasks 8, 9, 10. ✓
- `--coverage-threshold 0.90` (relaxation) — used consistently Tasks 9, 10; rationale in spec §3.5 + §5.4 + §7. ✓
- `--drop implied_team_total,spread` (swap mode) — used Tasks 9, 10. ✓

**Placeholder scan:** No "TBD", "TODO", "implement later", or "similar to Task N" in this plan.
