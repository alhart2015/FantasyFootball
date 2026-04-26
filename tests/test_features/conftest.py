"""Shared synthetic frames for feature-builder tests.

Each fixture returns a SCHEMA-VALID frame (already-normalized, already
typed) — these mimic the output of `read_partition`, not the raw
nfl_data_py response. Tests build features directly from these without
going through the ingest layer."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR


@pytest.fixture
def wr_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 3 WRs across 2 teams (MIN, KC).

    Designed so trailing-4 windows have round-number expectations:
    - Justin Jefferson (MIN, gsis_id=00-0036322): 12/10/8/6 targets weeks 1-4,
      14/12/10/8 weeks 5-8.
    - Jaylen Reed (MIN, gsis_id=00-0036323, made-up): 4/4/4/4 targets weeks 1-4,
      6/6/6/6 weeks 5-8.
    - Rashee Rice (KC, gsis_id=00-0034950): 8/8/8/8 targets weeks 1-4,
      10/10/10/10 weeks 5-8.

    All three play opponent rotation: weeks 1-4 vs DET, weeks 5-8 vs CHI."""
    rows = []
    for week in range(1, 9):
        opp = "DET" if week <= 4 else "CHI"

        # Jefferson (MIN)
        jef_targets = [12, 10, 8, 6, 14, 12, 10, 8][week - 1]
        rows.append(
            {
                "gsis_id": "00-0036322",
                "season": 2024,
                "week": week,
                "position": "WR",
                "team": "MIN",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": jef_targets - 2,
                "receiving_yards": float(jef_targets * 12),
                "receiving_tds": 1,
                "receiving_air_yards": float(jef_targets * 14),
                "targets": jef_targets,
                "fumbles_lost": 0,
            }
        )

        # Reed (MIN secondary WR — needed so target_share isn't 100%)
        reed_targets = 4 if week <= 4 else 6
        rows.append(
            {
                "gsis_id": "00-0036323",
                "season": 2024,
                "week": week,
                "position": "WR",
                "team": "MIN",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": reed_targets - 1,
                "receiving_yards": float(reed_targets * 8),
                "receiving_tds": 0,
                "receiving_air_yards": float(reed_targets * 9),
                "targets": reed_targets,
                "fumbles_lost": 0,
            }
        )

        # Rice (KC)
        rice_targets = 8 if week <= 4 else 10
        rows.append(
            {
                "gsis_id": "00-0034950",
                "season": 2024,
                "week": week,
                "position": "WR",
                "team": "KC",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": rice_targets - 2,
                "receiving_yards": float(rice_targets * 11),
                "receiving_tds": 0,
                "receiving_air_yards": float(rice_targets * 13),
                "targets": rice_targets,
                "fumbles_lost": 0,
            }
        )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_snap_counts(wr_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the same WRs/weeks. ~95% snap pct uniformly."""
    rows = []
    for _, r in wr_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 60,
                "offense_pct": 0.95,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 2,
                "st_pct": 0.05,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_depth_charts() -> pd.DataFrame:
    """Depth chart snapshot for week 5 of 2024 (the typical as_of_week
    for tests). Jefferson + Rice as WR1; Reed as WR2."""
    rows = [
        {
            "gsis_id": "00-0036322",
            "season": 2024,
            "week": 5,
            "team": "MIN",
            "position": "WR",
            "depth_team": "WR1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0036323",
            "season": 2024,
            "week": 5,
            "team": "MIN",
            "position": "WR",
            "depth_team": "WR2",
            "depth_rank": 2,
        },
        {
            "gsis_id": "00-0034950",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "WR",
            "depth_team": "WR1",
            "depth_rank": 1,
        },
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_ngs_receiving() -> pd.DataFrame:
    """NGS receiving snapshots through week 4 of 2024 for the 3 WRs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0036322",
                    "season": 2024,
                    "week": week,
                    "team": "MIN",
                    "position": "WR",
                    "avg_cushion": 5.0,
                    "avg_separation": 3.2,
                    "avg_intended_air_yards": 12.0,
                    "percent_share_of_intended_air_yards": 30.0,
                    "receptions": 9,
                    "targets": 12,
                    "catch_percentage": 75.0,
                    "yards": 110,
                    "rec_touchdowns": 1,
                    "avg_yac": 4.0,
                    "avg_expected_yac": 3.5,
                    "avg_yac_above_expectation": 0.5,
                },
                {
                    "gsis_id": "00-0036323",
                    "season": 2024,
                    "week": week,
                    "team": "MIN",
                    "position": "WR",
                    "avg_cushion": 6.5,
                    "avg_separation": 2.8,
                    "avg_intended_air_yards": 9.0,
                    "percent_share_of_intended_air_yards": 15.0,
                    "receptions": 3,
                    "targets": 4,
                    "catch_percentage": 75.0,
                    "yards": 32,
                    "rec_touchdowns": 0,
                    "avg_yac": 3.0,
                    "avg_expected_yac": 3.0,
                    "avg_yac_above_expectation": 0.0,
                },
                {
                    "gsis_id": "00-0034950",
                    "season": 2024,
                    "week": week,
                    "team": "KC",
                    "position": "WR",
                    "avg_cushion": 5.5,
                    "avg_separation": 3.0,
                    "avg_intended_air_yards": 10.0,
                    "percent_share_of_intended_air_yards": 25.0,
                    "receptions": 6,
                    "targets": 8,
                    "catch_percentage": 75.0,
                    "yards": 88,
                    "rec_touchdowns": 0,
                    "avg_yac": 4.5,
                    "avg_expected_yac": 4.0,
                    "avg_yac_above_expectation": 0.5,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: MIN @ CHI, KC @ CHI (made up to keep both
    WR teams pointing at the same opponent for trivially-checkable opponent
    proxy joins)."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_MIN_CHI", "2024_05_KC_CHI"], dtype=_PYARROW_STR),
            "home_team": pd.array(["CHI", "CHI"], dtype=_PYARROW_STR),
            "away_team": pd.array(["MIN", "KC"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            # nfl_data_py.import_schedules convention (verified empirically
            # against 2023 data): positive spread_line = HOME favored,
            # negative spread_line = AWAY favored. INVERTS standard sportsbook
            # convention. CHI is home dog vs MIN/KC favored on the road, so
            # spread_line is negative: -3.5 / -7.5.
            "spread_line": [-3.5, -7.5],
            "total_line": [48.5, 51.0],
            "home_moneyline": pd.array([155, 280], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180, -340], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([55, 55], dtype=pd.Int64Dtype()),
            "wind": pd.array([8, 8], dtype=pd.Int64Dtype()),
        }
    )


@pytest.fixture
def qb_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 2 QBs across 2 teams (KC, MIN).

    Designed so trailing-4 windows have round-number expectations:
    - Patrick Mahomes (KC, gsis_id=00-0034857): 36/38/40/42 attempts weeks 1-4,
      36/38/40/42 weeks 5-8. Trailing-4 mean attempts = 39.0 either way.
    - Kirk Cousins (MIN, gsis_id=00-0033106): 30/30/30/30 attempts uniformly.

    Both play opponent rotation: weeks 1-4 vs DEN, weeks 5-8 vs CHI.
    No rushing usage (pure pocket QBs in this fixture).
    """
    rows = []
    for week in range(1, 9):
        opp = "DEN" if week <= 4 else "CHI"

        # Mahomes (KC)
        mahomes_attempts = [36, 38, 40, 42, 36, 38, 40, 42][week - 1]
        rows.append(
            {
                "gsis_id": "00-0034857",
                "season": 2024,
                "week": week,
                "position": "QB",
                "team": "KC",
                "opponent": opp,
                "passing_yards": float(mahomes_attempts * 7.5),
                "passing_tds": 2,
                "interceptions": 1,
                "attempts": mahomes_attempts,
                "completions": int(mahomes_attempts * 0.65),
                "sacks": 2,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        )

        # Cousins (MIN)
        rows.append(
            {
                "gsis_id": "00-0033106",
                "season": 2024,
                "week": week,
                "position": "QB",
                "team": "MIN",
                "opponent": opp,
                "passing_yards": 250.0,
                "passing_tds": 1,
                "interceptions": 0,
                "attempts": 30,
                "completions": 20,
                "sacks": 1,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def qb_snap_counts(qb_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the same QBs/weeks. 100% snap pct (full-time starters)."""
    rows = []
    for _, r in qb_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 65,
                "offense_pct": 1.0,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 0,
                "st_pct": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def qb_depth_charts() -> pd.DataFrame:
    """Depth chart snapshot for week 5 of 2024. Both QBs as starters."""
    rows = [
        {
            "gsis_id": "00-0034857",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "QB",
            "depth_team": "QB1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0033106",
            "season": 2024,
            "week": 5,
            "team": "MIN",
            "position": "QB",
            "depth_team": "QB1",
            "depth_rank": 1,
        },
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def qb_ngs_passing() -> pd.DataFrame:
    """NGS passing snapshots through week 4 of 2024 for the 2 QBs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0034857",
                    "season": 2024,
                    "week": week,
                    "team": "KC",
                    "position": "QB",
                    "avg_time_to_throw": 2.71,
                    "avg_completed_air_yards": 6.2,
                    "avg_intended_air_yards": 8.1,
                    "avg_air_yards_differential": -1.9,
                    "aggressiveness": 12.5,
                    "max_completed_air_distance": 42.0,
                    "avg_air_yards_to_sticks": -0.4,
                    "completion_percentage": 68.5,
                    "expected_completion_percentage": 65.2,
                    "completion_percentage_above_expectation": 3.3,
                    "avg_air_distance": 9.5,
                    "max_air_distance": 55.0,
                },
                {
                    "gsis_id": "00-0033106",
                    "season": 2024,
                    "week": week,
                    "team": "MIN",
                    "position": "QB",
                    "avg_time_to_throw": 2.45,
                    "avg_completed_air_yards": 5.8,
                    "avg_intended_air_yards": 7.2,
                    "avg_air_yards_differential": -1.4,
                    "aggressiveness": 10.0,
                    "max_completed_air_distance": 38.0,
                    "avg_air_yards_to_sticks": -0.7,
                    "completion_percentage": 66.7,
                    "expected_completion_percentage": 66.0,
                    "completion_percentage_above_expectation": 0.7,
                    "avg_air_distance": 8.5,
                    "max_air_distance": 48.0,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def qb_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: KC @ CHI, MIN @ CHI (made up to share opponent)."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_KC_CHI", "2024_05_MIN_CHI"], dtype=_PYARROW_STR),
            "home_team": pd.array(["CHI", "CHI"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "MIN"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [
                -7.5,
                -3.5,
            ],  # KC favored by 7.5; MIN favored by 3.5 (away favored = negative)
            "total_line": [51.0, 48.5],
            "home_moneyline": pd.array([280, 155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-340, -180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([55, 55], dtype=pd.Int64Dtype()),
            "wind": pd.array([8, 8], dtype=pd.Int64Dtype()),
        }
    )


@pytest.fixture
def rb_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 2 RBs across 2 teams (PHI, SF).

    - Saquon Barkley (PHI, 00-0034796): 20 carries/game uniformly, 2 targets.
    - Christian McCaffrey (SF, 00-0036650): 14 carries/game, 6 targets/game (passing-down back).
    """
    rows = []
    for week in range(1, 9):
        opp = "DAL" if week <= 4 else "SEA"
        # Saquon — workhorse runner
        rows.append(
            {
                "gsis_id": "00-0034796",
                "season": 2024,
                "week": week,
                "position": "RB",
                "team": "PHI",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": 90.0,
                "rushing_tds": 1,
                "carries": 20,
                "receptions": 1,
                "receiving_yards": 8.0,
                "receiving_tds": 0,
                "receiving_air_yards": 5.0,
                "targets": 2,
                "fumbles_lost": 0,
            }
        )
        # CMC — pass-catching back
        rows.append(
            {
                "gsis_id": "00-0036650",
                "season": 2024,
                "week": week,
                "position": "RB",
                "team": "SF",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": 65.0,
                "rushing_tds": 0,
                "carries": 14,
                "receptions": 5,
                "receiving_yards": 42.0,
                "receiving_tds": 0,
                "receiving_air_yards": 28.0,
                "targets": 6,
                "fumbles_lost": 0,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def rb_snap_counts(rb_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the RBs. ~85% pct (workhorse/feature backs)."""
    rows = []
    for _, r in rb_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 55,
                "offense_pct": 0.85,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 5,
                "st_pct": 0.15,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def rb_depth_charts() -> pd.DataFrame:
    """Depth chart for week 5 of 2024. Both RBs as RB1."""
    rows = [
        {
            "gsis_id": "00-0034796",
            "season": 2024,
            "week": 5,
            "team": "PHI",
            "position": "RB",
            "depth_team": "RB1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0036650",
            "season": 2024,
            "week": 5,
            "team": "SF",
            "position": "RB",
            "depth_team": "RB1",
            "depth_rank": 1,
        },
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def rb_ngs_rushing() -> pd.DataFrame:
    """NGS rushing snapshots through week 4 of 2024 for the 2 RBs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0034796",
                    "season": 2024,
                    "week": week,
                    "team": "PHI",
                    "position": "RB",
                    "efficiency": 3.1,
                    "percent_attempts_gte_eight_defenders": 25.0,
                    "avg_time_to_los": 2.95,
                    "rush_attempts": 20,
                    "rush_yards": 90,
                    "expected_rush_yards": 80.0,
                    "rush_yards_over_expected": 10.0,
                    "avg_rush_yards": 4.5,
                    "rush_yards_over_expected_per_att": 0.5,
                    "rush_pct_over_expected": 12.5,
                },
                {
                    "gsis_id": "00-0036650",
                    "season": 2024,
                    "week": week,
                    "team": "SF",
                    "position": "RB",
                    "efficiency": 3.4,
                    "percent_attempts_gte_eight_defenders": 18.0,
                    "avg_time_to_los": 2.80,
                    "rush_attempts": 14,
                    "rush_yards": 65,
                    "expected_rush_yards": 60.0,
                    "rush_yards_over_expected": 5.0,
                    "avg_rush_yards": 4.6,
                    "rush_yards_over_expected_per_att": 0.4,
                    "rush_pct_over_expected": 8.0,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def rb_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: PHI @ SEA, SF @ SEA."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_PHI_SEA", "2024_05_SF_SEA"], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA", "SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["PHI", "SF"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [-2.5, -3.5],  # both away teams favored
            "total_line": [45.0, 48.5],
            "home_moneyline": pd.array([135, 155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-160, -180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf", "fieldturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([62, 62], dtype=pd.Int64Dtype()),
            "wind": pd.array([6, 6], dtype=pd.Int64Dtype()),
        }
    )


@pytest.fixture
def te_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 2 TEs + 1 supporting WR per team.

    The supporting WR (matching target volume) makes target_share within
    the pass-catching group = 0.5 for each TE — easy to verify by hand.

    - Travis Kelce (KC, 00-0030506): 8 targets/game uniformly.
    - George Kittle (SF, 00-0033084): 6 targets/game uniformly. Plan 3b
      treats this id as a Taysom-Hill-shape TE for rushing-feature
      coverage: 6 carries / 28 rushing yards / 1 rushing TD per game so
      the trailing-4 rushing-mean rollups have non-zero data to verify.
    - Rashee Rice (KC WR, 00-0034950): 8 targets/game (matches Kelce).
    - Brandon Aiyuk (SF WR, 00-0035716): 6 targets/game (matches Kittle).
    """
    rows = []
    for week in range(1, 9):
        opp = "DEN" if week <= 4 else "ARI"
        # Kelce (TE, KC)
        rows.append(
            _make_receiver_row(
                "00-0030506", "TE", "KC", opp, week, targets=8, recs=6, yds=70, tds=1
            )
        )
        # Kittle (TE, SF) — doubles as the Taysom-Hill-shape rushing TE for
        # Plan 3b's rushing-feature coverage.
        rows.append(
            _make_receiver_row(
                "00-0033084",
                "TE",
                "SF",
                opp,
                week,
                targets=6,
                recs=4,
                yds=55,
                tds=0,
                carries=6,
                rushing_yards=28.0,
                rushing_tds=1,
            )
        )
        # Rice (WR, KC)
        rows.append(
            _make_receiver_row(
                "00-0034950", "WR", "KC", opp, week, targets=8, recs=6, yds=88, tds=0
            )
        )
        # Aiyuk (WR, SF)
        rows.append(
            _make_receiver_row(
                "00-0035716", "WR", "SF", opp, week, targets=6, recs=4, yds=58, tds=0
            )
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


def _make_receiver_row(
    gsis_id: str,
    position: str,
    team: str,
    opp: str,
    week: int,
    *,
    targets: int,
    recs: int,
    yds: float,
    tds: int,
    carries: int = 0,
    rushing_yards: float = 0.0,
    rushing_tds: int = 0,
) -> dict[str, object]:
    """Helper to build a synthetic receiver-shaped weekly_stats row.

    Optional rushing fields default to zero so existing call sites are
    unaffected; Plan 3b uses them to give Kittle a non-zero rushing
    workload for the TE rushing-feature rollup tests.
    """
    return {
        "gsis_id": gsis_id,
        "season": 2024,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opp,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": float(rushing_yards),
        "rushing_tds": rushing_tds,
        "carries": carries,
        "receptions": recs,
        "receiving_yards": float(yds),
        "receiving_tds": tds,
        "receiving_air_yards": float(yds * 1.2),
        "targets": targets,
        "fumbles_lost": 0,
    }


@pytest.fixture
def te_snap_counts(te_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the TE-test cohort. ~92% pct uniformly."""
    rows = []
    for _, r in te_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 60,
                "offense_pct": 0.92,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 2,
                "st_pct": 0.05,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def te_depth_charts() -> pd.DataFrame:
    """Depth chart for week 5 of 2024. Both TEs as TE1."""
    rows = [
        {
            "gsis_id": "00-0030506",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "TE",
            "depth_team": "TE1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0033084",
            "season": 2024,
            "week": 5,
            "team": "SF",
            "position": "TE",
            "depth_team": "TE1",
            "depth_rank": 1,
        },
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def te_ngs_receiving() -> pd.DataFrame:
    """NGS receiving snapshots through week 4 of 2024 for the 2 TEs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0030506",
                    "season": 2024,
                    "week": week,
                    "team": "KC",
                    "position": "TE",
                    "avg_cushion": 4.5,
                    "avg_separation": 2.8,
                    "avg_intended_air_yards": 9.0,
                    "percent_share_of_intended_air_yards": 22.0,
                    "receptions": 6,
                    "targets": 8,
                    "catch_percentage": 75.0,
                    "yards": 70,
                    "rec_touchdowns": 1,
                    "avg_yac": 4.0,
                    "avg_expected_yac": 3.5,
                    "avg_yac_above_expectation": 0.5,
                },
                {
                    "gsis_id": "00-0033084",
                    "season": 2024,
                    "week": week,
                    "team": "SF",
                    "position": "TE",
                    "avg_cushion": 4.0,
                    "avg_separation": 3.2,
                    "avg_intended_air_yards": 8.0,
                    "percent_share_of_intended_air_yards": 18.0,
                    "receptions": 4,
                    "targets": 6,
                    "catch_percentage": 66.7,
                    "yards": 55,
                    "rec_touchdowns": 0,
                    "avg_yac": 5.0,
                    "avg_expected_yac": 4.5,
                    "avg_yac_above_expectation": 0.5,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def te_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: KC @ ARI, SF @ ARI."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_KC_ARI", "2024_05_SF_ARI"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ARI", "ARI"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "SF"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [-7.5, -1.5],
            "total_line": [49.0, 47.0],
            "home_moneyline": pd.array([280, 105], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-340, -125], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["closed", "closed"], dtype=_PYARROW_STR),  # ARI has retractable
            "temp": pd.array([72, 72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0, 0], dtype=pd.Int64Dtype()),
        }
    )
