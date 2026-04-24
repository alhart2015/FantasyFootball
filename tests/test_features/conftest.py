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
            # spread_line is the HOME team's spread (nfl_data_py convention).
            # CHI is home dog; MIN/KC (away) favored by 3.5/7.5 -> +3.5/+7.5.
            "spread_line": [3.5, 7.5],
            "total_line": [48.5, 51.0],
            "home_moneyline": pd.array([155, 280], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180, -340], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([55, 55], dtype=pd.Int64Dtype()),
            "wind": pd.array([8, 8], dtype=pd.Int64Dtype()),
        }
    )
