"""Pandera schema tests — verify schemas accept good DataFrames and reject bad ones."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.schemas import (
    _PYARROW_STR,
    IdMapSchema,
    ProjectionWeeklySchema,
    SchedulesSchema,
    WeeklyStatsSchema,
)


def _good_weekly_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "season": [2024],
            "week": [3],
            "position": ["WR"],
            "team": ["MIN"],
            "opponent": ["HOU"],
            "passing_yards": [0.0],
            "passing_tds": [0],
            "interceptions": [0],
            "rushing_yards": [0.0],
            "rushing_tds": [0],
            "carries": [0],
            "receptions": [9],
            "receiving_yards": [110.0],
            "receiving_tds": [1],
            "receiving_air_yards": [145.0],
            "targets": [12],
            "fumbles_lost": [0],
        }
    )


def test_weekly_stats_accepts_good_frame() -> None:
    WeeklyStatsSchema.validate(_good_weekly_stats())


def test_weekly_stats_rejects_bad_position() -> None:
    bad = _good_weekly_stats()
    bad["position"] = "FB"
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_weekly_stats_rejects_week_out_of_range() -> None:
    bad = _good_weekly_stats()
    bad["week"] = 99
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_weekly_stats_rejects_bad_gsis_id() -> None:
    bad = _good_weekly_stats()
    bad["gsis_id"] = "not-an-id"
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_id_map_schema() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "espn_id": ["4262921"],
            "sleeper_id": ["6794"],
            "pfr_id": ["JeffJu00"],
            "full_name": ["Justin Jefferson"],
            "position": ["WR"],
            "team": ["MIN"],
        }
    )
    IdMapSchema.validate(df)


def test_projection_weekly_schema() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "season": [2026],
            "week": [3],
            "position": ["WR"],
            "team": ["MIN"],
            "opponent": ["HOU"],
            "ruleset": ["ESPN_PPR"],
            "family": ["GAMMA"],
            "params": [b"\x00"],
            "mean": [18.4],
            "p10": [6.1],
            "p50": [17.2],
            "p90": [33.5],
            "model_id": ["baseline:abc123:2014-2025"],
            "generated_at": [pd.Timestamp("2026-09-01", tz="UTC")],
        }
    )
    ProjectionWeeklySchema.validate(df)


def test_schedules_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "game_id": pd.array(["2024_03_KC_ATL", "2024_03_MIN_HOU"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ATL", "HOU"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "MIN"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-09-22T17:00:00Z", "2024-09-22T17:00:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [3.5, -2.5],
            "total_line": [48.5, 44.0],
            "home_moneyline": pd.array([155, -125], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180, 105], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf", "matrixturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["dome", "dome"], dtype=_PYARROW_STR),
            "temp": pd.array([72, 72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0, 0], dtype=pd.Int64Dtype()),
        }
    )
    SchedulesSchema.validate(df)


def test_schedules_schema_rejects_unknown_team_code() -> None:
    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "game_id": pd.array(["2024_03_XX_ATL"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ATL"], dtype=_PYARROW_STR),
            "away_team": pd.array(["XX"], dtype=_PYARROW_STR),  # invalid
            "kickoff": pd.to_datetime(["2024-09-22T17:00:00Z"], utc=True).as_unit("us"),
            "spread_line": [3.5],
            "total_line": [48.5],
            "home_moneyline": pd.array([155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["dome"], dtype=_PYARROW_STR),
            "temp": pd.array([72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0], dtype=pd.Int64Dtype()),
        }
    )
    with pytest.raises(SchemaError):
        SchedulesSchema.validate(df)


def test_schedules_schema_allows_nullable_lines() -> None:
    """Future-week games may have NaN spread/total/kickoff."""
    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [18],
            "game_id": pd.array(["2024_18_TBD_TBD"], dtype=_PYARROW_STR),
            "home_team": pd.array(["KC"], dtype=_PYARROW_STR),
            "away_team": pd.array(["DEN"], dtype=_PYARROW_STR),
            "kickoff": pd.array([pd.NaT], dtype="datetime64[us, UTC]"),
            # spread_line/total_line are Series[float] (float64). Use float("nan")
            # (not pd.NA) so the column stays float64 rather than object — this
            # matches what nfl_data_py's import_schedules actually returns for
            # missing lines.
            "spread_line": [float("nan")],
            "total_line": [float("nan")],
            "home_moneyline": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "surface": pd.array([pd.NA], dtype=_PYARROW_STR),
            "roof": pd.array([pd.NA], dtype=_PYARROW_STR),
            "temp": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "wind": pd.array([pd.NA], dtype=pd.Int64Dtype()),
        }
    )
    SchedulesSchema.validate(df)
