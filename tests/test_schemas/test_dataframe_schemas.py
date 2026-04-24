"""Pandera schema tests — verify schemas accept good DataFrames and reject bad ones."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.schemas import (
    IdMapSchema,
    ProjectionWeeklySchema,
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
            "receptions": [9],
            "receiving_yards": [110.0],
            "receiving_tds": [1],
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
