"""Tests for src/projections/preseason/features.py."""

from __future__ import annotations

import pandas as pd

from projections.preseason.features import build_preseason_features
from projections.schemas import Position, Team


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
    """Each row: (gsis_id, week, position, team, depth_rank). Builder reads week=1."""
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
    return pd.DataFrame(rows, columns=["gsis_id", "season", "round", "pick"]).astype(
        {"season": "int32", "round": "Int64", "pick": "Int64"}
    )


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
    assert int(out["depth_chart_rank"].iloc[0]) == 1
