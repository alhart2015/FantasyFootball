"""Refresh per-season weekly stats from `nfl_data_py.import_weekly_data`.

Writes one parquet partition per season (further per-week splitting is
unnecessary at this scale — a season is small).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import _PYARROW_STR, Position, WeeklyStatsSchema, normalize_team_code
from projections.store import write_partition

_KEEP = [
    "gsis_id",
    "season",
    "week",
    "position",
    "team",
    "opponent",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
]

_RENAME = {
    "player_id": "gsis_id",
    "recent_team": "team",
    "opponent_team": "opponent",
}


def _fetch_raw_weekly(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_weekly_data(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # Coerce dtypes that nfl_data_py sometimes returns as floats.
    for int_col in (
        "passing_tds",
        "interceptions",
        "rushing_tds",
        "receptions",
        "receiving_tds",
        "fumbles_lost",
    ):
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype(int)

    for float_col in ("passing_yards", "rushing_yards", "receiving_yards"):
        if float_col in df.columns:
            df[float_col] = df[float_col].fillna(0.0).astype(float)

    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].map(_normalize_team).astype(_PYARROW_STR)

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    df = WeeklyStatsSchema.validate(df)
    return df


def refresh_weekly_stats(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write weekly stats for each season. One partition per season.
    Idempotent — re-running a season overwrites that partition only."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_weekly([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "weekly_stats", df, season=season, week=None)
        record_manifest(data_root, table="weekly_stats", season=season, df=df)
        written.append(path)
    return written
