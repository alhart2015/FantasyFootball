"""Refresh per-season schedule + Vegas line data from `nfl_data_py.import_schedules`.

One parquet partition per season (consistent with weekly_stats — schedules at
~272 rows/year are tiny so no further partitioning).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import _PYARROW_STR, SchedulesSchema, normalize_team_code
from projections.store import write_partition

_KEEP = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "surface",
    "roof",
    "temp",
    "wind",
]


def _fetch_raw_schedules(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_schedules(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _build_kickoff(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Combine `gameday` (date string) + `gametime` (HH:MM string) into a
    timezone-aware UTC timestamp series.

    `nfl_data_py.import_schedules` publishes `gametime` as ET wall-clock for
    every game (including international broadcasts), so the parse sequence is:
    parse naive, localize to ``America/New_York`` (zoneinfo handles the
    EDT/EST switch automatically across the Sep-Feb season span), then convert
    to UTC for storage.

    Missing gameday OR gametime → NaT (e.g., flex-scheduled weeks where
    kickoff hasn't been confirmed). DST transition windows (early Sunday
    morning) lie outside any NFL kickoff slot, but ambiguous/nonexistent
    timestamps are mapped to NaT defensively."""
    combined = gameday.astype(str) + " " + gametime.astype(str)
    # Parse naive; the string is ET wall-clock, not UTC.
    parsed = pd.to_datetime(combined, format="%Y-%m-%d %H:%M", errors="coerce")
    parsed = parsed.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    return parsed.astype("datetime64[us, UTC]")


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["kickoff"] = _build_kickoff(df["gameday"], df["gametime"])
    df["home_team"] = df["home_team"].map(_normalize_team).astype(_PYARROW_STR)
    df["away_team"] = df["away_team"].map(_normalize_team).astype(_PYARROW_STR)
    df["game_id"] = df["game_id"].astype(_PYARROW_STR)

    for str_col in ("surface", "roof"):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(_PYARROW_STR)

    for int_col in ("home_moneyline", "away_moneyline", "temp", "wind"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype(pd.Int64Dtype())

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = SchedulesSchema.validate(df)
    return df


def refresh_schedules(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write schedule data for each season. One partition per season.
    Idempotent — re-running a season overwrites that partition only."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_schedules([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "schedules", df, season=season, week=None)
        record_manifest(data_root, table="schedules", season=season, df=df)
        written.append(path)
    return written
