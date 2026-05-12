"""Refresh per-season weekly stats from `nflreadpy.load_player_stats`.

Writes one parquet partition per season (further per-week splitting is
unnecessary at this scale — a season is small).

`nflreadpy` is the official `nflverse` successor to `nfl_data_py` and
follows nflverse's post-2025 release path (``stats_player/stats_player_week_<season>.parquet``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nflreadpy
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
    "attempts",
    "completions",
    "sacks",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "targets",
    "fumbles_lost",
]

# The post-2025 nflverse release renamed several stat columns
# (`interceptions` → `passing_interceptions`, `sacks` → `sacks_suffered`)
# and replaced `recent_team` with `team`. The old names are kept here as
# no-op renames so the legacy-shape fake fixtures under tests/conftest.py
# continue to normalize correctly through the same path.
_RENAME = {
    "player_id": "gsis_id",
    "recent_team": "team",
    "opponent_team": "opponent",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
}


def _fetch_raw_weekly(seasons: list[int]) -> pd.DataFrame:
    return nflreadpy.load_player_stats(seasons=seasons).to_pandas()


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


_FUMBLE_LOST_SOURCES = (
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
)


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # Derive ``fumbles_lost`` (total) from the source-specific columns.
    # Upstream does not provide an aggregated column — only
    # ``rushing_fumbles_lost`` / ``receiving_fumbles_lost`` /
    # ``sack_fumbles_lost``. Sum what's present; default 0 if all are
    # missing (e.g., very old seasons).
    present = [c for c in _FUMBLE_LOST_SOURCES if c in df.columns]
    if present:
        df["fumbles_lost"] = df[list(present)].fillna(0).sum(axis=1)
    else:
        df["fumbles_lost"] = 0

    # Coerce dtypes that upstream returns as floats / int32.
    # WeeklyStatsSchema's Series[int] requires int64; upstream returns
    # int32 for season/week and floats for several stat columns.
    for int_col in (
        "season",
        "week",
        "passing_tds",
        "interceptions",
        "attempts",
        "completions",
        "sacks",
        "rushing_tds",
        "carries",
        "receptions",
        "receiving_tds",
        "targets",
        "fumbles_lost",
    ):
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype("int64")

    for float_col in ("passing_yards", "rushing_yards", "receiving_yards", "receiving_air_yards"):
        if float_col in df.columns:
            df[float_col] = df[float_col].fillna(0.0).astype(float)

    # Filter to offensive positions before normalizing team codes — the
    # post-2025 release format includes defensive/special-teams rows whose
    # `opponent` is sometimes NaN, and `normalize_team_code` rejects nulls.
    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[df["gsis_id"].notna()].copy()

    # Drop player-weeks with zero offensive touch (no attempt, no carry, no
    # target). nflreadpy's post-2025 release format includes inactive-roster
    # rows that the legacy nfl_data_py path omitted; without this filter,
    # baseline mean-stat predictions regress 10-15% downward because the
    # training set picks up ~10% more all-zero rows. See backtest-gate diff
    # captured during the nflreadpy migration. K is implicitly dropped here
    # because kickers never have attempts / carries / targets in the
    # offensive-stat sense — when K modeling lands it will need a separate
    # ingest path keyed off FG / PAT counts.
    df = df[(df["attempts"] > 0) | (df["carries"] > 0) | (df["targets"] > 0)].copy()

    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].map(_normalize_team).astype(_PYARROW_STR)

    df = df[[c for c in _KEEP if c in df.columns]].copy()
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
