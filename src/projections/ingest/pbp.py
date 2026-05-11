"""Refresh per-season play-by-play from `nflreadpy.load_pbp`.

Writes one parquet partition per season (curated subset of upstream's ~370
columns; see PbpSchema). Idempotent — re-running a season overwrites that
partition only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import nflreadpy
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import _PYARROW_STR, GSIS_ID_PATTERN, PbpSchema, normalize_team_code
from projections.store import write_partition

_KEEP: tuple[str, ...] = (
    "play_id",
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "qb_dropback",
    "qb_scramble",
    "sack",
    "rush_attempt",
    "pass_attempt",
    "epa",
    "wpa",
    "success",
    "air_yards",
    "yards_after_catch",
    "complete_pass",
    "xpass",
    "pass_oe",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
)

_GSIS_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")

# float64 coercion list: upstream sometimes returns float32 (the legacy
# nfl_data_py path explicitly downcast); PbpSchema's Series[float] fields
# require float64. Real-data drift caught in Plan 9 Phase 6 (synthetic
# fixture used Python floats which are float64).
_FLOAT64_COLS: tuple[str, ...] = (
    "qb_dropback",
    "qb_scramble",
    "sack",
    "rush_attempt",
    "pass_attempt",
    "epa",
    "wpa",
    "success",
    "air_yards",
    "yards_after_catch",
    "complete_pass",
    "xpass",
    "pass_oe",
    "down",
    "yardline_100",
    "half_seconds_remaining",
)


def _fetch_raw_pbp(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around nflreadpy; tests monkey-patch this."""
    return nflreadpy.load_pbp(seasons=seasons).to_pandas()


def _normalize_team_or_none(v: object) -> str | None:
    """Apply normalize_team_code, but pass None through for special-teams plays."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return normalize_team_code(str(v)).value


def _coerce_player_id(v: object) -> str | None:
    """Coerce upstream player id to canonical gsis_id format or None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v)
    return s if _GSIS_RE.fullmatch(s) else None


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Coerce season/week/play_id to int64. Upstream returns play_id as
    # float32 (the legacy path downcast floats for memory) and week as
    # int32; both need to be int64 for PbpSchema's Series[int] fields.
    for int_col in ("season", "week", "play_id"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype("int64")

    # ydstogo: nullable Int64 (PbpSchema field is Series[int] with nullable=True).
    # Upstream returns this as float32 with NaN for special-teams plays.
    if "ydstogo" in df.columns:
        df["ydstogo"] = df["ydstogo"].astype("Int64")

    # Numeric float columns: upstream may return float32; PbpSchema expects
    # float64. Coerce explicitly. (Covers every `Series[float]` field in
    # PbpSchema.)
    for float_col in _FLOAT64_COLS:
        if float_col in df.columns:
            df[float_col] = df[float_col].astype("float64")

    # Team codes — nullable string (kickoffs/punts have NaN posteam/defteam).
    for team_col in ("posteam", "defteam"):
        if team_col in df.columns:
            df[team_col] = df[team_col].map(_normalize_team_or_none).astype(_PYARROW_STR)

    # String columns — pyarrow-backed.
    for str_col in (
        "game_id",
        "play_type",
    ):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(_PYARROW_STR)

    # Player-id columns — coerce malformed values to None and apply pyarrow string dtype.
    for pid_col in ("passer_player_id", "rusher_player_id", "receiver_player_id"):
        if pid_col in df.columns:
            df[pid_col] = df[pid_col].map(_coerce_player_id).astype(_PYARROW_STR)

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = PbpSchema.validate(df)
    return df


def refresh_pbp(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write play-by-play data for each season.

    One partition per season. Idempotent — re-running a season overwrites
    that partition only.
    """
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_pbp([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "pbp", df, season=season, week=None)
        record_manifest(data_root, table="pbp", season=season, df=df)
        written.append(path)
    return written
