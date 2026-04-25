"""Refresh per-season NGS data from `nfl_data_py.import_ngs_data`.

Parameterized by `stat_type` ∈ {"passing", "rushing", "receiving"}; produces
three distinct partition tables (`ngs_passing`, `ngs_rushing`, `ngs_receiving`).

NGS returns season-to-date weekly snapshots (cumulative through each week),
which is naturally leakage-safe — feature builders filter by `week < as_of_week`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import nfl_data_py as nfl
import pandas as pd
import pandera.pandas as pa

from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    NgsPassingSchema,
    NgsReceivingSchema,
    NgsRushingSchema,
    Position,
    normalize_team_code,
)
from projections.store import write_partition

NgsStatType = Literal["passing", "rushing", "receiving"]
STAT_TYPES: tuple[NgsStatType, ...] = ("passing", "rushing", "receiving")

_RENAME = {
    "player_gsis_id": "gsis_id",
    "team_abbr": "team",
    "player_position": "position",
}

_KEEP_COMMON = ["gsis_id", "season", "week", "team", "position"]

_KEEP_PASSING = [
    *_KEEP_COMMON,
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "avg_air_yards_differential",
    "aggressiveness",
    "max_completed_air_distance",
    "avg_air_yards_to_sticks",
    "completion_percentage",
    "expected_completion_percentage",
    "completion_percentage_above_expectation",
    "avg_air_distance",
    "max_air_distance",
]

_KEEP_RUSHING = [
    *_KEEP_COMMON,
    "efficiency",
    "percent_attempts_gte_eight_defenders",
    "avg_time_to_los",
    "rush_attempts",
    "rush_yards",
    "expected_rush_yards",
    "rush_yards_over_expected",
    "avg_rush_yards",
    "rush_yards_over_expected_per_att",
    "rush_pct_over_expected",
]

_KEEP_RECEIVING = [
    *_KEEP_COMMON,
    "avg_cushion",
    "avg_separation",
    "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "receptions",
    "targets",
    "catch_percentage",
    "yards",
    "rec_touchdowns",
    "avg_yac",
    "avg_expected_yac",
    "avg_yac_above_expectation",
]

_KEEP_FOR: dict[NgsStatType, list[str]] = {
    "passing": _KEEP_PASSING,
    "rushing": _KEEP_RUSHING,
    "receiving": _KEEP_RECEIVING,
}

_INT_COLS_FOR: dict[NgsStatType, tuple[str, ...]] = {
    "passing": (),
    "rushing": ("rush_attempts", "rush_yards"),
    "receiving": ("receptions", "targets", "yards", "rec_touchdowns"),
}

_SCHEMA_FOR: dict[NgsStatType, type[pa.DataFrameModel]] = {
    "passing": NgsPassingSchema,
    "rushing": NgsRushingSchema,
    "receiving": NgsReceivingSchema,
}


def _fetch_raw_ngs(stat_type: NgsStatType, seasons: list[int]) -> pd.DataFrame:
    return nfl.import_ngs_data(stat_type, seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _normalize_one_season(stat_type: NgsStatType, raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # NGS int-typed stat columns are nullable in pandera (qualifying-threshold misses).
    for int_col in _INT_COLS_FOR[stat_type]:
        if int_col in df.columns:
            df[int_col] = df[int_col].astype(pd.Int64Dtype())

    # nfl_data_py returns int32 for season/week; pandera Series[int] requires int64.
    # NGS also includes season-summary rows with week=0 — drop them.
    df = df[df["season"].notna() & df["week"].notna()].copy()
    for int_col in ("season", "week"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype("int64")
    # NGS sometimes includes synthetic week numbers (0=season summary,
    # 23+=pro bowl / all-star). Schema declares week in [1, 22]; filter.
    df = df[(df["week"] >= 1) & (df["week"] <= 22)].copy()

    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP_FOR[stat_type] if c in df.columns]].copy()
    df = _SCHEMA_FOR[stat_type].validate(df)
    return df


def refresh_ngs(
    data_root: Path,
    *,
    stat_type: NgsStatType,
    seasons: Iterable[int],
) -> list[Path]:
    """Fetch and write NGS data for `stat_type` and each season. Idempotent.

    Writes to `data/raw/ngs_{stat_type}/season=YYYY/part.parquet`.
    """
    if stat_type not in STAT_TYPES:
        raise ValueError(f"stat_type must be one of {STAT_TYPES}, got {stat_type!r}")

    table = f"ngs_{stat_type}"
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_ngs(stat_type, [season])
        df = _normalize_one_season(stat_type, raw)
        path = write_partition(data_root / "raw", table, df, season=season, week=None)
        record_manifest(data_root, table=table, season=season, df=df)
        written.append(path)
    return written
