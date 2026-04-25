"""Refresh per-season snap counts from `nfl_data_py.import_snap_counts`.

`nfl_data_py` returns a `pfr_player_id` column rather than `gsis_id` for
snap counts. We join on the id_map (built by `build_id_map`) to resolve
`pfr_player_id` -> `gsis_id` before validation. Rows with no id_map match
(bench/practice players we don't track) are dropped silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import _PYARROW_STR, Position, SnapCountsSchema, normalize_team_code
from projections.store import read_partition, write_partition

_KEEP = [
    "gsis_id",
    "season",
    "week",
    "team",
    "opponent",
    "position",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
]


def _fetch_raw_snap_counts(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_snap_counts(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _resolve_gsis_via_id_map(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Inner-join `df.pfr_player_id` with id_map's `pfr_id` to attach `gsis_id`.
    Rows with no id_map match are dropped (bench/practice players)."""
    id_map = read_partition(data_root / "raw", "id_map", season=None)
    id_map_subset = id_map[["pfr_id", "gsis_id"]].dropna(subset=["pfr_id"])
    merged = df.merge(
        id_map_subset,
        left_on="pfr_player_id",
        right_on="pfr_id",
        how="inner",
    )
    return merged.drop(columns=["pfr_player_id", "pfr_id"])


def _normalize_one_season(raw: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    df = raw.copy()

    # Drop rows missing pfr_player_id (rare bench cases) before the join.
    df = df[df["pfr_player_id"].notna()].copy()

    # Resolve pfr_player_id -> gsis_id via id_map. Drops unmatched rows.
    df = _resolve_gsis_via_id_map(df, data_root)

    # Drop rows with NaN season/week before int64 coercion.
    df = df[df["season"].notna() & df["week"].notna()].copy()
    # nfl_data_py returns int32 for season/week; pandera Series[int] requires int64.
    for int_col in ("season", "week", "offense_snaps", "defense_snaps", "st_snaps"):
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype("int64")

    # *_pct columns can be NaN when team total for that side is 0; fill to 0.0
    # so the non-nullable schema accepts them.
    for float_col in ("offense_pct", "defense_pct", "st_pct"):
        if float_col in df.columns:
            df[float_col] = df[float_col].fillna(0.0).astype(float)

    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = SnapCountsSchema.validate(df)
    return df


def refresh_snap_counts(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write snap counts for each season. Idempotent.

    Requires `id_map.parquet` to already exist in `data_root/raw/` (built
    by `build_id_map`); raises `FileNotFoundError` if missing.
    """
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_snap_counts([season])
        df = _normalize_one_season(raw, data_root)
        path = write_partition(data_root / "raw", "snap_counts", df, season=season, week=None)
        record_manifest(data_root, table="snap_counts", season=season, df=df)
        written.append(path)
    return written
