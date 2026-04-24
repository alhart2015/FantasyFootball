"""Parquet partitioned read/write helpers. Layout is `{table}/season=YYYY/week=WW/part.parquet`.
Tables without season (e.g., id_map) are written to `{table}.parquet`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def _partition_dir(root: Path, table: str, season: int | None, week: int | None) -> Path:
    if season is None:
        if week is not None:
            raise ValueError("week cannot be set when season is None")
        return root / table
    p = root / table / f"season={season}"
    if week is not None:
        p = p / f"week={week:02d}"
    return p


def _partition_file(root: Path, table: str, season: int | None, week: int | None) -> Path:
    if season is None:
        return root / f"{table}.parquet"
    return _partition_dir(root, table, season, week) / "part.parquet"


def write_partition(
    root: Path,
    table: str,
    df: pd.DataFrame,
    *,
    season: int | None,
    week: int | None,
) -> Path:
    """Write `df` to the parquet partition for `(table, season, week)`. Idempotent:
    removes the existing partition file first if present."""
    target = _partition_file(root, table, season, week)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    df.to_parquet(target, index=False)
    return target


def read_partition(
    root: Path,
    table: str,
    *,
    season: int | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    """Read parquet partition(s). If `week` is None and `season` is set, reads all
    weeks under that season. If `season` is None, reads the unpartitioned table file."""
    if season is None:
        return pd.read_parquet(_partition_file(root, table, None, None))

    if week is not None:
        return pd.read_parquet(_partition_file(root, table, season, week))

    season_dir = _partition_dir(root, table, season, None)
    files = sorted(season_dir.rglob("part.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet partitions under {season_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def delete_partition(
    root: Path, table: str, *, season: int | None, week: int | None = None
) -> None:
    """Remove a partition directory or unpartitioned file. Used by tests and re-ingests."""
    if season is None:
        f = _partition_file(root, table, None, None)
        if f.exists():
            f.unlink()
        return
    target = _partition_dir(root, table, season, week)
    if target.exists():
        shutil.rmtree(target)
