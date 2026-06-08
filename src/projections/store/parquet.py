"""Parquet partitioned read/write helpers. Layout is `{table}/season=YYYY/week=WW/part.parquet`,
or `{table}/season=YYYY/asof=YYYY-MM-DD/part.parquet` for date-snapshotted tables.
Tables without season (e.g., id_map) are written to `{table}.parquet`."""

from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

_ASOF_DIR_RE = re.compile(r"^asof=\d{4}-\d{2}-\d{2}$")


def _partition_dir(
    root: Path, table: str, season: int | None, week: int | None, asof: date | None
) -> Path:
    if season is None:
        if week is not None or asof is not None:
            raise ValueError("week/asof cannot be set when season is None")
        return root / table
    if week is not None and asof is not None:
        raise ValueError("week and asof are mutually exclusive partition dimensions")
    p = root / table / f"season={season}"
    if week is not None:
        p = p / f"week={week:02d}"
    if asof is not None:
        p = p / f"asof={asof.isoformat()}"
    return p


def _partition_file(
    root: Path, table: str, season: int | None, week: int | None, asof: date | None
) -> Path:
    if season is None:
        return root / f"{table}.parquet"
    return _partition_dir(root, table, season, week, asof) / "part.parquet"


def write_partition(
    root: Path,
    table: str,
    df: pd.DataFrame,
    *,
    season: int | None,
    week: int | None = None,
    asof: date | None = None,
) -> Path:
    """Write `df` to the parquet partition for `(table, season, week, asof)`. Idempotent:
    removes the existing partition file first if present."""
    target = _partition_file(root, table, season, week, asof)
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
    asof: date | None = None,
) -> pd.DataFrame:
    """Read parquet partition(s). `season=None` reads the unpartitioned table file. With a
    season set: a specific `asof` (or `week`) reads that one partition; otherwise all
    `part.parquet` under the season (across week/asof subdirs) are concatenated.

    For asof-snapshotted tables, a season-only read concatenates EVERY dated snapshot under
    that season (use the in-row `asof` column to distinguish them, or `read_latest_partition`
    for just the newest)."""
    if season is None:
        return pd.read_parquet(_partition_file(root, table, None, None, None))
    if asof is not None or week is not None:
        return pd.read_parquet(_partition_file(root, table, season, week, asof))
    season_dir = _partition_dir(root, table, season, None, None)
    files = sorted(season_dir.rglob("part.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet partitions under {season_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_latest_partition(root: Path, table: str, *, season: int) -> pd.DataFrame:
    """Read only the newest `asof` snapshot under a season (ISO dates sort chronologically)."""
    season_dir = _partition_dir(root, table, season, None, None)
    asof_dirs = sorted(
        d for d in season_dir.glob("asof=*") if d.is_dir() and _ASOF_DIR_RE.match(d.name)
    )
    if not asof_dirs:
        raise FileNotFoundError(f"No asof snapshots under {season_dir}")
    return pd.read_parquet(asof_dirs[-1] / "part.parquet")


def delete_partition(
    root: Path, table: str, *, season: int | None, week: int | None = None, asof: date | None = None
) -> None:
    """Remove a partition directory or unpartitioned file. Used by tests and re-ingests."""
    if season is None:
        f = _partition_file(root, table, None, None, None)
        if f.exists():
            f.unlink()
        return
    target = _partition_dir(root, table, season, week, asof)
    if target.exists():
        shutil.rmtree(target)
