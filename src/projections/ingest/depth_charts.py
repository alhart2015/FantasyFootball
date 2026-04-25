"""Refresh per-season depth charts from `nfl_data_py.import_depth_charts`.

`nfl_data_py` raw column conventions vary across seasons:
- Pre-2018-ish: `depth_team` uses alignment labels (LWR, RWR, SWR).
- Newer seasons: `depth_team` uses rank labels (WR1, WR2) and `depth_position`
  contains a numeric rank.

`_parse_depth_rank` resolves these into a single canonical `depth_rank` int.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    DepthChartsSchema,
    Position,
    normalize_team_code,
)
from projections.store import write_partition

_log = logging.getLogger(__name__)

_KEEP = ["gsis_id", "season", "week", "team", "position", "depth_team", "depth_rank"]
_RENAME = {"club_code": "team"}
_TRAILING_DIGITS = re.compile(r"(\d+)$")


def _fetch_raw_depth_charts(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_depth_charts(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _parse_depth_rank(*, depth_team: str | None, depth_position: int | None) -> tuple[int, bool]:
    """Resolve a numeric `depth_rank` from raw inputs.

    Returns (rank, warned). `warned` is True if we had to fall back to 1
    because the inputs were unrankable, OR if the parsed rank was clamped
    from an out-of-range value, so the caller can log once with a
    representative example.
    """
    if depth_position is not None and not pd.isna(depth_position):
        try:
            return min(10, max(1, int(depth_position))), False
        except (ValueError, TypeError):
            pass
    if depth_team is not None and not pd.isna(depth_team):
        match = _TRAILING_DIGITS.search(str(depth_team))
        if match:
            parsed = int(match.group(1))
            if parsed >= 1:
                return min(10, parsed), parsed > 10
    return 1, True


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # Resolve depth_rank row-by-row; track if any rows fell back to 1 unranked.
    ranks: list[int] = []
    fallback_count = 0
    sample_label: str | None = None
    for _, row in df.iterrows():
        rank, warned = _parse_depth_rank(
            depth_team=row.get("depth_team"),
            depth_position=row.get("depth_position"),
        )
        ranks.append(rank)
        if warned:
            fallback_count += 1
            if sample_label is None:
                sample_label = str(row.get("depth_team"))
    if fallback_count:
        _log.warning(
            "Fell back to depth_rank=1 for %d rows (e.g., depth_team=%r). "
            "These are unrankable labels (alignment-based or out-of-range numeric).",
            fallback_count,
            sample_label,
        )
    df["depth_rank"] = ranks

    # Drop rows with NaN season/week (corrupt rows that would coerce to 0
    # and fail schema validation downstream).
    df = df[df["season"].notna() & df["week"].notna()].copy()
    # nfl_data_py returns int32 for season/week; pandera Series[int] requires int64.
    for int_col in ("season", "week", "depth_rank"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype("int64")

    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = DepthChartsSchema.validate(df)
    return df


def refresh_depth_charts(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write depth charts for each season. Idempotent."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_depth_charts([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "depth_charts", df, season=season, week=None)
        record_manifest(data_root, table="depth_charts", season=season, df=df)
        written.append(path)
    return written
