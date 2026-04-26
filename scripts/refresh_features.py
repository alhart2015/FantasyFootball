"""Plan 3c Phase 1 — refresh the feature cache from raw data.

For each (position, season) pair, iterate every week present in the season's
depth_charts partition, call the position's feature builder via
POSITION_DISPATCH, validate against the position's FeaturesSchema, and write
to data/features/{position}/season=YYYY/week=WW/part.parquet.

Per-week feature builds need raw data for the prior season too (rolling
windows of trailing 4 games can cross a season boundary at week 1-4 of a
season). The script always concatenates the prior season's raw data when
present.

Usage:
    python scripts/refresh_features.py wr --seasons 2018-2024
    python scripts/refresh_features.py all --seasons 2018-2024
    python scripts/refresh_features.py qb              # default seasons 2018-2024
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from projections.models import POSITION_DISPATCH
from projections.schemas import Position
from projections.store import read_partition, write_partition

_DEFAULT_SEASONS = range(2018, 2025)


def _parse_season_range(s: str) -> range:
    """`"2018-2024"` -> `range(2018, 2025)`; `"2024"` -> `range(2024, 2025)`."""
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return range(lo, hi + 1)
    n = int(s)
    return range(n, n + 1)


def _read_raw_with_prior(
    raw_root: Path, table: str, season: int, *, include_prior: bool = True
) -> pd.DataFrame:
    """Read raw partition for (table, season), optionally concatenated with
    (table, season-1) so trailing-4 rolling windows have history at week 1-4."""
    cur = read_partition(raw_root, table, season=season)
    if not include_prior:
        return cur
    try:
        prior = read_partition(raw_root, table, season=season - 1)
    except FileNotFoundError:
        return cur
    return pd.concat([prior, cur], ignore_index=True)


def _refresh_one(
    position: Position,
    season: int,
    *,
    raw_root: Path,
    features_root: Path,
) -> int:
    """Build + write every available week of features for (position, season).
    Returns the number of week partitions written."""
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {
        "passing": "ngs_passing",
        "rushing": "ngs_rushing",
        "receiving": "ngs_receiving",
    }[dispatch.ngs_stat_type]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    ws_full = _read_raw_with_prior(raw_root, "weekly_stats", season)
    sc_full = _read_raw_with_prior(raw_root, "snap_counts", season)
    ngs_full = _read_raw_with_prior(raw_root, ngs_table, season)
    dc = read_partition(raw_root, "depth_charts", season=season)
    sch = read_partition(raw_root, "schedules", season=season)

    weeks = sorted(int(w) for w in dc["week"].unique())
    written = 0
    table = position.value.lower()
    for week in weeks:
        kwargs: dict[str, Any] = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc,
            "schedules": sch,
            "season": season,
            "as_of_week": week,
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            continue
        feats = dispatch.feature_schema.validate(feats)
        write_partition(features_root, table, feats, season=season, week=week)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the feature cache for a position (or all)."
    )
    parser.add_argument(
        "position",
        choices=["qb", "rb", "te", "wr", "all"],
        help="Target position, or 'all' for QB/RB/TE/WR.",
    )
    parser.add_argument(
        "--seasons",
        default="2018-2024",
        help="Inclusive season range, e.g. '2018-2024' or '2024'. Default 2018-2024.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    seasons = _parse_season_range(args.seasons)
    raw_root = args.data_root / "raw"
    features_root = args.data_root / "features"

    positions: tuple[Position, ...] = (
        (Position.QB, Position.RB, Position.TE, Position.WR)
        if args.position == "all"
        else (Position(args.position.upper()),)
    )

    total = 0
    for position in positions:
        for season in seasons:
            n = _refresh_one(
                position,
                season,
                raw_root=raw_root,
                features_root=features_root,
            )
            print(f"  {position.value} {season}: wrote {n} week partition(s)")
            total += n
    print(f"\nTotal partitions written: {total}")


if __name__ == "__main__":
    main()
