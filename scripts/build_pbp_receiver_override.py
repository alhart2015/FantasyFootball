"""Build the PBP receiver override parquet for the WR/TE family probe.

One-shot CLI. Loads PBP across the requested season range (plus the prior
season for trailing-4 backfill) and depth_charts filtered to WR + TE, calls
build_pbp_receiver_overrides, writes the resulting frame to a parquet.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_pbp_receiver_override --seasons 2018-2024
    python -m scripts.build_pbp_receiver_override --seasons 2018-2024 --force
    python -m scripts.build_pbp_receiver_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md §6.2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.pbp_receiver_features import build_pbp_receiver_overrides
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/pbp_receiver.parquet")
_RECEIVER_POSITIONS: tuple[str, ...] = (Position.WR.value, Position.TE.value)


def _parse_season_range(s: str) -> range:
    """`'2018-2024'` -> `range(2018, 2025)`; `'2024'` -> `range(2024, 2025)`."""
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return range(lo, hi + 1)
    n = int(s)
    return range(n, n + 1)


def _read_concat(raw_root: Path, table: str, seasons: Sequence[int]) -> pd.DataFrame:
    """Read one partition per season and concat. Skip seasons without a partition."""
    frames: list[pd.DataFrame] = []
    for s in seasons:
        try:
            frames.append(read_partition(raw_root, table, season=s))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError(
            f"no partitions found for table={table!r} in seasons={list(seasons)}"
        )
    return pd.concat(frames, ignore_index=True)


def _build_receiver_index(depth_charts: pd.DataFrame, seasons: range) -> pd.DataFrame:
    """Filter depth_charts to WR + TE in the requested season range; dedupe
    on (gsis_id, season, week).

    Mirrors the team-level pattern (scripts/build_pbp_family_override.py:78-80)
    which uses depth_charts as the index source so the override's coverage
    matches the per-position baseline-feature parquet's coverage.
    """
    return (
        depth_charts[
            depth_charts["season"].isin(seasons)
            & depth_charts["position"].isin(_RECEIVER_POSITIONS)
        ][["gsis_id", "season", "week"]]
        .drop_duplicates(subset=["gsis_id", "season", "week"])
        .reset_index(drop=True)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--seasons",
        type=_parse_season_range,
        default=range(2018, 2025),
        help="Season range, e.g. '2018-2024' or '2024'. Default: 2018-2024.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root for raw and features partitions. Default: data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Override output parquet path. Default: {_DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output if it already exists.",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite.")

    seasons: range = args.seasons
    raw_root = args.data_root / "raw"
    pbp_seasons = range(seasons.start - 1, seasons.stop)  # +1 prior for backfill

    pbp = _read_concat(raw_root, "pbp", list(pbp_seasons))
    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))

    receiver_index = _build_receiver_index(depth_charts, seasons)
    overrides = build_pbp_receiver_overrides(pbp, receiver_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
