"""Build the PBP family override parquet for the family probe.

One-shot CLI. Loads PBP / weekly_stats / schedules across the requested
season range (plus the prior season for trailing-4 backfill at week 1-4),
calls build_pbp_family_overrides, writes the resulting frame to a parquet.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_pbp_family_override --seasons 2018-2024
    python -m scripts.build_pbp_family_override --seasons 2018-2024 --force
    python -m scripts.build_pbp_family_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md §6.2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.pbp_team_features import build_pbp_family_overrides
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/pbp_family.parquet")


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


def _build_player_team_week_index(
    weekly_stats: pd.DataFrame, schedules: pd.DataFrame, seasons: range
) -> pd.DataFrame:
    """Inner-join weekly_stats and schedules to produce
    ``(gsis_id, season, week, team, opp)``. Restrict to the target season range
    so the override is keyed only to the seasons being probed."""
    ws = weekly_stats[weekly_stats["season"].isin(seasons)][["gsis_id", "season", "week", "team"]]
    sch = schedules[schedules["season"].isin(seasons)][["season", "week", "home_team", "away_team"]]
    home = sch.rename(columns={"home_team": "team", "away_team": "opp"})
    away = sch.rename(columns={"away_team": "team", "home_team": "opp"})
    team_opp = pd.concat([home, away], ignore_index=True)[["season", "week", "team", "opp"]]
    return ws.merge(team_opp, on=["season", "week", "team"], how="inner")


def main(argv: Sequence[str] | None = None) -> int:
    assert __doc__ is not None  # module docstring is set at top of file
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
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
    weekly_stats = _read_concat(raw_root, "weekly_stats", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))

    idx = _build_player_team_week_index(weekly_stats, schedules, seasons)
    overrides = build_pbp_family_overrides(pbp, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
