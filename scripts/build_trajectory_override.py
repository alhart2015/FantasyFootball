"""Build the trajectory override parquet for the trajectory family probe.

One-shot CLI. Loads weekly_stats / snap_counts / depth_charts / schedules /
draft_picks across the requested season range (plus the prior season for
weekly_stats / snap_counts at week 1 trailing-8-game backfill), calls
build_trajectory_overrides, writes the resulting frame to a parquet.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_trajectory_override --seasons 2018-2024
    python -m scripts.build_trajectory_override --seasons 2018-2024 --force
    python -m scripts.build_trajectory_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.trajectory_features import (
    build_draft_lookup,
    build_trajectory_overrides,
)
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/trajectory.parquet")


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


_FANTASY_POSITIONS: tuple[str, ...] = tuple(
    p.value for p in (Position.QB, Position.RB, Position.WR, Position.TE)
)


def _build_player_team_week_index(
    depth_charts: pd.DataFrame, schedules: pd.DataFrame, seasons: range
) -> pd.DataFrame:
    """Inner-join depth_charts (filtered to fantasy positions) with schedules
    to produce ``(gsis_id, season, week, team, opp, position)``.

    Mirrors PR #24's helper, with ``position`` preserved (the trajectory
    assembler dispatches per-position volume_trend so it needs position on
    every row). Pins canonical dtypes on the output (Task 12 I-2 fix):
    ``gsis_id`` / ``team`` / ``opp`` / ``position`` -> StringDtype("pyarrow"),
    ``season`` / ``week`` -> Int64Dtype. Without this the override parquet
    would carry object-dtype string columns inherited from the raw frames
    even though weekly_stats / snap_counts produce nullable extension dtypes.

    Rows where the player's team has no schedule entry that week (bye
    weeks) drop out of the inner join — matches the per-position
    feature builders' bye-week filter.
    """
    dc = depth_charts[
        depth_charts["season"].isin(seasons) & depth_charts["position"].isin(_FANTASY_POSITIONS)
    ][["gsis_id", "season", "week", "team", "position"]].drop_duplicates(
        subset=["gsis_id", "season", "week"]
    )
    sch = schedules[schedules["season"].isin(seasons)][["season", "week", "home_team", "away_team"]]
    home = sch.rename(columns={"home_team": "team", "away_team": "opp"})
    away = sch.rename(columns={"away_team": "team", "home_team": "opp"})
    team_opp = pd.concat([home, away], ignore_index=True)[["season", "week", "team", "opp"]]
    result = dc.merge(team_opp, on=["season", "week", "team"], how="inner")
    return result.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "week": pd.Int64Dtype(),
            "team": pd.StringDtype("pyarrow"),
            "opp": pd.StringDtype("pyarrow"),
            "position": pd.StringDtype("pyarrow"),
        }
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
    history_seasons = range(seasons.start - 1, seasons.stop)  # +1 prior for trailing-8 backfill

    weekly_stats = _read_concat(raw_root, "weekly_stats", list(history_seasons))
    snap_counts = _read_concat(raw_root, "snap_counts", list(history_seasons))
    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))
    try:
        draft_picks = _read_concat(raw_root, "draft_picks", list(range(1980, seasons.stop)))
    except FileNotFoundError:
        draft_picks = pd.DataFrame(
            columns=[
                "gsis_id",
                "draft_year",
                "draft_round",
                "draft_overall_pick",
                "pfr_id",
                "draft_age",
            ]
        )

    idx = _build_player_team_week_index(depth_charts, schedules, seasons)
    draft_lookup = build_draft_lookup(draft_picks)
    overrides = build_trajectory_overrides(weekly_stats, snap_counts, draft_lookup, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
