"""Build the Vegas team-context override parquet for the 33c family probe.

One-shot CLI. Loads schedules + depth_charts across the requested season
range, builds the player-team-week index, calls
build_vegas_team_context_overrides, writes the resulting frame to a parquet.
Prints audit numbers (per-column coverage, week-1 NaN rate, unique
team-season count, histogram bounds) so a follow-up step can capture them
into reports/feature_probe_vegas_team_context_override_audit.md.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_vegas_team_context_override --seasons 2018-2024
    python -m scripts.build_vegas_team_context_override --seasons 2018-2024 --force
    python -m scripts.build_vegas_team_context_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.vegas_team_context_features import (
    build_vegas_team_context_overrides,
)
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/vegas_team_context.parquet")


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

    Mirrors PR #25 / PR #28's helper. Pins canonical dtypes on the output:
    ``gsis_id`` / ``team`` / ``opp`` / ``position`` -> StringDtype("pyarrow"),
    ``season`` / ``week`` -> Int64Dtype.
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args. Extracted for testability — same pattern as
    `scripts.build_weather_override.parse_args`."""
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
        help="Root for raw partitions. Default: data.",
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

    return args


def _print_audit(overrides: pd.DataFrame) -> None:
    """Print audit numbers for `reports/feature_probe_vegas_team_context_override_audit.md`.

    Numbers reported per spec §6.7:
        - Per-column coverage rate (% non-NaN).
        - Week-1 NaN rate on season_avg_* (expected ~6%).
        - Unique team-season count per season (expected 32).
        - Histogram bounds: min/max/mean of each feature col.
    """
    n = len(overrides)
    print(f"vegas_team_context override audit ({n} rows):")

    feature_cols = (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    )
    for col in feature_cols:
        coverage = overrides[col].notna().mean() * 100.0
        print(f"  {col} coverage: {coverage:.2f}%")

    # week-1 NaN rate on season_avg_*
    wk1 = overrides[overrides["week"] == 1]
    n_wk1 = len(wk1)
    if n_wk1 > 0:
        for col in ("season_avg_spread", "season_avg_implied_team_total"):
            wk1_nan_pct = wk1[col].isna().mean() * 100.0
            print(f"  {col} week-1 NaN rate: {wk1_nan_pct:.2f}% (expected ~100%)")

    # Unique team-season count
    for season, group in overrides.groupby("season"):
        n_unique = group[["preseason_spread"]].drop_duplicates().shape[0]
        print(f"  season {season}: {n_unique} unique preseason_spread values (expected 32)")

    # Histogram bounds
    for col in feature_cols:
        s = overrides[col].dropna()
        if len(s) > 0:
            print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seasons: range = args.seasons
    raw_root = args.data_root / "raw"

    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))

    idx = _build_player_team_week_index(depth_charts, schedules, seasons)
    overrides = build_vegas_team_context_overrides(schedules, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    _print_audit(overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
