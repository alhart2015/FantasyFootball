"""Build the weather override parquet for the weather family probe.

One-shot CLI. Loads schedules + depth_charts across the requested season
range, builds the player-team-week index, calls build_weather_overrides,
writes the resulting frame to a parquet. Prints audit numbers (dome rate,
outdoor-NaN rate, is_high_wind rate, is_grass_surface rate) so a follow-up
step can capture them into reports/feature_probe_weather_override_audit.md.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_weather_override --seasons 2018-2024
    python -m scripts.build_weather_override --seasons 2018-2024 --force
    python -m scripts.build_weather_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.weather_features import build_weather_overrides
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/weather.parquet")


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

    Mirrors PR #25's helper. Pins canonical dtypes on the output:
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
    `scripts.build_trajectory_override.main`."""
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


def _print_audit(overrides: pd.DataFrame, schedules: pd.DataFrame) -> None:
    """Print audit numbers for `reports/feature_probe_weather_override_audit.md`.

    Numbers reported:
        - Pooled dome / closed-roof game share (% of games).
        - Outdoor-NaN rate per weather feature (% of override rows).
        - Pooled is_high_wind rate (% of override rows where True).
        - Pooled is_grass_surface rate (% of override rows where True).
    """
    n = len(overrides)
    is_indoor = schedules["roof"].isin(["dome", "closed"]).fillna(False)
    n_indoor_games = int(is_indoor.sum())
    n_total_games = len(schedules)
    indoor_pct = (n_indoor_games / n_total_games * 100.0) if n_total_games else 0.0

    nan_rates = {
        col: overrides[col].isna().mean() * 100.0
        for col in (
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_grass_surface",
        )
    }
    high_wind_rate = overrides["is_high_wind"].fillna(0.0).mean() * 100.0
    grass_rate = overrides["is_grass_surface"].fillna(0.0).mean() * 100.0

    print(f"weather override audit ({n} rows):")
    print(f"  indoor games (dome+closed): {n_indoor_games}/{n_total_games} = {indoor_pct:.1f}%")
    for col, pct in nan_rates.items():
        print(f"  {col} NaN rate: {pct:.2f}%")
    print(f"  is_high_wind=1.0 rate (incl. dome): {high_wind_rate:.2f}%")
    print(f"  is_grass_surface=1.0 rate: {grass_rate:.2f}%")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seasons: range = args.seasons
    raw_root = args.data_root / "raw"

    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))

    idx = _build_player_team_week_index(depth_charts, schedules, seasons)
    overrides = build_weather_overrides(schedules, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    _print_audit(overrides, schedules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
