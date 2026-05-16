"""CLI wrapper for `projections.draft.vorp.generate_vorp_table`.

Reads a weekly-projections parquet partition for a given season, aggregates
to season totals, computes VORP under the given LeagueConfig, and writes the
resulting VORP table as CSV or parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from projections.aggregation.season import aggregate_to_season
from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import _POSITION_VALUES, ProjectionWeeklySchema


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a per-player VORP table.")
    parser.add_argument("--season", type=int, required=True, help="Target season (e.g. 2026).")
    parser.add_argument(
        "--league-config",
        type=Path,
        required=True,
        help="Path to LeagueConfig JSON.",
    )
    parser.add_argument(
        "--weekly-projections",
        type=Path,
        required=True,
        help=(
            "Path to the weekly-projections partition root "
            "(e.g. data/projections/weekly/ruleset=espn_ppr)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path; .csv or .parquet (sniffed by extension).",
    )
    return parser.parse_args()


def _read_weekly_partition(root: Path, season: int) -> pd.DataFrame:
    """Read every part.parquet under `root/season=YYYY/week=WW/`. Errors if zero files."""
    season_dir = root / f"season={season}"
    if not season_dir.exists():
        raise FileNotFoundError(
            f"No partition directory at {season_dir!s}. Did predict_*.py run for this season?"
        )
    paths = sorted(season_dir.glob("week=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No week=*/part.parquet files under {season_dir!s}.")
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _log_per_position_summary(
    vorp_table: pd.DataFrame,
    league_config: LeagueConfig,
    dropped_positions: dict[str, int],
) -> None:
    """Emit the eyeball-mitigation summary (spec §4)."""
    print()
    print(f"VORP table written: {len(vorp_table)} players, ruleset={league_config.ruleset.name}")
    print()
    print("Position summary (replacement_fpts | in-scope row count | top-3 by VORP):")
    for pos in _POSITION_VALUES:
        in_config = any(
            slot.value == pos and count > 0 for slot, count in league_config.roster_slots.items()
        )
        pos_rows = vorp_table[vorp_table["position"] == pos]
        if pos_rows.empty:
            if in_config:
                print(
                    f"  {pos:>3}  MISSING from projection input — "
                    f"required by LeagueConfig; auction-values will error."
                )
            continue
        replacement = float(pos_rows["replacement_fpts"].iloc[0])
        top3 = pos_rows.nlargest(3, "vorp")[["gsis_id", "vorp"]]
        top3_str = ", ".join(
            f"{row.gsis_id}(VORP{row.vorp:+.1f})" for row in top3.itertuples(index=False)
        )
        print(
            f"  {pos:>3}  replacement={replacement:>7.2f}  rows={len(pos_rows):>3}  top: {top3_str}"
        )
    for pos, n_dropped in sorted(dropped_positions.items()):
        print(f"  NOTE: dropped {n_dropped} row(s) at position {pos} (not in LeagueConfig).")
    print()


def main() -> int:
    args = _parse_args()
    league_config = LeagueConfig.model_validate_json(args.league_config.read_text())

    weekly = _read_weekly_partition(args.weekly_projections, args.season)
    weekly = ProjectionWeeklySchema.validate(weekly)
    weekly = weekly[weekly["ruleset"] == league_config.ruleset.name]
    if weekly.empty:
        print(
            f"ERROR: no rows in {args.weekly_projections!s} match "
            f"ruleset={league_config.ruleset.name}",
            file=sys.stderr,
        )
        return 1

    season_proj = aggregate_to_season(weekly, ruleset=league_config.ruleset)

    in_scope = {slot.value for slot, count in league_config.roster_slots.items() if count > 0}
    dropped_positions = (
        season_proj[~season_proj["position"].isin(in_scope)]["position"].value_counts().to_dict()
    )

    out_df = generate_vorp_table(season_proj, league_config)

    suffix = args.out.suffix.lower()
    if suffix == ".csv":
        out_df.sort_values("vorp", ascending=False).to_csv(args.out, index=False)
    elif suffix == ".parquet":
        out_df.to_parquet(args.out, index=False)
    else:
        print(
            f"ERROR: unsupported output extension {suffix!r}; use .csv or .parquet",
            file=sys.stderr,
        )
        return 1

    _log_per_position_summary(out_df, league_config, dropped_positions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
