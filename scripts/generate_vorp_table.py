"""CLI wrapper for `projections.draft.vorp.generate_vorp_table`.

Reads projections for a given season, aggregates to season totals, computes VORP
under the given LeagueConfig, and writes the resulting VORP table as CSV or parquet.

Supports two projection sources:
  --source weekly     (default) In-season model: reads a weekly-projections partition.
  --source consensus  Pre-season: reads the latest consensus-projections snapshot.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from projections.aggregation.season import aggregate_to_season
from projections.draft.consensus_source import consensus_to_season_projections
from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import (
    ConsensusProjectionSchema,
    Position,
    ProjectionWeeklySchema,
    VorpTableSchema,
)
from projections.store import read_latest_partition, read_partition


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a per-player VORP table.")
    parser.add_argument("--season", type=int, required=True, help="Target season (e.g. 2026).")
    parser.add_argument(
        "--league-config", type=Path, required=True, help="Path to LeagueConfig JSON."
    )
    parser.add_argument(
        "--source",
        choices=["weekly", "consensus"],
        default="weekly",
        help="Projection source: 'weekly' (in-season model, default) or 'consensus' (preseason).",
    )
    parser.add_argument(
        "--weekly-projections",
        type=Path,
        default=None,
        help="[--source weekly] Weekly-projections partition root "
        "(e.g. data/projections/weekly/ruleset=espn_ppr).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="[--source consensus] Store root; reads <root>/processed/consensus_projections/.",
    )
    parser.add_argument(
        "--asof",
        type=date.fromisoformat,
        default=None,
        help="[--source consensus] Snapshot date YYYY-MM-DD; defaults to the latest snapshot.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output path; .csv or .parquet (sniffed)."
    )
    return parser.parse_args()


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
    for pos_enum in Position:
        pos = pos_enum.value
        pos_rows = vorp_table[vorp_table["position"] == pos]
        if pos_rows.empty:
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


def _warn_dropped_draftable(consensus: pd.DataFrame, league_config: LeagueConfig) -> None:
    """Surface players the has_points filter drops whose ADP is inside the draftable range,
    so the coverage gap is an explicit eyeball check rather than silent (spec §3.3)."""
    dropped = consensus[~consensus["has_points"] & consensus["consensus_adp"].notna()]
    draftable = dropped[dropped["consensus_adp"] <= league_config.total_pool_size]
    if draftable.empty:
        print("0 draftable players dropped for missing points.", file=sys.stderr)
        return
    print(
        f"WARNING: {len(draftable)} draftable player(s) dropped — have an ADP inside the "
        f"top {league_config.total_pool_size} but no projected points:",
        file=sys.stderr,
    )
    for row in draftable.sort_values("consensus_adp").head(25).itertuples(index=False):
        print(
            f"  {row.full_name}  ADP={float(row.consensus_adp):.1f}  rank={row.consensus_rank}",
            file=sys.stderr,
        )


def main() -> int:
    args = _parse_args()
    league_config = LeagueConfig.model_validate_json(args.league_config.read_text())

    consensus: pd.DataFrame | None = None
    if args.source == "weekly":
        if args.weekly_projections is None:
            print("ERROR: --weekly-projections is required for --source weekly", file=sys.stderr)
            return 1
        weekly_root: Path = args.weekly_projections
        weekly = read_partition(weekly_root.parent, table=weekly_root.name, season=args.season)
        weekly = ProjectionWeeklySchema.validate(weekly)
        season_proj = aggregate_to_season(weekly, ruleset=league_config.ruleset)
    else:
        processed = args.data_root / "processed"
        if args.asof is not None:
            consensus = read_partition(
                processed, "consensus_projections", season=args.season, asof=args.asof
            )
        else:
            consensus = read_latest_partition(
                processed, "consensus_projections", season=args.season
            )
        consensus = ConsensusProjectionSchema.validate(consensus)
        _warn_dropped_draftable(consensus, league_config)
        season_proj = consensus_to_season_projections(consensus)

    in_scope = {slot.value for slot, count in league_config.roster_slots.items() if count > 0}
    dropped_positions = (
        season_proj[~season_proj["position"].isin(in_scope)]["position"].value_counts().to_dict()
    )

    out_df = generate_vorp_table(season_proj, league_config)

    if consensus is not None:
        adp = consensus[["gsis_id", "consensus_adp"]]
        out_df = out_df.merge(adp, on="gsis_id", how="left")
        out_df = VorpTableSchema.validate(out_df)

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
