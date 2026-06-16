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
    _PYARROW_STR,
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
        default=None,
        help="[--source consensus] Store root (default: data); reads "
        "<root>/processed/consensus_projections/.",
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


def _reject_irrelevant_flags(args: argparse.Namespace) -> str | None:
    """Reject flags that don't apply to the chosen ``--source`` so a wrong-source flag can't be
    silently ignored (e.g. ``--weekly-projections`` in consensus mode, which would otherwise be
    discarded while the run reads the default ``--data-root`` — a silent wrong-data footgun).

    Returns an error message, or None when the flags are consistent with ``--source``.
    """
    if args.source == "weekly":
        irrelevant = [
            name
            for name, value in (("--asof", args.asof), ("--data-root", args.data_root))
            if value is not None
        ]
        if irrelevant:
            verb = "is" if len(irrelevant) == 1 else "are"
            return f"{', '.join(irrelevant)} {verb} only valid with --source consensus"
    elif args.weekly_projections is not None:
        return "--weekly-projections is only valid with --source weekly"
    return None


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


def _merge_consensus_columns(out_df: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Attach the consensus market columns (consensus_adp, full_name) onto the VORP table.

    full_name is the player's display name (incl. placeholder-gsis rookies the live board
    must name). Returns a re-validated VorpTableSchema frame.
    """
    cols = consensus[["gsis_id", "consensus_adp", "full_name"]]
    merged = out_df.merge(cols, on="gsis_id", how="left")
    merged["gsis_id"] = merged["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(merged)


def main() -> int:
    args = _parse_args()
    flag_error = _reject_irrelevant_flags(args)
    if flag_error is not None:
        print(f"ERROR: {flag_error}", file=sys.stderr)
        return 1
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
        processed = (args.data_root or Path("data")) / "processed"
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
        out_df = _merge_consensus_columns(out_df, consensus)

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
