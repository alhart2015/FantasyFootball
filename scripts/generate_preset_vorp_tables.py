"""Generate the 9 preset VORP tables (scoring x size) for 2026, each correctly re-scored.

Mirrors src/projections/draft/backtest/draft_basis.build_draft_basis (build_consensus under
the league ruleset -> consensus_to_season_projections -> generate_vorp_table), but attaches
the published ESPN+Sleeper consensus_adp + full_name (the real 2026 market) and loops the grid.
Output: data/vorp_2026/{scoring}_{n}team.parquet (untracked artifacts).

Run:  python scripts/generate_preset_vorp_tables.py [--season 2026] [--data-root data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.consensus.blend import build_consensus
from projections.draft.assistant.presets import SCORING_KEYS, TEAM_SIZES, get_preset
from projections.draft.consensus_source import consensus_to_season_projections
from projections.draft.vorp import generate_vorp_table
from projections.schemas import (
    _PYARROW_STR,
    ConsensusProjectionSchema,
    ExternalProjectionSchema,
    Ruleset,
    VorpTableSchema,
)
from projections.store import read_latest_partition


def resolve_espn_auction_dollars(frame: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    """Resolve one per-player ESPN auction dollar from the consensus frame: crowd average when
    present and >0, else the ruleset's expert value (PPR for ESPN_PPR/ESPN_HALF, STANDARD for
    STANDARD), else NA. Vectorized; NA-safe Int64 (rounds the fractional crowd average). An
    absent input column is treated as all-NA. `frame` must be the consensus frame — the columns
    do not survive consensus_to_season_projections onto the VORP table."""
    n = len(frame)

    def _col(name: str) -> pd.Series:
        if name in frame.columns:
            return frame[name].astype("Float64")
        return pd.Series([pd.NA] * n, dtype="Float64", index=frame.index)

    crowd = _col("espn_auction_value_avg")
    if ruleset.name in ("ESPN_PPR", "ESPN_HALF"):
        expert = _col("espn_auction_value_ppr")
    elif ruleset.name == "STANDARD":
        expert = _col("espn_auction_value_std")
    else:
        expert = pd.Series([pd.NA] * n, dtype="Float64", index=frame.index)

    use_crowd = crowd.notna() & (crowd > 0)
    value = expert.where(~use_crowd, crowd)  # crowd where use_crowd, else expert
    return value.round().astype("Int64")


def build_preset_table(external: pd.DataFrame, scoring_key: str, n_teams: int) -> pd.DataFrame:
    """One preset's VORP table: re-score the external snapshot under the preset ruleset,
    compute VORP for the preset size, attach consensus_adp + full_name."""
    preset = get_preset(scoring_key, n_teams)
    consensus = ConsensusProjectionSchema.validate(
        build_consensus(external, preset.league_config.ruleset)
    )
    season_proj = consensus_to_season_projections(consensus)
    table = generate_vorp_table(season_proj, preset.league_config)
    consensus = consensus.copy()
    consensus["espn_auction_dollars"] = resolve_espn_auction_dollars(
        consensus, preset.league_config.ruleset
    )
    cols = consensus[["gsis_id", "consensus_adp", "full_name", "espn_auction_dollars"]]
    table = table.merge(cols, on="gsis_id", how="left")
    table["gsis_id"] = table["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(table)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the 9 preset VORP tables for 2026.")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    external = ExternalProjectionSchema.validate(
        read_latest_partition(args.data_root / "raw", "external_projections", season=args.season)
    )
    out_dir = args.data_root / "vorp_2026"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scoring_key in SCORING_KEYS:
        for n_teams in TEAM_SIZES:
            preset = get_preset(scoring_key, n_teams)
            table = build_preset_table(external, scoring_key, n_teams)
            # Direct .to_parquet (not store.write_partition): these are fixed-path artifact
            # outputs (data/vorp_2026/{scoring}_{n}team.parquet), not season/week/asof-keyed
            # store partitions, so the partition schema doesn't apply.
            table.to_parquet(preset.table_path, index=False)
            print(f"{preset.label}: {len(table)} players -> {preset.table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
