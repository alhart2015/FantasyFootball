"""Backtest draft basis: blended ESPN+Sleeper season projection + Sleeper-only ADP -> fixed-VORP.

Reuses build_consensus, which scores the per-field mean of each source's stat line under the
league ruleset (ESPN + Sleeper where both are stat-bearing; whichever is present otherwise — so
seasons where ESPN's historical season projection is missing fall back to Sleeper). ESPN ADP is a
useless sentinel for past seasons (~170 for unranked players), so consensus_adp comes from
Sleeper alone.
"""

from __future__ import annotations

import pandas as pd

from projections.consensus.blend import build_consensus
from projections.draft.assistant.pool_identity import reconcile_pool_gsis
from projections.draft.consensus_source import consensus_to_season_projections
from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import (
    _PYARROW_STR,
    ConsensusProjectionSchema,
    ProjectionSource,
    VorpTableSchema,
)


def sleeper_adp(external: pd.DataFrame) -> pd.Series:
    """Sleeper-only ADP per gsis_id (mean over Sleeper rows with adp > 0).

    Returns a Series indexed by gsis_id. Players with no valid Sleeper ADP row are absent.
    """
    sl = external[external["source"] == ProjectionSource.SLEEPER.value]
    sl = sl[sl["adp"].notna() & (sl["adp"] > 0)]
    return sl.groupby("gsis_id")["adp"].mean()


def build_draft_basis(
    external: pd.DataFrame,
    *,
    league_config: LeagueConfig,
    id_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the fixed-VORP draft basis table from one external_projections snapshot.

    Scores the blended ESPN+Sleeper stat line under `league_config.ruleset` (half-PPR when
    configured) via build_consensus, then attaches Sleeper-only ADP as `consensus_adp`. Returns a
    VorpTableSchema-validated DataFrame with the extra `consensus_adp` column (Optional in schema).

    Args:
        external: ExternalProjectionSchema-validated DataFrame (one snapshot: one season/asof).
        league_config: Frozen league configuration; ruleset must match the stat-line source.
        id_map: when given, reconcile any placeholder (99-) gsis to real ones via name+position
            (`reconcile_pool_gsis`) so the pool joins to weekly_stats (injury p) and id_map (byes)
            -- without it the availability model silently degrades. Off by default to keep the
            function pure of store reads; the backtest input loader passes it.

    Returns:
        VorpTableSchema-validated DataFrame extended with `consensus_adp` (Sleeper-only ADP).
    """
    consensus = ConsensusProjectionSchema.validate(build_consensus(external, league_config.ruleset))
    season_proj = consensus_to_season_projections(consensus)
    table = generate_vorp_table(season_proj, league_config)

    adp = sleeper_adp(external).rename("consensus_adp")
    # gsis_id in the VORP table is _PYARROW_STR; align the merge index dtype so the join
    # isn't silently empty when adp index is plain object/str.
    adp.index = adp.index.astype(_PYARROW_STR)

    table = table.merge(adp, on="gsis_id", how="left")
    table["gsis_id"] = table["gsis_id"].astype(_PYARROW_STR)
    if id_map is not None:
        # reconcile_pool_gsis keys on full_name; generate_vorp_table omits it, so attach it
        # from the consensus frame (gsis-keyed) before reconciling.
        names = consensus[["gsis_id", "full_name"]].drop_duplicates("gsis_id")
        names["gsis_id"] = names["gsis_id"].astype(_PYARROW_STR)
        table = table.merge(names, on="gsis_id", how="left")
        table = reconcile_pool_gsis(table, id_map)
    return VorpTableSchema.validate(table)


__all__ = ["build_draft_basis", "sleeper_adp"]
