"""Preseason feature builder.

Produces one row per (gsis_id, target_season) for every rostered player on
depth_charts_<target_season> week=1 in skill positions {QB, RB, WR, TE}.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §3.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from projections.schemas import (
    Position,
    PreseasonFeaturesSchema,
    Stat,
)

logger = logging.getLogger(__name__)

_SKILL_POSITIONS: Final = frozenset({Position.QB, Position.RB, Position.WR, Position.TE})

# Position -> tuple of stats to materialize prior_{1,2,3}_season_per_game columns for.
# Used in Tasks 7 (feature aggregation) and Task 13 (rookie GLM fit).
_STATS_BY_POSITION: Final[dict[Position, tuple[Stat, ...]]] = {
    Position.QB: (
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.INTERCEPTIONS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
    ),
    Position.RB: (
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    ),
    Position.WR: (
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
    ),
    Position.TE: (
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    ),
}


def _schema_stat_name(stat: Stat) -> str:
    """The schema renames `Stat.INTERCEPTIONS` -> `passing_interceptions` for
    disambiguation from defensive interceptions in future K/DST work."""
    if stat is Stat.INTERCEPTIONS:
        return "passing_interceptions"
    return stat.value


def build_preseason_features(
    *,
    weekly_stats: pd.DataFrame,
    depth_charts_target: pd.DataFrame,
    draft_picks: pd.DataFrame,
    id_map: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """Build the preseason feature frame for `target_season`.

    Returns a DataFrame validated against PreseasonFeaturesSchema. One row per
    rostered player on `depth_charts_target` at week=1 in {QB, RB, WR, TE}.

    Task 5 implements identity columns + position filter + dup detection.
    Task 6 adds player-profile columns (age, years_exp, is_rookie, draft pick).
    Task 7 adds prior-season per-game aggregates.
    Task 8 adds dropped-player side-channel CSV.
    """
    # Suppress unused-arg warnings for inputs Tasks 6-8 will consume.
    del weekly_stats, draft_picks, id_map

    # 1. Take the week-1 preseason snapshot of the depth chart.
    dc = depth_charts_target.loc[depth_charts_target["week"] == 1].copy()
    if dc.empty:
        raise ValueError(
            f"depth_charts_target has no week=1 rows for season={target_season}. "
            "v1 preseason builder reads the week-1 snapshot."
        )

    # 2. Position filter — skill positions only.
    skill_position_values = {p.value for p in _SKILL_POSITIONS}
    n_before = len(dc)
    dc = dc.loc[dc["position"].isin(skill_position_values)].copy()
    n_filtered = n_before - len(dc)
    if n_filtered:
        logger.info(
            "build_preseason_features: filtered %d non-skill-position rows (season=%d)",
            n_filtered,
            target_season,
        )

    # 3. Duplicate detection.
    dup_mask = dc.duplicated(subset=["gsis_id"], keep=False)
    if dup_mask.any():
        dup_ids = dc.loc[dup_mask, "gsis_id"].unique().tolist()
        raise ValueError(
            f"Duplicate gsis_id rows in depth_charts_target week=1: {dup_ids[:5]!r}. "
            "Upstream depth_charts dedup bug; never silently swallow."
        )

    # 4. Project identity columns.
    out = pd.DataFrame(
        {
            "gsis_id": dc["gsis_id"].astype("string[pyarrow]"),
            "season": pd.array([target_season] * len(dc), dtype="int32"),
            "position": dc["position"].astype("string[pyarrow]"),
            "team": dc["team"].astype("string[pyarrow]"),
            "depth_chart_rank": dc["depth_rank"].astype("Int64"),
        }
    ).reset_index(drop=True)

    # 5. Stub remaining required columns to satisfy the schema's required-column
    # check. Tasks 6-7 replace these with real computations.
    out["age"] = pd.array([pd.NA] * len(out), dtype="Float32")
    out["years_exp"] = pd.array([0] * len(out), dtype="Int64")
    out["is_rookie"] = pd.array([False] * len(out), dtype="bool")
    out["draft_round"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["draft_pick_overall"] = pd.array([pd.NA] * len(out), dtype="Int64")

    out = PreseasonFeaturesSchema.validate(out)
    return out
