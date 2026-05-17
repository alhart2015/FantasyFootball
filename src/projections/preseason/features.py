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

    # ---- Age (from id_map.birth_date) ----
    id_map_lookup = id_map.set_index("gsis_id")
    birth_dates = pd.to_datetime(
        out["gsis_id"].map(id_map_lookup["birth_date"]),
        errors="coerce",
    )
    out["age"] = pd.array(
        [float(target_season - bd.year) if pd.notna(bd) else None for bd in birth_dates],
        dtype="Float32",
    )

    # ---- Rookie detection ----
    # A player is a rookie if they have NO prior-season NFL history. "Prior
    # history" = at least one weekly_stats row in a prior season OR a
    # draft_picks row in a prior season. Players drafted in the target season
    # (with no prior history) ARE rookies. UDFA rookies (no draft_picks row,
    # no prior weekly_stats) also fall through to is_rookie=True.
    prior_weekly_ids = weekly_stats.loc[weekly_stats["season"] < target_season, "gsis_id"].unique()
    prior_draft_ids = draft_picks.loc[draft_picks["season"] < target_season, "gsis_id"].unique()
    has_prior_history = out["gsis_id"].isin(prior_weekly_ids) | out["gsis_id"].isin(prior_draft_ids)

    is_rookie = ~has_prior_history
    out["is_rookie"] = is_rookie.astype(bool)

    # ---- years_exp ----
    # Prefer weekly_stats.first_season; fall back to draft_picks.season when a
    # player has no weekly_stats rows (e.g. a player drafted in a prior season
    # whose stats history wasn't provided to the builder). Rookies forced to 0.
    if not weekly_stats.empty:
        first_season_by_player = weekly_stats.groupby("gsis_id")["season"].min()
        first_season_from_weekly = out["gsis_id"].map(first_season_by_player)
    else:
        first_season_from_weekly = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
    first_draft_by_player = draft_picks.groupby("gsis_id")["season"].min()
    first_season_from_draft = out["gsis_id"].map(first_draft_by_player)
    first_season_lookup = first_season_from_weekly.combine_first(first_season_from_draft)
    years_exp_raw = (target_season - first_season_lookup).fillna(0)
    # Rookies have years_exp = 0 regardless of any data lag.
    years_exp_arr = years_exp_raw.mask(out["is_rookie"], 0).astype("Int64")
    out["years_exp"] = years_exp_arr

    # ---- Draft pick (round + pick_overall) from earliest draft_picks row per player ----
    most_recent_pick = (
        draft_picks.sort_values("season")
        .drop_duplicates(subset=["gsis_id"], keep="first")
        .set_index("gsis_id")
    )
    out["draft_round"] = out["gsis_id"].map(most_recent_pick["round"]).astype("Int64")
    out["draft_pick_overall"] = out["gsis_id"].map(most_recent_pick["pick"]).astype("Int64")

    out = PreseasonFeaturesSchema.validate(out)
    return out
