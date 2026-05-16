"""Auction $ generator: converts per-player VORP to per-player auction dollars.

Public surface: `generate_auction_values(vorp_table, league_config, reference_prices=None)`.

Algorithm: standard Surplus-Of-Surplus (SOS) allocation. Reserve `min_bid` for every
drafted slot, then distribute the remaining budget proportionally to positive VORP
among the rostered pool. Strategy-agnostic; one $ per player.

Spec: docs/superpowers/specs/2026-05-16-auction-values-design.md
"""

from __future__ import annotations

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import Position, RosterSlot

# RosterSlot keys that consume position-specific picks at draft. FLEX, SUPER_FLEX, BENCH
# fill from the remainder. IR is excluded (post-draft).
_POSITION_SLOTS: tuple[RosterSlot, ...] = (
    RosterSlot.QB,
    RosterSlot.RB,
    RosterSlot.WR,
    RosterSlot.TE,
    RosterSlot.K,
    RosterSlot.DST,
)

# Position eligibility for filler slots.
_FLEX_ELIGIBLE: frozenset[Position] = frozenset({Position.RB, Position.WR, Position.TE})
_SUPER_FLEX_ELIGIBLE: frozenset[Position] = frozenset(
    {Position.QB, Position.RB, Position.WR, Position.TE}
)


def _select_pool(vorp_table: pd.DataFrame, league_config: LeagueConfig) -> list[str]:
    """Select the in-pool `gsis_id`s per the spec §3.1 algorithm.

    Returns a list of length `league_config.total_pool_size`. Selection order:
    position-specific slots, then FLEX, then SUPER_FLEX, then BENCH. Within each
    pass, players are ranked by `season_mean_fpts` desc, tie-broken by `vorp` desc
    then `gsis_id` asc.

    Raises `ValueError` if any required position is missing from `vorp_table`.
    """
    # Pre-sort once; we'll slice by position from this sorted view.
    sorted_df = vorp_table.sort_values(
        by=["season_mean_fpts", "vorp", "gsis_id"],
        ascending=[False, False, True],
        kind="mergesort",  # stable
    ).reset_index(drop=True)

    picked: list[str] = []
    picked_set: set[str] = set()

    # Pass 1: position-specific slots.
    for slot in _POSITION_SLOTS:
        wanted = league_config.roster_slots.get(slot, 0)
        if wanted <= 0:
            continue
        # The RosterSlot enum mirrors Position values for non-FLEX/BENCH slots.
        pos_value = slot.value
        position_pool = sorted_df[sorted_df["position"] == pos_value]
        needed = league_config.n_teams * wanted
        if len(position_pool) < needed:
            raise ValueError(
                f"VORP table has only {len(position_pool)} {pos_value} players "
                f"but league_config requires {needed} ({league_config.n_teams} teams "
                f"x {wanted} {pos_value} slots)."
            )
        for gid in position_pool["gsis_id"].head(needed).tolist():
            picked.append(gid)
            picked_set.add(gid)

    def _fill_filler_slot(slot: RosterSlot, eligible: frozenset[Position]) -> None:
        wanted = league_config.roster_slots.get(slot, 0)
        if wanted <= 0:
            return
        needed = league_config.n_teams * wanted
        eligible_values = {p.value for p in eligible}
        remaining = sorted_df[
            (sorted_df["position"].isin(eligible_values)) & (~sorted_df["gsis_id"].isin(picked_set))
        ]
        if len(remaining) < needed:
            raise ValueError(
                f"VORP table cannot fill {needed} {slot.value} slots: only "
                f"{len(remaining)} eligible players remain after position-specific picks."
            )
        for gid in remaining["gsis_id"].head(needed).tolist():
            picked.append(gid)
            picked_set.add(gid)

    # Pass 2: FLEX (RB/WR/TE).
    _fill_filler_slot(RosterSlot.FLEX, _FLEX_ELIGIBLE)

    # Pass 3: SUPER_FLEX (QB/RB/WR/TE).
    _fill_filler_slot(RosterSlot.SUPER_FLEX, _SUPER_FLEX_ELIGIBLE)

    # Pass 4: BENCH. Position-agnostic over positions the league recognizes.
    bench_count = league_config.roster_slots.get(RosterSlot.BENCH, 0)
    if bench_count > 0:
        needed = league_config.n_teams * bench_count
        # Bench can come from any position that has at least one slot in the league
        # (excluding IR). This guards against drafting K/DST onto bench in leagues
        # that don't roster them.
        league_positions = {
            slot.value
            for slot in league_config.roster_slots
            if slot in _POSITION_SLOTS and league_config.roster_slots[slot] > 0
        }
        remaining = sorted_df[
            (sorted_df["position"].isin(league_positions))
            & (~sorted_df["gsis_id"].isin(picked_set))
        ]
        if len(remaining) < needed:
            raise ValueError(
                f"VORP table cannot fill {needed} BENCH slots: only "
                f"{len(remaining)} eligible players remain after starter + flex picks."
            )
        for gid in remaining["gsis_id"].head(needed).tolist():
            picked.append(gid)
            picked_set.add(gid)

    return picked


def generate_auction_values(
    vorp_table: pd.DataFrame,
    league_config: LeagueConfig,
    reference_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Placeholder. Implemented in Task 5."""
    raise NotImplementedError


__all__ = ["generate_auction_values"]
