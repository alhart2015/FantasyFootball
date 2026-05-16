"""Pool-selection helper used by both auction-values and VORP.

The pool is the set of players who would actually be drafted under a given
`LeagueConfig`. Selection ranks by `season_mean_fpts` (descending) within each
position pass, tie-broken by `vorp` (descending) if present then `gsis_id`
(ascending). Filling order: position-specific slots → FLEX → SUPER_FLEX → BENCH.

Module-private (leading underscore). Direct callers are `auction.py` (passes a
vorp-populated frame) and `vorp.py` (passes a frame without `vorp`).
"""

from __future__ import annotations

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import Position, RosterSlot

_POSITION_SLOTS: tuple[RosterSlot, ...] = (
    RosterSlot.QB,
    RosterSlot.RB,
    RosterSlot.WR,
    RosterSlot.TE,
    RosterSlot.K,
    RosterSlot.DST,
)

_FLEX_ELIGIBLE: frozenset[Position] = frozenset({Position.RB, Position.WR, Position.TE})
_SUPER_FLEX_ELIGIBLE: frozenset[Position] = frozenset(
    {Position.QB, Position.RB, Position.WR, Position.TE}
)


def _take_top_n(
    sorted_df: pd.DataFrame,
    picked: list[str],
    picked_set: set[str],
    eligible_position_values: frozenset[str],
    needed: int,
    slot_label: str,
) -> None:
    """Pick the top `needed` not-yet-picked players whose position is in `eligible_position_values`.

    Mutates `picked` and `picked_set` in place. Raises ValueError if fewer than `needed`
    candidates remain.
    """
    if needed <= 0:
        return
    remaining = sorted_df[
        sorted_df["position"].isin(eligible_position_values)
        & ~sorted_df["gsis_id"].isin(picked_set)
    ]
    if len(remaining) < needed:
        raise ValueError(
            f"VORP table cannot fill {needed} {slot_label} slots: only "
            f"{len(remaining)} eligible players remain."
        )
    for gid in remaining["gsis_id"].head(needed).tolist():
        picked.append(gid)
        picked_set.add(gid)


def _select_pool(vorp_table: pd.DataFrame, league_config: LeagueConfig) -> list[str]:
    """Select the in-pool `gsis_id`s per the auction-values spec §3.1 algorithm.

    Returns a list of length `league_config.total_pool_size`. Selection order:
    position-specific slots, then FLEX, then SUPER_FLEX, then BENCH. Within each
    pass, players are ranked by `season_mean_fpts` desc, tie-broken by `vorp` desc
    (if column present) then `gsis_id` asc. Callers that don't yet have a VORP
    value (e.g. the VORP generator itself) can omit the `vorp` column entirely;
    the tie-break falls back to `gsis_id` alone.

    Raises `ValueError` if any required position is missing from `vorp_table`.
    """
    sort_cols = ["season_mean_fpts"]
    sort_asc = [False]
    if "vorp" in vorp_table.columns:
        sort_cols.append("vorp")
        sort_asc.append(False)
    sort_cols.append("gsis_id")
    sort_asc.append(True)

    sorted_df = vorp_table.sort_values(
        by=sort_cols,
        ascending=sort_asc,
        kind="mergesort",
    ).reset_index(drop=True)

    picked: list[str] = []
    picked_set: set[str] = set()

    for slot in _POSITION_SLOTS:
        wanted = league_config.roster_slots.get(slot, 0)
        if wanted <= 0:
            continue
        _take_top_n(
            sorted_df,
            picked,
            picked_set,
            eligible_position_values=frozenset({slot.value}),
            needed=league_config.n_teams * wanted,
            slot_label=slot.value,
        )

    _take_top_n(
        sorted_df,
        picked,
        picked_set,
        eligible_position_values=frozenset(p.value for p in _FLEX_ELIGIBLE),
        needed=league_config.n_teams * league_config.roster_slots.get(RosterSlot.FLEX, 0),
        slot_label=RosterSlot.FLEX.value,
    )

    _take_top_n(
        sorted_df,
        picked,
        picked_set,
        eligible_position_values=frozenset(p.value for p in _SUPER_FLEX_ELIGIBLE),
        needed=league_config.n_teams * league_config.roster_slots.get(RosterSlot.SUPER_FLEX, 0),
        slot_label=RosterSlot.SUPER_FLEX.value,
    )

    # Bench eligibility excludes positions the league doesn't roster (e.g. no K slots
    # → no K on bench). Guards against drafting K/DST into a K-less / DST-less league.
    bench_eligible = frozenset(
        slot.value
        for slot in league_config.roster_slots
        if slot in _POSITION_SLOTS and league_config.roster_slots[slot] > 0
    )
    _take_top_n(
        sorted_df,
        picked,
        picked_set,
        eligible_position_values=bench_eligible,
        needed=league_config.n_teams * league_config.roster_slots.get(RosterSlot.BENCH, 0),
        slot_label=RosterSlot.BENCH.value,
    )

    return picked
