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
from projections.schemas import AuctionValuesSchema, Position, RosterSlot

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
    """Convert per-player VORP into per-player auction dollars under `league_config`.

    Returns a DataFrame validated against `AuctionValuesSchema`. One row per player
    in `vorp_table`. Players not in the rostered pool get `auction_dollars=0` and
    `pool_rank=NA`. `reference_dollars` and `value_delta` are present in the output
    regardless of whether `reference_prices` was passed (all-NA when not passed).

    See spec §3 for the SOS algorithm and §6 for edge-case decisions.
    """
    # Input validation.
    if vorp_table["gsis_id"].duplicated().any():
        dup = vorp_table.loc[vorp_table["gsis_id"].duplicated(), "gsis_id"].iloc[0]
        raise ValueError(f"vorp_table has duplicate gsis_id rows (first duplicate: {dup}).")

    # Step 1 - build the rostered pool.
    pool_ids = _select_pool(vorp_table, league_config)
    pool_set = set(pool_ids)

    # Step 2 - compute the surplus.
    total_budget = league_config.total_budget
    reserve = league_config.total_pool_size * league_config.min_bid
    surplus = total_budget - reserve

    # Step 3 - allocate surplus to positive VORP.
    pool_df = vorp_table[vorp_table["gsis_id"].isin(pool_set)].copy()
    positive_vorp = pool_df["vorp"].clip(lower=0.0)
    positive_vorp_sum = float(positive_vorp.sum())

    if positive_vorp_sum > 0:
        extra_float = (positive_vorp / positive_vorp_sum) * surplus
    else:
        # Degenerate case: distribute surplus uniformly.
        extra_float = pd.Series(surplus / league_config.total_pool_size, index=pool_df.index)

    pool_df["_dollars_float"] = league_config.min_bid + extra_float

    # Step 4 - round and close drift.
    rounded = pool_df["_dollars_float"].round().astype("int64")
    drift = total_budget - int(rounded.sum())
    if drift != 0:
        fractional = pool_df["_dollars_float"] - pool_df["_dollars_float"].astype("int64")
        if drift > 0:
            # Add 1 to the rows whose float is closest to the next integer up.
            order = fractional.sort_values(ascending=False).index
        else:
            # Subtract 1 from the rows whose float is closest to the next integer down,
            # but exclude rows already at min_bid (the floor) — those can't absorb -1.
            adjustable_mask = rounded > league_config.min_bid
            order = fractional[adjustable_mask].sort_values(ascending=True).index
            if len(order) < abs(drift):
                # Pathologically small pool where the floor blocks closure. This shouldn't
                # happen on real auction inputs (it requires extreme rounding pressure),
                # but if it does, we surface it rather than silently violate the floor.
                raise ValueError(
                    f"Cannot close rounding drift of {drift} without violating min_bid "
                    f"floor of ${league_config.min_bid}. This usually indicates an extreme "
                    f"degenerate input (e.g., very small budget per slot)."
                )
        step = 1 if drift > 0 else -1
        for idx in order[: abs(drift)]:
            rounded.loc[idx] = rounded.loc[idx] + step
    pool_df["auction_dollars"] = rounded.astype(pd.Int64Dtype())

    # Step 5 - rank within pool.
    rank_sort = pool_df.sort_values(
        by=["auction_dollars", "vorp", "season_mean_fpts", "gsis_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    rank_sort["pool_rank"] = range(1, len(rank_sort) + 1)
    pool_df = pool_df.merge(
        rank_sort[["gsis_id", "pool_rank"]],
        on="gsis_id",
        how="left",
    )

    # Assemble the full output: pool + non-pool rows.
    non_pool_df = vorp_table[~vorp_table["gsis_id"].isin(pool_set)].copy()
    non_pool_df["auction_dollars"] = pd.array([0] * len(non_pool_df), dtype=pd.Int64Dtype())
    non_pool_df["pool_rank"] = pd.array([pd.NA] * len(non_pool_df), dtype=pd.Int64Dtype())

    out = pd.concat(
        [
            pool_df[
                [
                    "gsis_id",
                    "position",
                    "season_mean_fpts",
                    "vorp",
                    "auction_dollars",
                    "pool_rank",
                ]
            ],
            non_pool_df[
                [
                    "gsis_id",
                    "position",
                    "season_mean_fpts",
                    "vorp",
                    "auction_dollars",
                    "pool_rank",
                ]
            ],
        ],
        ignore_index=True,
    )
    out["in_pool"] = out["gsis_id"].isin(pool_set)

    # Step 6 - attach reference prices.
    if reference_prices is None:
        out["reference_dollars"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
        out["value_delta"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
    else:
        if reference_prices["gsis_id"].duplicated().any():
            dup = reference_prices.loc[reference_prices["gsis_id"].duplicated(), "gsis_id"].iloc[0]
            raise ValueError(
                f"reference_prices has duplicate gsis_id rows (first duplicate: {dup})."
            )
        ref = reference_prices[["gsis_id", "reference_dollars"]].copy()
        ref["reference_dollars"] = ref["reference_dollars"].astype(pd.Int64Dtype())
        out = out.merge(ref, on="gsis_id", how="left")
        # `merge` preserves Int64Dtype with NA for unmatched rows.
        out["value_delta"] = (out["auction_dollars"] - out["reference_dollars"]).astype(
            pd.Int64Dtype()
        )

    # Re-order columns to match AuctionValuesSchema.
    out = out[
        [
            "gsis_id",
            "position",
            "season_mean_fpts",
            "vorp",
            "in_pool",
            "auction_dollars",
            "pool_rank",
            "reference_dollars",
            "value_delta",
        ]
    ]

    return AuctionValuesSchema.validate(out)


__all__ = ["generate_auction_values"]
