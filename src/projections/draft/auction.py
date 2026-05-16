"""Auction $ generator: converts per-player VORP to per-player auction dollars.

Public surface: `generate_auction_values(vorp_table, league_config, reference_prices=None)`.

Algorithm: standard Surplus-Of-Surplus (SOS) allocation. Reserve `min_bid` for every
drafted slot, then distribute the remaining budget proportionally to positive VORP
among the rostered pool. Strategy-agnostic; one $ per player.

Spec: docs/superpowers/specs/2026-05-16-auction-values-design.md
"""

from __future__ import annotations

import pandas as pd

from projections.draft._pool import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.schemas import AuctionValuesSchema

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "gsis_id",
    "position",
    "season_mean_fpts",
    "vorp",
    "in_pool",
    "auction_dollars",
    "pool_rank",
    "reference_dollars",
    "value_delta",
)


def _reject_duplicate_gsis_ids(df: pd.DataFrame, label: str) -> None:
    if df["gsis_id"].duplicated().any():
        dup = df.loc[df["gsis_id"].duplicated(), "gsis_id"].iloc[0]
        raise ValueError(f"{label} has duplicate gsis_id rows (first duplicate: {dup}).")


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
    _reject_duplicate_gsis_ids(vorp_table, "vorp_table")

    pool_ids = _select_pool(vorp_table, league_config)
    pool_set = set(pool_ids)
    pool_mask = vorp_table["gsis_id"].isin(pool_set)

    total_budget = league_config.total_budget
    reserve = league_config.total_pool_size * league_config.min_bid
    surplus = total_budget - reserve

    pool_df = vorp_table.loc[pool_mask].copy()
    positive_vorp = pool_df["vorp"].clip(lower=0.0)
    positive_vorp_sum = float(positive_vorp.sum())

    if positive_vorp_sum > 0:
        extra_float = (positive_vorp / positive_vorp_sum) * surplus
    else:
        # Every in-pool player has VORP <= 0 — distribute the surplus uniformly so the
        # sum invariant still holds and no player exceeds another by accident of rounding.
        extra_float = pd.Series(surplus / league_config.total_pool_size, index=pool_df.index)

    dollars_float = league_config.min_bid + extra_float
    rounded = dollars_float.round().astype("int64")

    drift = total_budget - int(rounded.sum())
    if drift != 0:
        fractional = dollars_float - dollars_float.astype("int64")
        if drift > 0:
            order = fractional.sort_values(ascending=False).index
        else:
            # Floor-protection: rows already at min_bid can't absorb -1 without
            # violating the spec §3 step 4 invariant. Exclude them from candidates.
            adjustable_mask = rounded > league_config.min_bid
            order = fractional[adjustable_mask].sort_values(ascending=True).index
            if len(order) < abs(drift):
                raise ValueError(
                    f"Cannot close rounding drift of {drift} without violating min_bid "
                    f"floor of ${league_config.min_bid}. This usually indicates an extreme "
                    f"degenerate input (e.g., very small budget per slot)."
                )
        step = 1 if drift > 0 else -1
        for idx in order[: abs(drift)]:
            rounded.loc[idx] = rounded.loc[idx] + step
    pool_df["auction_dollars"] = rounded.astype(pd.Int64Dtype())

    pool_df = pool_df.sort_values(
        by=["auction_dollars", "vorp", "season_mean_fpts", "gsis_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    pool_df["pool_rank"] = pd.array(range(1, len(pool_df) + 1), dtype=pd.Int64Dtype())

    non_pool_df = vorp_table.loc[~pool_mask].copy()
    non_pool_df["auction_dollars"] = pd.array([0] * len(non_pool_df), dtype=pd.Int64Dtype())
    non_pool_df["pool_rank"] = pd.array([pd.NA] * len(non_pool_df), dtype=pd.Int64Dtype())

    out = pd.concat([pool_df, non_pool_df], ignore_index=True)
    out["in_pool"] = out["gsis_id"].isin(pool_set)

    if reference_prices is None:
        out["reference_dollars"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
        out["value_delta"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
    else:
        _reject_duplicate_gsis_ids(reference_prices, "reference_prices")
        ref = reference_prices[["gsis_id", "reference_dollars"]].copy()
        ref["reference_dollars"] = ref["reference_dollars"].astype(pd.Int64Dtype())
        out = out.merge(ref, on="gsis_id", how="left")
        out["value_delta"] = (out["auction_dollars"] - out["reference_dollars"]).astype(
            pd.Int64Dtype()
        )

    return AuctionValuesSchema.validate(out[list(_OUTPUT_COLUMNS)])


__all__ = ["generate_auction_values"]
