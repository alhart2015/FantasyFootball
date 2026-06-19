"""Auction $ generator: converts per-player VORP to per-player auction dollars.

Public surface: `generate_auction_values(vorp_table, league_config, reference_prices=None)`.

Algorithm: standard Surplus-Of-Surplus (SOS) allocation. Reserve `min_bid` for every
drafted slot, then distribute the remaining budget proportionally to positive VORP
among the rostered pool. Strategy-agnostic; one $ per player.

Spec: docs/superpowers/specs/2026-05-16-auction-values-design.md
"""

from __future__ import annotations

import pandas as pd

from projections.draft._pool import _reject_duplicate_gsis_ids, _select_pool
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


def _allocate_surplus(value_signal: pd.Series, config: LeagueConfig) -> pd.Series:
    """Split the auction surplus across in-pool players in proportion to a non-negative value
    signal, returning whole-dollar prices that sum to ``config.total_budget``.

    ``value_signal`` is a non-null ``float64`` Series indexed over the in-pool players (one entry
    per drafted slot). Every entry is floored at ``min_bid``; the surplus
    ``total_budget - total_pool_size*min_bid`` is distributed proportionally to ``value_signal``
    (uniformly if it sums to <= 0). The index is preserved; the result is ``Int64``. Shared by
    ``generate_auction_values`` (VORP signal) and ``espn_anchored_bot_prices`` (ESPN $ signal).
    """
    total_budget = config.total_budget
    reserve = config.total_pool_size * config.min_bid
    surplus = total_budget - reserve

    signal_sum = float(value_signal.sum())
    if signal_sum > 0:
        extra_float = (value_signal / signal_sum) * surplus
    else:
        extra_float = pd.Series(surplus / config.total_pool_size, index=value_signal.index)

    dollars_float = config.min_bid + extra_float
    rounded = dollars_float.round().astype("int64")

    drift = total_budget - int(rounded.sum())
    if drift != 0:
        fractional = dollars_float - dollars_float.astype("int64")
        if drift > 0:
            order = fractional.sort_values(ascending=False).index
        else:
            adjustable_mask = rounded > config.min_bid
            order = fractional[adjustable_mask].sort_values(ascending=True).index
            if len(order) < abs(drift):
                raise ValueError(
                    f"Cannot close rounding drift of {drift} without violating min_bid "
                    f"floor of ${config.min_bid}. This usually indicates an extreme "
                    f"degenerate input (e.g., very small budget per slot)."
                )
        step = 1 if drift > 0 else -1
        for idx in order[: abs(drift)]:
            rounded.loc[idx] = rounded.loc[idx] + step
    return rounded.astype(pd.Int64Dtype())


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

    pool_df = vorp_table.loc[pool_mask].copy()
    pool_df["auction_dollars"] = _allocate_surplus(pool_df["vorp"].clip(lower=0.0), league_config)

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


def espn_anchored_bot_prices(pool: pd.DataFrame, config: LeagueConfig) -> pd.Series:
    """Per-player bot reference dollars anchored on real ESPN auction values (TODO #49c Slice 2).

    Returns a ``gsis_id``-indexed ``Int64`` Series over EVERY row of ``pool``: in-pool players get
    an SOS allocation of the budget over ``espn_auction_dollars`` (NA / absent column -> 0 weight,
    so unpriced players park at ``min_bid``); out-of-pool players get 0 (bots reading it bid
    ``min_bid``, as today). Call on the same ``pool`` frame passed to ``generate_auction_values``.

    Raises ``ValueError`` on degenerate drift (propagated from ``_allocate_surplus``); espn-mode
    callers catch it and fall back to model pricing.
    """
    _reject_duplicate_gsis_ids(pool, "pool")
    pool_set = set(_select_pool(pool, config))
    pool_mask = pool["gsis_id"].isin(pool_set)
    pool_df = pool.loc[pool_mask].copy()

    if "espn_auction_dollars" in pool_df.columns:
        value_signal = (
            pool_df["espn_auction_dollars"]
            .astype("Float64")
            .fillna(0)
            .clip(lower=0.0)
            .astype("float64")
        )
    else:
        value_signal = pd.Series(0.0, index=pool_df.index, dtype="float64")
    in_pool_dollars = _allocate_surplus(value_signal, config)

    out = pd.Series(
        pd.array([0] * len(pool), dtype=pd.Int64Dtype()),
        index=pd.Index(pool["gsis_id"], name="gsis_id"),
    )
    out.loc[pd.Index(pool_df["gsis_id"])] = in_pool_dollars.to_numpy()
    return out


__all__ = ["espn_anchored_bot_prices", "generate_auction_values"]
