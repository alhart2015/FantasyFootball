"""Auction bid models (the tournament contestants) and the read-only hero view.

Spec docs/superpowers/specs/2026-06-17-auction-strategy-tournament-design.md §3.4.
Each model returns a *desired* max bid (any int); the engine clamps it to
[min_bid, feasible_max] (§3.2), so models never re-implement the reserve or the floor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.league_config import LeagueConfig


@dataclass(frozen=True)
class AuctionView:
    """Read-only snapshot of the auction for the hero seat (built by simulate_auction)."""

    my_budget: int
    my_open_slots: int
    my_positions: Counter[str]
    my_roster: pd.DataFrame  # pool rows for the hero's drafted gsis_ids
    drafted: frozenset[str]
    budgets_by_seat: tuple[int, ...]
    baseline_dollars: pd.DataFrame  # full AuctionValuesSchema frame, indexed by gsis_id


@runtime_checkable
class AuctionBidStrategy(Protocol):
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int: ...


def _total_open_slots(view: AuctionView, config: LeagueConfig) -> int:
    return config.n_teams * config.roster_size - len(view.drafted)


@dataclass(frozen=True)
class StaticDollarBid:
    """v1 — bid straight to the static SOS dollar."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        return int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])


@dataclass(frozen=True)
class InflationBid:
    """v2 — re-price the static dollar by live surplus inflation (spec §3.4)."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        money = sum(view.budgets_by_seat) - min_bid * _total_open_slots(view, config)
        bd = view.baseline_dollars
        undrafted_in_pool = bd[bd["in_pool"] & ~bd.index.isin(view.drafted)]
        value = float((undrafted_in_pool["auction_dollars"] - min_bid).sum())
        inflation = money / value if value > 0 else 1.0
        base = int(bd.loc[player["gsis_id"], "auction_dollars"])
        return round(min_bid + (base - min_bid) * inflation)


@dataclass(frozen=True)
class MarginalValueBid:
    """v3 — bid to the player's marginal optimal-lineup lift at the live market rate (spec §3.4)."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        slots = config.roster_slots
        base_pts = optimal_lineup_points(view.my_roster, slots)
        cand = pool[pool["gsis_id"] == player["gsis_id"]]
        with_player = pd.concat([view.my_roster, cand], ignore_index=True)
        lift = optimal_lineup_points(with_player, slots) - base_pts
        if lift <= 0.0:
            return min_bid
        money = sum(view.budgets_by_seat) - min_bid * _total_open_slots(view, config)
        bd = view.baseline_dollars
        undrafted_in_pool_ids = bd[bd["in_pool"] & ~bd.index.isin(view.drafted)].index
        # Lift of a single player to an EMPTY lineup == its season_mean_fpts (every in-pool
        # position has a starting slot), so the board's surplus value in lineup-points is the
        # sum of undrafted in-pool projected points. Cheap, and equal to summing single-player
        # optimal_lineup_points one by one.
        on_board = pool[pool["gsis_id"].isin(undrafted_in_pool_ids)]
        value_points = float(on_board["season_mean_fpts"].sum())
        points_per_dollar = value_points / money if money > 0 else 0.0
        if points_per_dollar <= 0.0:
            return min_bid
        return round(min_bid + lift / points_per_dollar)
