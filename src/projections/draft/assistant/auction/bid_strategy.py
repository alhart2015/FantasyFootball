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


def _surplus_money(view: AuctionView, config: LeagueConfig) -> int:
    """Room-wide budget left beyond the min-bid reserve for every open slot (the inflation/marginal
    denominator). Shared by v2 and v3 so the surplus definition can't drift between them."""
    return sum(view.budgets_by_seat) - config.min_bid * _total_open_slots(view, config)


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
        money = _surplus_money(view, config)
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
        money = _surplus_money(view, config)
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


def _undrafted(pool: pd.DataFrame, drafted: frozenset[str]) -> pd.DataFrame:
    """Pool rows whose gsis_id is not yet drafted (same isin pattern as InflationBid)."""
    return pool[~pool["gsis_id"].isin(drafted)]


def _vorp_threshold(pool: pd.DataFrame, k: int) -> float:
    """The k-th highest `vorp` in the pool — the cutoff for 'top-k by VORP'. If the pool has
    fewer than k players, the pool minimum (so every player clears the bar). k<=0 -> +inf
    (nothing clears)."""
    if k <= 0:
        return float("inf")
    vorps = pool["vorp"]
    if len(vorps) <= k:
        return float(vorps.min())
    return float(vorps.nlargest(k).iloc[-1])


@dataclass(frozen=True)
class AnchorBudgetBid:
    """Stars-and-scrubs: pour budget into `n_anchors` top-VORP players, $1 the rest (spec §B)."""

    n_anchors: int = 4

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        threshold = _vorp_threshold(pool, self.n_anchors * config.n_teams)
        anchors_held = int((view.my_roster["vorp"] >= threshold).sum())
        anchors_remaining = max(0, self.n_anchors - anchors_held)
        open_slots = view.my_open_slots
        feasible_max = view.my_budget - min_bid * (open_slots - 1)
        if float(player["vorp"]) >= threshold and anchors_remaining > 0:
            reserve = min_bid * max(0, open_slots - anchors_remaining)
            cap = (view.my_budget - reserve) / anchors_remaining
            return round(min(cap, float(feasible_max)))
        return min_bid


@dataclass(frozen=True)
class OverbidValueBid:
    """Pay up for studs (top-VORP), plain value for others; engine clamp handles broke (spec §B)."""

    k: float = 1.3
    stud_count: int | None = None

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        stud_count = self.stud_count if self.stud_count is not None else 3 * config.n_teams
        threshold = _vorp_threshold(pool, stud_count)
        value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        if float(player["vorp"]) >= threshold:
            return round(value * self.k)
        return value


@dataclass(frozen=True)
class VorpShareBid:
    """Allocate the remaining budget proportionally to VORP across the top-`open_slots` targets."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        targets = _undrafted(pool, view.drafted).nlargest(view.my_open_slots, "vorp")
        if str(player["gsis_id"]) not in {str(g) for g in targets["gsis_id"]}:
            return min_bid
        denom = float(targets["vorp"].clip(lower=0.0).sum())
        if denom <= 0.0:
            return min_bid
        share = max(0.0, float(player["vorp"])) / denom
        return round(view.my_budget * share)
