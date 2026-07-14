"""Auction bid models (the tournament contestants) and the read-only hero view.

Spec docs/superpowers/specs/2026-06-17-auction-strategy-tournament-design.md §3.4.
Each model returns a *desired* max bid (any int); the engine clamps it to
[min_bid, feasible_max] (§3.2), so models never re-implement the reserve or the floor.
"""

from __future__ import annotations

import math
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
    baseline_dollars: pd.DataFrame  # indexed engine frame: AuctionValuesSchema cols + bot_dollars


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
        base = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        return round(base * _budget_urgency(view, config))


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
        bid = min_bid + (base - min_bid) * inflation
        return round(bid * _budget_urgency(view, config))


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
        base: float = float(min_bid)
        if lift > 0.0:
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
            if points_per_dollar > 0.0:
                base = min_bid + lift / points_per_dollar
        return round(base * _budget_urgency(view, config))


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


def _tier_cuts(pool: pd.DataFrame, stud_frac: float, scrub_frac: float) -> tuple[float, float]:
    """The (stud_cut, scrub_cut) VORP thresholds for a fraction-of-pool tiering: `stud_frac` of the
    pool clears stud_cut, `scrub_frac` falls below scrub_cut. Shared by PatientValueBid and
    StudsAndDepthBid so the tier definition can't drift between them."""
    n = len(pool)
    stud_cut = _vorp_threshold(pool, round(stud_frac * n))
    scrub_cut = _vorp_threshold(pool, round((1.0 - scrub_frac) * n))
    return stud_cut, scrub_cut


URGENCY_GAIN = 3.0


def _budget_urgency(view: AuctionView, config: LeagueConfig) -> float:
    """Late-draft budget-deployment factor (spec §A). Exactly 1.0 at the draft start
    (`my_open_slots == roster_size` -> progress 0) and when broke (no surplus beyond the
    $1-per-open-slot floor); escalates toward 1.0 + URGENCY_GAIN as the roster fills *and* idle
    cash remains. Bounded [1.0, 1.0 + URGENCY_GAIN) for my_open_slots >= 1 (both factors in [0,1));
    the engine's [min_bid, feasible_max] clamp bounds the resulting bid. The surplus<=0 guard runs
    before the surplus/my_budget term, so my_budget==0 never divides by zero."""
    surplus = view.my_budget - config.min_bid * view.my_open_slots
    if surplus <= 0:
        return 1.0
    progress = 1.0 - view.my_open_slots / config.roster_size
    return 1.0 + URGENCY_GAIN * progress * (surplus / view.my_budget)


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
        base: float = float(min_bid)
        if float(player["vorp"]) >= threshold and anchors_remaining > 0:
            reserve = min_bid * max(0, open_slots - anchors_remaining)
            # unclamped desire; engine clamps to [min_bid, feasible_max] (module contract)
            base = (view.my_budget - reserve) / anchors_remaining
        return round(base * _budget_urgency(view, config))


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
        base = value * self.k if float(player["vorp"]) >= threshold else float(value)
        return round(base * _budget_urgency(view, config))


@dataclass(frozen=True)
class VorpShareBid:
    """Allocate the remaining budget proportionally to VORP across the top-`open_slots` targets."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        targets = _undrafted(pool, view.drafted).nlargest(view.my_open_slots, "vorp")
        base: float = float(min_bid)
        if str(player["gsis_id"]) in {str(g) for g in targets["gsis_id"]}:
            denom = float(targets["vorp"].clip(lower=0.0).sum())
            if denom > 0.0:
                share = max(0.0, float(player["vorp"])) / denom
                base = view.my_budget * share
        return round(base * _budget_urgency(view, config))


@dataclass(frozen=True)
class PatientValueBid:
    """Holds budget through the stud frenzy; pays up for mid-tier value when reserve remains."""

    midtier_premium: float = 0.35
    stud_frac: float = 0.10
    scrub_frac: float = 0.50

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        stud_cut, scrub_cut = _tier_cuts(pool, self.stud_frac, self.scrub_frac)
        v = float(player["vorp"])
        base: float = float(min_bid)
        if stud_cut > v >= scrub_cut:  # mid-tier: not a stud (let go), not a scrub
            value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
            bid = round(value * (1.0 + self.midtier_premium))
            reserve = view.my_budget - min_bid * (view.my_open_slots - 1)
            if reserve >= bid:
                base = float(bid)
        return round(base * _budget_urgency(view, config))


@dataclass(frozen=True)
class StudsAndDepthBid:
    """The 'good bot as a hero' (spec §C): secure a few studs near fair value (a modest premium to
    actually win the anchor), bid fair value across mid-tier depth (no $1-dumping), $1 the scrubs —
    then deploy the whole budget via _budget_urgency as the draft winds down."""

    stud_premium: float = 0.2
    stud_frac: float = 0.10
    scrub_frac: float = 0.20

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        stud_cut, scrub_cut = _tier_cuts(pool, self.stud_frac, self.scrub_frac)
        v = float(player["vorp"])
        auction_dollars = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        if v >= stud_cut:  # stud: modest premium to actually win the anchor (unlike static)
            base: float = auction_dollars * (1.0 + self.stud_premium)
        elif v < scrub_cut:  # scrub: floor it
            base = float(min_bid)
        else:  # mid-tier depth: fair value, no $1-dumping
            base = float(auction_dollars)
        return round(base * _budget_urgency(view, config))


@dataclass(frozen=True)
class BalancedValueBid:
    """Balanced-breadth hero: spread the whole budget into a full roster by bidding up to a LOW
    per-player cap (`pace` x the even per-slot share) on every startable player, instead of
    concentrating on studs. Deliberately does NOT apply _budget_urgency (the ramp over-pays late
    scrubs).

    `premium` scales the fair-value bid so that in an INFLATED market (e.g. ESPN-anchored bots,
    where the mid-tier clears above fair value) the bid still reaches the cap and wins the
    contested mid-tier; `premium=1.0` bids the cap on any player worth more than a full per-slot
    share. The low cap is what forces the spread — raising it backfires (it lets the hero chase
    over-priced studs and starve the roster). Defaults from the 2026-07-14 cap-vs-premium sweep:
    premium=1.0 is a ~2x ESPN-market win (playoff 0.24 -> 0.44) and neutral in the un-inflated
    model market; the low cap wins both. See reports/auction_tournament_validation_2026.md.

    `non_increasing_cap=True` clamps the pace cap to the OPENING per-slot share so it can't
    self-inflate as the hero wins cheap players (the Slice 1 fix, 2026-07-14 robust-win-hero spec);
    default False keeps the inflating control byte-for-byte."""

    premium: float = 1.0
    pace: float = 2.0
    non_increasing_cap: bool = False

    def __post_init__(self) -> None:
        if not (self.premium >= 0.0 and math.isfinite(self.premium)):
            raise ValueError(f"premium must be finite and >= 0; got {self.premium}")
        if not (self.pace > 0.0 and math.isfinite(self.pace)):
            raise ValueError(f"pace must be finite and > 0; got {self.pace}")

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        fair = float(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        per_slot = view.my_budget / max(1, view.my_open_slots)
        if self.non_increasing_cap:
            # Never let the cap rise above the OPENING per-slot pace. As a breadth hero wins players
            # cheaper than its per-slot share, budget/open_slots ratchets up and the inflating cap
            # balloons (overpays late, lopsided roster — the diagnosed bug). Clamping to the constant
            # opening share kills the ratchet while still retreating below it when the hero is broke.
            per_slot = min(per_slot, config.budget / config.roster_size)
        cap = self.pace * per_slot
        return round(min(fair * (1.0 + self.premium), cap))
