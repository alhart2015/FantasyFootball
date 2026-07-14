"""Noisy-WTP bot bid policy and second-price clearing (spec §3.5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import Position

DEFAULT_PRICE_JITTER: float = 0.15  # fractional WTP spread; auction analog of adp_jitter


@dataclass(frozen=True)
class SeatView:
    """Minimal per-seat view the bot reads (the engine handles feasible_max + eligibility)."""

    open_slots: int
    # default = all positions; Task 5 passes the real per-bot set
    eligible_positions: frozenset[Position] = frozenset(Position)
    budget: int = 0


def bot_max_bid(
    seat_view: SeatView,
    player: pd.Series,
    baseline_dollars: pd.DataFrame,
    config: LeagueConfig,
    rng: np.random.Generator,
    *,
    price_jitter: float,
) -> int:
    """Value-rational WTP centered on the market dollar; abstain (0) if full or position-gated."""
    if seat_view.open_slots <= 0:
        return 0
    if Position(player["position"]) not in seat_view.eligible_positions:
        return 0
    base = float(baseline_dollars.loc[player["gsis_id"], "bot_dollars"])
    wtp = base * (1.0 + rng.normal(0.0, price_jitter))
    return round(max(float(config.min_bid), wtp))


def resolve_bids(bids: dict[int, int], min_bid: int, rng: np.random.Generator) -> tuple[int, int]:
    """English (second-price + min_bid) clearing. `bids` maps seat -> clamped max bid.

    Winner is the argmax bid; ties are broken UNIFORMLY AT RANDOM among the top bidders (via `rng`),
    not by seat index. A lowest-index tie-break systematically dumped every min-bid ($1) tie on seat
    0, so whoever sat there hoarded ~4 junk players/draft (~120 pts, ~0.10 reg-win%) — a pure seat
    artifact. Price is one tick over the runner-up's ceiling, never above the winner's own; a lone
    bidder pays min_bid.
    """
    max_bid = max(bids.values())
    tied = sorted(s for s, b in bids.items() if b == max_bid)
    winner_seat = tied[0] if len(tied) == 1 else int(rng.choice(tied))
    if len(bids) == 1:
        return winner_seat, min_bid
    second_max = sorted(bids.values(), reverse=True)[1]
    return winner_seat, min(max_bid, second_max + min_bid)


def resolve_unbid(
    nominee_pos: Position,
    nominator: int,
    open_seats: Sequence[int],
    seat_eligible: Mapping[int, frozenset[Position]],
    min_bid: int,
) -> tuple[int, int]:
    """Clear a zero-bid nomination — the sibling of `resolve_bids` for the no-bidder case.

    The nominator takes its nominee at `min_bid` if it can roster the position; otherwise the
    lowest-index open seat that can. On the non-forced path the room-union nomination rule
    guarantees at least one open seat is eligible for the nominee, so the search never fails
    (asserted rather than silently mis-awarding to an ineligible seat, which would violate a
    position cap). `open_seats` must be ascending for the lowest-index tiebreak; every entry (and
    `nominator`) is a key of `seat_eligible`.
    """
    if nominee_pos in seat_eligible[nominator]:
        return nominator, min_bid
    winner = next((s for s in open_seats if nominee_pos in seat_eligible[s]), None)
    assert winner is not None, (
        "non-forced nominee must be rosterable by some open seat (room-union rule)"
    )
    return winner, min_bid


# ---------------------------------------------------------------------------
# Bot archetypes
# ---------------------------------------------------------------------------


@runtime_checkable
class BotArchetype(Protocol):
    def max_bid(
        self,
        seat_view: SeatView,
        player: pd.Series,
        baseline_dollars: pd.DataFrame,
        config: LeagueConfig,
        rng: np.random.Generator,
        *,
        price_jitter: float,
    ) -> int: ...


def _value_tier(
    value: float,
    baseline_dollars: pd.DataFrame,
    stud_frac: float,
    scrub_frac: float,
) -> str:
    """'stud' | 'mid' | 'scrub' by rank of `value` among in-pool bot_dollars (desc)."""
    inpool = baseline_dollars.loc[baseline_dollars["in_pool"], "bot_dollars"]
    n = len(inpool)
    rank = int((inpool > value).sum())  # 0-based rank, higher value -> lower rank
    if rank < stud_frac * n:
        return "stud"
    if rank >= (1.0 - scrub_frac) * n:
        return "scrub"
    return "mid"


@dataclass(frozen=True)
class AggressiveBot:
    """Today's bot: value*(1+noise), blows budget early. Delegates to bot_max_bid."""

    def max_bid(
        self,
        seat_view: SeatView,
        player: pd.Series,
        baseline_dollars: pd.DataFrame,
        config: LeagueConfig,
        rng: np.random.Generator,
        *,
        price_jitter: float,
    ) -> int:
        return bot_max_bid(
            seat_view, player, baseline_dollars, config, rng, price_jitter=price_jitter
        )


@dataclass(frozen=True)
class PatientValueBot:
    """Underbids studs (reserves budget), pays a premium for mid-tier value when it has reserve."""

    understud: float = 0.5
    midtier_premium: float = 0.35
    stud_frac: float = 0.10
    scrub_frac: float = 0.50

    def max_bid(
        self,
        seat_view: SeatView,
        player: pd.Series,
        baseline_dollars: pd.DataFrame,
        config: LeagueConfig,
        rng: np.random.Generator,
        *,
        price_jitter: float,
    ) -> int:
        pos = Position(player["position"])
        if seat_view.open_slots <= 0 or pos not in seat_view.eligible_positions:
            return 0
        value = float(baseline_dollars.loc[player["gsis_id"], "bot_dollars"])
        tier = _value_tier(value, baseline_dollars, self.stud_frac, self.scrub_frac)
        noise = 1.0 + rng.normal(0.0, price_jitter)
        if tier == "stud":
            return round(max(float(config.min_bid), value * self.understud * noise))
        reserve = seat_view.budget - config.min_bid * (seat_view.open_slots - 1)
        if tier == "mid" and reserve > value:  # value-aware reserve (spec §Part 2)
            return round(max(float(config.min_bid), value * (1.0 + self.midtier_premium) * noise))
        return config.min_bid


@dataclass(frozen=True)
class BalancedBot:
    """Aggressive WTP, but paced: never spends more than `pace` x its even per-slot share."""

    pace: float = 2.0

    def max_bid(
        self,
        seat_view: SeatView,
        player: pd.Series,
        baseline_dollars: pd.DataFrame,
        config: LeagueConfig,
        rng: np.random.Generator,
        *,
        price_jitter: float,
    ) -> int:
        pos = Position(player["position"])
        if seat_view.open_slots <= 0 or pos not in seat_view.eligible_positions:
            return 0
        value = float(baseline_dollars.loc[player["gsis_id"], "bot_dollars"])
        wtp = value * (1.0 + rng.normal(0.0, price_jitter))
        cap = self.pace * (seat_view.budget / seat_view.open_slots)
        return round(max(float(config.min_bid), min(wtp, cap)))


def assign_bot_archetypes(n_bots: int, mix: Sequence[BotArchetype]) -> list[BotArchetype]:
    """Round-robin `mix` across `n_bots` seats — exact, reproducible composition."""
    return [mix[i % len(mix)] for i in range(n_bots)]
