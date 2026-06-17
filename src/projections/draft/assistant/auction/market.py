"""Noisy-WTP bot bid policy and second-price clearing (spec §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig

DEFAULT_PRICE_JITTER: float = 0.15  # fractional WTP spread; auction analog of adp_jitter


@dataclass(frozen=True)
class SeatView:
    """Minimal per-seat view the bot reads (the engine handles feasible_max)."""

    open_slots: int


def bot_max_bid(
    seat_view: SeatView,
    player: pd.Series,
    baseline_dollars: pd.DataFrame,
    config: LeagueConfig,
    rng: np.random.Generator,
    *,
    price_jitter: float,
) -> int:
    """Value-rational WTP centered on the market dollar, with multiplicative noise."""
    if seat_view.open_slots <= 0:
        return 0
    base = float(baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
    wtp = base * (1.0 + rng.normal(0.0, price_jitter))
    return round(max(float(config.min_bid), wtp))


def resolve_bids(bids: dict[int, int], min_bid: int) -> tuple[int, int]:
    """English (second-price + min_bid) clearing. `bids` maps seat -> clamped max bid.

    Winner is the argmax bid (ties -> lowest seat index). Price is one tick over the
    runner-up's ceiling, never above the winner's own; a lone bidder pays min_bid.
    """
    ordered = sorted(bids.items(), key=lambda kv: (-kv[1], kv[0]))
    winner_seat, winner_max = ordered[0]
    if len(ordered) == 1:
        return winner_seat, min_bid
    second_max = ordered[1][1]
    return winner_seat, min(winner_max, second_max + min_bid)
