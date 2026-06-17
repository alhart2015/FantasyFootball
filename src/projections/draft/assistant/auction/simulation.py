"""One full auction: nominate -> bid -> award, hero via a bid model, rest via bots (spec §3.6).

Returns every seat's roster as {seat(1-based): [gsis_id, ...]} — the project_draft input.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import validate_pool_size
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy, AuctionView
from projections.draft.assistant.auction.market import SeatView, bot_max_bid, resolve_bids
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import bot_eligible, bot_position_bounds
from projections.schemas import Position


def validate_auction_inputs(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Pool-size and budget-solvency preconditions (spec §3.1)."""
    validate_pool_size(pool, config)
    if config.budget < config.min_bid * config.roster_size:
        raise ValueError(
            f"budget {config.budget} < min_bid*roster_size "
            f"({config.min_bid}*{config.roster_size}); a seat can't afford min_bid for every slot"
        )


@dataclass
class AuctionState:
    budgets: list[int]  # 0-based per seat
    rosters: list[list[tuple[str, str, int]]]  # (gsis_id, position, price) per seat, 0-based
    drafted: set[str]
    nominator: int  # 0-based

    @classmethod
    def initial(cls, config: LeagueConfig) -> AuctionState:
        n = config.n_teams
        return cls([config.budget] * n, [[] for _ in range(n)], set(), 0)


def _open_slots(state: AuctionState, seat: int, roster_size: int) -> int:
    return roster_size - len(state.rosters[seat])


def _feasible_max(state: AuctionState, seat: int, roster_size: int, min_bid: int) -> int:
    return state.budgets[seat] - min_bid * (_open_slots(state, seat, roster_size) - 1)


def _build_view(
    state: AuctionState, hero0: int, pool: pd.DataFrame, bd: pd.DataFrame, config: LeagueConfig
) -> AuctionView:
    my_ids = [g for (g, _p, _pr) in state.rosters[hero0]]
    return AuctionView(
        my_budget=state.budgets[hero0],
        my_open_slots=_open_slots(state, hero0, config.roster_size),
        my_positions=Counter(p for (_g, p, _pr) in state.rosters[hero0]),
        my_roster=pool[pool["gsis_id"].isin(my_ids)],
        drafted=frozenset(state.drafted),
        budgets_by_seat=tuple(state.budgets),
        baseline_dollars=bd,
    )


def _simulate_to_state(
    strategy: AuctionBidStrategy,
    my_seat: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    baseline_dollars: pd.DataFrame,
    price_jitter: float,
    rng: np.random.Generator,
) -> AuctionState:
    """Run the full auction loop; return the final AuctionState (budgets + priced rosters)."""
    validate_auction_inputs(pool, config)
    n = config.n_teams
    rs = config.roster_size
    min_bid = config.min_bid
    hero0 = my_seat - 1  # the single 1-based -> 0-based conversion (spec §3.2)

    minimums, maximums = bot_position_bounds(config.roster_slots)
    pos_by_id = {
        str(g): Position(str(p)) for g, p in zip(pool["gsis_id"], pool["position"], strict=True)
    }
    bd = baseline_dollars.set_index("gsis_id")
    nominate_order = bd.sort_values("auction_dollars", ascending=False).index.tolist()
    all_positions = frozenset(Position)
    state = AuctionState.initial(config)

    while any(_open_slots(state, s, rs) > 0 for s in range(n)):
        # advance the nominator pointer to a seat that still has an open slot
        while _open_slots(state, state.nominator, rs) == 0:
            state.nominator = (state.nominator + 1) % n

        # eligible positions per open seat (hero is ungated -> all positions); union over the room
        seat_eligible: dict[int, frozenset[Position]] = {}
        union: set[Position] = set()
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            if seat == hero0:
                seat_eligible[seat] = all_positions
                union |= all_positions
            else:
                counts = {
                    Position(p): c
                    for p, c in Counter(p for (_g, p, _pr) in state.rosters[seat]).items()
                }
                elig = bot_eligible(
                    counts, _open_slots(state, seat, rs), minimums=minimums, maximums=maximums
                )
                seat_eligible[seat] = elig
                union |= elig

        # nominate the highest-baseline undrafted player the room can roster; else forced (un-gated)
        nominee_id = next(
            (g for g in nominate_order if g not in state.drafted and pos_by_id[str(g)] in union),
            None,
        )
        forced = nominee_id is None
        if forced:
            nominee_id = next(g for g in nominate_order if g not in state.drafted)
            warnings.warn(
                f"auction: no open seat can roster a remaining position; forcing nominee "
                f"{nominee_id} ({pos_by_id[str(nominee_id)].value}) ungated (pool thin).",
                stacklevel=2,
            )
        assert nominee_id is not None  # guaranteed: pool is non-empty while any seat has open slots
        player = pool.loc[pool["gsis_id"] == nominee_id].iloc[0]

        # collect bids: hero always bids; bots abstain (dropped) if position-gated unless forced
        bids: dict[int, int] = {}
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            fmax = _feasible_max(state, seat, rs, min_bid)
            if seat == hero0:
                desired = strategy.max_bid(
                    _build_view(state, hero0, pool, bd, config), player, pool, config
                )
                bids[seat] = max(min_bid, min(int(desired), fmax))
            else:
                elig = all_positions if forced else seat_eligible[seat]
                desired = bot_max_bid(
                    SeatView(open_slots=_open_slots(state, seat, rs), eligible_positions=elig),
                    player,
                    bd,
                    config,
                    rng,
                    price_jitter=price_jitter,
                )
                if desired <= 0:  # abstain -> dropped before the clamp
                    continue
                bids[seat] = max(min_bid, min(int(desired), fmax))

        assert bids, "resolve_bids requires >=1 bid; forced-pick path guarantees it"
        winner, price = resolve_bids(bids, min_bid)
        state.budgets[winner] -= price
        state.rosters[winner].append((nominee_id, str(player["position"]), price))
        state.drafted.add(nominee_id)
        state.nominator = (state.nominator + 1) % n

    return state


def simulate_auction(
    strategy: AuctionBidStrategy,
    my_seat: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    baseline_dollars: pd.DataFrame,
    price_jitter: float,
    rng: np.random.Generator,
) -> dict[int, list[str]]:
    """One full auction; return every seat's roster {seat(1-based): [gsis_id, ...]}."""
    state = _simulate_to_state(
        strategy,
        my_seat,
        pool,
        config,
        baseline_dollars=baseline_dollars,
        price_jitter=price_jitter,
        rng=rng,
    )
    return {seat + 1: [g for (g, _p, _pr) in state.rosters[seat]] for seat in range(config.n_teams)}
