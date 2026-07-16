"""One full auction: nominate -> bid -> award, hero via a bid model, rest via bots (spec §3.6).

Returns every seat's roster as {seat(1-based): [gsis_id, ...]} — the project_draft input.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import validate_pool_size
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy, AuctionView
from projections.draft.assistant.auction.market import (
    AggressiveBot,
    BotArchetype,
    SeatView,
    assign_bot_archetypes,
    resolve_bids,
    resolve_unbid,
)
from projections.draft.assistant.auction.nomination import HeroNominator, NominationContext
from projections.draft.assistant.auction.snake_bot import SnakeBoard, adp_usable
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import bot_eligible, bot_position_bounds
from projections.schemas import Position


def _sample_nominee(
    candidates: list[str], val_by_id: dict[str, float], temp: float, rng: np.random.Generator
) -> str:
    """Pick the next nominee. temp<=0 -> the highest-value candidate (candidates are pre-sorted
    value-desc), consuming no RNG. temp>0 -> sample with weight max(value, 0.5)**(1/temp)."""
    if temp <= 0.0:
        return candidates[0]
    weights = np.array(
        [max(val_by_id[str(g)], 0.5) ** (1.0 / temp) for g in candidates], dtype=float
    )
    return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]


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


@dataclass(frozen=True)
class PickRecord:
    """One awarded nomination, appended to an opt-in `trace` for post-hoc market analysis.

    Records the global pick order the final `AuctionState` cannot reconstruct (rosters are stored
    per seat, not in draft order). `value` is the model auction_dollars (the shared fair value the
    model-market bots price on), so `price - value` is the clearing surplus/premium; `room_budget`
    is the sum of every seat's remaining budget AFTER the award (a proxy for how drained the room is
    when the player clears). No behavior change — populated only when a caller passes a `trace`.
    """

    pick: int  # 1-based global pick index (== len(drafted) after the award)
    gsis_id: str
    position: str
    value: float  # model auction_dollars (fair value)
    price: int  # clearing price paid
    winner_seat: int  # 0-based
    room_budget: int  # sum of all seats' budgets after this award
    forced: bool  # the ungated pool-thin fallback nomination (no eligible-position gate)


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
    snake_rng: np.random.Generator | None = None,
    nomination_temp: float = 0.0,
    bot_archetypes: Sequence[BotArchetype] | None = None,
    bot_dollars: pd.Series | None = None,
    trace: list[PickRecord] | None = None,
    hero_nominator: HeroNominator | None = None,
) -> AuctionState:
    """Run the full auction loop; return the final AuctionState (budgets + priced rosters).

    If `trace` is provided, one `PickRecord` per award is appended in draft order (diagnostics only;
    None leaves the hot path unchanged). If `hero_nominator` is provided, it chooses the hero's
    nominee on the hero's own non-forced turns (Slice 2 probe); None keeps `_sample_nominee` for all
    seats. Bots, the snake-broke path, and the forced pool-thin fallback are unaffected either way.
    """
    validate_auction_inputs(pool, config)
    n = config.n_teams
    rs = config.roster_size
    min_bid = config.min_bid
    hero0 = my_seat - 1  # the single 1-based -> 0-based conversion (spec §3.2)

    bot_seats = [s for s in range(n) if s != hero0]
    if bot_archetypes is None:
        seat_arch: dict[int, BotArchetype] = {s: AggressiveBot() for s in bot_seats}
    else:
        _assigned = assign_bot_archetypes(len(bot_seats), bot_archetypes)
        seat_arch = {s: _assigned[i] for i, s in enumerate(bot_seats)}

    if snake_rng is None:
        snake_rng = rng.spawn(1)[0]  # CRN-safe: spawn advances the seed-sequence, not rng's stream
    adp_ok = adp_usable(pool)
    # Per-bot fixed noisy-ADP boards; broke bots consume these to snipe their snake target (Task 4).
    snake_boards: dict[int, SnakeBoard] = (
        {s: SnakeBoard(pool, snake_rng) for s in bot_seats} if adp_ok else {}
    )

    minimums, maximums = bot_position_bounds(config.roster_slots)
    pos_by_id = {
        str(g): Position(str(p)) for g, p in zip(pool["gsis_id"], pool["position"], strict=True)
    }
    val_by_id = {
        str(g): float(v)
        for g, v in zip(
            baseline_dollars["gsis_id"], baseline_dollars["auction_dollars"], strict=True
        )
    }
    # Row lookup by id, built once — avoids an O(pool) boolean scan to fetch the nominee each round.
    pool_by_id = {str(g): row for g, row in pool.set_index("gsis_id", drop=False).iterrows()}
    bd = baseline_dollars.set_index("gsis_id")
    nominate_order = bd.sort_values("auction_dollars", ascending=False).index.tolist()
    if bot_dollars is None:
        bd["bot_dollars"] = bd["auction_dollars"]
    else:
        bd["bot_dollars"] = (
            bot_dollars.reindex(bd.index).fillna(bd["auction_dollars"]).astype(pd.Int64Dtype())
        )
    # value the room bids on — only the opt-in hero_nominator reads it, so skip the O(pool) build
    # when the hook is off (mirrors the `trace` guard; the default None path pays nothing here).
    # When bot_dollars is None the room bids on auction_dollars, so reuse val_by_id verbatim.
    if hero_nominator is None:
        bot_by_id: dict[str, float] = {}
    elif bot_dollars is None:
        bot_by_id = val_by_id
    else:
        bot_by_id = {str(g): float(v) for g, v in bd["bot_dollars"].items()}
    all_positions = frozenset(Position)
    state = AuctionState.initial(config)

    while any(_open_slots(state, s, rs) > 0 for s in range(n)):
        # advance the nominator pointer to a seat that still has an open slot
        while _open_slots(state, state.nominator, rs) == 0:
            state.nominator = (state.nominator + 1) % n

        # eligible positions per OPEN seat — hero included — via the SAME bot rule (spec R1); union
        seat_eligible: dict[int, frozenset[Position]] = {}
        union: set[Position] = set()
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
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
        candidates = [
            g for g in nominate_order if g not in state.drafted and pos_by_id[str(g)] in union
        ]
        forced = not candidates
        # state.drafted changes once per pick (after the award); build the frozenset the snake board
        # needs once here rather than per broke seat.
        drafted_fs = frozenset(state.drafted)
        if forced:
            nominee_id = next(g for g in nominate_order if g not in state.drafted)
            warnings.warn(
                f"auction: no open seat can roster a remaining position; forcing nominee "
                f"{nominee_id} ({pos_by_id[str(nominee_id)].value}) ungated (pool thin).",
                stacklevel=2,
            )
        else:
            # `snake_boards` holds only bot seats, and only when adp is usable; this is the
            # non-forced branch — so `nom in snake_boards and fmax == min_bid` is exactly "nom is a
            # broke bot." A broke nominator nominates its own snake target; a None target (its
            # eligible positions are pool-exhausted) or a flush/hero nominator falls back to central
            # sampling.
            nom = state.nominator
            nom_fmax = _feasible_max(state, nom, rs, min_bid)
            nominee_id = None
            if nom in snake_boards and nom_fmax == min_bid:
                nominee_id = snake_boards[nom].best_available(drafted_fs, seat_eligible[nom])
            if nominee_id is None:
                # Draw the central nominee unconditionally so the shared rng advances identically
                # whether or not a hero_nominator overrides the pick. Without this, at
                # nomination_temp>0 the override path skips _sample_nominee's rng.choice and the
                # stream desyncs after the hero's first nomination — which would break the CRN
                # pairing the probe's control-vs-poison verdict depends on.
                nominee_id = _sample_nominee(candidates, val_by_id, nomination_temp, rng)
                if nom == hero0 and hero_nominator is not None:
                    ctx = NominationContext(
                        hero_positions=Counter(
                            Position(p) for (_g, p, _pr) in state.rosters[hero0]
                        ),
                        value_by_id=bot_by_id,
                        position_by_id=pos_by_id,
                        position_minimums=minimums,
                    )
                    override = hero_nominator(candidates, ctx)
                    # Hard check (not assert): guards the pluggable hook even under `python -O` — a
                    # non-candidate would else KeyError or dup-draft downstream.
                    if override not in candidates:
                        raise ValueError("hero_nominator must return a member of candidates")
                    nominee_id = override
        assert nominee_id is not None  # guaranteed: pool is non-empty while any seat has open slots
        player = pool_by_id[str(nominee_id)]

        # collect bids: hero and bots alike abstain on an ineligible position unless forced
        bids: dict[int, int] = {}
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            fmax = _feasible_max(state, seat, rs, min_bid)
            elig = all_positions if forced else seat_eligible[seat]
            if seat == hero0:
                if pos_by_id[str(nominee_id)] not in elig:
                    continue  # hero is now gated like a bot (spec R2)
                desired = strategy.max_bid(
                    _build_view(state, hero0, pool, bd, config), player, pool, config
                )
                bids[seat] = max(min_bid, min(int(desired), fmax))
            else:
                # `seat in snake_boards` == "adp-usable bot seat" (boards exist only for bot seats);
                # combined with `not forced` and `fmax == min_bid` this is "seat is a broke bot."
                broke = not forced and seat in snake_boards and fmax == min_bid
                if broke:
                    target = snake_boards[seat].best_available(drafted_fs, seat_eligible[seat])
                    if target is None or str(nominee_id) != str(target):
                        continue  # abstain: not this broke bot's snake target
                    bids[seat] = min_bid  # broke ⇒ feasible_max == min_bid
                else:
                    desired = seat_arch[seat].max_bid(
                        SeatView(
                            open_slots=_open_slots(state, seat, rs),
                            eligible_positions=elig,
                            budget=state.budgets[seat],
                        ),
                        player,
                        bd,
                        config,
                        rng,
                        price_jitter=price_jitter,
                    )
                    if desired <= 0:  # abstain -> dropped before the clamp
                        continue
                    bids[seat] = max(min_bid, min(int(desired), fmax))

        if not bids:
            # Nobody bid (reachable on the non-forced path once broke bots abstain): the nominator
            # takes its nominee at min_bid, else the lowest-index open seat that can roster it.
            open_seats = [s for s in range(n) if _open_slots(state, s, rs) > 0]
            winner, price = resolve_unbid(
                pos_by_id[str(nominee_id)], state.nominator, open_seats, seat_eligible, min_bid
            )
        else:
            winner, price = resolve_bids(bids, min_bid, rng)
        state.budgets[winner] -= price
        state.rosters[winner].append((nominee_id, str(player["position"]), price))
        state.drafted.add(nominee_id)
        if trace is not None:
            trace.append(
                PickRecord(
                    pick=len(state.drafted),
                    gsis_id=str(nominee_id),
                    position=str(player["position"]),
                    value=val_by_id[str(nominee_id)],
                    price=price,
                    winner_seat=winner,
                    room_budget=sum(state.budgets),
                    forced=forced,
                )
            )
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
    snake_rng: np.random.Generator | None = None,
    nomination_temp: float = 0.0,
    bot_archetypes: Sequence[BotArchetype] | None = None,
    bot_dollars: pd.Series | None = None,
    hero_nominator: HeroNominator | None = None,
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
        snake_rng=snake_rng,
        nomination_temp=nomination_temp,
        bot_archetypes=bot_archetypes,
        bot_dollars=bot_dollars,
        hero_nominator=hero_nominator,
    )
    return {seat + 1: [g for (g, _p, _pr) in state.rosters[seat]] for seat in range(config.n_teams)}
