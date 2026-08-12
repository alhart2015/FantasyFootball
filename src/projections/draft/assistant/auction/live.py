"""LiveAuctionSession — the live auction board's controller (testable; Streamlit-free).

The auction analogue of `live.LiveDraftSession`. It holds the mutable auction truth (the
ordered list of who bought whom for how much) and delegates every decision to the engine
pieces the tournament already measured:

- **how much to bid** — an `AuctionBidStrategy` scored against the same `AuctionView` the
  simulation builds, then clamped to `[min_bid, feasible_max]` exactly as the engine clamps it,
  so the board's number is the number the measured strategy would have bid;
- **who may bid on what** — `bot_eligible` / `bot_position_bounds`, the same roster-discipline
  gate the engine applies to every seat (the hero included);
- **who to nominate** — the engine's own value-first rule, plus the tested (not adopted)
  poison nominators from `auction.nomination`.

`scripts/auction_board.py` is a thin view over this.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from projections.draft.assistant.auction.bid_strategy import (
    AuctionBidStrategy,
    AuctionView,
    build_engine_dollars,
    surplus_inflation,
)
from projections.draft.assistant.auction.nomination import (
    HeroNominator,
    NominationContext,
    drain_max,
    drain_off_position,
)
from projections.draft.assistant.auction.registry import ALL_BID_MODELS
from projections.draft.assistant.auction.simulation import validate_auction_inputs
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import SeatProjection, project_draft
from projections.draft.assistant.live import attach_names, build_player_names
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import (
    allocate_roster_slots,
    bot_eligible,
    bot_position_bounds,
)
from projections.schemas import (
    _PYARROW_STR,
    GsisId,
    Position,
    RosterSlot,
    validate_gsis_id,
)

# The board's bid-model menu — the tournament roster plus the validated opt-ins.
BOARD_BID_MODELS: Mapping[str, AuctionBidStrategy] = ALL_BID_MODELS
BOARD_BID_MODEL_NAMES: tuple[str, ...] = tuple(BOARD_BID_MODELS)
# `balanced` (BalancedValueBid, premium 0.0) is the shipped default: the robust win% leader in
# BOTH the model-priced and ESPN-anchored markets across all 12 seats (PR #95, Run N).
DEFAULT_BID_MODEL = "balanced"

# Slot label for a rostered player who has no allocatable slot. `record_purchase` records what
# the room did without enforcing positional eligibility, so this state is reachable.
_NO_SLOT = "(no slot)"

# Nomination modes the board offers. `value` is the engine's own rule; the other two are the
# Slice 2 poison probes, retained as tested opt-ins with a NO-GO verdict (see auction.nomination).
NOMINATION_MODES: tuple[str, ...] = ("value", "drain_off_position", "drain_max")
_POISON_NOMINATORS: dict[str, HeroNominator] = {
    "drain_off_position": drain_off_position,
    "drain_max": drain_max,
}
NOMINATION_NOTES: dict[str, str] = {
    "value": "Engine default: nominate the most valuable player the room can still roster.",
    "drain_off_position": (
        "Probe (NOT adopted): priciest player at a position you have already filled — the money "
        "drained is your rivals'. Measured a real edge in the model market only."
    ),
    "drain_max": (
        "Probe (NOT adopted): priciest player left. Measured HARMFUL; shown for reference."
    ),
}


@dataclass(frozen=True)
class Purchase:
    """One awarded lot: `seat` (1-based) bought `gsis_id` for `price`."""

    gsis_id: GsisId
    seat: int
    price: int


@dataclass(frozen=True)
class BidAdvice:
    """What to do about one player, right now.

    `max_bid` is the number to stop at: the strategy's desire clamped to the engine's
    `[min_bid, feasible_max]` window. `room_ceiling` is the most any *opponent* who can still
    roster the position is able to bid — when your `max_bid` clears it, the lot is yours if you
    want it. `eligible` is False when your own roster can no longer take the position, in which
    case `max_bid` is 0 (pass).
    """

    gsis_id: str
    full_name: str
    position: str
    max_bid: int
    desired: int  # the model's unclamped desire, before the engine clamp
    fair_value: int  # our model auction dollars
    market_value: int  # what the room anchors on (ESPN-anchored when available)
    feasible_max: int  # your own solvency ceiling
    room_ceiling: int  # richest opponent who can roster this position
    eligible: bool

    @property
    def i_want(self) -> bool:
        """Worth buying: your roster can take him and your ceiling reaches the room's price."""
        return self.eligible and self.max_bid >= self.market_value

    @property
    def uncontested(self) -> bool:
        """You out-reach every rival the model expects to bid on this position."""
        return self.eligible and self.max_bid > self.room_ceiling


@dataclass(frozen=True)
class _SeatState:
    """Everything derived from the purchase list, computed once per mutation (see `_seat_state`)."""

    rosters: dict[int, list[str]]
    spent: dict[int, int]
    drafted: frozenset[str]
    eligible: dict[int, frozenset[Position]]


@dataclass(frozen=True)
class AuctionRosterView:
    """My roster: filled slots (with prices), remaining open starting slots, money spent."""

    filled: pd.DataFrame  # columns: slot, gsis_id, full_name, position, price
    open_slots: dict[RosterSlot, int]
    spent: int


def build_market_dollars(
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    market: Literal["espn", "model"] = "espn",
    unranked_discount: float | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """`(baseline_dollars, bot_dollars)` for a live auction — the tournament's own market setup.

    `baseline_dollars` is our SOS valuation of `pool`; `bot_dollars` is what the ROOM is expected
    to pay (real ESPN auction values re-allocated over the budget when `market="espn"` and the
    pool carries them, else None = the room bids our values). Mirrors `run_auction_tournament`,
    including its fallbacks: an ESPN request on a pool without usable ESPN prices warns and falls
    back to model pricing rather than failing the draft.
    """
    if market not in ("espn", "model"):
        raise ValueError(f"market must be 'espn' or 'model'; got {market!r}")
    baseline = generate_auction_values(pool, config)
    if market == "model":
        return baseline, None
    if not has_usable_espn_prices(pool):
        warnings.warn(
            "market='espn' but the pool has no usable espn_auction_dollars; "
            "falling back to model (shared-value) pricing.",
            stacklevel=2,
        )
        return baseline, None
    try:
        bot = espn_anchored_bot_prices(
            pool, config, model_values=baseline, unranked_discount=unranked_discount
        )
    except ValueError as exc:  # degenerate rounding drift — same fallback the tournament takes
        warnings.warn(
            f"espn_anchored_bot_prices failed ({exc}); falling back to model.", stacklevel=2
        )
        return baseline, None
    return baseline, bot


@dataclass
class LiveAuctionSession:
    """Mutable, Streamlit-free controller for one live auction draft."""

    league: LeagueConfig
    my_seat: int  # 1-based
    id_map: pd.DataFrame
    pool: pd.DataFrame
    strategy: AuctionBidStrategy
    strategy_name: str
    market: Literal["espn", "model"] = "espn"
    unranked_discount: float | None = None
    nomination_mode: str = "value"
    season: int = 2026
    purchases: list[Purchase] = field(default_factory=list)
    team_names: tuple[str, ...] = ()  # optional per-seat labels, seat 1 first
    # Persistence-only paths (defaults keep core tests path-free).
    league_config_path: Path = field(default=Path("."))
    vorp_path: Path = field(default=Path("."))
    id_map_path: Path = field(default=Path("."))
    # Lazily built so constructing a session doesn't re-run the SOS allocation.
    _market: tuple[pd.DataFrame, pd.Series | None] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _seat_cache: tuple[tuple[tuple[str, int, int], ...], _SeatState] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not 1 <= self.my_seat <= self.league.n_teams:
            raise ValueError(f"my_seat must be in 1..{self.league.n_teams}; got {self.my_seat}")
        if self.nomination_mode not in NOMINATION_MODES:
            raise ValueError(
                f"unknown nomination_mode {self.nomination_mode!r}; expected {NOMINATION_MODES}"
            )
        # The same preconditions `run_auction_tournament` and every simulation entry point
        # apply (spec §3.1): enough players to fill the room, and a budget that can afford
        # min_bid for every slot. Skipping them let a short pool start a live draft that could
        # never reach `is_complete` — it dead-ends mid-room with no error.
        validate_auction_inputs(self.pool, self.league)

    # ------------------------------------------------------------------ market data

    @property
    def _market_pair(self) -> tuple[pd.DataFrame, pd.Series | None]:
        if self._market is None:
            self._market = build_market_dollars(
                self.pool,
                self.league,
                market=self.market,
                unranked_discount=self.unranked_discount,
            )
        return self._market

    @property
    def baseline_dollars(self) -> pd.DataFrame:
        """Our SOS valuation of the pool (`AuctionValuesSchema`, one row per player)."""
        return self._market_pair[0]

    @cached_property
    def engine_dollars(self) -> pd.DataFrame:
        """The gsis-indexed frame the bid models read (`auction_dollars` + `bot_dollars`)."""
        baseline, bot = self._market_pair
        return build_engine_dollars(baseline, bot)

    @cached_property
    def player_names(self) -> dict[str, str]:
        return build_player_names(self.id_map, self.pool)

    def name(self, gsis_id: str) -> str:
        return self.player_names.get(str(gsis_id), "—")

    @cached_property
    def _position_by_id(self) -> dict[str, Position]:
        return {
            str(g): Position(str(p))
            for g, p in zip(self.pool["gsis_id"], self.pool["position"], strict=True)
        }

    @cached_property
    def _pool_row_by_id(self) -> dict[str, pd.Series]:
        """Pool row per gsis_id, built once — a bid model is called per player per render, and a
        boolean scan of the pool for each would be O(pool x board)."""
        return {str(g): row for g, row in self.pool.set_index("gsis_id", drop=False).iterrows()}

    @cached_property
    def _position_bounds(self) -> tuple[dict[Position, int], dict[Position, int]]:
        return bot_position_bounds(self.league.roster_slots)

    def team_label(self, seat: int) -> str:
        """Display label for a seat: the caller's name if given, else 'Team N' ('You' for mine)."""
        if 1 <= seat <= len(self.team_names) and self.team_names[seat - 1].strip():
            return self.team_names[seat - 1].strip()
        return "You" if seat == self.my_seat else f"Team {seat}"

    # ------------------------------------------------------------------ auction state

    @property
    def seats(self) -> range:
        return range(1, self.league.n_teams + 1)

    @property
    def state_key(self) -> tuple[tuple[str, int, int], ...]:
        """The purchase list — the only mutable state, and the key every memo hangs on.

        Public because callers outside this class cache off it too: `scripts/auction_board.py`
        keys its projected-eval cache on this. The seat matters as much as the player — an
        auction's rosters are (player, seat) pairs, unlike a snake draft where pick order
        alone determines the seat — and so does the price, since budgets follow from it.
        (Length alone would collide: an undo followed by a different award is the same length.)
        """
        return tuple((str(p.gsis_id), p.seat, p.price) for p in self.purchases)

    @property
    def _fingerprint(self) -> tuple[tuple[str, int, int], ...]:
        return self.state_key

    def _seat_state(self) -> _SeatState:
        """Per-seat rosters/spend/eligibility, recomputed only when a purchase changes.

        Every render prices ~40 players, and each pricing needs each seat's budget and eligible
        positions; without this each of those would rescan the whole purchase list.
        """
        key = self._fingerprint
        if self._seat_cache is None or self._seat_cache[0] != key:
            rosters: dict[int, list[str]] = {s: [] for s in self.seats}
            spent: dict[int, int] = dict.fromkeys(self.seats, 0)
            for p in self.purchases:
                rosters[p.seat].append(str(p.gsis_id))
                spent[p.seat] += p.price
            minimums, maximums = self._position_bounds
            eligible = {
                s: bot_eligible(
                    Counter(self._position_by_id[g] for g in rosters[s]),
                    self.league.roster_size - len(rosters[s]),
                    minimums=minimums,
                    maximums=maximums,
                )
                for s in self.seats
            }
            drafted = frozenset(g for ids in rosters.values() for g in ids)
            self._seat_cache = (key, _SeatState(rosters, spent, drafted, eligible))
        return self._seat_cache[1]

    @property
    def drafted_ids(self) -> frozenset[str]:
        return self._seat_state().drafted

    @property
    def is_complete(self) -> bool:
        return len(self.purchases) >= self.league.total_pool_size

    def roster_ids(self, seat: int) -> list[str]:
        return list(self._seat_state().rosters[seat])  # copy: the memo's list is not the caller's

    def spent(self, seat: int) -> int:
        return self._seat_state().spent[seat]

    def budget(self, seat: int) -> int:
        return self.league.budget - self.spent(seat)

    def open_slots(self, seat: int) -> int:
        return self.league.roster_size - len(self._seat_state().rosters[seat])

    def feasible_max(self, seat: int) -> int:
        """The engine's solvency ceiling: what a seat can bid and still $1 out its roster.

        Zero for a seat with no open slot — it cannot bid at all. Without that guard the
        formula reads `budget + min_bid` there, i.e. more than the seat has, which the board's
        status bar printed as "Your max bid" while the budget ledger (which guards the same
        expression) showed 0 for the same seat on the same screen.
        """
        open_ = self.open_slots(seat)
        if open_ <= 0:
            return 0
        return self.budget(seat) - self.league.min_bid * (open_ - 1)

    def positions(self, seat: int) -> Counter[Position]:
        return Counter(self._position_by_id[g] for g in self._seat_state().rosters[seat])

    def eligible_positions(self, seat: int) -> frozenset[Position]:
        """Positions this seat may still buy, under the engine's roster-discipline rule."""
        return self._seat_state().eligible[seat]

    @property
    def nominating_seat(self) -> int:
        """Whose turn it is to nominate, by the engine's rotation (seat 1 opens), skipping full
        rosters. Advisory only — a real room may rotate differently; the board just displays it."""
        seat = len(self.purchases) % self.league.n_teams + 1
        for _ in self.seats:
            if self.open_slots(seat) > 0:
                return seat
            seat = seat % self.league.n_teams + 1
        return seat

    @property
    def is_my_nomination(self) -> bool:
        return not self.is_complete and self.nominating_seat == self.my_seat

    def available_pool(self) -> pd.DataFrame:
        drafted = self.drafted_ids
        return self.pool[~self.pool["gsis_id"].isin(drafted)].reset_index(drop=True)

    # ------------------------------------------------------------------ recording

    def record_purchase(self, gsis_id: str, seat: int, price: int) -> None:
        """Record that `seat` bought `gsis_id` for `price`. Raises on any illegal award.

        Validated against the same invariants the engine maintains: the player is in the pool and
        undrafted, the seat has an open slot, and the price is between `min_bid` and that seat's
        solvency ceiling (so no seat can be left unable to fill its roster).
        """
        gid = validate_gsis_id(str(gsis_id))
        if gid not in self._position_by_id:
            raise ValueError(f"{gid} is not in the draft pool")
        if gid in self.drafted_ids:
            raise ValueError(f"{self.name(gid)} is already drafted")
        if seat not in self.seats:
            raise ValueError(f"seat must be in 1..{self.league.n_teams}; got {seat}")
        if self.open_slots(seat) <= 0:
            raise ValueError(f"{self.team_label(seat)} has a full roster")
        if price < self.league.min_bid:
            raise ValueError(f"price must be at least ${self.league.min_bid}; got ${price}")
        ceiling = self.feasible_max(seat)
        if price > ceiling:
            raise ValueError(
                f"${price} leaves {self.team_label(seat)} unable to fill its roster "
                f"(max ${ceiling})"
            )
        self.purchases.append(Purchase(gsis_id=gid, seat=seat, price=int(price)))

    def undo(self) -> Purchase | None:
        return self.purchases.pop() if self.purchases else None

    # ------------------------------------------------------------------ bid advice

    def _view(self) -> AuctionView:
        mine = self.roster_ids(self.my_seat)
        return AuctionView(
            my_budget=self.budget(self.my_seat),
            my_open_slots=self.open_slots(self.my_seat),
            my_positions=Counter(self._position_by_id[g].value for g in mine),
            my_roster=self.pool[self.pool["gsis_id"].isin(mine)],
            drafted=self.drafted_ids,
            budgets_by_seat=tuple(self.budget(s) for s in self.seats),
            baseline_dollars=self.engine_dollars,
        )

    def room_ceiling(self, position: Position) -> int:
        """The most any OPPONENT who can still roster `position` is able to bid.

        Your `max_bid` above this number wins the lot outright; below it, expect a fight. Zero
        when no opponent can take the position (the lot is yours at `min_bid` if you want it).
        """
        return max(
            (
                self.feasible_max(s)
                for s in self.seats
                if s != self.my_seat
                and self.open_slots(s) > 0
                and position in self.eligible_positions(s)
            ),
            default=0,
        )

    def advise(self, gsis_id: str, *, view: AuctionView | None = None) -> BidAdvice:
        """What to bid on one player, right now."""
        gid = str(gsis_id)
        if gid not in self._position_by_id:
            raise ValueError(f"{gid} is not in the draft pool")
        if gid in self.drafted_ids:
            raise ValueError(f"{self.name(gid)} is already drafted")
        view = self._view() if view is None else view
        pos = self._position_by_id[gid]
        bd = self.engine_dollars
        fair = int(bd.loc[gid, "auction_dollars"]) if gid in bd.index else 0
        market = int(bd.loc[gid, "bot_dollars"]) if gid in bd.index else 0
        # On a forced lot the engine drops the positional gate for every seat (`elig =
        # all_positions if forced`), so the hero is ungated too — otherwise the board tells you
        # to PASS on a lot the room is about to sell you.
        eligible = self.is_forced_lot or pos in self.eligible_positions(self.my_seat)
        ceiling = self.feasible_max(self.my_seat)
        desired = 0
        clamped = 0
        if eligible and self.open_slots(self.my_seat) > 0:
            player = self._pool_row_by_id[gid]
            desired = int(self.strategy.max_bid(view, player, self.pool, self.league))
            # The same clamp the engine applies to a hero's desire (simulation §3.2).
            clamped = max(self.league.min_bid, min(desired, ceiling))
        return BidAdvice(
            gsis_id=gid,
            full_name=self.name(gid),
            position=pos.value,
            max_bid=clamped,
            desired=desired,
            fair_value=fair,
            market_value=market,
            feasible_max=ceiling,
            room_ceiling=self.room_ceiling(pos),
            eligible=eligible,
        )

    def bid_board(
        self, position: Position | None = None, query: str = "", top: int = 40
    ) -> pd.DataFrame:
        """The priced board of available players — one row per player, `max_bid` first.

        Filters to `position` (None = all) and to names containing `query` (case-insensitive),
        keeps the `top` most valuable survivors, then prices each one through the bid model.
        The `top` cut is by our own `auction_dollars` (cheap and monotone-ish in the bid) BEFORE
        the per-player model call, which for `marginal` costs a lineup solve each; the returned
        rows are then sorted by `max_bid` descending.
        """
        avail = self.available_pool()
        if position is not None:
            avail = avail[avail["position"] == position.value]
        if "full_name" in avail.columns:
            avail = avail.drop(columns=["full_name"])
        named = attach_names(avail, self.player_names)
        if query:
            named = named[named["full_name"].str.contains(query, case=False, na=False, regex=False)]
        if named.empty:
            return pd.DataFrame(
                columns=[
                    "full_name",
                    "position",
                    "max_bid",
                    "value",
                    "market",
                    "edge",
                    "room_max",
                    "vorp",
                    "adp",
                    "gsis_id",
                ]
            )
        bd = self.engine_dollars
        values = [
            int(bd.loc[g, "auction_dollars"]) if g in bd.index else 0 for g in named["gsis_id"]
        ]
        named = named.assign(_value=values)
        named = named.sort_values(["_value", "vorp"], ascending=False).head(top)
        view = self._view()  # built once: the view is per-board-state, not per-player
        rows = []
        for gid in named["gsis_id"]:
            a = self.advise(str(gid), view=view)
            rows.append(
                {
                    "full_name": a.full_name,
                    "position": a.position,
                    "max_bid": a.max_bid,
                    "value": a.fair_value,
                    "market": a.market_value,
                    # >0: we outbid what the room is anchored on — a lot we can win.
                    "edge": a.max_bid - a.market_value,
                    "room_max": a.room_ceiling,
                    "gsis_id": a.gsis_id,
                }
            )
        # `rows` carries plain `str` ids (object dtype); `named` carries them as
        # `pd.StringDtype("pyarrow")`. Merging on mismatched key dtypes can raise, or silently
        # produce all-null right-hand columns — a bid board with blank vorp/adp on every row
        # that no length-or-order assertion would catch. Align the key before joining.
        left = pd.DataFrame(rows)
        left["gsis_id"] = left["gsis_id"].astype(_PYARROW_STR)
        out = left.merge(
            named[[c for c in ("gsis_id", "vorp", "consensus_adp") if c in named.columns]],
            on="gsis_id",
            how="left",
        )
        out = out.rename(columns={"consensus_adp": "adp"})
        if "adp" not in out.columns:
            out["adp"] = pd.array([pd.NA] * len(out), dtype=pd.Float64Dtype())
        cols = [
            "full_name",
            "position",
            "max_bid",
            "value",
            "market",
            "edge",
            "room_max",
            "vorp",
            "adp",
            "gsis_id",
        ]
        return out[cols].sort_values(["max_bid", "value"], ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ nomination

    def _undrafted_in_value_order(self) -> list[str]:
        """Every undrafted pool player, most valuable first — the engine's `nominate_order`."""
        order = self.engine_dollars.sort_values("auction_dollars", ascending=False).index
        drafted = self.drafted_ids
        return [str(g) for g in order if str(g) not in drafted and str(g) in self._position_by_id]

    @property
    def is_forced_lot(self) -> bool:
        """True when no open seat can roster any remaining position, but lots remain.

        The engine's pool-thin branch (`simulation._simulate_to_state`): rather than dead-end,
        it forces the top undrafted player and lets every seat bid **ungated**. Late in a real
        draft this is reachable — every open seat holding only unmet-TE deficits with no TEs
        left — and the engine emits a UserWarning for it.

        False once the auction is complete: the pool still holds undrafted players then, but
        there is nothing left to nominate and no seat that could bid.
        """
        if self.is_complete:
            return False
        return not self._gated_candidates() and bool(self._undrafted_in_value_order())

    def _gated_candidates(self) -> list[str]:
        """Undrafted players SOME open seat can still roster — the engine's union gate."""
        union: set[Position] = set()
        for seat in self.seats:
            if self.open_slots(seat) > 0:
                union |= self.eligible_positions(seat)
        return [g for g in self._undrafted_in_value_order() if self._position_by_id[g] in union]

    def _nomination_candidates(self) -> list[str]:
        """Undrafted players SOME open seat can still roster, most valuable first — the engine's
        own candidate rule (`simulation._simulate_to_state`), including its union-of-eligible gate.

        When that gate empties while lots remain, fall back the way the engine does: a single
        forced nominee, ungated. Without it the board reported "Nothing left to nominate" and a
        $0 max bid on every player while the room was still selling lots you had to bid on.
        """
        gated = self._gated_candidates()
        if gated:
            return gated
        return self._undrafted_in_value_order()[:1] if self.is_forced_lot else []

    def suggested_nomination(self) -> str | None:
        """Who to put up, under `nomination_mode`. None once nothing is nominable."""
        candidates = self._nomination_candidates()
        if not candidates:
            return None
        if self.nomination_mode == "value":
            return candidates[0]  # engine default: highest value the room can roster
        minimums, _ = self._position_bounds
        bd = self.engine_dollars
        ctx = NominationContext(
            hero_positions=self.positions(self.my_seat),
            value_by_id={str(g): float(v) for g, v in bd["bot_dollars"].items()},
            position_by_id=self._position_by_id,
            position_minimums=minimums,
        )
        return _POISON_NOMINATORS[self.nomination_mode](candidates, ctx)

    def nomination_board(self, top: int = 12) -> pd.DataFrame:
        """The nomination shortlist: the most valuable nominable players and what they cost you.

        `i_want` marks a player your own roster can still take; `room_max` is the richest rival
        ceiling for the position. A high-`market`, low-`i_want` player is the classic drain
        nomination — the room's money goes on someone you were never buying.
        """
        candidates = self._nomination_candidates()[:top]
        if not candidates:
            return pd.DataFrame(
                columns=[
                    "full_name",
                    "position",
                    "value",
                    "market",
                    "max_bid",
                    "room_max",
                    "i_want",
                    "gsis_id",
                ]
            )
        view = self._view()
        rows = []
        for gid in candidates:
            a = self.advise(gid, view=view)
            rows.append(
                {
                    "full_name": a.full_name,
                    "position": a.position,
                    "value": a.fair_value,
                    "market": a.market_value,
                    "max_bid": a.max_bid,
                    "room_max": a.room_ceiling,
                    "i_want": a.i_want,
                    "gsis_id": a.gsis_id,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ my team

    def my_roster_view(self) -> AuctionRosterView:
        prices = {str(p.gsis_id): p.price for p in self.purchases if p.seat == self.my_seat}
        mine = self.roster_ids(self.my_seat)
        placements, open_, _ = allocate_roster_slots(
            ((g, self._position_by_id[g]) for g in mine), self.league.roster_slots
        )
        rows = [
            {
                "slot": slot.value,
                "gsis_id": gid,
                "full_name": self.name(gid),
                "position": pos.value,
                "price": prices[gid],
            }
            for gid, pos, slot in placements
        ]
        # `allocate_roster_slots` omits a player with no open slot, but `record_purchase`
        # deliberately does not enforce positional eligibility — it records what the room
        # actually did. So a third QB in a thin-bench league is on the roster and counted in
        # `spent`, yet has no placement. Show him rather than losing him silently.
        placed = {gid for gid, _, _ in placements}
        rows += [
            {
                "slot": _NO_SLOT,
                "gsis_id": gid,
                "full_name": self.name(gid),
                "position": self._position_by_id[gid].value,
                "price": prices[gid],
            }
            for gid in mine
            if gid not in placed
        ]
        filled = pd.DataFrame(rows, columns=["slot", "gsis_id", "full_name", "position", "price"])
        open_slots: dict[RosterSlot, int] = {
            s: c for s, c in open_.items() if c > 0 and s != RosterSlot.BENCH
        }
        return AuctionRosterView(
            filled=filled, open_slots=open_slots, spent=self.spent(self.my_seat)
        )

    def budget_table(self) -> pd.DataFrame:
        """Every seat's money and roster state — the board's ledger."""
        rows = []
        for seat in self.seats:
            open_ = self.open_slots(seat)
            budget = self.budget(seat)
            rows.append(
                {
                    "seat": seat,
                    "team": self.team_label(seat),
                    "you": "★" if seat == self.my_seat else "",
                    "budget": budget,
                    "players": self.league.roster_size - open_,
                    "open": open_,
                    "max_bid": self.feasible_max(seat) if open_ > 0 else 0,
                    "per_slot": round(budget / open_, 1) if open_ > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def purchase_log(self) -> pd.DataFrame:
        """Every lot awarded so far, in order, with the value we had on the player."""
        bd = self.engine_dollars
        rows = []
        for i, p in enumerate(self.purchases):
            gid = str(p.gsis_id)
            value = int(bd.loc[gid, "auction_dollars"]) if gid in bd.index else 0
            rows.append(
                {
                    "#": i + 1,
                    "player": self.name(gid),
                    "team": self.team_label(p.seat),
                    "price": p.price,
                    "value": value,
                    "over": p.price - value,
                    "mine": "★" if p.seat == self.my_seat else "",
                }
            )
        return pd.DataFrame(rows, columns=["#", "player", "team", "price", "value", "over", "mine"])

    def inflation(self) -> float:
        """Live market inflation (>1 = the board left will clear above our values)."""
        return surplus_inflation(self._view(), self.league)

    # ------------------------------------------------------------------ end of draft

    def project_league_outcomes(
        self,
        *,
        n_sims: int = 2000,
        seed: int = 0,
        availability: PlayerAvailability | None = None,
        params: VarianceParams | None = None,
        data_root: Path = Path("data"),
    ) -> dict[int, SeatProjection]:
        """Projected-vs-projected per-seat season metrics for the COMPLETED auction.

        Same league sim the snake board runs (injury + performance draws, optimal lineup both
        sides, fixed top-6/top-2-bye bracket). Raises if any roster is still unfilled.
        """
        if not self.is_complete:
            raise ValueError("auction must be complete to project league outcomes")
        pool = attach_is_rookie(self.pool, season=self.season, data_root=data_root)
        if availability is None:
            from projections.draft.assistant.availability_loader import load_store_availability

            availability = load_store_availability(pool, season=self.season, data_root=data_root)
        if params is None:
            params = VarianceParams.load()
        rosters = {seat: self.roster_ids(seat) for seat in self.seats}
        return project_draft(
            rosters=rosters,
            pool=pool,
            availability=availability,
            params=params,
            league_config=self.league,
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
        )

    # ------------------------------------------------------------------ persistence

    def to_state_dict(self) -> dict[str, object]:
        return {
            "league_config": str(self.league_config_path),
            "my_seat": self.my_seat,
            "purchases": [
                {"gsis_id": str(p.gsis_id), "seat": p.seat, "price": p.price}
                for p in self.purchases
            ],
            "strategy_name": self.strategy_name,
            "market": self.market,
            "unranked_discount": self.unranked_discount,
            "nomination_mode": self.nomination_mode,
            "season": self.season,
            "team_names": list(self.team_names),
            "vorp_table": str(self.vorp_path),
            "id_map": str(self.id_map_path),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, *, id_map: pd.DataFrame, pool: pd.DataFrame) -> LiveAuctionSession:
        """Rebuild a session from a saved state dict (bid model via the shared registry)."""
        data = json.loads(path.read_text())
        league = LeagueConfig.model_validate_json(Path(data["league_config"]).read_text())
        name = str(data["strategy_name"])
        if name not in BOARD_BID_MODELS:
            raise ValueError(f"unknown bid model {name!r}; expected one of {BOARD_BID_MODEL_NAMES}")
        # Validate rather than coerce. `"espn" if x == "espn" else "model"` turned a corrupted
        # or hand-edited `"market"` into a model-priced room with no error, silently repricing
        # every market_value, every `edge`, `i_want`, and both nomination poison rules on our own
        # values instead of ESPN's. `build_market_dollars` raises on the same input; match it.
        saved_market = str(data.get("market", "espn"))
        if saved_market not in ("espn", "model"):
            raise ValueError(f"saved market must be 'espn' or 'model'; got {saved_market!r}")
        market: Literal["espn", "model"] = "espn" if saved_market == "espn" else "model"
        sess = cls(
            league=league,
            my_seat=int(data["my_seat"]),
            id_map=id_map,
            pool=pool,
            strategy=BOARD_BID_MODELS[name],
            strategy_name=name,
            market=market,
            unranked_discount=data.get("unranked_discount"),
            nomination_mode=str(data.get("nomination_mode", "value")),
            season=int(data.get("season", 2026)),
            team_names=tuple(str(t) for t in data.get("team_names", [])),
            league_config_path=Path(data["league_config"]),
            vorp_path=Path(data.get("vorp_table", ".")),
            id_map_path=Path(data.get("id_map", ".")),
        )
        for rec in data.get("purchases", []):
            sess.record_purchase(str(rec["gsis_id"]), int(rec["seat"]), int(rec["price"]))
        return sess
