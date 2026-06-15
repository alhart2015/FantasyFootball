"""Slot↔position eligibility for draft tooling — one source of truth.

Promoted out of `_pool.py`'s private symbols so both the pool selector and the
draft assistant share the same FLEX/SUPER_FLEX/bench rules. Adds a greedy
allocation that, given my league's roster slots and the positions I've already
drafted, reports which positions I can still roster and whether each still has
an open *starting* (non-bench) slot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from projections.schemas import Position, RosterSlot

# Position-specific starting slots (a slot whose label is also a Position).
POSITION_SLOTS: tuple[RosterSlot, ...] = (
    RosterSlot.QB,
    RosterSlot.RB,
    RosterSlot.WR,
    RosterSlot.TE,
    RosterSlot.K,
    RosterSlot.DST,
)

FLEX_ELIGIBLE: frozenset[Position] = frozenset({Position.RB, Position.WR, Position.TE})
SUPER_FLEX_ELIGIBLE: frozenset[Position] = frozenset(
    {Position.QB, Position.RB, Position.WR, Position.TE}
)

# Flex-type slots in ascending eligibility breadth (FLEX subset of SUPER_FLEX). Order is
# load-bearing: narrowest eligibility first keeps restrictive-first greedy lineup fills optimal.
# Single source consumed by roster_score, backtest.lineup, and the season-value sampler.
FLEX_SLOTS: tuple[tuple[RosterSlot, frozenset[Position]], ...] = (
    (RosterSlot.FLEX, FLEX_ELIGIBLE),
    (RosterSlot.SUPER_FLEX, SUPER_FLEX_ELIGIBLE),
)


def bench_eligible_positions(roster_slots: Mapping[RosterSlot, int]) -> frozenset[Position]:
    """Positions the league actually rosters (so the shared bench can hold them).

    Excludes positions with no position slot — e.g. a K-less league never benches
    a kicker. Source of truth for bench eligibility; `_pool.py`'s pool selector
    delegates here.
    """
    return frozenset(
        Position(slot.value) for slot in POSITION_SLOTS if roster_slots.get(slot, 0) > 0
    )


def _open_slots_after(
    roster_slots: Mapping[RosterSlot, int], my_roster: Iterable[Position]
) -> tuple[Counter[RosterSlot], frozenset[Position]]:
    """Per-team open slots remaining after greedily placing my drafted players.

    Fill priority per player: own position slot → FLEX → SUPER_FLEX → BENCH.
    A player with no open slot (roster overflow) is left unplaced (no negatives).
    Also returns the bench-eligible set it computes, so the caller need not
    recompute it.
    """
    open_: Counter[RosterSlot] = Counter(
        {slot: count for slot, count in roster_slots.items() if slot != RosterSlot.IR and count > 0}
    )
    benchable = bench_eligible_positions(roster_slots)
    for pos in my_roster:
        own = RosterSlot(pos.value)
        candidates = (
            (own, True),
            (RosterSlot.FLEX, pos in FLEX_ELIGIBLE),
            (RosterSlot.SUPER_FLEX, pos in SUPER_FLEX_ELIGIBLE),
            (RosterSlot.BENCH, pos in benchable),
        )
        for slot, eligible in candidates:
            if eligible and open_.get(slot, 0) > 0:
                open_[slot] -= 1
                break
    return open_, benchable


def _has_open_starting(pos: Position, open_: Counter[RosterSlot]) -> bool:
    """Is there an open *non-bench* slot this position could occupy?"""
    if open_.get(RosterSlot(pos.value), 0) > 0:
        return True
    if pos in FLEX_ELIGIBLE and open_.get(RosterSlot.FLEX, 0) > 0:
        return True
    if pos in SUPER_FLEX_ELIGIBLE and open_.get(RosterSlot.SUPER_FLEX, 0) > 0:
        return True
    return False


def eligible_positions(
    roster_slots: Mapping[RosterSlot, int], my_roster: Iterable[Position]
) -> dict[Position, bool]:
    """Map every still-rosterable position to its starting-need tier.

    Returns `{position: fills_starting_slot}`:
      - a position is a key iff I can still roster a player there (an open
        position/FLEX/SUPER_FLEX slot, or open BENCH capacity if benchable);
      - the value is True iff it still has an open *starting* (non-bench) slot.
    Positions I can no longer roster are absent (the caller drops them).
    """
    my_roster = list(my_roster)
    open_, benchable = _open_slots_after(roster_slots, my_roster)
    bench_open = open_.get(RosterSlot.BENCH, 0) > 0
    result: dict[Position, bool] = {}
    for pos in Position:
        starting = _has_open_starting(pos, open_)
        rosterable = starting or (pos in benchable and bench_open)
        if rosterable:
            result[pos] = starting
    return result
