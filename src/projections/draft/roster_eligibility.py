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
from typing import TypeVar

from projections.schemas import Position, RosterSlot

_Player = TypeVar("_Player")

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


def allocate_roster_slots(
    players: Iterable[tuple[_Player, Position]],
    roster_slots: Mapping[RosterSlot, int],
) -> tuple[list[tuple[_Player, Position, RosterSlot]], Counter[RosterSlot], frozenset[Position]]:
    """Greedily place each player into a roster slot; return placements, open slots, benchable.

    Fill priority per player: own position slot → FLEX → SUPER_FLEX → BENCH. A player
    with no open slot (roster overflow) is omitted from placements (no negatives). Each
    player is a `(key, position)` pair; the opaque `key` (e.g. a gsis_id) is carried
    through to the placement so callers can label rows. Also returns the bench-eligible
    set it computes, so wrappers need not recompute it. The single source of truth for
    the restrictive-first fill rule, shared by `_open_slots_after` and the live board.
    """
    open_: Counter[RosterSlot] = Counter(
        {slot: count for slot, count in roster_slots.items() if slot != RosterSlot.IR and count > 0}
    )
    benchable = bench_eligible_positions(roster_slots)
    placements: list[tuple[_Player, Position, RosterSlot]] = []
    for key, pos in players:
        candidates = (
            (RosterSlot(pos.value), True),
            (RosterSlot.FLEX, pos in FLEX_ELIGIBLE),
            (RosterSlot.SUPER_FLEX, pos in SUPER_FLEX_ELIGIBLE),
            (RosterSlot.BENCH, pos in benchable),
        )
        for slot, eligible in candidates:
            if eligible and open_.get(slot, 0) > 0:
                open_[slot] -= 1
                placements.append((key, pos, slot))
                break
    return placements, open_, benchable


def _open_slots_after(
    roster_slots: Mapping[RosterSlot, int], my_roster: Iterable[Position]
) -> tuple[Counter[RosterSlot], frozenset[Position]]:
    """Per-team open slots remaining after greedily placing my drafted players.

    Thin wrapper over `allocate_roster_slots` that discards placements and forwards the
    bench-eligible set it already computed (so the caller need not recompute it).
    """
    _, open_, benchable = allocate_roster_slots(((pos, pos) for pos in my_roster), roster_slots)
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


def bot_eligible(
    counts: Mapping[Position, int],
    picks_left: int,
    *,
    minimums: Mapping[Position, int],
    maximums: Mapping[Position, int],
) -> frozenset[Position]:
    """Positions a roster-disciplined bot may take now (the snake draft_field rule, generalized).

    Reserve the final picks for unmet minimums; otherwise allow any position still under its cap.
    The eligible set is drawn strictly from the `minimums`/`maximums` keysets, so a position
    present in `counts` but absent from the bound maps (e.g. K/DST when the bounds omit them)
    is never returned.
    """
    deficit = {p: max(0, minimums.get(p, 0) - counts.get(p, 0)) for p in minimums}
    if picks_left <= sum(deficit.values()):
        return frozenset(p for p, d in deficit.items() if d > 0)
    return frozenset(p for p in maximums if counts.get(p, 0) < maximums[p])
