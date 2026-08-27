"""Slot↔position eligibility for draft tooling — one source of truth.

Promoted out of `_pool.py`'s private symbols so both the pool selector and the
draft assistant share the same FLEX/SUPER_FLEX/bench rules. Adds a greedy
allocation that, given my league's roster slots and the positions I've already
drafted, reports which positions I can still roster and whether each still has
an open *starting* (non-bench) slot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import ceil
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


def choose_starters(
    players: Sequence[_Player],
    roster_slots: Mapping[RosterSlot, int],
    *,
    value: Callable[[_Player], float | None],
    position: Callable[[_Player], str],
) -> list[int]:
    """Indices of the players who start, best lineup first. Restrictive slots, then flex.

    **The selection, extracted.** Three implementations of this greedy already existed --
    `assistant.roster_score.optimal_lineup_points` over a DataFrame of season points,
    `backtest.lineup.weekly_lineup_points` over mappings of weekly points, and a vectorised
    sampler in `assistant.season_value` -- and each returned only a total. The waiver
    recommender needs to know *which* players started, because the player it should suggest
    dropping is by definition one who did not. Rather than write a fourth copy, the choice
    lives here, beside `POSITION_SLOTS` and `FLEX_SLOTS`, which is the taxonomy it walks.

    `value` returning `None` means unstartable — a bye week, or a player with no projection.
    Distinct from returning `0.0`, which is a real projection of nothing and can still fill a
    slot no one else is eligible for.

    Fill order is load-bearing: single-position slots first (most restrictive), then flex tiers
    narrowest-first. Greedy is optimal under that order for these slot structures, which is why
    every copy of this has used it.

    Ties break on the earlier index, so a caller that sorts its input gets a deterministic
    lineup. Returned in fill order rather than sorted, so the caller can see which slot each
    player filled by position in the list.
    """
    startable: list[tuple[int, float, Position]] = []
    for index, player in enumerate(players):
        points = value(player)
        if points is None:
            continue
        try:
            pos = Position(position(player))
        except ValueError:
            # A position the league does not roster (an IDP slot, an ESPN oddity). Unstartable
            # rather than an error: one unrecognised player must not stop a lineup being set.
            continue
        startable.append((index, float(points), pos))

    by_pos: dict[Position, list[tuple[int, float]]] = {pos: [] for pos in Position}
    for index, points, pos in startable:
        by_pos[pos].append((index, points))
    for pos in by_pos:
        # Descending by points, then ascending by index so the tie-break is the caller's order.
        by_pos[pos].sort(key=lambda pair: (-pair[1], pair[0]))

    cursor: dict[Position, int] = {pos: 0 for pos in Position}
    chosen: list[int] = []

    for slot in POSITION_SLOTS:
        pos = Position(slot.value)
        for _ in range(roster_slots.get(slot, 0)):
            if cursor[pos] < len(by_pos[pos]):
                chosen.append(by_pos[pos][cursor[pos]][0])
                cursor[pos] += 1

    for slot, eligible in FLEX_SLOTS:
        for _ in range(roster_slots.get(slot, 0)):
            best_pos: Position | None = None
            best_value = float("-inf")
            for pos in sorted(eligible, key=lambda p: p.value):
                if cursor[pos] < len(by_pos[pos]) and by_pos[pos][cursor[pos]][1] > best_value:
                    best_pos, best_value = pos, by_pos[pos][cursor[pos]][1]
            if best_pos is not None:
                chosen.append(by_pos[best_pos][cursor[best_pos]][0])
                cursor[best_pos] += 1

    return chosen


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


def bot_position_bounds(
    roster_slots: Mapping[RosterSlot, int],
) -> tuple[dict[Position, int], dict[Position, int]]:
    """League-driven per-position (minimums, maximums) for a roster-disciplined bot.

    min = the DEDICATED starting slots only. A flex slot is not a requirement for any one position,
    so FLEX/SUPER_FLEX add nothing here.

    max = min + flex capacity + bench share. A flex-eligible position could in principle fill every
    FLEX slot, so its cap carries the full FLEX count (and every super-flex-eligible position the
    full SUPER_FLEX count); the bench is distributed proportionally to the minimums, rounded up so
    the caps always permit a full roster (Σmax >= roster_size).

    This previously anchored FLEX to RB and SUPER_FLEX to QB unconditionally, which made RB's
    minimum 4 (not 2) in a 2RB+2FLEX league AND handed RB a larger share of the bench, capping WR at
    4. That cap bound on 98-99% of rosters while RB's bound on 3-5%, so every seat was structurally
    forbidden a 5th WR -- while the valuation layer (`vorp._starter_demand`, which allocates FLEX by
    actually filling the slots) had WR absorbing MORE of the flex than RB, 3.14 vs 2.82 starters per
    team on the 2026 table. The two layers disagreed and this one won at draft time. See issue #143
    and docs/superpowers/specs/2026-08-13-auction-flex-position-bounds-design.md.
    """
    minimums: dict[Position, int] = {}
    for slot in POSITION_SLOTS:
        count = roster_slots.get(slot, 0)
        if count > 0:
            minimums[Position(slot.value)] = count
    flex = roster_slots.get(RosterSlot.FLEX, 0)
    superflex = roster_slots.get(RosterSlot.SUPER_FLEX, 0)
    # A flex-eligible position with no dedicated slot still needs an entry: `bot_eligible` draws its
    # eligible set strictly from these keysets, so omitting it would ban the position outright.
    for pos in FLEX_ELIGIBLE if flex else ():
        minimums.setdefault(pos, 0)
    for pos in SUPER_FLEX_ELIGIBLE if superflex else ():
        minimums.setdefault(pos, 0)

    sum_min = sum(minimums.values())
    bench = roster_slots.get(RosterSlot.BENCH, 0)
    maximums = {
        pos: m
        + (flex if pos in FLEX_ELIGIBLE else 0)
        + (superflex if pos in SUPER_FLEX_ELIGIBLE else 0)
        + (ceil(bench * m / sum_min) if sum_min > 0 else 0)
        for pos, m in minimums.items()
    }
    return minimums, maximums


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
