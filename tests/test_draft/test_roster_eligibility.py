"""Tests for projections.draft.roster_eligibility."""

from __future__ import annotations

from projections.draft.roster_eligibility import (
    FLEX_ELIGIBLE,
    SUPER_FLEX_ELIGIBLE,
    allocate_roster_slots,
    bot_eligible,
    bot_position_bounds,
    eligible_positions,
)
from projections.schemas import Position, RosterSlot

_SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 2,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 3,
}


def test_eligibility_sets() -> None:
    assert FLEX_ELIGIBLE == frozenset({Position.RB, Position.WR, Position.TE})
    assert SUPER_FLEX_ELIGIBLE == frozenset({Position.QB, Position.RB, Position.WR, Position.TE})


def test_empty_roster_all_positions_start() -> None:
    elig = eligible_positions(_SLOTS, [])
    # QB/RB/WR/TE all have an open starting (position) slot.
    assert elig[Position.QB] is True
    assert elig[Position.RB] is True
    assert elig[Position.WR] is True
    assert elig[Position.TE] is True
    # K/DST are not rostered by this config → not eligible at all.
    assert Position.K not in elig
    assert Position.DST not in elig


def test_filled_position_slot_falls_to_flex_then_bench() -> None:
    # Two RB position slots filled by two RBs; RB can still start via FLEX.
    elig = eligible_positions(_SLOTS, [Position.RB, Position.RB])
    assert elig[Position.RB] is True  # FLEX is an open *starting* slot

    # A third RB consumes FLEX; now RB is bench-only (not a starting slot).
    elig3 = eligible_positions(_SLOTS, [Position.RB, Position.RB, Position.RB])
    assert Position.RB in elig3
    assert elig3[Position.RB] is False  # only BENCH remains


def test_position_fully_filled_is_dropped() -> None:
    # QB:1 filled by a QB, no FLEX/SUPER_FLEX, and BENCH:0 → no room for a
    # 2nd QB anywhere, so QB must be absent from eligible_positions.
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.BENCH: 0}
    elig = eligible_positions(slots, [Position.QB])
    # QB position slot filled, no FLEX/SUPER_FLEX/BENCH → QB ineligible.
    assert Position.QB not in elig
    assert elig[Position.RB] is True


def test_allocate_roster_slots_places_restrictive_first_and_carries_key() -> None:
    # Three RBs into RB:2 + FLEX:1: first two take RB slots, third spills to FLEX;
    # the opaque key (here a gsis-like string) is carried through to each placement.
    players = [("a", Position.RB), ("b", Position.RB), ("c", Position.RB)]
    placements, open_, benchable = allocate_roster_slots(players, _SLOTS)
    assert placements == [
        ("a", Position.RB, RosterSlot.RB),
        ("b", Position.RB, RosterSlot.RB),
        ("c", Position.RB, RosterSlot.FLEX),
    ]
    assert open_[RosterSlot.RB] == 0
    assert open_[RosterSlot.FLEX] == 0
    assert open_[RosterSlot.QB] == 1  # untouched
    # benchable is the set of positions the league rosters (QB/RB/WR/TE all have slots).
    assert benchable == frozenset({Position.QB, Position.RB, Position.WR, Position.TE})


def test_allocate_roster_slots_overflow_player_omitted() -> None:
    # One QB slot, no FLEX/SUPER_FLEX/BENCH → a 2nd QB has nowhere to go.
    slots = {RosterSlot.QB: 1, RosterSlot.BENCH: 0}
    placements, open_, _ = allocate_roster_slots([("a", Position.QB), ("b", Position.QB)], slots)
    assert placements == [("a", Position.QB, RosterSlot.QB)]  # 2nd QB omitted
    assert open_[RosterSlot.QB] == 0


_MN = {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
_MX = {Position.QB: 3, Position.RB: 6, Position.WR: 6, Position.TE: 3}


def test_bot_eligible_reserves_final_picks_for_deficits() -> None:
    counts = {Position.QB: 1, Position.RB: 1, Position.WR: 3, Position.TE: 1}  # RB deficit = 2
    # picks_left == Σdeficit (2) -> forced: only positions still below minimum
    assert bot_eligible(counts, 2, minimums=_MN, maximums=_MX) == frozenset({Position.RB})


def test_bot_eligible_cap_branch_above_the_deficit_boundary() -> None:
    counts = {Position.QB: 3, Position.RB: 1, Position.WR: 1, Position.TE: 0}  # QB at max (3)
    # picks_left (10) > Σdeficit -> cap branch: every position still under its max, QB excluded
    assert bot_eligible(counts, 10, minimums=_MN, maximums=_MX) == frozenset(
        {Position.RB, Position.WR, Position.TE}
    )


def test_bot_eligible_ignores_positions_absent_from_bounds() -> None:
    counts = {Position.QB: 1, Position.K: 2}  # K not in the bound maps
    assert Position.K not in bot_eligible(
        counts, 10, minimums={Position.QB: 1}, maximums={Position.QB: 3}
    )


_SKILL = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 3,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 9,
}


def test_bounds_skill_roster_min_and_max() -> None:
    mn, mx = bot_position_bounds(_SKILL)
    # FLEX anchored to RB
    assert mn == {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
    # min + ceil bench share
    assert mx == {Position.QB: 3, Position.RB: 7, Position.WR: 7, Position.TE: 3}


def test_bounds_superflex_anchors_to_qb() -> None:
    slots = dict(_SKILL)
    del slots[RosterSlot.FLEX]
    slots[RosterSlot.SUPER_FLEX] = 1
    mn, _ = bot_position_bounds(slots)
    assert mn[Position.QB] == 2  # 1 strict + 1 super-flex


def test_bounds_sigma_max_at_least_roster_size() -> None:
    _, mx = bot_position_bounds(_SKILL)
    roster_size = sum(c for s, c in _SKILL.items() if s != RosterSlot.IR)
    assert sum(mx.values()) >= roster_size  # caps always permit a full roster
