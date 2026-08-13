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
    # UPDATED for issue #143: these assertions were correct about incorrect behaviour. FLEX used to
    # be anchored to RB (mn RB 3, mx RB 7 / WR 7); a flex slot is not a requirement for any one
    # position, so it now contributes to CAPS only.
    mn, mx = bot_position_bounds(_SKILL)
    assert mn == {Position.QB: 1, Position.RB: 2, Position.WR: 3, Position.TE: 1}  # dedicated only
    # min + flex capacity (RB/WR/TE, not QB) + ceil bench share
    assert mx == {Position.QB: 3, Position.RB: 6, Position.WR: 8, Position.TE: 4}


def test_bounds_superflex_raises_the_cap_of_every_eligible_position() -> None:
    # UPDATED for issue #143: previously asserted SUPER_FLEX anchored to QB's MINIMUM (mn QB == 2).
    # A super-flex slot no more requires a 2nd QB than a FLEX requires a 3rd RB.
    slots = dict(_SKILL)
    del slots[RosterSlot.FLEX]
    slots[RosterSlot.SUPER_FLEX] = 1
    mn, mx = bot_position_bounds(slots)
    assert mn[Position.QB] == 1  # dedicated QB slots only; nothing forces a 2nd
    assert mx[Position.QB] == 4  # 1 dedicated + 1 super-flex + 2 bench share
    assert mx[Position.RB] == 6  # RB is super-flex eligible too, so its cap rises as well


def test_bounds_are_symmetric_across_equally_slotted_flex_positions() -> None:
    """R1 — the regression guard for #143.

    RB and WR carry identical dedicated slots here, so nothing about the league distinguishes them
    and their bounds must match. The old anchor gave RB min 4 / max 7 against WR's 2 / 4, which is
    exactly the asymmetry that hard-capped every seat at 4 WR.
    """
    slots = {
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.FLEX: 2,
        RosterSlot.BENCH: 5,
    }  # Will's league
    mn, mx = bot_position_bounds(slots)
    assert mn[Position.RB] == mn[Position.WR]
    assert mx[Position.RB] == mx[Position.WR]
    assert (mn[Position.RB], mx[Position.RB]) == (2, 6)


def test_bounds_flex_eligible_position_without_a_dedicated_slot_is_still_rosterable() -> None:
    """A TE-less league still has to let a bot take a TE into its FLEX.

    `bot_eligible` draws its eligible set strictly from these keysets, so a missing entry bans the
    position outright rather than merely not requiring it.
    """
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.FLEX: 1}
    mn, mx = bot_position_bounds(slots)
    assert mn[Position.TE] == 0  # nothing forces a TE
    assert mx[Position.TE] == 1  # ...but one can fill the flex
    # 6 picks (the full roster) vs Σdeficit 5 -> the cap branch, where TE must be allowed. At
    # exactly 5 the forced branch correctly excludes it: unmet minimums come first.
    assert Position.TE in bot_eligible({}, 6, minimums=mn, maximums=mx)


def test_bounds_leave_room_to_fill_every_flex_slot() -> None:
    """R2 — after the minimums are met, the remaining picks must be able to fill the FLEX slots.

    Guards the one way lowering the minimums could break a roster: if the non-flex-eligible caps
    (QB) could absorb all the leftover picks, a seat could finish unable to field a lineup.
    """
    slots = {
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.FLEX: 2,
        RosterSlot.BENCH: 5,
    }
    mn, mx = bot_position_bounds(slots)
    roster_size = sum(slots.values())
    flex_needed = slots[RosterSlot.FLEX]
    non_flex_headroom = sum(mx[p] - mn[p] for p in mx if p not in FLEX_ELIGIBLE)
    spare = roster_size - sum(mn.values())
    assert spare - non_flex_headroom >= flex_needed


def test_bounds_sigma_max_at_least_roster_size() -> None:
    _, mx = bot_position_bounds(_SKILL)
    roster_size = sum(c for s, c in _SKILL.items() if s != RosterSlot.IR)
    assert sum(mx.values()) >= roster_size  # caps always permit a full roster
