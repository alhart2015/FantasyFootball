"""Tests for projections.draft.roster_eligibility."""

from __future__ import annotations

from projections.draft.roster_eligibility import (
    FLEX_ELIGIBLE,
    SUPER_FLEX_ELIGIBLE,
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
