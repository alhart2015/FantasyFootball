"""Tests for snake-order pick timing."""

from __future__ import annotations

from projections.draft.assistant.pick_timing import (
    my_next_pick,
    my_upcoming_picks,
    picks_until_next,
    slot_for,
)

N = 12
ROUNDS = 4  # picks per team for these tests


def test_slot_for_snake_wrap() -> None:
    assert slot_for(7, N) == 7  # round 1, straight order
    assert slot_for(12, N) == 12  # end of round 1
    assert slot_for(13, N) == 12  # round 2 reverses → slot 12 picks back-to-back
    assert slot_for(18, N) == 7  # round 2, reversed
    assert slot_for(24, N) == 1
    assert slot_for(25, N) == 1  # round 3 straight again


def test_my_upcoming_includes_current_when_mine() -> None:
    # Slot 7 of 12: my picks are 7, 18, 31, 42.
    assert my_upcoming_picks(7, my_slot=7, n_teams=N, rounds=ROUNDS) == [7, 18, 31, 42]
    # Standing at pick 8 (not mine): current pick excluded.
    assert my_upcoming_picks(8, my_slot=7, n_teams=N, rounds=ROUNDS) == [18, 31, 42]


def test_my_next_pick_is_strictly_after_current() -> None:
    assert my_next_pick(7, my_slot=7, n_teams=N, rounds=ROUNDS) == 18
    assert my_next_pick(18, my_slot=7, n_teams=N, rounds=ROUNDS) == 31
    # On my final pick there is no next pick.
    assert my_next_pick(42, my_slot=7, n_teams=N, rounds=ROUNDS) is None


def test_picks_until_next_counts_opponents() -> None:
    assert picks_until_next(7, my_slot=7, n_teams=N, rounds=ROUNDS) == 10  # picks 8..17
    assert picks_until_next(42, my_slot=7, n_teams=N, rounds=ROUNDS) is None
