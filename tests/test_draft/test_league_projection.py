"""Tests for the projected-vs-projected league simulation."""

from __future__ import annotations

import pytest

from projections.draft.assistant.league_projection import gauntlet_schedule


@pytest.mark.parametrize("n_teams", [10, 12, 16])
def test_gauntlet_schedule_is_a_valid_round_robin(n_teams: int) -> None:
    sched = gauntlet_schedule(n_teams, n_weeks=13)
    assert len(sched) == 13
    for week in sched:
        seats = [s for pair in week for s in pair]
        assert sorted(seats) == list(range(1, n_teams + 1))  # everyone plays exactly once
    # slot 1 plays slot 2 in wk1, slot 3 in wk2 (the rotating gauntlet)
    assert (1, 2) in [tuple(sorted(p)) for p in sched[0]]
    assert (1, 3) in [tuple(sorted(p)) for p in sched[1]]


def test_gauntlet_schedule_rejects_odd() -> None:
    with pytest.raises(ValueError, match="even"):
        gauntlet_schedule(11, n_weeks=13)
