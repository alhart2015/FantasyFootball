"""Tests for the projected-vs-projected league simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import gauntlet_schedule, team_weekly_points
from projections.draft.assistant.performance_variance import VarianceParams
from projections.schemas import _PYARROW_STR, RosterSlot

_SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 3,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 9,
}


def _roster(gsis: list[str], pos: list[str], mean: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": mean,
            "is_rookie": [False] * len(gsis),
        }
    )


def test_team_weekly_points_shape_and_higher_means_score_more() -> None:
    params = VarianceParams.load()
    weeks = list(range(1, 14))
    ids = [f"00-000{i:04d}" for i in range(1, 9)]
    pos = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB"]
    avail = PlayerAvailability(p={g: 1.0 for g in ids}, bye={})
    strong = team_weekly_points(
        _roster(ids, pos, [300.0] * 8),
        avail,
        params,
        n_sims=400,
        weeks=weeks,
        roster_slots=_SLOTS,
        rng=np.random.default_rng(0),
    )
    weak = team_weekly_points(
        _roster(ids, pos, [120.0] * 8),
        avail,
        params,
        n_sims=400,
        weeks=weeks,
        roster_slots=_SLOTS,
        rng=np.random.default_rng(0),
    )
    assert strong.shape == (400, 13)
    assert strong.mean() > weak.mean()  # higher projections -> more lineup points


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
