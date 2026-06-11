"""Tests for the RosterValuer seam."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.valuer import RosterValuer, SeasonValuer, StartersValuer
from projections.schemas import _PYARROW_STR, RosterSlot


def _roster(players: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array([p[0] for p in players], dtype=_PYARROW_STR),
            "position": pd.array([p[1] for p in players], dtype=_PYARROW_STR),
            "season_mean_fpts": [p[2] for p in players],
        }
    )


def test_both_satisfy_protocol() -> None:
    avail = PlayerAvailability(p={}, bye={})
    assert isinstance(StartersValuer(), RosterValuer)
    assert isinstance(SeasonValuer(availability=avail, n_sims=10, base_seed=0), RosterValuer)


def test_starters_valuer_equals_optimal_lineup_points() -> None:
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    assert StartersValuer().value(roster, slots) == optimal_lineup_points(roster, slots)


def test_season_valuer_is_deterministic_per_roster() -> None:
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "RB", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    avail = PlayerAvailability(p={"00-0000001": 0.7, "00-0000002": 0.7}, bye={})
    v = SeasonValuer(availability=avail, n_sims=100, base_seed=0)
    assert v.value(roster, slots) == v.value(roster, slots)  # same roster -> same value


def test_season_valuer_differs_from_starters_under_risk() -> None:
    # The two metrics genuinely differ: with sub-1.0 availability the risk-aware value
    # is strictly below the no-risk starters value (each starter plays only ~p of weeks).
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 180.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = PlayerAvailability(p={"00-0000001": 0.7, "00-0000002": 0.7}, bye={})
    season = SeasonValuer(availability=avail, n_sims=2000, base_seed=0).value(roster, slots)
    starters = StartersValuer().value(roster, slots)
    assert season < starters  # ~ (200+180)*0.7 ≈ 266 < 380
