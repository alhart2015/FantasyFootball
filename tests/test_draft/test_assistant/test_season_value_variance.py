from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.season_value import (
    _lineup_points_sampled,
    _roster_fill_meta,
    _vectorized_lineup_points,
)
from projections.schemas import RosterSlot


def _roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": [f"00-000000{i}" for i in range(5)],
            "position": ["QB", "RB", "RB", "WR", "WR"],
            "season_mean_fpts": [300.0, 250.0, 200.0, 220.0, 180.0],
        }
    )


def test_sampled_fill_matches_fixed_when_points_constant() -> None:
    roster = _roster()
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    meta = _roster_fill_meta(roster, slots)
    avail = np.ones((6, len(roster)), bool)
    fixed = _vectorized_lineup_points(avail, meta)
    pos = roster["position"].to_numpy().astype(str)
    pts = np.broadcast_to(roster["season_mean_fpts"].to_numpy()[None, :], (6, len(roster)))
    sampled = _lineup_points_sampled(pts, avail, pos, slots)
    assert np.allclose(sampled, fixed)


def test_sampled_fill_starts_the_weekly_best() -> None:
    # Two RBs, one RB slot: each row should start whichever RB scored more THAT row.
    roster = pd.DataFrame(
        {"gsis_id": ["a", "b"], "position": ["RB", "RB"], "season_mean_fpts": [10.0, 10.0]}
    )
    slots = {RosterSlot.RB: 1}
    pos = roster["position"].to_numpy().astype(str)
    pts = np.array([[5.0, 20.0], [30.0, 1.0]])  # row0 -> b(20), row1 -> a(30)
    avail = np.ones((2, 2), bool)
    out = _lineup_points_sampled(pts, avail, pos, slots)
    assert out[0] == 20.0 and out[1] == 30.0
