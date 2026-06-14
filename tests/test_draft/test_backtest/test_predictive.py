from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.backtest.league import Calendar
from projections.draft.backtest.predictive import predictive_outcomes, sample_actual_lookup
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot


def _vp_zero() -> VarianceParams:
    return VarianceParams(
        {"default": {"a": 0.0, "b": 1e-7}},
        {"default|veteran": 1e-7, "default|rookie": 1e-7},
    )


def test_sample_actual_lookup_shape_and_zerovar() -> None:
    pool = pd.DataFrame(
        {
            "gsis_id": ["a", "b"],
            "position": ["QB", "RB"],
            "season_mean_fpts": [340.0, 170.0],
            "is_rookie": [False, False],
        }
    )
    lut = sample_actual_lookup(pool, _vp_zero(), weeks=range(1, 15), rng=np.random.default_rng(0))
    assert abs(lut[("a", 1)] - 340.0 / 17) < 0.5  # zero-var -> ~projected per game
    assert ("b", 14) in lut
    assert len(lut) == 2 * 14


def _cfg6() -> LeagueConfig:
    return LeagueConfig(
        name="t6",
        n_teams=6,
        budget=200,
        min_bid=1,
        roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 1},
        ruleset="espn_half",  # type: ignore[arg-type]
    )


def _pool6() -> pd.DataFrame:
    rows = []
    for i in range(40):
        pos = "QB" if i % 2 == 0 else "RB"
        rows.append(
            {
                "gsis_id": f"00-{i + 1:07d}",
                "position": pos,
                "season_mean_fpts": 220.0 - i,
                "vorp": 120.0 - i,
                "replacement_fpts": 80.0,
                "consensus_adp": float(i + 1),
                "is_rookie": False,
            }
        )
    return pd.DataFrame(rows)


def test_predictive_outcomes_returns_per_strategy_arrays() -> None:
    pool = _pool6()
    cfg = _cfg6()
    cal = Calendar(regular_weeks=(1, 2, 3, 4, 5), playoff_weeks=(6, 7, 8), playoff_size=6)
    weeks = range(1, 9)
    proj = {
        (str(g), wk): float(m)
        for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=True)
        for wk in weeks
    }
    labels = {1: "season_value", **{s: "bot" for s in range(2, 7)}}
    out = predictive_outcomes(
        pool,
        cfg,
        proj,
        _vp_zero(),
        seat_strategies={s: None for s in range(1, 7)},
        strategy_labels=labels,
        calendar=cal,
        jitter=8.0,
        draft_seeds=[0],
        n_predictive_sims=3,
        rng=np.random.default_rng(0),
    )
    assert set(out) == {"season_value", "bot"}
    assert len(out["season_value"]["champ"]) == 3  # 1 seat * 3 sims
    assert set(np.unique(out["season_value"]["champ"])) <= {0.0, 1.0}
    # exactly one champion per sim across the 6 seats
    total_champs = out["season_value"]["champ"].sum() + out["bot"]["champ"].sum()
    assert total_champs == 3
