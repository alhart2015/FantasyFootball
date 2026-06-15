from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
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


def _avail_all(gsis: list[str]) -> PlayerAvailability:
    return PlayerAvailability(p={g: 1.0 for g in gsis}, bye={})


def _vp_zero() -> VarianceParams:
    return VarianceParams(
        {"default": {"a": 0.0, "b": 1e-7}},
        {"default|veteran": 1e-7, "default|rookie": 1e-7},
    )


def test_var_reduces_to_deterministic_at_zero_variance() -> None:
    from projections.draft.assistant.season_value import expected_season_points_var

    roster = pd.DataFrame(
        {
            "gsis_id": ["a", "b", "c"],
            "position": ["QB", "RB", "WR"],
            "season_mean_fpts": [300.0, 250.0, 200.0],
            "is_rookie": [False, False, False],
        }
    )
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.WR: 1}
    v = expected_season_points_var(
        roster,
        slots,
        _avail_all(["a", "b", "c"]),
        _vp_zero(),
        n_sims=200,
        rng=np.random.default_rng(0),
        weeks=range(1, 15),
    )
    assert abs(v - (750.0 / 17 * 14)) < 5.0


def test_var_bye_forces_starter_out_for_that_week() -> None:
    # A starter's bye must drop them from the lineup that week. With zero variance, a QB/RB/WR
    # roster (one slot each, no backup) and the RB on bye in week 5 of the 14-week season: 13
    # full weeks + 1 week with the RB slot empty. Exercises the bye path the mask refactor
    # rewrote (the other var tests use bye={}, so this is the only bye coverage).
    from projections.draft.assistant.season_value import expected_season_points_var

    roster = pd.DataFrame(
        {
            "gsis_id": ["a", "b", "c"],
            "position": ["QB", "RB", "WR"],
            "season_mean_fpts": [300.0, 250.0, 200.0],
            "is_rookie": [False, False, False],
        }
    )
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = PlayerAvailability(p={"a": 1.0, "b": 1.0, "c": 1.0}, bye={"b": 5})
    v = expected_season_points_var(
        roster,
        slots,
        avail,
        _vp_zero(),
        n_sims=200,
        rng=np.random.default_rng(0),
        weeks=range(1, 15),
    )
    expected = (13 * 750.0 + 500.0) / 17  # RB (250 season pts) absent for exactly one week
    assert abs(v - expected) < 5.0


def test_marginal_crn_ranks_better_candidate_first() -> None:
    from projections.draft.assistant.season_value import marginal_season_values_var

    base = pd.DataFrame(
        {"gsis_id": ["a"], "position": ["QB"], "season_mean_fpts": [300.0], "is_rookie": [False]}
    )
    cands = pd.DataFrame(
        {
            "gsis_id": ["x", "y"],
            "position": ["RB", "RB"],
            "season_mean_fpts": [250.0, 120.0],
            "is_rookie": [False, False],
        }
    )
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 1}
    out = marginal_season_values_var(
        base,
        cands,
        slots,
        _avail_all(["a", "x", "y"]),
        _vp_zero(),
        n_sims=200,
        rng=np.random.default_rng(0),
        weeks=range(1, 15),
    )
    assert out["x"] > out["y"] > 0


def test_season_value_var_strategy_differs_from_deterministic() -> None:
    # season_value_var (risk_aware=True) routes to the variance MC and differs from the
    # deterministic season_value on the same seed; both produce valid recommendations.
    from projections.draft.assistant.state import DraftState
    from projections.draft.assistant.strategy import SeasonValueStrategy
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import _PYARROW_STR, RecommendationSchema, Ruleset

    cfg = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.FLEX: 1, RosterSlot.BENCH: 5},
        ruleset=Ruleset.espn_ppr(),
    )
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000010", "00-0000011", "00-0000020", "00-0000021"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["RB", "RB", "WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 240.0, 252.0, 230.0],
            "vorp": [50.0, 40.0, 52.0, 30.0],
            "replacement_fpts": [200.0, 200.0, 200.0, 200.0],
            "consensus_adp": pd.array([5.0, 6.0, 7.0, 8.0], dtype=pd.Float64Dtype()),
            "is_rookie": [False, False, True, False],
        }
    )
    state = DraftState(my_slot=7, n_teams=12, rounds=9, picks=(), my_roster=())
    avail = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    det = SeasonValueStrategy(avail, n_sims=400, base_seed=0).recommend(state, pool, cfg)
    var = SeasonValueStrategy(avail, n_sims=400, base_seed=0, risk_aware=True).recommend(
        state, pool, cfg
    )
    RecommendationSchema.validate(var)
    det_s = det.set_index("gsis_id")["score"].sort_index().to_numpy()
    var_s = var.set_index("gsis_id")["score"].sort_index().to_numpy()
    assert not np.allclose(det_s, var_s)  # the variance path produces a different valuation
