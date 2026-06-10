"""Tests for the strategy tournament + sigma tuning."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import NowOrNeverStrategy
from projections.draft.assistant.survival import LogisticSurvival
from projections.draft.assistant.tournament import (
    _paired_diff_ci,
    _validate_pool,
    run_tournament,
    tune_sigma,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config(n_teams: int = 2) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 12) -> pd.DataFrame:
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )


class _BestFpts:
    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[False, True])


class _WorstFpts:
    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[True, True])


def test_validate_pool_rejects_all_null_adp() -> None:
    pool = _pool()
    pool["consensus_adp"] = pd.array([pd.NA] * len(pool), dtype=pd.Float64Dtype())
    with pytest.raises(ValueError, match="consensus_adp"):
        _validate_pool(pool, _config())


def test_validate_pool_rejects_too_small_pool() -> None:
    cfg = _config(n_teams=2)  # needs 6 players
    with pytest.raises(ValueError, match="need >="):
        _validate_pool(_pool(n=4), cfg)


def test_validate_pool_accepts_valid() -> None:
    _validate_pool(_pool(), _config())  # no raise


def test_paired_diff_ci_constant_edge_excludes_zero() -> None:
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0] * 4)
    b = a - 3.0  # A beats B by a constant 3 every paired seed
    ci = _paired_diff_ci(a, b, n_bootstrap=500, seed=0)
    assert ci.point == pytest.approx(3.0)
    assert ci.lo_95 > 0


def test_paired_diff_ci_zero_edge_brackets_zero() -> None:
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0] * 4)
    ci = _paired_diff_ci(a, a.copy(), n_bootstrap=500, seed=0)
    assert ci.lo_95 <= 0 <= ci.hi_95


def test_run_tournament_declares_better_strategy() -> None:
    # Single WR starting slot removes the position-balance confound (both fakes are
    # position-blind, but optimal_lineup_points rewards balance). _BestFpts
    # deterministically takes the global best-fpts player (gsis 00-0000001, a WR,
    # fpts 200) at pick 1, so its lineup scores 200 every seed; _WorstFpts cannot.
    cfg = LeagueConfig(
        name="t",
        n_teams=2,
        roster_slots={RosterSlot.WR: 1, RosterSlot.BENCH: 2},
        ruleset=Ruleset.espn_ppr(),
    )
    result = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()},
        pool=_pool(),
        config=cfg,
        my_slot=1,
        n_seeds=40,
        adp_jitter=3.0,
        base_seed=0,
    )
    assert result.summaries["best"].point > result.summaries["worst"].point
    assert result.winner == "best"
    assert result.diff is not None and result.diff.lo_95 > 0


def test_tune_sigma_returns_argmax() -> None:
    result = tune_sigma(
        [1.0, 8.0],
        pool=_pool(),
        config=_config(),
        my_slot=1,
        n_seeds=20,
        adp_jitter=3.0,
        base_seed=0,
    )
    assert len(result.grid) == 2
    assert result.best_sigma in (1.0, 8.0)
    assert result.best_sigma == max(result.grid, key=lambda r: r[1])[0]


def test_sigma_changes_now_or_never_top_pick() -> None:
    # Spec §4 guard: the survival sigma must actually flow into NowOrNeverStrategy's
    # ranking (catches the "p_available silently NaN / sigma ignored" wiring bug).
    # adp_jitter is a separate knob that lives only in simulate_draft's bot path --
    # NowOrNeverStrategy never sees it, and RawVorpStrategy has no sigma at all.
    #
    # Crafted so sigma flips the top pick deterministically. At pick 1 (my_slot=1,
    # n_teams=2, rounds=3) the hero's next pick is #4, so survival is evaluated at
    # pick 4. RB1 has higher vorp (100) but is safe (adp 100); WR1 has lower vorp
    # (90) but is gone soon (adp 1).
    #   tight sigma (0.5): RB1 surely survives -> opportunity cost ~= its vorp ->
    #     RB1 score ~= 0; WR1 surely gone -> WR1 score ~= 90 -> TOP = WR1.
    #   loose sigma (1000): survival ~flat ~0.5 -> raw vorp dominates ->
    #     RB1 score ~47.6 > WR1 ~45.0 -> TOP = RB1.
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [100.0, 90.0],
            "vorp": [100.0, 90.0],
            "replacement_fpts": [0.0, 0.0],
            "consensus_adp": pd.array([100.0, 1.0], dtype=pd.Float64Dtype()),
        }
    )
    cfg = LeagueConfig(
        name="t",
        n_teams=2,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
    state = DraftState(my_slot=1, n_teams=2, rounds=3, picks=(), my_roster=())
    tight = NowOrNeverStrategy(LogisticSurvival(sigma=0.5))
    loose = NowOrNeverStrategy(LogisticSurvival(sigma=1000.0))
    top_tight = tight.recommend(state, pool, cfg).iloc[0]["gsis_id"]
    top_loose = loose.recommend(state, pool, cfg).iloc[0]["gsis_id"]
    assert top_tight == "00-0000002"  # WR1: opportunity cost dominates under tight sigma
    assert top_loose == "00-0000001"  # RB1: raw vorp dominates under loose sigma
    assert top_tight != top_loose


def test_league_driven_two_roster_shapes_same_pool() -> None:
    # Same pool, two different roster shapes (FLEX vs SUPER_FLEX) -- the harness must
    # run off LeagueConfig alone, with no hardcoded slots. We assert it produces valid
    # finite results for both shapes (not a winner ordering: the position-blind fakes
    # don't guarantee best>worst on a balance-sensitive metric).
    pool = _pool(n=24)
    flex = LeagueConfig(
        name="flex",
        n_teams=2,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
    superflex = LeagueConfig(
        name="sf",
        n_teams=2,
        roster_slots={
            RosterSlot.RB: 1,
            RosterSlot.WR: 1,
            RosterSlot.SUPER_FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    for cfg in (flex, superflex):
        result = run_tournament(
            {"best": _BestFpts(), "worst": _WorstFpts()},
            pool=pool,
            config=cfg,
            my_slot=1,
            n_seeds=10,
            adp_jitter=2.0,
            base_seed=0,
        )
        assert set(result.summaries) == {"best", "worst"}
        assert all(math.isfinite(ci.point) for ci in result.summaries.values())
