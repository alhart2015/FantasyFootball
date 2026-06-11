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
    _bootstrap_mean,
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


def test_validate_pool_empty_reports_size_not_adp() -> None:
    # An empty/mis-filtered pool should report the pool-size problem, not the
    # vacuously-true "no ADP signal" (code-review #6).
    empty = _pool(n=0)
    with pytest.raises(ValueError, match="need >="):
        _validate_pool(empty, _config())


def test_run_tournament_rejects_negative_adp_jitter() -> None:
    # A negative jitter would crash deep in numpy ("scale < 0"); fail loud, named.
    with pytest.raises(ValueError, match="adp_jitter must be >= 0"):
        run_tournament(
            {"best": _BestFpts(), "worst": _WorstFpts()},
            pool=_pool(),
            config=_config(),
            my_slot=1,
            n_seeds=5,
            adp_jitter=-1.0,
            base_seed=0,
        )


def test_run_tournament_allows_zero_adp_jitter() -> None:
    # 0 is a valid (deterministic, zero-noise) field, not an error.
    result = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()},
        pool=_pool(),
        config=_config(),
        my_slot=1,
        n_seeds=5,
        adp_jitter=0.0,
        base_seed=0,
    )
    assert set(result.summaries) == {"best", "worst"}


def test_tune_sigma_rejects_non_positive_sigma() -> None:
    # LogisticSurvival needs sigma > 0; the engine rejects the whole grid up front
    # (protects programmatic callers, not just the CLI).
    with pytest.raises(ValueError, match="must all be > 0"):
        tune_sigma(
            [1.0, 0.0, 2.0],
            pool=_pool(),
            config=_config(),
            my_slot=1,
            n_seeds=5,
            adp_jitter=2.0,
            base_seed=0,
        )


def test_tune_sigma_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="sigma_grid must be non-empty"):
        tune_sigma(
            [],
            pool=_pool(),
            config=_config(),
            my_slot=1,
            n_seeds=5,
            adp_jitter=2.0,
            base_seed=0,
        )


def test_run_tournament_rejects_out_of_range_my_slot() -> None:
    # An out-of-range slot owns no snake pick -> hero drafts nobody, scores 0, and a
    # bogus "winner" would be declared. Must fail loud instead (final-review #1).
    with pytest.raises(ValueError, match="my_slot must be in"):
        run_tournament(
            {"best": _BestFpts(), "worst": _WorstFpts()},
            pool=_pool(),
            config=_config(n_teams=2),
            my_slot=99,
            n_seeds=5,
            adp_jitter=2.0,
            base_seed=0,
        )


def test_tune_sigma_rejects_non_positive_seeds() -> None:
    with pytest.raises(ValueError, match="n_seeds must be >= 1"):
        tune_sigma(
            [1.0, 2.0],
            pool=_pool(),
            config=_config(),
            my_slot=1,
            n_seeds=0,
            adp_jitter=2.0,
            base_seed=0,
        )


def test_bootstrap_mean_paired_constant_edge_excludes_zero() -> None:
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0] * 4)
    b = a - 3.0  # A beats B by a constant 3 every paired seed
    ci = _bootstrap_mean(a - b, n_bootstrap=500, seed=0)
    assert ci.point == pytest.approx(3.0)
    assert ci.lo_95 > 0


def test_bootstrap_mean_paired_zero_edge_brackets_zero() -> None:
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0] * 4)
    ci = _bootstrap_mean(a - a.copy(), n_bootstrap=500, seed=0)
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


def test_default_valuer_matches_optimal_lineup_points() -> None:
    # The default-valuer tournament must equal the pre-change behavior: StartersValuer
    # is optimal_lineup_points, so run_tournament's numbers are unchanged.
    from projections.draft.assistant.valuer import StartersValuer

    kwargs = dict(
        pool=_pool(), config=_config(), my_slot=1, n_seeds=20, adp_jitter=2.0, base_seed=0
    )
    default_run = run_tournament({"best": _BestFpts(), "worst": _WorstFpts()}, **kwargs)
    explicit = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()}, valuer=StartersValuer(), **kwargs
    )
    assert default_run.summaries["best"].point == explicit.summaries["best"].point
    assert default_run.summaries["worst"].point == explicit.summaries["worst"].point


def test_season_valuer_runs_in_tournament() -> None:
    # A tournament scored by the season valuer produces a valid result (smoke + shape).
    from projections.draft.assistant.availability import PlayerAvailability
    from projections.draft.assistant.valuer import SeasonValuer

    pool = _pool()
    avail = PlayerAvailability(p={str(g): 0.8 for g in pool["gsis_id"]}, bye={})
    valuer = SeasonValuer(availability=avail, n_sims=50, base_seed=0)
    result = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()},
        pool=pool,
        config=_config(),
        my_slot=1,
        n_seeds=10,
        adp_jitter=2.0,
        base_seed=0,
        valuer=valuer,
    )
    assert set(result.summaries) == {"best", "worst"}
    assert all(ci.point > 0 for ci in result.summaries.values())


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
