"""score_distribution: convert per-stat distributions into a fantasy-points distribution."""

from __future__ import annotations

import math
import time

import pytest

from projections.distributions import Distribution, ParametricGamma, ParametricNormal
from projections.schemas import Ruleset, Stat
from projections.scoring import score_distribution


def test_passing_yards_only_distribution() -> None:
    # If only passing_yards is uncertain (mean=300, std=50), the points
    # distribution should have mean ~ 12 and std ~ 50/25 = 2.
    stat_dists = {Stat.PASSING_YARDS: ParametricNormal(mean=300.0, std=50.0)}
    out = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=20_000, seed=42)
    assert math.isclose(out.mean(), 12.0, abs_tol=0.05)
    assert math.isclose(out.std(), 2.0, abs_tol=0.05)


def test_combined_receiving_distribution() -> None:
    # Receptions ~ Gamma(shape=8, scale=1) => mean 8, var 8
    # Rec yards ~ Normal(mean=100, std=20)
    # Rec TDs constant 0
    # In PPR: pts = 1*rec + rec_yds/10 => mean = 8 + 10 = 18, std ~ sqrt(8 + 4) = sqrt(12)
    stat_dists: dict[Stat, Distribution] = {
        Stat.RECEPTIONS: ParametricGamma(shape=8.0, scale=1.0),
        Stat.RECEIVING_YARDS: ParametricNormal(mean=100.0, std=20.0),
    }
    out = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=50_000, seed=0)
    assert math.isclose(out.mean(), 18.0, abs_tol=0.1)
    assert math.isclose(out.std(), math.sqrt(12), abs_tol=0.1)


def test_score_distribution_is_deterministic_with_seed() -> None:
    stat_dists = {Stat.PASSING_YARDS: ParametricNormal(mean=300.0, std=50.0)}
    a = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=1000, seed=7)
    b = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=1000, seed=7)
    assert a.mean() == pytest.approx(b.mean())
    assert a.std() == pytest.approx(b.std())


def test_score_distribution_mean_matches_linear_combination_of_stat_means() -> None:
    """For a linear scoring rule like ESPN_PPR, the mean of the points
    distribution should equal sum(coef[stat] * mean(samples_per_stat[stat]))
    -- regardless of whether the implementation is iterative or vectorized.
    Verifies the scoring math without coupling to internal layout."""
    rng_seed = 42
    stat_dists: dict[Stat, Distribution] = {
        Stat.RECEPTIONS: ParametricNormal(mean=5.0, std=1.5),
        Stat.RECEIVING_YARDS: ParametricNormal(mean=70.0, std=20.0),
        Stat.RECEIVING_TDS: ParametricNormal(mean=0.5, std=0.4),
    }
    ruleset = Ruleset.espn_ppr()
    out = score_distribution(stat_dists, ruleset, n_samples=20_000, seed=rng_seed)

    # Expected mean: rec_pts * 5 + (1/recv_yds_per_pt) * 70 + recv_td * 0.5
    # (Note: receptions/receiving_tds are integer-rounded -- nudges the mean
    # slightly relative to the unrounded multiply, so use a wider tolerance.)
    expected_unrounded = (
        ruleset.reception_pts * 5.0
        + 70.0 / ruleset.receiving_yds_per_pt
        + ruleset.receiving_td_pts * 0.5
    )
    assert abs(out.mean() - expected_unrounded) < 1.0  # within 1 PPR point


def test_score_distribution_runs_fast_enough_for_backtest() -> None:
    """Backtest scale: 10k samples per row, 16 cells x ~1700 rows = ~272M
    underlying ops. The vectorized implementation should complete a single
    10k-sample call in under 50ms on a modern dev box. The pre-vectorization
    Python loop took several seconds at this scale.

    Test budget is 200ms (4x the target) to absorb CI noise without losing
    the 100x+ regression-detection signal."""
    stat_dists: dict[Stat, Distribution] = {
        Stat.RECEPTIONS: ParametricNormal(mean=5.0, std=1.5),
        Stat.RECEIVING_YARDS: ParametricNormal(mean=70.0, std=20.0),
        Stat.RECEIVING_TDS: ParametricNormal(mean=0.5, std=0.4),
        Stat.RUSHING_YARDS: ParametricNormal(mean=2.0, std=2.5),
        Stat.RUSHING_TDS: ParametricNormal(mean=0.05, std=0.2),
        Stat.FUMBLES_LOST: ParametricNormal(mean=0.05, std=0.15),
    }
    ruleset = Ruleset.espn_ppr()

    start = time.perf_counter()
    score_distribution(stat_dists, ruleset, n_samples=10_000, seed=42)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, f"score_distribution took {elapsed:.3f}s, target <0.2s"
