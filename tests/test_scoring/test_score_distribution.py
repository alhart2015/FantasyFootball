"""score_distribution: convert per-stat distributions into a fantasy-points distribution."""

from __future__ import annotations

import math

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
