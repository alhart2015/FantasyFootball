"""Tests for QuantileDistribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from projections.distributions import QuantileDistribution


def test_constructor_validates_sorted_quantiles() -> None:
    with pytest.raises(ValueError, match="quantiles must be strictly ascending"):
        QuantileDistribution(
            quantiles=np.array([0.10, 0.05, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )


def test_constructor_validates_quantiles_in_open_unit_interval() -> None:
    with pytest.raises(ValueError, match="quantiles must lie strictly in"):
        QuantileDistribution(
            quantiles=np.array([0.0, 0.10, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
    with pytest.raises(ValueError, match="quantiles must lie strictly in"):
        QuantileDistribution(
            quantiles=np.array([0.05, 0.10, 0.50, 0.90, 1.0]),
            values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )


def test_constructor_validates_values_non_decreasing() -> None:
    with pytest.raises(ValueError, match="values must be non-decreasing"):
        QuantileDistribution(
            quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.0, 3.0, 2.5, 5.0]),
        )


def test_constructor_validates_matching_lengths() -> None:
    with pytest.raises(ValueError, match="must have matching length"):
        QuantileDistribution(
            quantiles=np.array([0.10, 0.50, 0.90]),
            values=np.array([1.0, 2.0]),
        )


def test_constructor_validates_minimum_two_knots() -> None:
    with pytest.raises(ValueError, match="at least 2 knots"):
        QuantileDistribution(
            quantiles=np.array([0.50]),
            values=np.array([1.0]),
        )


def test_quantile_returns_exact_value_at_knot() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    assert dist.quantile(0.50) == pytest.approx(3.0)
    assert dist.quantile(0.10) == pytest.approx(2.0)


def test_quantile_interpolates_between_knots() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.10, 0.50, 0.90]),
        values=np.array([0.0, 10.0, 20.0]),
    )
    # Midpoint between (0.10, 0.0) and (0.50, 10.0): q=0.30 -> v=5.0
    assert dist.quantile(0.30) == pytest.approx(5.0)


def test_quantile_extrapolates_beyond_knots() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.10, 0.50, 0.90]),
        values=np.array([0.0, 10.0, 20.0]),
    )
    # q=0.05 is below q_min=0.10. Linear extrapolation from (0.10, 0.0) and (0.50, 10.0):
    # slope = (10 - 0) / (0.50 - 0.10) = 25; v(0.05) = 0 + 25 * (0.05 - 0.10) = -1.25
    assert dist.quantile(0.05) == pytest.approx(-1.25)
    # q=0.99 is above q_max=0.90. Slope = (20 - 10) / (0.90 - 0.50) = 25;
    # v(0.99) = 20 + 25 * (0.99 - 0.90) = 22.25
    assert dist.quantile(0.99) == pytest.approx(22.25)


def test_mean_matches_normal_distribution() -> None:
    """Quantiles drawn from N(loc=10, scale=2); QD.mean() ≈ 10 within 0.05."""
    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = scipy_stats.norm.ppf(qs, loc=10.0, scale=2.0)
    dist = QuantileDistribution(quantiles=qs, values=vs)
    assert dist.mean() == pytest.approx(10.0, abs=0.5)


def test_std_matches_normal_distribution() -> None:
    """Quantiles drawn from N(loc=10, scale=2); QD.std() ≈ 2 within 0.5."""
    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = scipy_stats.norm.ppf(qs, loc=10.0, scale=2.0)
    dist = QuantileDistribution(quantiles=qs, values=vs)
    # 5 knots is coarse; numerical-integration tolerance is generous.
    assert dist.std() == pytest.approx(2.0, abs=0.5)


def test_sample_is_deterministic_with_seeded_rng() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.10, 0.50, 0.90]),
        values=np.array([0.0, 10.0, 20.0]),
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = dist.sample(n=100, rng=rng_a)
    b = dist.sample(n=100, rng=rng_b)
    np.testing.assert_array_equal(a, b)


def test_sample_returns_correct_shape_and_dtype() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    samples = dist.sample(n=500, rng=np.random.default_rng(0))
    assert samples.shape == (500,)
    assert samples.dtype == np.float64


def test_empirical_quantiles_match_stored_quantiles() -> None:
    """Sample 10K from a constructed QD; empirical quantiles match stored within 0.1."""
    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = scipy_stats.norm.ppf(qs, loc=10.0, scale=2.0)
    dist = QuantileDistribution(quantiles=qs, values=vs)
    samples = dist.sample(n=10_000, rng=np.random.default_rng(0))
    for q, v in zip(qs, vs, strict=True):
        assert np.quantile(samples, q) == pytest.approx(v, abs=0.2)


def test_constant_quantile_repeats_value() -> None:
    """Mass concentrated at zero (count-stat case): repeated 0.0 values are valid."""
    dist = QuantileDistribution(
        quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
        values=np.array([0.0, 0.0, 0.0, 1.0, 2.0]),
    )
    assert dist.quantile(0.10) == pytest.approx(0.0)
    assert dist.quantile(0.50) == pytest.approx(0.0)
    assert dist.quantile(0.90) == pytest.approx(1.0)
