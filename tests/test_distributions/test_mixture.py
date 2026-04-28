"""Plan 6 Phase 1 — MixtureDistribution math: mean / std / cdf / quantile / sample."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import (
    MixtureDistribution,
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    QuantileDistribution,
)


def _make_quantile_dist() -> QuantileDistribution:
    return QuantileDistribution(
        quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64),
        values=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
    )


def test_constructor_rejects_boundary_weights() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=10.0, std=1.0)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=0.0)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=1.0)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=-0.1)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=1.1)


def test_mean_is_linear_combination() -> None:
    a = ParametricNormal(mean=5.0, std=2.0)
    b = ParametricNormal(mean=15.0, std=3.0)
    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
        expected = w * 5.0 + (1.0 - w) * 15.0
        assert mix.mean() == pytest.approx(expected, abs=1e-12)


def test_variance_uses_mixture_formula() -> None:
    """variance = w*var_A + (1-w)*var_B + w*(1-w)*(mean_A - mean_B)^2."""
    a = ParametricNormal(mean=5.0, std=2.0)  # var = 4
    b = ParametricNormal(mean=15.0, std=3.0)  # var = 9
    for w in [0.2, 0.5, 0.8]:
        mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
        expected_var = w * 4.0 + (1 - w) * 9.0 + w * (1 - w) * (5.0 - 15.0) ** 2
        assert mix.std() ** 2 == pytest.approx(expected_var, abs=1e-9)


def test_cdf_is_linear_pool() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricGamma(shape=2.0, scale=2.0)
    for w in [0.25, 0.5, 0.75]:
        mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
        for x in [-2.0, 0.0, 1.0, 5.0, 10.0]:
            expected = w * a.cdf(x) + (1 - w) * b.cdf(x)
            assert mix.cdf(x) == pytest.approx(expected, abs=1e-12)


def test_quantile_round_trips_through_cdf() -> None:
    """For each q, cdf(quantile(q)) ~ q."""
    a = ParametricNormal(mean=2.0, std=1.0)
    b = ParametricNormal(mean=8.0, std=2.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.4)
    for q in [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]:
        x = mix.quantile(q)
        assert mix.cdf(x) == pytest.approx(q, abs=1e-6)


def test_quantile_invalid_q() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=5.0, std=1.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    with pytest.raises(ValueError, match="q"):
        mix.quantile(0.0)
    with pytest.raises(ValueError, match="q"):
        mix.quantile(1.0)
    with pytest.raises(ValueError, match="q"):
        mix.quantile(-0.1)


def test_quantile_raises_when_q_outside_joint_support() -> None:
    """When both children's cdfs clamp at a common minimum, q below that minimum
    cannot be represented and quantile() raises rather than silently clipping."""
    # Build a mixture of two QuantileDistributions both clamping at q=0.05 below.
    a = QuantileDistribution(
        quantiles=np.array([0.05, 0.5, 0.95], dtype=np.float64),
        values=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
    b = QuantileDistribution(
        quantiles=np.array([0.05, 0.5, 0.95], dtype=np.float64),
        values=np.array([15.0, 25.0, 35.0], dtype=np.float64),
    )
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    # q=0.02 is below both children's lowest stored quantile (0.05); cdf clamps
    # at 0.05 for x below the lowest stored value, so cdf(lo) >= 0.05 > 0.02.
    with pytest.raises(ValueError, match="below joint support"):
        mix.quantile(0.02)
    # Symmetric: q=0.98 is above both children's highest stored quantile (0.95).
    with pytest.raises(ValueError, match="above joint support"):
        mix.quantile(0.98)


def test_sample_converges_to_analytic_moments() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=5.0, std=2.0)
    w = 0.4
    mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
    rng = np.random.default_rng(seed=42)
    draws = mix.sample(n=20000, rng=rng)
    assert draws.shape == (20000,)
    expected_mean = w * 0.0 + (1 - w) * 5.0
    expected_var = w * 1.0 + (1 - w) * 4.0 + w * (1 - w) * (0.0 - 5.0) ** 2
    expected_std = np.sqrt(expected_var)
    assert draws.mean() == pytest.approx(expected_mean, abs=0.1)
    assert draws.std() == pytest.approx(expected_std, abs=0.1)


def test_sample_with_count_component() -> None:
    """Mixture with NB component samples ints from one side, floats from the other."""
    a = ParametricNegativeBinomial(mean=1.5, dispersion=4.0)
    b = ParametricNormal(mean=20.0, std=5.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    rng = np.random.default_rng(seed=0)
    draws = mix.sample(n=5000, rng=rng)
    assert draws.shape == (5000,)
    # Roughly half should be small integers (NB samples), half larger floats (Normal).
    small = (draws < 5.0).sum()
    large = (draws > 10.0).sum()
    assert 0.4 * 5000 < small < 0.6 * 5000
    assert 0.4 * 5000 < large < 0.6 * 5000


def test_sample_with_quantile_component() -> None:
    a = _make_quantile_dist()
    b = ParametricNormal(mean=100.0, std=5.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.3)
    rng = np.random.default_rng(seed=1)
    draws = mix.sample(n=10000, rng=rng)
    assert draws.shape == (10000,)
    a_mean = a.mean()
    expected_mean = 0.3 * a_mean + 0.7 * 100.0
    assert draws.mean() == pytest.approx(expected_mean, abs=2.0)


def test_quantile_extreme_q_within_safe_bracket() -> None:
    """Quantile inversion at q in (1e-3, 1 - 1e-3) should converge cleanly."""
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=10.0, std=2.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    for q in [0.001, 0.01, 0.99, 0.999]:
        x = mix.quantile(q)
        assert np.isfinite(x)
        assert mix.cdf(x) == pytest.approx(q, abs=1e-5)
