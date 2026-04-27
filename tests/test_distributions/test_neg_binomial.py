"""Tests for ParametricNegativeBinomial."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import ParametricNegativeBinomial


def test_mean_matches_constructor_arg() -> None:
    dist = ParametricNegativeBinomial(mean=2.0, dispersion=4.0)
    assert dist.mean() == pytest.approx(2.0)


def test_std_overdispersed_vs_poisson() -> None:
    """NB std exceeds Poisson std (sqrt(mean)) when dispersion is finite."""
    dist = ParametricNegativeBinomial(mean=2.0, dispersion=4.0)
    # var = mean + mean^2 / dispersion = 2 + 4/4 = 3; std = sqrt(3)
    assert dist.std() == pytest.approx(np.sqrt(3.0))


def test_quantile_monotonic() -> None:
    dist = ParametricNegativeBinomial(mean=1.0, dispersion=2.0)
    assert dist.quantile(0.1) <= dist.quantile(0.5) <= dist.quantile(0.9)


def test_quantile_rejects_out_of_range() -> None:
    dist = ParametricNegativeBinomial(mean=1.0, dispersion=2.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(0.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(1.0)


def test_sample_returns_correct_shape_and_dtype() -> None:
    dist = ParametricNegativeBinomial(mean=1.0, dispersion=2.0)
    rng = np.random.default_rng(42)
    samples = dist.sample(n=500, rng=rng)
    assert samples.shape == (500,)
    assert samples.dtype == np.float64
    # NB samples are non-negative integers (cast to float64).
    assert (samples >= 0).all()


def test_sample_mean_approximates_param_mean() -> None:
    """Law of large numbers — sample mean is close to the parameterized mean."""
    dist = ParametricNegativeBinomial(mean=0.3, dispersion=2.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(n=10_000, rng=rng)
    assert samples.mean() == pytest.approx(0.3, abs=0.05)


def test_constructor_rejects_non_positive_mean() -> None:
    with pytest.raises(ValueError, match="mean must be positive"):
        ParametricNegativeBinomial(mean=0.0, dispersion=2.0)
    with pytest.raises(ValueError, match="mean must be positive"):
        ParametricNegativeBinomial(mean=-0.5, dispersion=2.0)


def test_constructor_rejects_non_positive_dispersion() -> None:
    with pytest.raises(ValueError, match="dispersion must be positive"):
        ParametricNegativeBinomial(mean=1.0, dispersion=0.0)
