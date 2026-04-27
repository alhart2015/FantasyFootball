"""Tests for ParametricStudentT."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import ParametricStudentT


def test_mean_matches_loc() -> None:
    """For df > 1, Student-t mean is loc."""
    dist = ParametricStudentT(loc=10.0, scale=2.0, df=4.0)
    assert dist.mean() == pytest.approx(10.0)


def test_std_matches_formula() -> None:
    """For df > 2, var = scale^2 * df / (df - 2)."""
    dist = ParametricStudentT(loc=0.0, scale=2.0, df=4.0)
    expected_std = 2.0 * np.sqrt(4.0 / 2.0)
    assert dist.std() == pytest.approx(expected_std)


def test_quantile_symmetric_around_loc() -> None:
    """Student-t is symmetric: P(X <= loc) = 0.5."""
    dist = ParametricStudentT(loc=5.0, scale=1.0, df=3.0)
    assert dist.quantile(0.5) == pytest.approx(5.0)


def test_quantile_rejects_out_of_range() -> None:
    dist = ParametricStudentT(loc=0.0, scale=1.0, df=4.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(0.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(1.0)


def test_sample_shape_and_dtype() -> None:
    dist = ParametricStudentT(loc=0.0, scale=1.0, df=4.0)
    rng = np.random.default_rng(42)
    samples = dist.sample(n=500, rng=rng)
    assert samples.shape == (500,)
    assert samples.dtype == np.float64


def test_sample_mean_approximates_loc() -> None:
    dist = ParametricStudentT(loc=5.0, scale=2.0, df=10.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(n=10_000, rng=rng)
    assert samples.mean() == pytest.approx(5.0, abs=0.1)


def test_constructor_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        ParametricStudentT(loc=0.0, scale=0.0, df=4.0)


def test_constructor_rejects_low_df() -> None:
    """Phase 1+ implementations need df > 2 for finite variance — std() is
    undefined otherwise. Reject at construction time."""
    with pytest.raises(ValueError, match="df must be greater than 2"):
        ParametricStudentT(loc=0.0, scale=1.0, df=2.0)
    with pytest.raises(ValueError, match="df must be greater than 2"):
        ParametricStudentT(loc=0.0, scale=1.0, df=1.0)
