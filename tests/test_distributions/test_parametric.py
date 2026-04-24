"""Parametric Distribution tests — math correctness and sampling determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projections.distributions import ParametricGamma, ParametricNormal


def test_normal_mean_std_quantile() -> None:
    d = ParametricNormal(mean=10.0, std=2.0)
    assert d.mean() == pytest.approx(10.0)
    assert d.std() == pytest.approx(2.0)
    assert d.quantile(0.5) == pytest.approx(10.0)
    # ~68% within 1 std => p84.13 ~ 12.0
    assert d.quantile(0.8413) == pytest.approx(12.0, abs=0.01)


def test_normal_sample_shape_and_determinism() -> None:
    d = ParametricNormal(mean=10.0, std=2.0)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1 = d.sample(1000, rng=rng1)
    s2 = d.sample(1000, rng=rng2)
    assert s1.shape == (1000,)
    assert np.array_equal(s1, s2)


def test_normal_sample_summary() -> None:
    d = ParametricNormal(mean=10.0, std=2.0)
    rng = np.random.default_rng(0)
    s = d.sample(100_000, rng=rng)
    assert math.isclose(float(s.mean()), 10.0, abs_tol=0.05)
    assert math.isclose(float(s.std()), 2.0, abs_tol=0.05)


def test_gamma_positive_support() -> None:
    d = ParametricGamma(shape=4.0, scale=3.0)
    assert d.mean() == pytest.approx(12.0)  # shape * scale
    assert d.std() == pytest.approx(math.sqrt(36))  # sqrt(shape) * scale
    rng = np.random.default_rng(1)
    s = d.sample(10_000, rng=rng)
    assert (s >= 0).all()


def test_gamma_quantile_monotonic() -> None:
    d = ParametricGamma(shape=2.0, scale=5.0)
    q10 = d.quantile(0.1)
    q50 = d.quantile(0.5)
    q90 = d.quantile(0.9)
    assert q10 < q50 < q90


def test_normal_rejects_nonpositive_std() -> None:
    with pytest.raises(ValueError, match="std must be positive"):
        ParametricNormal(mean=0.0, std=0.0)
    with pytest.raises(ValueError, match="std must be positive"):
        ParametricNormal(mean=0.0, std=-1.0)


def test_gamma_rejects_nonpositive_shape() -> None:
    with pytest.raises(ValueError, match="shape must be positive"):
        ParametricGamma(shape=0.0, scale=1.0)
    with pytest.raises(ValueError, match="shape must be positive"):
        ParametricGamma(shape=-1.0, scale=1.0)


def test_gamma_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        ParametricGamma(shape=1.0, scale=0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        ParametricGamma(shape=1.0, scale=-1.0)


@pytest.mark.parametrize("q", [-0.1, 0.0, 1.0, 1.1])
def test_quantile_rejects_out_of_range_q(q: float) -> None:
    n = ParametricNormal(mean=0.0, std=1.0)
    g = ParametricGamma(shape=1.0, scale=1.0)
    with pytest.raises(ValueError, match="q must be in"):
        n.quantile(q)
    with pytest.raises(ValueError, match="q must be in"):
        g.quantile(q)
