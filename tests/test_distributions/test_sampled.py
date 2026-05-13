"""Tests for src/projections/distributions/sampled.py.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md §3.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions.sampled import FrozenSampledDistribution


def test_sample_returns_underlying_array_when_n_equals_len() -> None:
    """The n == len branch is the load-bearing property for cross-stat coherence.

    score_distribution calls .sample(n_samples=10_000) on each per-stat distribution;
    when n matches len(samples), FrozenSampledDistribution returns the underlying
    array directly (exact identity, not a view), preserving any cross-stat
    correlation baked into the array.
    """
    rng = np.random.default_rng(seed=42)
    samples = rng.normal(loc=10.0, scale=2.0, size=100)
    dist = FrozenSampledDistribution(samples=samples)
    out = dist.sample(n=100)
    assert out is dist.samples
    assert np.array_equal(out, samples)


def test_sample_resamples_when_n_differs_from_len() -> None:
    """Fallback to rng.choice when n != len(samples)."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = FrozenSampledDistribution(samples=samples)
    rng = np.random.default_rng(seed=0)
    out = dist.sample(n=20, rng=rng)
    assert out.shape == (20,)
    assert set(out.tolist()).issubset(set(samples.tolist()))


def test_sample_resamples_use_provided_rng_for_determinism() -> None:
    samples = np.arange(100, dtype=np.float64)
    dist = FrozenSampledDistribution(samples=samples)
    out1 = dist.sample(n=50, rng=np.random.default_rng(seed=7))
    out2 = dist.sample(n=50, rng=np.random.default_rng(seed=7))
    assert np.array_equal(out1, out2)


def test_mean_std_quantile_cdf_match_numpy_reference() -> None:
    rng = np.random.default_rng(seed=1)
    samples = rng.normal(loc=5.0, scale=1.5, size=10_000)
    dist = FrozenSampledDistribution(samples=samples)
    assert dist.mean() == pytest.approx(float(samples.mean()))
    assert dist.std() == pytest.approx(float(samples.std()))
    assert dist.quantile(0.5) == pytest.approx(float(np.quantile(samples, 0.5)))
    assert dist.cdf(5.0) == pytest.approx(float((samples <= 5.0).mean()))


def test_quantile_rejects_endpoints() -> None:
    dist = FrozenSampledDistribution(samples=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="must be in"):
        dist.quantile(0.0)
    with pytest.raises(ValueError, match="must be in"):
        dist.quantile(1.0)


def test_satisfies_distribution_protocol() -> None:
    """Runtime structural isinstance check against the Distribution Protocol.

    Distribution is @runtime_checkable; isinstance checks attribute presence.
    """
    from projections.distributions.base import Distribution

    dist = FrozenSampledDistribution(samples=np.array([1.0, 2.0, 3.0]))
    assert isinstance(dist, Distribution)


def test_samples_array_is_read_only() -> None:
    """The frozen dataclass marks the underlying array read-only so in-place
    mutation through the n == len zero-copy branch raises rather than silently
    corrupting cross-stat correlation."""
    samples = np.array([1.0, 2.0, 3.0])
    dist = FrozenSampledDistribution(samples=samples)
    out = dist.sample(n=3)
    with pytest.raises(ValueError):
        out[0] = 99.0


def test_construction_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FrozenSampledDistribution(samples=np.array([], dtype=np.float64))


def test_construction_rejects_multidim_samples() -> None:
    with pytest.raises(ValueError, match="1-D"):
        FrozenSampledDistribution(samples=np.array([[1.0, 2.0], [3.0, 4.0]]))
