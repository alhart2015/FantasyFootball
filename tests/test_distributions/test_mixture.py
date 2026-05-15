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


def test_quantile_with_quantile_distribution_component_handles_tail_q() -> None:
    """MixtureDistribution.quantile(q) returns a finite extrapolated value
    for q outside the QuantileDistribution component's persisted knot range.

    Previously, QuantileDistribution.cdf clamped at qs[0] / qs[-1] while
    QuantileDistribution.quantile linearly extrapolated past the knots; this
    inconsistency capped the joint mixture cdf below 1.0 (and above 0.0 on
    the lower tail), causing brentq to fail to bracket q values in the tails.

    With cdf now extrapolating to match quantile (clipped to [0, 1]), tail
    quantiles round-trip finitely. The first production code path that
    surfaces this is the wr_ensemble_decomposed factory's
    MixtureDistribution(QuantileDistribution-from-decomposed-baseline,
    parametric-from-lgb-nb) under EnsembleModel.predict_distribution.
    """
    a = QuantileDistribution(
        quantiles=np.array([0.05, 0.5, 0.95], dtype=np.float64),
        values=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
    b = QuantileDistribution(
        quantiles=np.array([0.05, 0.5, 0.95], dtype=np.float64),
        values=np.array([15.0, 25.0, 35.0], dtype=np.float64),
    )
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    for q in (0.02, 0.10, 0.50, 0.90, 0.98):
        x = mix.quantile(q)
        assert np.isfinite(x), f"mix.quantile({q}) returned non-finite: {x}"
    # Lower-tail q=0.02 should land at a value below the lower stored value
    # of either component (vs[0]=10 for component_a, vs[0]=15 for component_b).
    assert mix.quantile(0.02) < 10.0
    # Upper-tail q=0.98 should land above the upper stored values.
    assert mix.quantile(0.98) > 35.0


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


def test_quantile_distribution_plus_negative_binomial_tail_q_finite() -> None:
    """Direct regression of the mixture-tail edge case the wr_ensemble_decomposed
    integration surfaces: MixtureDistribution(QuantileDistribution,
    ParametricNegativeBinomial).quantile(q) at tail q ∈ {0.10, 0.50, 0.90, 0.99}
    returns finite values. Pre-fix: cdf of the QuantileDistribution component
    clamped at qs[-1]=0.95, capping the joint cdf below 1.0 and breaking the
    brentq inversion at q=0.99. Post-fix: cdf extrapolates linearly past knots
    (clipped to [0, 1]) so the joint cdf reaches 1.0 cleanly.
    """
    quantile_dist = QuantileDistribution(
        quantiles=np.arange(0.05, 0.96, 0.05),
        values=np.linspace(0.0, 10.0, 19),
    )
    nb_dist = ParametricNegativeBinomial(mean=5.0, dispersion=2.0)
    mix = MixtureDistribution(component_a=quantile_dist, component_b=nb_dist, weight=0.5)
    for q in (0.10, 0.50, 0.90, 0.99):
        x = mix.quantile(q)
        assert np.isfinite(x), f"mix.quantile({q}) returned non-finite: {x}"
        # Round-trip: cdf(quantile(q)) ~ q. Tolerance is loose because NB's
        # cdf is a right-continuous step function (discrete distribution), so
        # the joint mixture cdf has piecewise-constant segments where brentq
        # locks onto the step's left edge — the cdf at the returned x can sit
        # at the next step up. NB step heights at low dispersion / moderate
        # mean are ~0.01-0.05 in this regime.
        assert mix.cdf(x) == pytest.approx(q, abs=0.05)
