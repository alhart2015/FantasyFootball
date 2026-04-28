"""Plan 6 Phase 0 — cdf(x) parity with scipy on each existing parametric distribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
    QuantileDistribution,
)


@pytest.mark.parametrize(
    "x",
    [-5.0, -1.0, 0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0],
)
def test_normal_cdf_matches_scipy(x: float) -> None:
    dist = ParametricNormal(mean=10.0, std=4.0)
    expected = float(stats.norm.cdf(x, loc=10.0, scale=4.0))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x",
    [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0],
)
def test_gamma_cdf_matches_scipy(x: float) -> None:
    dist = ParametricGamma(shape=4.0, scale=2.5)
    expected = float(stats.gamma.cdf(x, a=4.0, scale=2.5))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x",
    [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0],
)
def test_nb_cdf_matches_scipy(x: float) -> None:
    dist = ParametricNegativeBinomial(mean=2.5, dispersion=5.0)
    n = 5.0
    p = 5.0 / (5.0 + 2.5)
    expected = float(stats.nbinom.cdf(x, n=n, p=p))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x",
    [-50.0, -10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0, 50.0, 200.0],
)
def test_student_t_cdf_matches_scipy(x: float) -> None:
    dist = ParametricStudentT(loc=5.0, scale=3.0, df=4.5)
    expected = float(stats.t.cdf(x, df=4.5, loc=5.0, scale=3.0))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


def test_cdf_endpoints() -> None:
    """cdf(-large) ~ 0, cdf(+large) ~ 1 for unbounded distributions."""
    # Annotate as list[Distribution] so mypy keeps the protocol type across the
    # heterogeneous list — without this, mypy widens to `object` and loses cdf().
    dists: list[Distribution] = [
        ParametricNormal(mean=0.0, std=1.0),
        ParametricStudentT(loc=0.0, scale=1.0, df=5.0),
    ]
    for dist in dists:
        assert dist.cdf(-1e6) == pytest.approx(0.0, abs=1e-12)
        assert dist.cdf(1e6) == pytest.approx(1.0, abs=1e-12)


def test_cdf_monotone() -> None:
    """cdf is non-decreasing across a fine x grid for each parametric family."""
    grid = np.linspace(-10.0, 50.0, 200)
    families: list[Distribution] = [
        ParametricNormal(mean=5.0, std=3.0),
        ParametricGamma(shape=2.0, scale=4.0),
        ParametricNegativeBinomial(mean=3.0, dispersion=4.0),
        ParametricStudentT(loc=5.0, scale=2.0, df=4.0),
    ]
    for dist in families:
        cdfs = np.array([dist.cdf(float(x)) for x in grid])
        assert np.all(np.diff(cdfs) >= -1e-12), f"{type(dist).__name__} cdf not monotone"


def _make_quantile_dist() -> QuantileDistribution:
    """Symmetric 5-knot fixture spanning [10, 50] at quantiles
    [0.05, 0.25, 0.5, 0.75, 0.95]."""
    return QuantileDistribution(
        quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64),
        values=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
    )


def test_quantile_cdf_at_stored_knots() -> None:
    """cdf(value_at_qk) == qk exactly for each stored knot."""
    dist = _make_quantile_dist()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    vs = [10.0, 20.0, 30.0, 40.0, 50.0]
    for q, v in zip(qs, vs, strict=True):
        assert dist.cdf(v) == pytest.approx(q, abs=1e-12)


def test_quantile_cdf_piecewise_linear_between_knots() -> None:
    """Between knots cdf is linear: at midpoint of two stored values, cdf is
    midpoint of two stored quantiles."""
    dist = _make_quantile_dist()
    # midpoint between (0.25, 20) and (0.5, 30) is value 25, cdf 0.375
    assert dist.cdf(25.0) == pytest.approx(0.375, abs=1e-12)
    # midpoint between (0.5, 30) and (0.75, 40) is value 35, cdf 0.625
    assert dist.cdf(35.0) == pytest.approx(0.625, abs=1e-12)


def test_quantile_cdf_clamps_at_endpoints() -> None:
    """Below the lowest stored value, cdf clamps at the lowest stored quantile.
    Above the highest stored value, cdf clamps at the highest stored quantile.
    """
    dist = _make_quantile_dist()
    assert dist.cdf(-100.0) == pytest.approx(0.05, abs=1e-12)
    assert dist.cdf(0.0) == pytest.approx(0.05, abs=1e-12)
    assert dist.cdf(1000.0) == pytest.approx(0.95, abs=1e-12)


def test_quantile_cdf_monotone_on_grid() -> None:
    """cdf is non-decreasing across a fine value grid."""
    dist = _make_quantile_dist()
    grid = np.linspace(0.0, 60.0, 200)
    cdfs = np.array([dist.cdf(float(x)) for x in grid])
    assert np.all(np.diff(cdfs) >= -1e-12)


def test_quantile_cdf_round_trip_with_quantile() -> None:
    """For q in (q_min, q_max), cdf(quantile(q)) == q."""
    dist = _make_quantile_dist()
    for q in [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]:
        v = dist.quantile(q)
        assert dist.cdf(v) == pytest.approx(q, abs=1e-9)
