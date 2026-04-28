"""Plan 6 Phase 3 — EnsembleModel weight fitting math."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import (
    ParametricNegativeBinomial,
    ParametricNormal,
    QuantileDistribution,
)
from projections.models.ensemble import _fit_weight_for_stat, _pinball


def test_pinball_loss_underestimate() -> None:
    """When q_pred < actual, pinball = q*(actual - q_pred) for q in (0, 1)."""
    # actual = 10, q_pred = 7, q = 0.9 → q_pred too low → loss = 0.9 * (10 - 7) = 2.7
    assert _pinball(actual=10.0, q_pred=7.0, q=0.9) == pytest.approx(2.7, abs=1e-12)


def test_pinball_loss_overestimate() -> None:
    """When q_pred > actual, pinball = (1-q)*(q_pred - actual)."""
    # actual = 10, q_pred = 13, q = 0.9 → q_pred too high → loss = 0.1 * 3 = 0.3
    assert _pinball(actual=10.0, q_pred=13.0, q=0.9) == pytest.approx(0.3, abs=1e-12)


def test_pinball_loss_exact_match() -> None:
    """When q_pred == actual, pinball = 0 regardless of q."""
    for q in [0.05, 0.1, 0.5, 0.9, 0.95]:
        assert _pinball(actual=10.0, q_pred=10.0, q=q) == pytest.approx(0.0, abs=1e-12)


def test_fit_weight_recovers_known_optimum_a_dominant() -> None:
    """When component A perfectly matches actuals and B is far off, w → 1."""
    rng = np.random.default_rng(seed=7)
    n = 500
    actuals = rng.normal(loc=10.0, scale=2.0, size=n)
    components_a = [ParametricNormal(mean=10.0, std=2.0) for _ in range(n)]
    components_b = [ParametricNormal(mean=100.0, std=2.0) for _ in range(n)]

    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert w_star > 0.9, f"expected w near 1.0 (A wins), got {w_star}"


def test_fit_weight_recovers_known_optimum_b_dominant() -> None:
    """Symmetric: B dominant → w → 0."""
    rng = np.random.default_rng(seed=11)
    n = 500
    actuals = rng.normal(loc=10.0, scale=2.0, size=n)
    components_a = [ParametricNormal(mean=100.0, std=2.0) for _ in range(n)]
    components_b = [ParametricNormal(mean=10.0, std=2.0) for _ in range(n)]

    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert w_star < 0.1, f"expected w near 0.0 (B wins), got {w_star}"


def test_fit_weight_handles_quantile_component() -> None:
    """No crash with QuantileDistribution as component_b (Plan 5/5c shape)."""
    n = 200
    rng = np.random.default_rng(seed=3)
    actuals = rng.gamma(shape=2.0, scale=4.0, size=n)
    components_a = [ParametricNormal(mean=8.0, std=4.0) for _ in range(n)]
    components_b = [
        QuantileDistribution(
            quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
            values=np.array([1.0, 4.0, 8.0, 14.0, 25.0]),
        )
        for _ in range(n)
    ]
    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.0 < w_star < 1.0


def test_fit_weight_handles_count_actuals() -> None:
    """Count-stat shape: NB component_b, integer actuals."""
    rng = np.random.default_rng(seed=5)
    n = 300
    actuals = rng.poisson(lam=1.5, size=n).astype(float)
    components_a = [ParametricNormal(mean=1.5, std=1.5) for _ in range(n)]
    components_b = [ParametricNegativeBinomial(mean=1.5, dispersion=4.0) for _ in range(n)]
    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.001 <= w_star <= 0.999


def test_fit_weight_clip_bounds() -> None:
    """Optimizer output is clipped into [0.001, 0.999]."""
    n = 100
    actuals = np.full(n, 10.0)
    components_a = [ParametricNormal(mean=10.0, std=0.001) for _ in range(n)]
    components_b = [ParametricNormal(mean=10.0, std=10.0) for _ in range(n)]
    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.001 <= w_star <= 0.999


def test_fit_weight_handles_length_mismatch() -> None:
    """Raises ValueError when component lists / actuals lengths differ."""
    a = [ParametricNormal(mean=0.0, std=1.0)] * 10
    b = [ParametricNormal(mean=5.0, std=1.0)] * 10
    actuals = np.zeros(11)  # off-by-one
    with pytest.raises(ValueError, match="length mismatch"):
        _fit_weight_for_stat(components_a=a, components_b=b, actuals=actuals)


def test_fit_weight_empty_input_returns_default() -> None:
    """Zero-length input returns 0.5 with RuntimeWarning."""
    actuals = np.zeros(0)
    with pytest.warns(RuntimeWarning, match="zero-length"):
        result = _fit_weight_for_stat(
            components_a=[],
            components_b=[],
            actuals=actuals,
        )
    assert result == 0.5
