"""Tests for the Tweedie yards_per_target probe.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.backtest.tweedie_yards_per_target_probe import (
    _RIDGE_ALPHAS,
    _fit_ridge_efficiency,
    _fit_shared_volume,
)


def test_fit_shared_volume_returns_fitted_ridgecv() -> None:
    """Shared volume fit on a linear-target synthetic frame recovers the slope."""
    rng = np.random.default_rng(seed=2026)
    x = rng.uniform(0.0, 5.0, size=(200, 3)).astype(np.float64)
    # true: targets = 2 * x[:, 0] + 1 * x[:, 1] - 0.5 * x[:, 2] + noise
    targets = (2.0 * x[:, 0] + 1.0 * x[:, 1] - 0.5 * x[:, 2] + rng.normal(0, 0.2, size=200)).astype(
        np.float64
    )

    ridge = _fit_shared_volume(x, targets)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    # Coefficient recovery sanity.
    assert abs(ridge.coef_[0] - 2.0) < 0.1
    assert abs(ridge.coef_[1] - 1.0) < 0.1


def test_fit_ridge_efficiency_matches_pinned_alpha_grid() -> None:
    """Ridge efficiency fit uses the same alpha grid as BaselineModel.fit."""
    rng = np.random.default_rng(seed=2027)
    x = rng.uniform(0.0, 1.0, size=(150, 2)).astype(np.float64)
    # ratio = yards_per_target ~ 8 + 4 * x[:, 0] (unbounded positive)
    ratio = (8.0 + 4.0 * x[:, 0] + rng.normal(0, 0.5, size=150)).astype(np.float64)

    ridge = _fit_ridge_efficiency(x, ratio)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    # Coefficient recovery sanity for the unbounded yards-per-target response.
    assert abs(ridge.coef_[0] - 4.0) < 0.5
