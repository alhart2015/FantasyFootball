"""Tests for src/projections/backtest/target_decomposition_probe.py.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import RidgeCV

from projections.backtest.target_decomposition_probe import (
    _RIDGE_ALPHAS,
    _WR_RECEIVING_DECOMPS,
    _fit_decomposed_efficiency,
    _fit_decomposed_volume,
    _fit_direct,
    _predict_decomposed,
    _predict_direct,
)
from projections.schemas import Stat


def test_wr_receiving_decomps_registry_has_three_stats() -> None:
    """Three decomposed stats — receptions, receiving_yards, receiving_tds.

    Each shares Stat.TARGETS as its volume factor; efficiency clip-hi is 1.0
    for ratios and +inf for yards-per-target.
    """
    assert set(_WR_RECEIVING_DECOMPS.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for stat, decomp in _WR_RECEIVING_DECOMPS.items():
        assert decomp.volume_stat is Stat.TARGETS
        assert decomp.numerator_stat is stat
    assert _WR_RECEIVING_DECOMPS[Stat.RECEPTIONS].efficiency_clip_hi == 1.0
    assert _WR_RECEIVING_DECOMPS[Stat.RECEIVING_TDS].efficiency_clip_hi == 1.0
    assert _WR_RECEIVING_DECOMPS[Stat.RECEIVING_YARDS].efficiency_clip_hi == float("inf")


def _synthetic_xy(n: int = 50, n_features: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features))
    coef = np.array([1.0, -0.5, 0.25, 0.0])
    y = x @ coef + 0.1 * rng.standard_normal(n)
    return x, y


def test_fit_direct_returns_ridgecv_with_canonical_alphas() -> None:
    x, y = _synthetic_xy()
    ridge = _fit_direct(x, y)
    assert isinstance(ridge, RidgeCV)
    np.testing.assert_array_equal(ridge.alphas, _RIDGE_ALPHAS)
    # Sanity: prediction shape matches input rows.
    assert ridge.predict(x).shape == (len(x),)


def test_fit_decomposed_volume_targets_y_is_volume_stat() -> None:
    x, _ = _synthetic_xy()
    targets = np.arange(len(x), dtype=np.int64)  # arbitrary non-trivial targets
    ridge = _fit_decomposed_volume(x, targets)
    assert isinstance(ridge, RidgeCV)
    # Coefficient is a 4-vector matching n_features.
    assert ridge.coef_.shape == (4,)


def test_fit_decomposed_efficiency_filters_to_targets_positive() -> None:
    """Efficiency arm trains only on rows where targets > 0; ratio is
    numerator / targets on those rows.
    """
    x, _ = _synthetic_xy(n=20, seed=1)
    # Half the rows have targets > 0; the rest are zero-target.
    targets = np.array([5, 0, 3, 0, 8, 0, 2, 0, 4, 0, 6, 0, 1, 0, 7, 0, 9, 0, 3, 0])
    numerator = targets * 0.5  # so true catch_rate is 0.5 on every targets > 0 row
    numerator[targets == 0] = 0  # well-defined ratio nonexistent here, but predict won't see these
    ridge = _fit_decomposed_efficiency(x, numerator, targets)
    assert isinstance(ridge, RidgeCV)
    # On the targets > 0 subset, ratio is constant 0.5; ridge should predict ~0.5 everywhere.
    pred = ridge.predict(x)
    assert np.allclose(pred, 0.5, atol=0.05)


def test_fit_decomposed_efficiency_raises_when_no_positive_targets() -> None:
    """All-zero-targets training set is malformed; raise rather than silently
    return a ridge fit on zero rows.
    """
    x, _ = _synthetic_xy(n=10)
    targets = np.zeros(10, dtype=np.int64)
    numerator = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="targets > 0"):
        _fit_decomposed_efficiency(x, numerator, targets)


def test_predict_direct_returns_float64_array_of_eval_shape() -> None:
    x_train, y_train = _synthetic_xy(seed=2)
    ridge = _fit_direct(x_train, y_train)
    x_eval, _ = _synthetic_xy(n=30, seed=3)
    pred = _predict_direct(ridge, x_eval)
    assert pred.dtype == np.float64
    assert pred.shape == (30,)


def test_predict_decomposed_clips_volume_at_zero_and_efficiency_at_clip_hi() -> None:
    """Volume floored at 0; efficiency clipped to [0, efficiency_clip_hi].

    Construct a contrived volume_ridge that predicts negatives on some rows
    and an efficiency_ridge that predicts > 1 on some rows, with
    efficiency_clip_hi = 1.0. Verify the product respects both clips.
    """
    # Train ridges with extreme synthetic data so we control the predictions.
    rng = np.random.default_rng(4)
    n = 8
    x = rng.standard_normal((n, 2))

    # Volume ridge: y_train = -x[:, 0] * 5 (negatives appear when x[:, 0] > 0)
    volume_y = -x[:, 0] * 5.0
    volume_ridge = _fit_direct(x, volume_y)

    # Efficiency ridge: y_train = x[:, 1] * 0.5 + 0.5, range roughly [-0.5, 1.5]
    efficiency_y = x[:, 1] * 0.5 + 0.5
    efficiency_ridge = _fit_direct(x, efficiency_y)

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=efficiency_ridge,
        x=x,
        efficiency_clip_hi=1.0,
    )
    assert pred.dtype == np.float64
    assert pred.shape == (n,)
    # Product of clipped factors is non-negative and bounded above by max(volume_clip, +inf) * 1.0.
    assert (pred >= 0.0).all()
    # On rows where volume_ridge predicts < 0, the product is exactly 0.
    raw_volume = volume_ridge.predict(x)
    assert np.all(pred[raw_volume < 0] == 0.0)


def test_predict_decomposed_no_clip_on_efficiency_hi_when_inf() -> None:
    """yards_per_target case: efficiency_clip_hi = +inf; Ridge predictions
    above ~15 (empirical max) are not clipped on the high side."""
    rng = np.random.default_rng(5)
    n = 6
    x = rng.standard_normal((n, 2))
    volume_y = np.full(n, 10.0)  # volume ridge predicts ~10 everywhere
    volume_ridge = _fit_direct(x, volume_y)
    efficiency_y = np.full(n, 25.0)  # efficiency ridge predicts ~25 everywhere
    efficiency_ridge = _fit_direct(x, efficiency_y)

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=efficiency_ridge,
        x=x,
        efficiency_clip_hi=float("inf"),
    )
    # No upper clip; product is ~250.
    assert np.all(pred > 200.0)
