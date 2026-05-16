"""Tests for src/projections/backtest/rb_decomposition_probe.py.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.backtest.rb_decomposition_probe import (
    _RB_DECOMPS,
    _RIDGE_ALPHAS,
    _fit_decomposed_efficiency,
    _fit_decomposed_volume,
    _fit_direct,
    _predict_decomposed,
    _predict_direct,
    _StatDecomp,
)
from projections.schemas import Stat


def test_rb_decomps_registry_has_five_stats_across_two_volume_axes() -> None:
    """Five composed stats: 2 rushing (carries axis) + 3 receiving (targets axis)."""
    assert set(_RB_DECOMPS.keys()) == {
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    # Rushing axis.
    assert _RB_DECOMPS[Stat.RUSHING_YARDS].volume_stat is Stat.CARRIES
    assert _RB_DECOMPS[Stat.RUSHING_TDS].volume_stat is Stat.CARRIES
    # Receiving axis.
    assert _RB_DECOMPS[Stat.RECEPTIONS].volume_stat is Stat.TARGETS
    assert _RB_DECOMPS[Stat.RECEIVING_YARDS].volume_stat is Stat.TARGETS
    assert _RB_DECOMPS[Stat.RECEIVING_TDS].volume_stat is Stat.TARGETS

    # Clip semantics: rate factors -> 1.0; unbounded efficiency -> +inf.
    assert _RB_DECOMPS[Stat.RUSHING_YARDS].efficiency_clip_hi == float("inf")
    assert _RB_DECOMPS[Stat.RUSHING_TDS].efficiency_clip_hi == 1.0
    assert _RB_DECOMPS[Stat.RECEPTIONS].efficiency_clip_hi == 1.0
    assert _RB_DECOMPS[Stat.RECEIVING_YARDS].efficiency_clip_hi == float("inf")
    assert _RB_DECOMPS[Stat.RECEIVING_TDS].efficiency_clip_hi == 1.0

    # numerator_stat == the key for every entry (composition invariant).
    for stat, decomp in _RB_DECOMPS.items():
        assert isinstance(decomp, _StatDecomp)
        assert decomp.numerator_stat is stat


def test_fit_direct_fits_ridgecv_on_synthetic_linear_relationship() -> None:
    """Direct fit recovers a clean linear slope within tolerance."""
    rng = np.random.default_rng(seed=2026)
    x = rng.uniform(0.0, 5.0, size=(200, 3)).astype(np.float64)
    # true: y = 2 * x[:, 0] + 1 * x[:, 1] - 0.5 * x[:, 2] + noise
    y = (2.0 * x[:, 0] + 1.0 * x[:, 1] - 0.5 * x[:, 2] + rng.normal(0, 0.2, size=200)).astype(
        np.float64
    )

    ridge = _fit_direct(x, y)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    assert abs(ridge.coef_[0] - 2.0) < 0.1
    assert abs(ridge.coef_[1] - 1.0) < 0.1


def test_fit_decomposed_volume_fits_on_unfiltered_rows() -> None:
    """Volume sub-model trains on ALL rows including volume == 0
    (zero-volume rows are legitimate observations of low-volume players).
    """
    rng = np.random.default_rng(seed=2027)
    x = rng.uniform(0.0, 1.0, size=(150, 2)).astype(np.float64)
    # carries integer with ~30% zeros
    carries = np.where(
        rng.uniform(0, 1, size=150) < 0.3,
        0,
        rng.poisson(10, size=150),
    ).astype(np.int64)

    ridge = _fit_decomposed_volume(x, carries)

    assert isinstance(ridge, RidgeCV)
    pred = ridge.predict(x)
    assert pred.shape == (150,)


def test_fit_decomposed_efficiency_fits_only_on_positive_volume_rows() -> None:
    """Efficiency sub-model trains only on rows where volume > 0."""
    rng = np.random.default_rng(seed=2028)
    x = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    volume = np.where(
        rng.uniform(0, 1, size=100) < 0.5,
        0,
        rng.poisson(8, size=100),
    ).astype(np.int64)
    # numerator ~ volume * (4 + 2 * x[:, 0])
    rate = 4.0 + 2.0 * x[:, 0]
    numerator = (volume.astype(np.float64) * rate).astype(np.int64)

    ridge = _fit_decomposed_efficiency(x, numerator, volume)

    assert isinstance(ridge, RidgeCV)
    # The slope on x[:, 0] should be ~2.0 (the rate's coefficient).
    assert abs(ridge.coef_[0] - 2.0) < 0.5


def test_fit_decomposed_efficiency_raises_if_no_positive_volume() -> None:
    """If every training row has volume == 0, the efficiency fit cannot
    proceed (division-by-zero in the ratio).
    """
    import pytest

    x = np.zeros((10, 3), dtype=np.float64)
    volume = np.zeros(10, dtype=np.int64)
    numerator = np.zeros(10, dtype=np.int64)

    with pytest.raises(ValueError, match=r"no training rows with volume > 0"):
        _fit_decomposed_efficiency(x, numerator, volume)


def test_predict_direct_passes_through_no_clipping() -> None:
    """Direct prediction does NOT clip (mirrors BaselineModel; downstream
    Distribution constructor handles family floors).
    """
    rng = np.random.default_rng(seed=2029)
    x_train = rng.uniform(0.0, 1.0, size=(50, 2)).astype(np.float64)
    # Force negative predictions: train y = -1 + ...
    y_train = (-1.0 + rng.normal(0, 0.05, size=50)).astype(np.float64)
    ridge = _fit_direct(x_train, y_train)

    x_eval = rng.uniform(0.0, 1.0, size=(5, 2)).astype(np.float64)
    pred = _predict_direct(ridge, x_eval)

    assert pred.shape == (5,)
    # Predictions are around -1 — no clip applied.
    assert (pred < 0).all()


def test_predict_decomposed_applies_two_sided_clip_on_rate_efficiency() -> None:
    """Decomposed prediction clips efficiency to [0, clip_hi]; volume to [0, +inf).

    Two-sided clip engages on rate factors (clip_hi = 1.0); only low-side on
    yards_per_volume (clip_hi = +inf).
    """
    rng = np.random.default_rng(seed=2030)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # Volume Ridge fit on a synthetic carries response.
    volume_y = (10.0 + 5.0 * x_train[:, 0] + rng.normal(0, 0.5, size=100)).astype(np.float64)
    volume_ridge = _fit_direct(x_train, volume_y)
    # Efficiency Ridge that will predict > 1 on high-x_eval (forces clip_hi=1.0).
    rate_y = (0.5 + 1.5 * x_train[:, 0] + rng.normal(0, 0.05, size=100)).astype(np.float64)
    rate_ridge = _fit_direct(x_train, rate_y)

    x_eval = np.array([[1.0, 0.5]], dtype=np.float64)  # rate prediction ~ 2.0 unclipped.

    # Rate factor with clip_hi=1.0: predicted efficiency caps at 1.0.
    pred_clipped = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=rate_ridge,
        x=x_eval,
        efficiency_clip_hi=1.0,
    )
    # Volume at x=1.0 is ~15; efficiency clipped to 1.0 -> result ~ 15.0
    assert 10.0 < pred_clipped[0] < 20.0

    # Same data, clip_hi=+inf: predicted efficiency is unclipped.
    pred_unclipped = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=rate_ridge,
        x=x_eval,
        efficiency_clip_hi=float("inf"),
    )
    # Now efficiency ~ 2.0 -> result ~ 30.
    assert pred_unclipped[0] > pred_clipped[0]


def test_predict_decomposed_clips_negative_volume_to_zero() -> None:
    """Volume predictions below 0 are clipped (can't have negative carries)."""
    rng = np.random.default_rng(seed=2031)
    x_train = rng.uniform(0.0, 1.0, size=(50, 2)).astype(np.float64)
    # Force negative volume predictions on the eval-low-x side.
    volume_y = (-5.0 + 10.0 * x_train[:, 0] + rng.normal(0, 0.5, size=50)).astype(np.float64)
    volume_ridge = _fit_direct(x_train, volume_y)
    eff_y = (5.0 + 2.0 * x_train[:, 0] + rng.normal(0, 0.1, size=50)).astype(np.float64)
    eff_ridge = _fit_direct(x_train, eff_y)

    x_eval = np.array([[0.0, 0.5]], dtype=np.float64)  # volume ~ -5 unclipped.

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=eff_ridge,
        x=x_eval,
        efficiency_clip_hi=float("inf"),
    )
    # Volume clipped to 0 -> result is 0.
    assert pred[0] == 0.0
