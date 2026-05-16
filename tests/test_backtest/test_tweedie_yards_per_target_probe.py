"""Tests for the Tweedie yards_per_target probe.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV, TweedieRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from projections.backtest.tweedie_yards_per_target_probe import (
    _RIDGE_ALPHAS,
    _TWEEDIE_ALPHAS,
    _TWEEDIE_POWER,
    _fit_ridge_efficiency,
    _fit_shared_volume,
    _fit_tweedie_efficiency,
    _predict_yards_ridge,
    _predict_yards_tweedie,
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


def _synthetic_tweedie_fixture(
    rng: np.random.Generator,
    n: int,
    b0: float,
    b1: float,
    *,
    phi: float = 1.0,
    power: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y, true_mu) from a compound-Poisson-Gamma Tweedie with
    a known relationship `mu = exp(b0 + b1 * x)`.

    For Tweedie p in (1, 2): if N ~ Pois(lambda) and each claim G_i iid
    Gamma(alpha, scale), the sum y = sum_{i=1..N} G_i is Tweedie-distributed.
    Closed-form params:
        lambda = mu^(2-p) / (phi * (2-p))
        alpha  = (2-p) / (p-1)
        scale  = phi * (p-1) * mu^(p-1)
    y has a point mass at 0 when N=0 (and continuous positive support otherwise).
    """
    x = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float64)
    mu = np.exp(b0 + b1 * x[:, 0])
    lam = mu ** (2 - power) / (phi * (2 - power))
    alpha = (2 - power) / (power - 1)
    n_claims = rng.poisson(lam)
    y = np.zeros(n, dtype=np.float64)
    nonzero = n_claims > 0
    scale_nonzero = phi * (power - 1) * mu[nonzero] ** (power - 1)
    # rng.gamma's `shape` argument is alpha * n_claims (sum of n Gamma(alpha, scale)
    # is Gamma(n*alpha, scale) when scale matches).
    y[nonzero] = rng.gamma(alpha * n_claims[nonzero], scale_nonzero)
    return x, y, mu


def test_fit_tweedie_efficiency_recovers_true_mu() -> None:
    """Generate (X, y) from a known Tweedie GLM; verify the Pipeline.predict
    recovers true mu within tolerance.
    """
    rng = np.random.default_rng(seed=2031)
    # Larger n for Tweedie (variance is intrinsic to the family).
    x, y, true_mu = _synthetic_tweedie_fixture(rng, n=500, b0=2.0, b1=0.5)

    pipeline = _fit_tweedie_efficiency(x, y)

    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert "gscv" in pipeline.named_steps
    inner = pipeline.named_steps["gscv"]
    assert isinstance(inner, GridSearchCV)
    assert isinstance(inner.best_estimator_, TweedieRegressor)
    assert inner.best_params_["alpha"] in _TWEEDIE_ALPHAS
    assert inner.best_estimator_.power == _TWEEDIE_POWER
    assert inner.best_estimator_.link == "log"

    pred_mu = pipeline.predict(x).astype(np.float64)
    # Tweedie variance is intrinsic; relative MAE up to 30% on average.
    relative_errors = np.abs(pred_mu - true_mu) / np.maximum(true_mu, 1e-6)
    assert float(relative_errors.mean()) < 0.30, (
        f"Tweedie fit relative MAE {float(relative_errors.mean()):.4f} too large"
    )
    # All predictions strictly positive (log link guarantees this).
    assert (pred_mu > 0).all()


def test_fit_tweedie_efficiency_handles_zero_yards_rows() -> None:
    """Tweedie p=1.5 must handle yards_per_target == 0 rows without raising.

    Pins the entire motivation for choosing Tweedie over Gamma (spec §1.4 #8).
    """
    rng = np.random.default_rng(seed=2032)
    n = 300
    x = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float64)
    # ~40% zero rows (mirrors realistic incompletion rate among targets-positive
    # WR-weeks); ~60% positive yards_per_target rows.
    zero_mask = rng.uniform(0, 1, size=n) < 0.4
    y = np.where(
        zero_mask,
        0.0,
        rng.gamma(shape=2.5, scale=4.0, size=n),
    ).astype(np.float64)

    pipeline = _fit_tweedie_efficiency(x, y)

    pred_mu = pipeline.predict(x).astype(np.float64)
    assert (pred_mu > 0).all()
    # Predictions in a reasonable yards_per_target range (mean of positive
    # rows is shape*scale = 10; with 40% zeros the overall mean is ~6).
    assert 1.0 < float(pred_mu.mean()) < 25.0


def test_predict_yards_ridge_clips_negative_to_zero() -> None:
    """Ridge predictions below 0 are clipped before multiplying by volume.

    yards_per_target is bounded below by 0 (no negative yards in a target's
    average); the Ridge incumbent's predict-time clip enforces this.
    """
    rng = np.random.default_rng(seed=2033)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # ratio = -2 + 4 * x[:, 0]: negative predictions on the low-x test rows,
    # so the >= 0 clip is exercised.
    ratio_train = (-2.0 + 4.0 * x_train[:, 0]).astype(np.float64)
    ridge_eff = _fit_ridge_efficiency(x_train, ratio_train)

    x_eval = np.array([[0.0, 0.5], [0.1, 0.5]], dtype=np.float64)
    mu_targets = np.array([10.0, 8.0], dtype=np.float64)

    pred = _predict_yards_ridge(mu_targets, x_eval, ridge_eff)

    # mu_eff at x=0.0 is ~-2.0 (unclipped) -> clipped to 0 -> yards = 0.
    assert pred[0] == 0.0
    # mu_eff at x=0.1 is ~-1.6 (unclipped) -> clipped to 0 -> yards = 0.
    assert pred[1] == 0.0


def test_predict_yards_tweedie_uses_inverse_log_link() -> None:
    """Tweedie prediction equals mu_targets * pipeline.predict(X), AND the
    pipeline applies the inverse-log link exp(scaled_X @ coef + intercept).
    """
    rng = np.random.default_rng(seed=2034)
    n = 250
    x, y, _ = _synthetic_tweedie_fixture(rng, n=n, b0=1.8, b1=0.4)

    pipeline = _fit_tweedie_efficiency(x, y)

    x_eval = rng.uniform(-1.0, 1.0, size=(10, 1)).astype(np.float64)
    mu_targets = rng.uniform(2.0, 10.0, size=10).astype(np.float64)

    pred = _predict_yards_tweedie(mu_targets, x_eval, pipeline)

    # Composition arm: _predict_yards_tweedie multiplies mu_targets by the
    # Pipeline's mean prediction.
    expected = mu_targets * pipeline.predict(x_eval).astype(np.float64)
    assert np.allclose(pred, expected)

    # Inverse-log-link arm: reconstruct Pipeline.predict by manually applying
    # exp(scaler.transform(x_eval) @ coef_ + intercept_). Pins that the log
    # link is actually applied (would catch a regression to link="identity").
    scaler = pipeline.named_steps["scaler"]
    gscv = pipeline.named_steps["gscv"]
    tweedie = gscv.best_estimator_
    x_scaled = scaler.transform(x_eval)
    linear_pred = x_scaled @ tweedie.coef_ + tweedie.intercept_
    reconstructed = np.exp(linear_pred)
    assert np.allclose(pipeline.predict(x_eval).astype(np.float64), reconstructed)

    # All Tweedie predictions strictly positive (log link guarantees this).
    assert (pred > 0).all()
