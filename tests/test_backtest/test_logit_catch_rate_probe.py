"""Tests for the logit catch_rate probe.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline

from projections.backtest.logit_catch_rate_probe import (
    _LOGIT_CS,
    _RIDGE_ALPHAS,
    PerStatVerdict,
    ProbeResults,
    _expand_to_trials,
    _fit_logit_efficiency,
    _fit_ridge_efficiency,
    _fit_shared_volume,
    _predict_receptions_logit,
    _predict_receptions_ridge,
    compute_verdict,
    walk_forward_residuals,
)


def test_expand_to_trials_basic_shape_and_labels() -> None:
    """3 rows with (T, S) = (4, 3), (2, 0), (5, 5). Expansion yields 11 trial
    rows: 3+1+0+2+5+0 = 7 successes and 1+2+0 = 3 failures, sharing X.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    successes = np.array([3, 0, 5], dtype=np.int64)
    trials = np.array([4, 2, 5], dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (11, 2)
    assert y_trials.shape == (11,)
    assert np.allclose(x_trials[:4], np.tile([1.0, 2.0], (4, 1)))
    assert np.array_equal(y_trials[:4], np.array([1, 1, 1, 0]))
    assert np.allclose(x_trials[4:6], np.tile([3.0, 4.0], (2, 1)))
    assert np.array_equal(y_trials[4:6], np.array([0, 0]))
    assert np.allclose(x_trials[6:11], np.tile([5.0, 6.0], (5, 1)))
    assert np.array_equal(y_trials[6:11], np.array([1, 1, 1, 1, 1]))


def test_expand_to_trials_zero_trials_dropped() -> None:
    """A row with T=0 must be dropped from the expansion entirely
    (rather than panicking on shape mismatch in the per-row alloc).
    """
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    successes = np.array([2, 0, 1], dtype=np.int64)
    trials = np.array([2, 0, 3], dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (5, 1)
    assert y_trials.shape == (5,)
    assert np.allclose(x_trials[:2], np.array([[1.0], [1.0]]))
    assert np.array_equal(y_trials[:2], np.array([1, 1]))
    assert np.allclose(x_trials[2:], np.array([[3.0], [3.0], [3.0]]))
    assert np.array_equal(y_trials[2:], np.array([1, 0, 0]))


def test_expand_to_trials_validates_successes_le_trials() -> None:
    """successes[i] > trials[i] is a bug in the caller. Raise ValueError
    rather than producing a corrupt expansion.
    """
    import pytest

    x = np.array([[1.0]], dtype=np.float64)
    successes = np.array([5], dtype=np.int64)
    trials = np.array([3], dtype=np.int64)

    with pytest.raises(ValueError, match=r"successes\[0\]=5 > trials\[0\]=3"):
        _expand_to_trials(x, successes, trials)


def test_expand_to_trials_handles_empty_input() -> None:
    """Empty (X, successes, trials) returns empty arrays of the right shape."""
    x = np.empty((0, 3), dtype=np.float64)
    successes = np.empty((0,), dtype=np.int64)
    trials = np.empty((0,), dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (0, 3)
    assert y_trials.shape == (0,)


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
    ratio = (0.6 + 0.3 * x[:, 0] + rng.normal(0, 0.05, size=150)).astype(np.float64)

    ridge = _fit_ridge_efficiency(x, ratio)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS


def test_fit_logit_efficiency_recovers_true_p() -> None:
    """Generate (X, S, T) from a known logit; verify predict_proba on a holdout
    matches the true catch_rate within tolerance.
    """
    rng = np.random.default_rng(seed=2028)
    n_rows = 400
    x = rng.uniform(-1.0, 1.0, size=(n_rows, 2)).astype(np.float64)
    # true: logit(p) = -0.4 + 1.2 * x[:, 0] - 0.6 * x[:, 1]
    logit_p = -0.4 + 1.2 * x[:, 0] - 0.6 * x[:, 1]
    true_p = 1.0 / (1.0 + np.exp(-logit_p))
    trials = rng.integers(1, 15, size=n_rows).astype(np.int64)
    successes = rng.binomial(trials, true_p).astype(np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)
    logit = _fit_logit_efficiency(x_trials, y_trials)

    # Pipeline wraps StandardScaler + LogisticRegressionCV per spec §5 risk #6.
    assert isinstance(logit, Pipeline)
    assert "scaler" in logit.named_steps
    inner = logit.named_steps["logit"]
    assert isinstance(inner, LogisticRegressionCV)
    assert inner.C_[0] in _LOGIT_CS

    # predict_proba on the Pipeline transparently scales x before the logit step.
    pred_p = logit.predict_proba(x)[:, 1]
    mae = float(np.abs(pred_p - true_p).mean())
    assert mae < 0.05, f"binomial-logit MAE on true_p too large: {mae}"


def test_predict_receptions_ridge_clips_to_unit_interval() -> None:
    """Ridge predictions outside [0, 1] are clipped before multiplying by volume."""
    rng = np.random.default_rng(seed=2029)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # Synthetic ratio = 0.5 + 1.5 * x[:, 0]: forces unclipped predictions > 1
    # on the high-x test rows, so the clip is exercised.
    ratio_train = (0.5 + 1.5 * x_train[:, 0]).astype(np.float64)
    ridge_eff = _fit_ridge_efficiency(x_train, ratio_train)

    x_eval = np.array([[1.0, 0.5], [0.9, 0.5]], dtype=np.float64)
    mu_targets = np.array([10.0, 8.0], dtype=np.float64)

    pred = _predict_receptions_ridge(mu_targets, x_eval, ridge_eff)

    # mu_eff at x=1.0 is ~2.0 (unclipped) -> clipped to 1.0 -> receptions = 10.0
    assert pred[0] == 10.0
    # mu_eff at x=0.9 is ~1.85 (unclipped) -> clipped to 1.0 -> receptions = 8.0
    assert pred[1] == 8.0


def test_predict_receptions_logit_uses_predict_proba() -> None:
    """Logit prediction equals mu_targets * predict_proba(X)[:, 1]."""
    rng = np.random.default_rng(seed=2030)
    n_rows = 200
    x = rng.uniform(-1.0, 1.0, size=(n_rows, 2)).astype(np.float64)
    trials = rng.integers(2, 10, size=n_rows).astype(np.int64)
    true_p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x[:, 0])))
    successes = rng.binomial(trials, true_p).astype(np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)
    logit_eff = _fit_logit_efficiency(x_trials, y_trials)

    x_eval = rng.uniform(-1.0, 1.0, size=(10, 2)).astype(np.float64)
    mu_targets = rng.uniform(2.0, 10.0, size=10).astype(np.float64)

    pred = _predict_receptions_logit(mu_targets, x_eval, logit_eff)

    expected = mu_targets * logit_eff.predict_proba(x_eval)[:, 1]
    assert np.allclose(pred, expected)
    assert (pred >= 0).all()


def _synthetic_wr_inputs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic WR (features, weekly_stats) pair for walk-forward
    integration testing.

    4 seasons x 4 weeks x 8 players = 128 rows. Features are uniform random;
    truth uses a known catch_rate-from-features generative model with targets
    correlated with one feature so volume_ridge has signal.
    """
    from projections.schemas import _PYARROW_STR, WeeklyStatsSchema

    rng = np.random.default_rng(seed=seed)
    rows: list[dict[str, object]] = []
    for season in range(2018, 2022):
        for week in range(1, 5):
            for p in range(8):
                rows.append(
                    {
                        "gsis_id": f"00-{1_000_000 + p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    # Use the WR feature schema's columns; fill plausibles for the rest.
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position

    feature_schema = POSITION_DISPATCH[Position.WR].feature_schema
    schema_cols = feature_schema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        elif col_name == "age":
            df[col_name] = rng.uniform(22.0, 30.0, size=len(df)).astype(np.float64)
        else:
            df[col_name] = rng.uniform(0.0, 1.0, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "WR"
    n = len(ws)
    target_lambda = rng.uniform(3.0, 12.0, size=n)
    targets = np.maximum(1, rng.poisson(target_lambda)).astype(np.int64)
    true_p = 1.0 / (1.0 + np.exp(-(0.4 + 0.5 * target_lambda / 10.0)))
    receptions = np.array(
        [int(rng.binomial(t, p)) for t, p in zip(targets, true_p, strict=True)],
        dtype=np.int64,
    )
    ws["targets"] = targets
    ws["receptions"] = receptions
    ws["receiving_yards"] = np.maximum(0.0, receptions * rng.normal(11.0, 3.0, size=n)).astype(
        np.float64
    )
    ws["receiving_tds"] = np.where(
        rng.uniform(0, 1, size=n) < np.minimum(targets * 0.05, 1.0), 1, 0
    ).astype(np.int64)
    ws["rushing_yards"] = np.zeros(n, dtype=np.float64)
    ws["rushing_tds"] = np.zeros(n, dtype=np.int64)
    ws["fumbles_lost"] = np.zeros(n, dtype=np.int64)
    ws["passing_yards"] = 0.0
    ws["passing_tds"] = np.int64(0)
    ws["interceptions"] = np.int64(0)

    schema_cols_ws = WeeklyStatsSchema.to_schema().columns
    for col_name, col in schema_cols_ws.items():
        if col_name in ws.columns:
            continue
        dtype_str = str(col.dtype)
        if "int" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.int64)
        elif "float" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.float64)
        else:
            ws[col_name] = 0
    return features, WeeklyStatsSchema.validate(ws)


def test_walk_forward_residuals_produces_paired_arrays() -> None:
    """Both ridge and logit prediction buffers are populated and length-matched
    on every eval year.
    """
    features, weekly_stats = _synthetic_wr_inputs()
    results = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    assert isinstance(results, ProbeResults)
    assert len(results.actual_receptions) > 0
    assert results.actual_receptions.shape == results.pred_ridge.shape
    assert results.actual_receptions.shape == results.pred_logit.shape
    assert results.year.shape == results.actual_receptions.shape
    # Coverage stats present for each eval year.
    assert set(results.coverage_per_year.keys()) == {2020, 2021}
    # All predictions non-negative.
    assert (results.pred_ridge >= 0).all()
    assert (results.pred_logit >= 0).all()


def test_walk_forward_residuals_arms_differ_on_some_rows() -> None:
    """The two arms should not produce identical predictions everywhere. This
    sanity-checks that the logit arm is actually exercised (a no-op fall-through
    to the ridge arm would be a silent bug).
    """
    features, weekly_stats = _synthetic_wr_inputs(seed=12345)
    results = walk_forward_residuals(features, weekly_stats, eval_years=(2021,))

    n_different = int(np.sum(np.abs(results.pred_ridge - results.pred_logit) > 1e-6))
    assert n_different > 0, "ridge and logit arms produced identical predictions"


def test_compute_verdict_signal_when_ci_strictly_negative() -> None:
    """Synthetic ProbeResults where the logit clearly beats the ridge -> SIGNAL."""
    rng = np.random.default_rng(seed=2031)
    n = 500
    actual = rng.uniform(0.0, 10.0, size=n)
    # ridge has 0.5 systematic bias; logit is unbiased.
    pred_ridge = actual + 0.5 + rng.normal(0, 0.1, size=n)
    pred_logit = actual + rng.normal(0, 0.1, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_receptions=actual,
        pred_ridge=pred_ridge,
        pred_logit=pred_logit,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert isinstance(verdict, PerStatVerdict)
    assert verdict.verdict == "SIGNAL"
    assert verdict.rmse_delta.hi_95 < 0


def test_compute_verdict_null_when_ci_brackets_zero() -> None:
    """Random noise -> NULL."""
    rng = np.random.default_rng(seed=2032)
    n = 300
    actual = rng.uniform(0.0, 10.0, size=n)
    pred_ridge = actual + rng.normal(0, 1.0, size=n)
    pred_logit = actual + rng.normal(0, 1.0, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_receptions=actual,
        pred_ridge=pred_ridge,
        pred_logit=pred_logit,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert verdict.verdict == "NULL"
    assert verdict.rmse_delta.lo_95 < 0 < verdict.rmse_delta.hi_95


def test_compute_verdict_regression_when_ci_strictly_positive() -> None:
    """Logit systematically worse than Ridge -> REGRESSION."""
    rng = np.random.default_rng(seed=2033)
    n = 500
    actual = rng.uniform(0.0, 10.0, size=n)
    pred_ridge = actual + rng.normal(0, 0.1, size=n)
    pred_logit = actual + 0.5 + rng.normal(0, 0.1, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_receptions=actual,
        pred_ridge=pred_ridge,
        pred_logit=pred_logit,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert verdict.verdict == "REGRESSION"
    assert verdict.rmse_delta.lo_95 > 0
