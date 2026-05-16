"""Logit catch_rate sub-model probe — factor-appropriate efficiency factor.

Two-arm probe comparing the production catch_rate sub-model class (RidgeCV
on the ratio with predict-time clipping to [0, 1]) against a binomial-logit
fit (LogisticRegressionCV via Bernoulli-trial row expansion). Per-stat
receptions Delta-CV-RMSE, walk-forward eval window 2021-2024, paired-bootstrap
CI on pooled residuals.

Mirrors `target_decomposition_probe.py`'s shape; reuses
`paired_bootstrap_rmse_delta` and `BootstrapDelta` from `adoption_gate.py`.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same alpha grid as `BaselineModel.fit` (src/projections/models/baseline.py) and
# `target_decomposition_probe._fit_direct` so the two probe arms differ only in
# the catch_rate sub-model class, not the regularization scale.
_RIDGE_ALPHAS: Final[np.ndarray] = np.logspace(-3, 3, 13)

# Cs grid for LogisticRegressionCV. C = 1 / alpha (sklearn's inverse-penalty
# convention). 5 points spanning 3 orders of magnitude — matches the
# effective regularization range of the Ridge alpha grid for the row-expanded
# Bernoulli trials.
_LOGIT_CS: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0)


def _expand_to_trials(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand each row into individual Bernoulli trials for a binomial-logit fit.

    For row i with `trials[i] = T` and `successes[i] = S`, emit T copies of
    `x[i]` — the first S with `y=1`, the remaining (T - S) with `y=0`. Rows
    with `trials[i] = 0` are dropped entirely.

    The expanded (X_trials, y_trials) pair is the LogisticRegressionCV input
    that recovers the same MLE as a binomial-logit GLM by likelihood
    factorization.

    Args:
        x: (n, n_features) feature matrix.
        successes: (n,) int array; count of successful trials per row.
        trials: (n,) int array; total trials per row.

    Returns:
        (x_trials, y_trials) where x_trials has shape (sum(trials), n_features)
        and y_trials has shape (sum(trials),) with int 0/1 values.

    Raises:
        ValueError: if any successes[i] > trials[i].
    """
    if x.shape[0] != successes.shape[0] or x.shape[0] != trials.shape[0]:
        raise ValueError(
            f"row count mismatch: x={x.shape[0]}, successes={successes.shape[0]}, "
            f"trials={trials.shape[0]}"
        )
    overflow = successes > trials
    if overflow.any():
        bad = int(np.argmax(overflow))
        raise ValueError(
            f"successes[{bad}]={int(successes[bad])} > trials[{bad}]={int(trials[bad])}"
        )

    keep = trials > 0
    x_kept = x[keep]
    successes_kept = successes[keep].astype(np.int64)
    trials_kept = trials[keep].astype(np.int64)

    # Repeat each kept row T times along axis 0.
    x_trials = np.repeat(x_kept, trials_kept, axis=0)

    # Build y per kept row: S ones followed by (T - S) zeros.
    failures_kept = trials_kept - successes_kept
    y_trials_parts: list[np.ndarray] = []
    for s, f in zip(successes_kept, failures_kept, strict=True):
        y_trials_parts.append(np.ones(int(s), dtype=np.int64))
        y_trials_parts.append(np.zeros(int(f), dtype=np.int64))
    y_trials = np.concatenate(y_trials_parts) if y_trials_parts else np.empty((0,), dtype=np.int64)

    return x_trials, y_trials


def _fit_shared_volume(x: np.ndarray, targets: np.ndarray) -> RidgeCV:
    """Fit the shared volume sub-model: `targets ~ X` via RidgeCV.

    Trained on all rows (no targets > 0 filter); zero-target rows are valid
    observations of low-volume players and the volume model must predict them.
    Identical recipe to `target_decomposition_probe._fit_decomposed_volume`.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, targets.astype(np.float64))
    return ridge


def _fit_ridge_efficiency(x_pos: np.ndarray, ratio: np.ndarray) -> RidgeCV:
    """Fit the Ridge-on-ratio efficiency sub-model (incumbent arm).

    Matches the catch_rate fit in `decomposed_baseline.py` exactly: RidgeCV
    on `receptions / targets` (computed only on rows with `targets > 0`).
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x_pos, ratio.astype(np.float64))
    return ridge


def _fit_logit_efficiency(x_trials: np.ndarray, y_trials: np.ndarray) -> Pipeline:
    """Fit the binomial-logit efficiency sub-model (candidate arm).

    Expects row-expanded Bernoulli trials from `_expand_to_trials`. The fit is
    mathematically equivalent to a binomial-logit GLM on (S, T-S) via MLE.

    Wraps StandardScaler + LogisticRegressionCV in a sklearn Pipeline per
    spec §5 risk #6 mitigation: LogisticRegression's L2 penalty is
    scale-dependent (whereas Ridge's CV-selected alpha is approximately
    scale-invariant); scaling the features stops the regularization scale
    from becoming a confounder between sub-model class and feature scale.
    Scaler is fit on the trial-expanded rows and persisted inside the
    Pipeline for predict-time use.

    Uses L2 regularization (matching Ridge's penalty family) and 5-fold CV
    across the `_LOGIT_CS` grid. Default solver `lbfgs` works well for L2
    logistic on this row count.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegressionCV(
                    Cs=list(_LOGIT_CS),
                    cv=5,
                    penalty="l2",
                    scoring="neg_log_loss",
                    solver="lbfgs",
                    max_iter=1000,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_trials, y_trials)
    return pipeline


def _predict_receptions_ridge(
    mu_targets: np.ndarray, x_eval: np.ndarray, ridge_eff: RidgeCV
) -> np.ndarray:
    """Incumbent-arm receptions prediction: mu_targets * clip(mu_ratio, 0, 1).

    Matches the production `decomposed_baseline.py` predict path for the
    mean of the receptions distribution (the predict-time variance/sampling
    is not exercised here — the probe is mean-only).
    """
    mu_ratio: np.ndarray = ridge_eff.predict(x_eval).astype(np.float64)
    mu_ratio_clipped = np.clip(mu_ratio, 0.0, 1.0)
    result: np.ndarray = mu_targets * mu_ratio_clipped
    return result


def _predict_receptions_logit(
    mu_targets: np.ndarray, x_eval: np.ndarray, logit_eff: Pipeline
) -> np.ndarray:
    """Candidate-arm receptions prediction: mu_targets * P(success | x).

    Uses Pipeline.predict_proba, which applies the fitted StandardScaler to
    x_eval before LogisticRegressionCV.predict_proba. The second column is
    P(y=1) under sklearn's binary classification convention.
    """
    p: np.ndarray = logit_eff.predict_proba(x_eval)[:, 1].astype(np.float64)
    result: np.ndarray = mu_targets * p
    return result
