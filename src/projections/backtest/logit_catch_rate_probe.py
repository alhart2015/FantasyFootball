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

import numpy as np


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
