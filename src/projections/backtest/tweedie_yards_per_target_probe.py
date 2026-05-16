"""Tweedie yards_per_target sub-model probe — factor-appropriate efficiency factor.

Two-arm probe comparing a Ridge-on-ratio efficiency sub-model (incumbent;
the recipe used by DecomposedBaselineModel when configured for unbounded
efficiency factors per src/projections/models/decomposed_baseline.py) against
a TweedieRegressor(power=1.5, link="log") fit on the same ratio (candidate).
Per-stat receiving_yards Delta-CV-RMSE, walk-forward eval window 2021-2024,
paired-bootstrap CI on pooled residuals.

Mirrors logit_catch_rate_probe.py's shape; reuses paired_bootstrap_rmse_delta
and BootstrapDelta from adoption_gate.py.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.linear_model import RidgeCV, TweedieRegressor
from sklearn.metrics import make_scorer, mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projections.models.baseline import _RIDGE_ALPHA_GRID

# Reuse the canonical Ridge alpha grid from `BaselineModel`. The probe's
# incumbent arm mirrors `DecomposedBaselineModel`'s unbounded-efficiency fit;
# the alphas MUST match or the comparison is no longer ridge-vs-ridge.
_RIDGE_ALPHAS: Final[np.ndarray] = _RIDGE_ALPHA_GRID


def _fit_shared_volume(x: np.ndarray, targets: np.ndarray) -> RidgeCV:
    """Fit the shared volume sub-model: targets ~ X via RidgeCV.

    Trained on all rows (no targets > 0 filter); zero-target rows are valid
    observations of low-volume players and the volume model must predict them.
    Identical recipe to logit_catch_rate_probe._fit_shared_volume and to
    target_decomposition_probe._fit_decomposed_volume.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, targets.astype(np.float64))
    return ridge


def _fit_ridge_efficiency(x_pos: np.ndarray, ratio: np.ndarray) -> RidgeCV:
    """Fit the Ridge-on-ratio efficiency sub-model (incumbent arm).

    Matches the unbounded-efficiency code path in decomposed_baseline.py
    (efficiency_clip_hi = float("inf")): RidgeCV on
    `receiving_yards / targets` (computed only on rows with `targets > 0`).
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x_pos, ratio.astype(np.float64))
    return ridge


# Tweedie alpha grid: 7 points spanning 6 orders of magnitude. Ridge uses 13
# because RidgeCV is fast; Tweedie fits are ~5-10x slower per fit (iterative
# Newton/LBFGS on the GLM likelihood), so we trade resolution for runtime.
_TWEEDIE_ALPHAS: Final[tuple[float, ...]] = tuple(float(a) for a in np.logspace(-3, 3, 7))

# Tweedie variance power. p=1.5 is the standard compound-Poisson-Gamma default
# (point mass at 0 + continuous positive support), matching yards_per_target's
# distribution shape. Fixed (not CV-searched) per spec §1.4 #7.
_TWEEDIE_POWER: Final[float] = 1.5


def _fit_tweedie_efficiency(x_pos: np.ndarray, ratio: np.ndarray) -> Pipeline:
    """Fit the Tweedie GLM efficiency sub-model (candidate arm).

    Wraps StandardScaler + GridSearchCV(TweedieRegressor) in a sklearn Pipeline.

    Why a Pipeline:
    - StandardScaler upstream: TweedieRegressor's L2 penalty is scale-dependent
      (whereas Ridge's CV-selected alpha is approximately scale-invariant);
      scaling stops the regularization scale from confounding the sub-model-
      class comparison. The scaler is fit on the full training rows here (same
      pattern as PR #39's logit probe); the inner-CV-fold leakage on a stable
      StandardScaler mean/std on ~5K-8K rows is negligible.
    - GridSearchCV downstream: TweedieRegressor lacks a built-in CV variant
      (no TweedieRegressorCV); GridSearchCV with `refit=True` produces a final
      estimator trained on the full train fold at the CV-selected alpha. The
      inner 5-fold CV is on the training fold only — no leakage with the outer
      walk-forward eval-year split.

    Scoring: mean_tweedie_deviance with matching power=1.5 (deviance is loss-
    style, so `greater_is_better=False`). This is the canonical Tweedie GLM
    fit objective.

    Solver: TweedieRegressor's default is `lbfgs`; max_iter=200 (sklearn
    default 100) for safety against convergence warnings on rows with
    extreme features.
    """
    scorer = make_scorer(mean_tweedie_deviance, power=_TWEEDIE_POWER, greater_is_better=False)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gscv",
                GridSearchCV(
                    estimator=TweedieRegressor(
                        power=_TWEEDIE_POWER,
                        link="log",
                        max_iter=200,
                    ),
                    param_grid={"alpha": list(_TWEEDIE_ALPHAS)},
                    cv=5,
                    scoring=scorer,
                    refit=True,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_pos, ratio.astype(np.float64))
    return pipeline


def _predict_yards_ridge(
    mu_targets: np.ndarray, x_eval: np.ndarray, ridge_eff: RidgeCV
) -> np.ndarray:
    """Incumbent-arm receiving_yards prediction: mu_targets * clip(mu_ratio, 0, +inf).

    Matches the unbounded-efficiency predict path in decomposed_baseline.py
    (efficiency_clip_hi = float("inf")): the >=0 floor still applies because
    yards_per_target cannot be negative; the upper clip is a no-op.
    """
    mu_ratio: np.ndarray = ridge_eff.predict(x_eval).astype(np.float64)
    mu_ratio_clipped = np.maximum(mu_ratio, 0.0)
    result: np.ndarray = mu_targets * mu_ratio_clipped
    return result


def _predict_yards_tweedie(
    mu_targets: np.ndarray, x_eval: np.ndarray, tweedie_eff: Pipeline
) -> np.ndarray:
    """Candidate-arm receiving_yards prediction: mu_targets * exp(scaled X @ beta).

    Pipeline.predict applies the fitted StandardScaler to x_eval, then the
    GridSearchCV.best_estimator_ (a refit TweedieRegressor with the CV-selected
    alpha) applies the inverse-log link (`exp(scaled_X @ beta)`). No manual
    exp() needed.
    """
    mu_ratio: np.ndarray = tweedie_eff.predict(x_eval).astype(np.float64)
    result: np.ndarray = mu_targets * mu_ratio
    return result
