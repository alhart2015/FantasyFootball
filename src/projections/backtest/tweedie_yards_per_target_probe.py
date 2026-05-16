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
from sklearn.linear_model import RidgeCV

# Same alpha grid as BaselineModel.fit (src/projections/models/baseline.py) and
# logit_catch_rate_probe._RIDGE_ALPHAS so the volume + Ridge arm differ from
# production only in the per-stat dispatch, not the regularization scale.
_RIDGE_ALPHAS: Final[np.ndarray] = np.logspace(-3, 3, 13)


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
