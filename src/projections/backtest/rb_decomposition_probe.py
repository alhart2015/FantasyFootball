"""RB rushing + receiving decomposition probe — model architecture probe.

Two-arm probe extending PR #32's target_decomposition_probe to RB across two
shared volume axes:
  - rushing: carries x (yards_per_carry, td_rate_per_carry)
  - receiving: targets x (catch_rate, yards_per_target, td_rate_per_target)

Five composed stats, two shared volume sub-models per training window
(carries, targets). Per-stat Delta-RMSE x 5 stats x walk-forward eval window
2021-2024, paired-bootstrap CI on pooled residuals.

Sub-model = RidgeCV everywhere with predict-time clipping; any SIGNAL is
attributable to decomposition itself, not a model-class change.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.models.baseline import _RIDGE_ALPHA_GRID
from projections.schemas import Stat

# Canonical Ridge alpha grid shared with BaselineModel.fit. The probe's
# incumbent (direct) arm and decomposed arm both use this grid, so any SIGNAL
# is attributable to the decomposition recipe, not a regularization-scale
# difference. PR #44 review-fix: import from baseline, do not redefine.
_RIDGE_ALPHAS: Final[np.ndarray] = _RIDGE_ALPHA_GRID


@dataclass(frozen=True, slots=True)
class _StatDecomp:
    """Per-stat decomposition spec.

    The probe measures one decomposed prediction per stat:
        mu_decomposed[stat] = clip(volume_ridge.predict(X), 0, +inf)
                            * clip(efficiency_ridge.predict(X), 0, efficiency_clip_hi)

    Volume sub-models (one per unique `volume_stat`) are fit once per training
    window. Efficiency sub-models are fit on the volume_stat > 0 row subset.
    """

    volume_stat: Stat
    efficiency_label: str
    efficiency_clip_hi: float
    numerator_stat: Stat


_RB_DECOMPS: Final[dict[Stat, _StatDecomp]] = {
    Stat.RUSHING_YARDS: _StatDecomp(
        volume_stat=Stat.CARRIES,
        efficiency_label="yards_per_carry",
        efficiency_clip_hi=float("inf"),
        numerator_stat=Stat.RUSHING_YARDS,
    ),
    Stat.RUSHING_TDS: _StatDecomp(
        volume_stat=Stat.CARRIES,
        efficiency_label="td_rate_per_carry",
        efficiency_clip_hi=1.0,
        numerator_stat=Stat.RUSHING_TDS,
    ),
    Stat.RECEPTIONS: _StatDecomp(
        volume_stat=Stat.TARGETS,
        efficiency_label="catch_rate",
        efficiency_clip_hi=1.0,
        numerator_stat=Stat.RECEPTIONS,
    ),
    Stat.RECEIVING_YARDS: _StatDecomp(
        volume_stat=Stat.TARGETS,
        efficiency_label="yards_per_target",
        efficiency_clip_hi=float("inf"),
        numerator_stat=Stat.RECEIVING_YARDS,
    ),
    Stat.RECEIVING_TDS: _StatDecomp(
        volume_stat=Stat.TARGETS,
        efficiency_label="td_rate_per_target",
        efficiency_clip_hi=1.0,
        numerator_stat=Stat.RECEIVING_TDS,
    ),
}


def _fit_direct(x: np.ndarray, y: np.ndarray) -> RidgeCV:
    """Fit RidgeCV(_RIDGE_ALPHAS) on (x, y).

    Caller is responsible for: NaN drop, bool-to-int8 coercion of x cols,
    volume-positive filtering when fitting an efficiency factor. Pure: arrays
    in, fitted ridge out.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, y.astype(np.float64))
    return ridge


def _fit_decomposed_volume(x: np.ndarray, volume: np.ndarray) -> RidgeCV:
    """Fit a shared volume sub-model on `volume` directly.

    Trained on UN-FILTERED training rows (zero-volume rows are legitimate
    observations of low-volume / out-of-rotation players).

    Called twice per training window: once with volume = carries (for the
    rushing axis), once with volume = targets (for the receiving axis).
    """
    return _fit_direct(x, volume.astype(np.float64))


def _fit_decomposed_efficiency(
    x: np.ndarray,
    numerator: np.ndarray,
    volume: np.ndarray,
) -> RidgeCV:
    """Fit an efficiency sub-model on rows where volume > 0.

    Ratio = numerator / volume on the masked subset. Caller passes `numerator`
    already aligned with `x` and `volume`; this helper handles the masking.

    Raises:
        ValueError: no rows in the training set have volume > 0.
    """
    mask = volume > 0
    if not mask.any():
        raise ValueError(
            "Cannot fit efficiency factor: no training rows with volume > 0. "
            "Check the training-window filter."
        )
    x_pos = x[mask]
    volume_pos = volume[mask].astype(np.float64)
    numerator_pos = numerator[mask].astype(np.float64)
    ratio = numerator_pos / volume_pos
    return _fit_direct(x_pos, ratio)


def _predict_direct(ridge: RidgeCV, x: np.ndarray) -> np.ndarray:
    """Direct per-row mu prediction. No clipping (matches BaselineModel)."""
    pred: np.ndarray = ridge.predict(x).astype(np.float64)
    return pred


def _predict_decomposed(
    *,
    volume_ridge: RidgeCV,
    efficiency_ridge: RidgeCV,
    x: np.ndarray,
    efficiency_clip_hi: float,
) -> np.ndarray:
    """Decomposed per-row mu prediction.

    mu = clip(volume.predict(x), 0, +inf) * clip(efficiency.predict(x), 0, hi)

    Volume clip floors at 0 (negative carries / targets impossible). Efficiency
    clip floors at 0 for all factors; ceiling is 1.0 for rate factors
    (catch_rate, td_rate_per_carry, td_rate_per_target) and +inf for unbounded
    efficiency (yards_per_carry, yards_per_target).
    """
    volume_raw: np.ndarray = volume_ridge.predict(x).astype(np.float64)
    volume = np.maximum(volume_raw, 0.0)
    eff_raw: np.ndarray = efficiency_ridge.predict(x).astype(np.float64)
    eff = np.clip(eff_raw, 0.0, efficiency_clip_hi)
    result: np.ndarray = volume * eff
    return result
