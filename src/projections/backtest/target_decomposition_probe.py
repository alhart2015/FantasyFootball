"""Target decomposition probe — model architecture probe.

Tests whether decomposing WR receiving stats into a shared `targets` volume
factor times a per-stat efficiency factor beats per-stat direct RidgeCV on
out-of-sample mean prediction. Per-stat Delta-CV-RMSE x 3 stats x walk-forward
eval window 2021-2024, paired-bootstrap CI on pooled residuals.

Architecturally distinct from `feature_probe.py`: that module probes feature
additions via override parquets; this module probes a prediction recipe
change with no override layer. Reuses `paired_bootstrap_rmse_delta` and
`BootstrapDelta` from `adoption_gate.py`.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.schemas import Stat

# Same alpha grid as BaselineModel.fit (src/projections/models/baseline.py:563).
_RIDGE_ALPHAS: Final = np.logspace(-3, 3, 13)


@dataclass(frozen=True, slots=True)
class _StatDecomp:
    """Per-stat decomposition spec.

    The probe measures one decomposed prediction per stat:
        mu_decomposed[stat] = clip(volume_ridge.predict(X), 0, +inf)
                            * clip(efficiency_ridge.predict(X), 0, efficiency_clip_hi)

    The shared volume sub-model is fit once per training window (against
    `volume_stat` directly); each efficiency sub-model is fit on rows where
    `volume_stat > 0` against `numerator_stat / volume_stat`.
    """

    volume_stat: Stat
    efficiency_label: str
    efficiency_clip_hi: float
    numerator_stat: Stat


_WR_RECEIVING_DECOMPS: Final[dict[Stat, _StatDecomp]] = {
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
    targets-positive filtering when fitting an efficiency factor. This helper
    is pure: arrays in, fitted ridge out.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, y.astype(np.float64))
    return ridge


def _fit_decomposed_volume(x: np.ndarray, targets: np.ndarray) -> RidgeCV:
    """Fit the shared volume sub-model on `targets` directly.

    Trained on the un-filtered training rows (zero-target rows are legitimate
    observations of low-volume players, and the volume model needs to predict
    them).
    """
    return _fit_direct(x, targets.astype(np.float64))


def _fit_decomposed_efficiency(
    x: np.ndarray,
    numerator: np.ndarray,
    targets: np.ndarray,
) -> RidgeCV:
    """Fit an efficiency sub-model on rows where targets > 0.

    Ratio = numerator / targets on those rows. Caller passes `numerator`
    already aligned with `x` and `targets`; this helper handles the masking.

    Raises:
        ValueError: no rows in the training set have targets > 0.
    """
    mask = targets > 0
    if not mask.any():
        raise ValueError(
            "Cannot fit efficiency factor: no training rows with targets > 0. "
            "Check the training-window filter."
        )
    x_pos = x[mask]
    targets_pos = targets[mask].astype(np.float64)
    numerator_pos = numerator[mask].astype(np.float64)
    ratio = numerator_pos / targets_pos
    return _fit_direct(x_pos, ratio)


def _predict_direct(ridge: RidgeCV, x: np.ndarray) -> np.ndarray:
    """Direct per-row mu prediction; matches BaselineModel.fit's predict semantics
    (no clipping; downstream Distribution constructor handles family floors).
    """
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

    Both clips engage on the *low* side; the high side engages only for
    bounded-rate efficiency factors (catch_rate, td_rate_per_target with
    efficiency_clip_hi=1.0). yards_per_target uses efficiency_clip_hi=+inf so
    the high side is a no-op.
    """
    volume_raw: np.ndarray = volume_ridge.predict(x).astype(np.float64)
    volume = np.maximum(volume_raw, 0.0)
    eff_raw: np.ndarray = efficiency_ridge.predict(x).astype(np.float64)
    eff = np.clip(eff_raw, 0.0, efficiency_clip_hi)
    result: np.ndarray = volume * eff
    return result
