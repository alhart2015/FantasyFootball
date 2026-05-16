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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.models.baseline import _RB_FEATURE_COLUMNS, _RIDGE_ALPHA_GRID
from projections.schemas import Position, Stat, WeeklyStatsSchema

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


@dataclass(frozen=True, slots=True)
class StatResiduals:
    """Pooled residuals for one decomposed stat across all eval years.

    Each array is row-aligned: row i is the same eval-year row in both arms.
    """

    actual: np.ndarray
    mu_direct: np.ndarray
    mu_decomposed: np.ndarray
    n_paired: int


@dataclass(frozen=True, slots=True)
class FactorResidualsByYear:
    """Per-eval-year (volume_residual, efficiency_residual) pairs for one stat.

    Used for the spec section 5 risk #2 orthogonality diagnostic. Volume residual
    is actual_volume - predicted_volume; efficiency residual is
    actual_efficiency_ratio - predicted_efficiency on rows with volume > 0.
    """

    eval_year: int
    volume_residuals: np.ndarray
    efficiency_residuals: np.ndarray


@dataclass(frozen=True, slots=True)
class WalkForwardOutput:
    """Bundle of all walk-forward outputs.

    Attributes:
        per_stat: {stat: StatResiduals} for the 5 entries in _RB_DECOMPS.
        factor_residuals_by_year: {stat: [FactorResidualsByYear per eval year]}.
        coverage_carries_by_year: per-eval-year carries > 0 rate (eval rows).
        coverage_targets_by_year: per-eval-year targets > 0 rate (eval rows).
        eval_years: the eval years actually computed (skipped years with no
            training data are absent from the coverage dicts).
    """

    per_stat: Mapping[Stat, StatResiduals]
    factor_residuals_by_year: Mapping[Stat, Sequence[FactorResidualsByYear]]
    coverage_carries_by_year: dict[int, float]
    coverage_targets_by_year: dict[int, float]
    eval_years: tuple[int, ...]


def walk_forward_residuals(
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    eval_years: Iterable[int],
) -> WalkForwardOutput:
    """For each eval year, train both arms on prior seasons, predict on the
    eval year, collect per-row residuals + per-volume-axis coverage stats.

    Two shared volume sub-models per training window (carries, targets);
    five efficiency sub-models on their respective volume > 0 subsets;
    five direct comparator sub-models on the full train rows. Eval predictions
    use the same X matrix for both arms.

    Spec: probe-design section 3.3.
    """
    eval_years_list = sorted(int(y) for y in eval_years)
    features_validated = features  # caller is responsible for schema validation
    ws = WeeklyStatsSchema.validate(weekly_stats)
    ws_rb = ws[ws["position"] == Position.RB.value].copy()

    per_stat_buffers: dict[Stat, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        stat: [] for stat in _RB_DECOMPS
    }
    factor_residuals_by_year: dict[Stat, list[FactorResidualsByYear]] = {
        stat: [] for stat in _RB_DECOMPS
    }
    coverage_carries_by_year: dict[int, float] = {}
    coverage_targets_by_year: dict[int, float] = {}

    all_seasons = sorted(int(s) for s in features_validated["season"].unique())
    feat_cols = list(_RB_FEATURE_COLUMNS)
    # The seven stat columns the harness joins from weekly_stats.
    stat_cols = [
        Stat.CARRIES.value,
        Stat.TARGETS.value,
        Stat.RUSHING_YARDS.value,
        Stat.RUSHING_TDS.value,
        Stat.RECEPTIONS.value,
        Stat.RECEIVING_YARDS.value,
        Stat.RECEIVING_TDS.value,
    ]

    def _join_and_filter(seasons: list[int]) -> pd.DataFrame | None:
        """Inner-join features <-> weekly_stats, drop NaN feature rows."""
        if not seasons:
            return None
        feat_slice = features_validated[features_validated["season"].isin(seasons)]
        ws_slice = ws_rb[ws_rb["season"].isin(seasons)]
        joined = feat_slice.merge(
            ws_slice[["gsis_id", "season", "week", *stat_cols]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            return None
        keep_mask = joined[feat_cols].notna().all(axis=1)
        joined = joined.loc[keep_mask]
        return joined if not joined.empty else None

    for eval_year in eval_years_list:
        train_seasons = [s for s in all_seasons if s < eval_year]
        train_join = _join_and_filter(train_seasons)
        eval_join = _join_and_filter([eval_year])
        if train_join is None or eval_join is None:
            continue

        # Boolean -> int8 coercion (matches BaselineModel.fit recipe).
        train_x_frame = train_join[feat_cols].copy()
        eval_x_frame = eval_join[feat_cols].copy()
        for col in train_x_frame.columns:
            if train_x_frame[col].dtype == bool:
                train_x_frame[col] = train_x_frame[col].astype(np.int8)
                eval_x_frame[col] = eval_x_frame[col].astype(np.int8)
        x_train = train_x_frame.to_numpy(dtype=np.float64)
        x_eval = eval_x_frame.to_numpy(dtype=np.float64)

        carries_train = train_join[Stat.CARRIES.value].to_numpy(dtype=np.int64)
        targets_train = train_join[Stat.TARGETS.value].to_numpy(dtype=np.int64)
        carries_eval = eval_join[Stat.CARRIES.value].to_numpy(dtype=np.int64)
        targets_eval = eval_join[Stat.TARGETS.value].to_numpy(dtype=np.int64)

        # Two shared volume sub-models.
        volume_carries = _fit_decomposed_volume(x_train, carries_train)
        volume_targets = _fit_decomposed_volume(x_train, targets_train)
        volume_ridges: dict[Stat, RidgeCV] = {
            Stat.CARRIES: volume_carries,
            Stat.TARGETS: volume_targets,
        }

        # Per-stat: fit direct + efficiency, predict on eval rows.
        for stat, decomp in _RB_DECOMPS.items():
            numerator_train = train_join[stat.value].to_numpy(dtype=np.float64)
            volume_train = train_join[decomp.volume_stat.value].to_numpy(dtype=np.int64)

            direct_ridge = _fit_direct(x_train, numerator_train)
            efficiency_ridge = _fit_decomposed_efficiency(
                x_train, numerator_train.astype(np.int64), volume_train
            )

            mu_direct = _predict_direct(direct_ridge, x_eval)
            mu_decomposed = _predict_decomposed(
                volume_ridge=volume_ridges[decomp.volume_stat],
                efficiency_ridge=efficiency_ridge,
                x=x_eval,
                efficiency_clip_hi=decomp.efficiency_clip_hi,
            )
            actual_eval = eval_join[stat.value].to_numpy(dtype=np.float64)

            per_stat_buffers[stat].append((actual_eval, mu_direct, mu_decomposed))

            # Factor residuals on eval rows where the relevant volume > 0.
            volume_eval = carries_eval if decomp.volume_stat is Stat.CARRIES else targets_eval
            mask_pos = volume_eval > 0
            if mask_pos.any():
                volume_pred: np.ndarray = (
                    volume_ridges[decomp.volume_stat].predict(x_eval[mask_pos]).astype(np.float64)
                )
                volume_resid = volume_eval[mask_pos].astype(np.float64) - volume_pred
                actual_ratio = actual_eval[mask_pos] / volume_eval[mask_pos].astype(np.float64)
                eff_pred: np.ndarray = efficiency_ridge.predict(x_eval[mask_pos]).astype(np.float64)
                predicted_ratio = np.clip(eff_pred, 0.0, decomp.efficiency_clip_hi)
                efficiency_resid = actual_ratio - predicted_ratio
            else:
                volume_resid = np.array([], dtype=np.float64)
                efficiency_resid = np.array([], dtype=np.float64)

            factor_residuals_by_year[stat].append(
                FactorResidualsByYear(
                    eval_year=eval_year,
                    volume_residuals=volume_resid,
                    efficiency_residuals=efficiency_resid,
                )
            )

        # Coverage on eval rows.
        coverage_carries_by_year[eval_year] = (
            float((carries_eval > 0).mean()) if carries_eval.size > 0 else 0.0
        )
        coverage_targets_by_year[eval_year] = (
            float((targets_eval > 0).mean()) if targets_eval.size > 0 else 0.0
        )

    per_stat_residuals: dict[Stat, StatResiduals] = {}
    for stat, buffers in per_stat_buffers.items():
        if not buffers:
            per_stat_residuals[stat] = StatResiduals(
                actual=np.array([], dtype=np.float64),
                mu_direct=np.array([], dtype=np.float64),
                mu_decomposed=np.array([], dtype=np.float64),
                n_paired=0,
            )
            continue
        per_stat_residuals[stat] = StatResiduals(
            actual=np.concatenate([b[0] for b in buffers]),
            mu_direct=np.concatenate([b[1] for b in buffers]),
            mu_decomposed=np.concatenate([b[2] for b in buffers]),
            n_paired=int(sum(len(b[0]) for b in buffers)),
        )

    return WalkForwardOutput(
        per_stat=per_stat_residuals,
        factor_residuals_by_year=factor_residuals_by_year,
        coverage_carries_by_year=coverage_carries_by_year,
        coverage_targets_by_year=coverage_targets_by_year,
        eval_years=tuple(eval_years_list),
    )
