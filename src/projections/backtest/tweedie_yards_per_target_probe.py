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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, TweedieRegressor
from sklearn.metrics import make_scorer, mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    paired_bootstrap_rmse_delta,
)
from projections.models.baseline import _RIDGE_ALPHA_GRID, _WR_FEATURE_COLUMNS
from projections.schemas import Position, Stat, WeeklyStatsSchema

VerdictLabel = Literal["SIGNAL", "NULL", "REGRESSION"]

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


@dataclass(slots=True)
class ProbeResults:
    """Pooled per-row buffers from a walk-forward run.

    Attributes:
        actual_yards: (N,) ground-truth receiving_yards (float64).
        pred_ridge: (N,) incumbent-arm receiving_yards predictions.
        pred_tweedie: (N,) candidate-arm receiving_yards predictions.
        year: (N,) eval year per row (int64).
        coverage_per_year: per-eval-year fraction of WR rows with targets > 0.
    """

    actual_yards: np.ndarray
    pred_ridge: np.ndarray
    pred_tweedie: np.ndarray
    year: np.ndarray
    coverage_per_year: dict[int, float]


@dataclass(slots=True, frozen=True)
class PerStatVerdict:
    """Per-stat verdict on the receiving_yards Delta-RMSE (tweedie minus ridge).

    Mirrors `logit_catch_rate_probe.PerStatVerdict` but is local to this module
    so the probe is self-contained.
    """

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    verdict: VerdictLabel


def _verdict_from_delta(delta: BootstrapDelta) -> VerdictLabel:
    """Map a paired-bootstrap RMSE delta to a verdict label.

    SIGNAL iff hi_95 < 0 (tweedie strictly improves over ridge).
    REGRESSION iff lo_95 > 0 (tweedie strictly regresses).
    NULL otherwise (CI brackets zero).
    """
    if delta.hi_95 < 0:
        return "SIGNAL"
    if delta.lo_95 > 0:
        return "REGRESSION"
    return "NULL"


def walk_forward_residuals(
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    eval_years: Iterable[int],
) -> ProbeResults:
    """For each eval year, train both arms on prior seasons, predict on the
    eval year, collect per-row residuals.

    Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md
    section 3.1 walk_forward_residuals.

    The caller is responsible for schema-validating `features`. `weekly_stats`
    is re-validated here so the join-key columns and dtypes are guaranteed.

    Returns:
        ProbeResults with pooled per-row predictions/actuals across all eval
        years and a per-eval-year `targets > 0` coverage map. If an eval year
        has no rows after the (features, weekly_stats) inner-join, the year is
        skipped and omitted from `coverage_per_year`.
    """
    eval_years_list = sorted(int(y) for y in eval_years)
    actual_buffer: list[np.ndarray] = []
    ridge_buffer: list[np.ndarray] = []
    tweedie_buffer: list[np.ndarray] = []
    year_buffer: list[np.ndarray] = []
    coverage_per_year: dict[int, float] = {}

    ws = WeeklyStatsSchema.validate(weekly_stats)
    ws_wr = ws[ws["position"] == Position.WR.value]

    all_seasons = sorted(int(s) for s in features["season"].unique())
    feat_cols = list(_WR_FEATURE_COLUMNS)

    def _join_and_filter(season_mask: pd.Series) -> pd.DataFrame | None:
        """Inner-join WR features <-> weekly_stats on (gsis_id, season, week),
        then drop rows with any NaN feature column. Returns None if the join
        is empty after filtering (so the caller can skip the fold)."""
        feat_slice = features.loc[season_mask]
        ws_slice = ws_wr.loc[ws_wr["season"].isin(feat_slice["season"].unique())]
        joined = feat_slice.merge(
            ws_slice[["gsis_id", "season", "week", "targets", "receiving_yards"]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        joined = joined.loc[joined[feat_cols].notna().all(axis=1)]
        return joined if not joined.empty else None

    for eval_year in eval_years_list:
        train_seasons = [s for s in all_seasons if s < eval_year]
        if not train_seasons:
            continue

        train_join = _join_and_filter(features["season"].isin(train_seasons))
        if train_join is None:
            continue

        x_train = train_join[feat_cols].to_numpy(dtype=np.float64)
        targets_train = train_join["targets"].to_numpy(dtype=np.int64)
        yards_train = train_join["receiving_yards"].to_numpy(dtype=np.float64)

        # Shared volume fit on all rows (no targets > 0 filter); zero-target
        # rows are valid low-volume observations.
        volume = _fit_shared_volume(x_train, targets_train)

        # Efficiency fits on rows with targets > 0 (ratio undefined at zero).
        pos_mask = targets_train > 0
        x_pos = x_train[pos_mask]
        targets_pos = targets_train[pos_mask].astype(np.float64)
        yards_pos = yards_train[pos_mask]
        if x_pos.shape[0] == 0:
            continue
        ratio_pos = yards_pos / targets_pos

        ridge_eff = _fit_ridge_efficiency(x_pos, ratio_pos)
        tweedie_eff = _fit_tweedie_efficiency(x_pos, ratio_pos)

        eval_join = _join_and_filter(features["season"] == eval_year)
        if eval_join is None:
            continue

        x_eval = eval_join[feat_cols].to_numpy(dtype=np.float64)
        eval_targets = eval_join["targets"].to_numpy(dtype=np.int64)
        eval_yards = eval_join["receiving_yards"].to_numpy(dtype=np.float64)

        mu_targets = volume.predict(x_eval).astype(np.float64)
        pred_ridge = _predict_yards_ridge(mu_targets, x_eval, ridge_eff)
        pred_tweedie = _predict_yards_tweedie(mu_targets, x_eval, tweedie_eff)

        actual_buffer.append(eval_yards)
        ridge_buffer.append(pred_ridge)
        tweedie_buffer.append(pred_tweedie)
        year_buffer.append(np.full(eval_yards.shape, eval_year, dtype=np.int64))

        # Coverage: fraction of eval rows with targets > 0.
        coverage_per_year[eval_year] = (
            float((eval_targets > 0).mean()) if eval_targets.size > 0 else 0.0
        )

    return ProbeResults(
        actual_yards=(
            np.concatenate(actual_buffer) if actual_buffer else np.array([], dtype=np.float64)
        ),
        pred_ridge=(
            np.concatenate(ridge_buffer) if ridge_buffer else np.array([], dtype=np.float64)
        ),
        pred_tweedie=(
            np.concatenate(tweedie_buffer) if tweedie_buffer else np.array([], dtype=np.float64)
        ),
        year=np.concatenate(year_buffer) if year_buffer else np.array([], dtype=np.int64),
        coverage_per_year=coverage_per_year,
    )


def compute_verdict(
    results: ProbeResults, *, n_bootstrap: int = 1000, seed: int = 42
) -> PerStatVerdict:
    """Pooled paired-bootstrap CI on the receiving_yards Delta-RMSE (tweedie minus ridge).

    The signed residuals fed to `paired_bootstrap_rmse_delta` are (actual - pred);
    that function computes RMSE on each arm and returns (candidate - incumbent),
    which matches our convention (tweedie - ridge).
    """
    inc_residuals = results.actual_yards - results.pred_ridge
    cand_residuals = results.actual_yards - results.pred_tweedie
    rmse_delta = paired_bootstrap_rmse_delta(
        inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
    )

    return PerStatVerdict(
        stat=Stat.RECEIVING_YARDS,
        n_paired=int(results.actual_yards.shape[0]),
        rmse_delta=rmse_delta,
        verdict=_verdict_from_delta(rmse_delta),
    )
