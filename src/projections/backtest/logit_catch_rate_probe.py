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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
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
# incumbent arm mirrors `DecomposedBaselineModel`'s catch_rate fit; the alphas
# MUST match or the comparison is no longer ridge-vs-ridge.
_RIDGE_ALPHAS: Final[np.ndarray] = _RIDGE_ALPHA_GRID

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

    x_trials = np.repeat(x_kept, trials_kept, axis=0)

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


@dataclass(slots=True)
class ProbeResults:
    """Pooled per-row buffers from a walk-forward run.

    Attributes:
        actual_receptions: (N,) ground-truth receptions (float64 for residual math).
        pred_ridge: (N,) incumbent-arm receptions predictions.
        pred_logit: (N,) candidate-arm receptions predictions.
        year: (N,) eval year per row (int64).
        coverage_per_year: per-eval-year fraction of WR rows with targets > 0.
    """

    actual_receptions: np.ndarray
    pred_ridge: np.ndarray
    pred_logit: np.ndarray
    year: np.ndarray
    coverage_per_year: dict[int, float]


@dataclass(slots=True, frozen=True)
class PerStatVerdict:
    """Per-stat verdict on the receptions Delta-RMSE (logit minus ridge).

    Mirrors `feature_probe.PerStatVerdict` but is local to this module so the
    probe is self-contained.
    """

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    verdict: VerdictLabel


def _verdict_from_delta(delta: BootstrapDelta) -> VerdictLabel:
    """Map a paired-bootstrap RMSE delta to a verdict label.

    SIGNAL iff hi_95 < 0 (logit strictly improves over ridge).
    REGRESSION iff lo_95 > 0 (logit strictly regresses).
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

    Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md
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
    logit_buffer: list[np.ndarray] = []
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
            ws_slice[["gsis_id", "season", "week", "targets", "receptions"]],
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
        receptions_train = train_join["receptions"].to_numpy(dtype=np.int64)

        volume = _fit_shared_volume(x_train, targets_train)

        # Efficiency fits on rows with targets > 0 (catch_rate is undefined
        # at zero targets, matching DecomposedBaselineModel).
        pos_mask = targets_train > 0
        x_pos = x_train[pos_mask]
        targets_pos = targets_train[pos_mask]
        receptions_pos = receptions_train[pos_mask]
        if x_pos.shape[0] == 0:
            continue
        ratio_pos = receptions_pos.astype(np.float64) / targets_pos.astype(np.float64)

        ridge_eff = _fit_ridge_efficiency(x_pos, ratio_pos)

        x_trials, y_trials = _expand_to_trials(x_pos, receptions_pos, targets_pos)
        logit_eff = _fit_logit_efficiency(x_trials, y_trials)

        eval_join = _join_and_filter(features["season"] == eval_year)
        if eval_join is None:
            continue

        x_eval = eval_join[feat_cols].to_numpy(dtype=np.float64)
        eval_targets = eval_join["targets"].to_numpy(dtype=np.int64)
        eval_receptions = eval_join["receptions"].to_numpy(dtype=np.float64)

        mu_targets = volume.predict(x_eval).astype(np.float64)
        pred_ridge = _predict_receptions_ridge(mu_targets, x_eval, ridge_eff)
        pred_logit = _predict_receptions_logit(mu_targets, x_eval, logit_eff)

        actual_buffer.append(eval_receptions)
        ridge_buffer.append(pred_ridge)
        logit_buffer.append(pred_logit)
        year_buffer.append(np.full(eval_receptions.shape, eval_year, dtype=np.int64))

        # Coverage: fraction of eval rows with targets > 0.
        coverage_per_year[eval_year] = (
            float((eval_targets > 0).mean()) if eval_targets.size > 0 else 0.0
        )

    return ProbeResults(
        actual_receptions=(
            np.concatenate(actual_buffer) if actual_buffer else np.array([], dtype=np.float64)
        ),
        pred_ridge=(
            np.concatenate(ridge_buffer) if ridge_buffer else np.array([], dtype=np.float64)
        ),
        pred_logit=(
            np.concatenate(logit_buffer) if logit_buffer else np.array([], dtype=np.float64)
        ),
        year=np.concatenate(year_buffer) if year_buffer else np.array([], dtype=np.int64),
        coverage_per_year=coverage_per_year,
    )


def compute_verdict(
    results: ProbeResults, *, n_bootstrap: int = 1000, seed: int = 42
) -> PerStatVerdict:
    """Pooled paired-bootstrap CI on the receptions Delta-RMSE (logit minus ridge).

    The signed residuals fed to `paired_bootstrap_rmse_delta` are (actual - pred);
    that function computes RMSE on each arm and returns (candidate - incumbent),
    which matches our convention (logit - ridge).
    """
    inc_residuals = results.actual_receptions - results.pred_ridge
    cand_residuals = results.actual_receptions - results.pred_logit
    rmse_delta = paired_bootstrap_rmse_delta(
        inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
    )

    return PerStatVerdict(
        stat=Stat.RECEPTIONS,
        n_paired=int(results.actual_receptions.shape[0]),
        rmse_delta=rmse_delta,
        verdict=_verdict_from_delta(rmse_delta),
    )
