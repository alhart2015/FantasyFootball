# src/projections/backtest/feature_probe.py
"""Feature signal probe — pre-spec screening tool.

Two-phase probe of a candidate feature column or column-set against the
production baseline features. Phase 1 fits per-stat Ridge regressors and
emits a paired-bootstrap CI on the per-stat Δ-CV-RMSE; Phase 2 (gated on
any Phase-1 SIGNAL cell) runs the configured production model class
(BaselineModel or LightGBMNbModel) on both feature sets walk-forward and
emits an adoption-gate-shaped composite verdict.

Pure numpy/scipy/pandas/sklearn. Reuses
``src/projections/backtest/adoption_gate.py``'s paired-bootstrap helpers
unchanged. Consumed by ``scripts/probe_feature_signal.py``.

Spec: docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.linear_model import RidgeCV

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    PositionVerdict,
    paired_bootstrap_rmse_delta,
)
from projections.models import POSITION_DISPATCH
from projections.models.baseline import BaselineModel
from projections.schemas import Position, Stat

# Same alpha grid as BaselineModel.fit (src/projections/models/baseline.py:528).
_RIDGE_ALPHAS = np.logspace(-3, 3, 13)

PerStatLabel = Literal["SIGNAL", "NULL", "REGRESSION"]


@dataclass(frozen=True, slots=True)
class PerStatVerdict:
    """Per-stat per-(year-or-pooled) screening verdict.

    ``verdict == "SIGNAL"`` iff ``rmse_delta.hi_95 < 0`` (candidate strictly
    improves CV-RMSE on this cell). ``REGRESSION`` iff ``lo_95 > 0``. ``NULL``
    otherwise — the bootstrap CI brackets zero, so the per-stat effect is
    indistinguishable from sampling noise on this dataset.
    """

    position: Position
    stat: Stat
    year_or_pooled: int | Literal["pooled"]
    n_paired: int
    rmse_delta: BootstrapDelta
    r_squared_delta: float  # in-sample, diagnostic only
    verdict: PerStatLabel


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Bundled probe report for rendering. ``phase2`` is None iff Phase 1
    returned no SIGNAL cell or the user passed ``--no-composite``."""

    candidate_name: str
    model_class: str
    baseline_features_path: str
    override_paths: tuple[str, ...]
    drop_columns: tuple[str, ...]
    phase1: list[PerStatVerdict]
    phase2: list[PositionVerdict] | None


def phase1_should_fire_phase2(verdicts: list[PerStatVerdict]) -> bool:
    """Return True iff any per-cell or pooled verdict in the Phase-1 result
    set is ``SIGNAL``. NULL and REGRESSION cells do not fire Phase 2 — the
    probe runs Phase 2 only when there's plausibly a real effect to evaluate
    at the composite level."""
    return any(v.verdict == "SIGNAL" for v in verdicts)


def _verdict_for_per_stat(rmse_delta: BootstrapDelta) -> PerStatLabel:
    """SIGNAL if CI strictly below 0; REGRESSION if strictly above 0; NULL otherwise.

    NaN bootstraps (degenerate fits) downgrade to NULL — same conservative
    fallback the adoption gate uses for NaN spearman.
    """
    if (
        not np.isfinite(rmse_delta.point)
        or not np.isfinite(rmse_delta.lo_95)
        or not np.isfinite(rmse_delta.hi_95)
    ):
        return "NULL"
    if rmse_delta.hi_95 < 0.0:
        return "SIGNAL"
    if rmse_delta.lo_95 > 0.0:
        return "REGRESSION"
    return "NULL"


def _coerce_bools(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror BaselineModel._x_frame_with_bool_coercion: bool → int8.

    Probe-side coercion so synthetic candidate columns of dtype bool don't
    poison the Ridge fit.
    """
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype(np.int8)
    return out


def _fit_predict_residuals(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit RidgeCV on (train_x, train_y), predict on test_x, return
    (test residuals, in-sample R² on training set)."""
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(train_x, train_y)
    test_pred = ridge.predict(test_x).astype(np.float64)
    test_residuals = test_y - test_pred
    train_pred = ridge.predict(train_x).astype(np.float64)
    train_resid = train_y - train_pred
    train_y_var = float(((train_y - train_y.mean()) ** 2).sum())
    if train_y_var == 0.0:
        in_sample_r2 = 0.0
    else:
        in_sample_r2 = 1.0 - float((train_resid**2).sum()) / train_y_var
    return test_residuals, in_sample_r2


def _probe_one_stat_one_window(
    *,
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    position: Position,
    stat: Stat,
    baseline_cols: tuple[str, ...],
    candidate_cols: tuple[str, ...],
    train_seasons: tuple[int, ...],
    test_seasons: tuple[int, ...],
    n_bootstrap: int,
    seed: int,
) -> tuple[BootstrapDelta, float]:
    """Inner kernel: inner-join, dropna, fit two Ridges, return
    (paired-bootstrap rmse delta, ΔR²).

    Drops rows with NaN in any baseline-or-candidate feature column or in the
    target stat — both Ridge fits must train and predict on the identical
    post-dropna row set so the bootstrap stays paired.

    ``train_seasons`` and ``test_seasons`` are the train/test season filters;
    callers control whether this is single-year (e.g., test=(2022,)) or
    pooled across multiple years."""
    ws = weekly_stats[weekly_stats["position"] == position.value]
    joined = features.merge(
        ws[["gsis_id", "season", "week", stat.value]],
        on=["gsis_id", "season", "week"],
        how="inner",
        validate="one_to_one",
    )
    needed_cols = sorted({*baseline_cols, *candidate_cols, stat.value})
    joined = joined.dropna(subset=needed_cols)

    train_mask = joined["season"].isin(train_seasons).to_numpy()
    test_mask = joined["season"].isin(test_seasons).to_numpy()

    x_baseline = _coerce_bools(joined[list(baseline_cols)]).to_numpy(dtype=np.float64)
    x_candidate = _coerce_bools(joined[list(candidate_cols)]).to_numpy(dtype=np.float64)
    y = joined[stat.value].to_numpy(dtype=np.float64)

    base_resid, base_r2 = _fit_predict_residuals(
        train_x=x_baseline[train_mask],
        train_y=y[train_mask],
        test_x=x_baseline[test_mask],
        test_y=y[test_mask],
    )
    cand_resid, cand_r2 = _fit_predict_residuals(
        train_x=x_candidate[train_mask],
        train_y=y[train_mask],
        test_x=x_candidate[test_mask],
        test_y=y[test_mask],
    )
    rmse_delta = paired_bootstrap_rmse_delta(
        base_resid, cand_resid, n_bootstrap=n_bootstrap, seed=seed
    )
    return rmse_delta, cand_r2 - base_r2


def probe_per_stat(
    *,
    position: Position,
    features_baseline_cols: tuple[str, ...],
    features_candidate_cols: tuple[str, ...],
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    target_stats: tuple[Stat, ...],
    holdout_years: tuple[int, ...],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[PerStatVerdict]:
    """Per-stat Ridge ΔRMSE bootstrap.

    For each (stat, holdout_year) and one ``year_or_pooled="pooled"`` row per
    stat, fit Ridge on ``[min(features.season), holdout_year - 1]`` rows with
    ``baseline_cols`` and again with ``candidate_cols``, predict on
    ``holdout_year`` rows, and emit a paired-bootstrap CI on the per-stat
    Δ-CV-RMSE. The pooled row resamples all ``holdout_years`` together.

    Both feature sets are evaluated on the **same** post-dropna row set —
    a row that is NaN on the candidate but valid on the baseline is dropped
    from both sides so the bootstrap stays paired.
    """
    min_season = int(features["season"].min())
    out: list[PerStatVerdict] = []
    for stat in target_stats:
        for holdout in holdout_years:
            train_seasons = tuple(range(min_season, holdout))
            rmse_delta, r2_delta = _probe_one_stat_one_window(
                features=features,
                weekly_stats=weekly_stats,
                position=position,
                stat=stat,
                baseline_cols=features_baseline_cols,
                candidate_cols=features_candidate_cols,
                train_seasons=train_seasons,
                test_seasons=(holdout,),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            out.append(
                PerStatVerdict(
                    position=position,
                    stat=stat,
                    year_or_pooled=holdout,
                    n_paired=rmse_delta.n_paired_rows,
                    rmse_delta=rmse_delta,
                    r_squared_delta=r2_delta,
                    verdict=_verdict_for_per_stat(rmse_delta),
                )
            )

        pooled_train_seasons = tuple(range(min_season, min(holdout_years)))
        pooled_test_seasons = tuple(holdout_years)
        rmse_delta, r2_delta = _probe_one_stat_one_window(
            features=features,
            weekly_stats=weekly_stats,
            position=position,
            stat=stat,
            baseline_cols=features_baseline_cols,
            candidate_cols=features_candidate_cols,
            train_seasons=pooled_train_seasons,
            test_seasons=pooled_test_seasons,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        out.append(
            PerStatVerdict(
                position=position,
                stat=stat,
                year_or_pooled="pooled",
                n_paired=rmse_delta.n_paired_rows,
                rmse_delta=rmse_delta,
                r_squared_delta=r2_delta,
                verdict=_verdict_for_per_stat(rmse_delta),
            )
        )
    return out


def _loosened_features_schema(base: type[pa.DataFrameModel]) -> type[pa.DataFrameModel]:
    """Return a DataFrameModel subclass of ``base`` with ``Config.strict = False``.

    Production schemas use ``strict="filter"``, which silently drops columns
    not declared in the schema. The probe needs candidate columns (which are
    not declared) to survive validation, so it swaps in a loosened schema for
    the duration of one probe invocation. The production schema is untouched.
    """

    class _Loosened(base):  # type: ignore[misc, valid-type]
        class Config(base.Config):  # type: ignore[name-defined, misc]
            strict = False

    return _Loosened


def _build_factory_with_columns(
    *,
    position: Position,
    model_class: str,
    columns: tuple[str, ...],
    feature_schema: type[pa.DataFrameModel],
) -> Callable[[], BaselineModel]:
    """Return a zero-arg factory that produces an unfitted production model
    with overridden ``feature_columns`` and ``feature_schema``.

    The base factory is ``POSITION_DISPATCH[position].factories[model_class]``.
    Each call to the returned factory builds a fresh instance — no shared
    mutable state across calls or with the production factory.
    """
    base_factory = POSITION_DISPATCH[position].factories[model_class]

    def _factory() -> BaselineModel:
        # Cast: factories[model_class] is typed as Callable[[], Model] (the
        # Protocol), but every concrete model class in this codebase carries
        # `feature_columns` and `feature_schema` as mutable @dataclass fields.
        # We pin to BaselineModel for the type signature; LightGBMModel et al.
        # share the same attribute shape so the override works identically.
        model = cast(BaselineModel, base_factory())
        # Concrete models are non-frozen dataclasses; the override pattern
        # deliberately exercises that mutability so the probe can swap in
        # candidate feature_columns / a loosened schema without subclassing.
        model.feature_columns = columns
        model.feature_schema = feature_schema
        return model

    return _factory
