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
from dataclasses import replace as dataclass_replace
from typing import Final, Literal, cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.linear_model import RidgeCV

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    PositionVerdict,
    paired_bootstrap_rmse_delta,
    paired_bootstrap_spearman_delta,
    verdict_for_position,
)
from projections.models import POSITION_DISPATCH
from projections.models.base import Model
from projections.models.baseline import BaselineModel
from projections.models.lightgbm import LightGBMModel
from projections.schemas import Position, Ruleset, Stat

# Same alpha grid as BaselineModel.fit (src/projections/models/baseline.py:528).
_RIDGE_ALPHAS = np.logspace(-3, 3, 13)

# Default effect-size floor for SIGNAL/REGRESSION verdicts (fpts). Plan 8
# measured the per-cell RMSE noise floor at ~0.08 fpts; 0.05 is a conservative
# threshold that catches all genuine effects while excluding the
# noise-floor wobble that surfaced in Plan 9 retro (e.g., -0.0009 fpts on
# n=3723, statistically significant but ~90x smaller than the noise floor).
_DEFAULT_EFFECT_SIZE_FLOOR: Final[float] = 0.05

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
    """Bundled probe report for rendering.

    ``phase2`` is None iff Phase 2 was skipped, in which case
    ``phase2_skip_reason`` is one of:

    - ``"no_signal"`` — no SIGNAL cells in Phase 1 at all; probe predicts
      DO_NOT_ADOPT.
    - ``"no_pooled_signal"`` — at least one SIGNAL cell, but only at the
      per-year level. Default gating skips Phase 2 because the adoption gate
      operates on pooled statistics; pass ``--force-composite`` to run Phase 2
      anyway (e.g., to test a non-Ridge model class on the same feature set).
    - ``"user_disabled"`` — user passed ``--no-composite``.
    - ``None`` — Phase 2 ran (``phase2`` is non-None).
    """

    candidate_name: str
    model_class: str
    baseline_features_path: str
    override_paths: tuple[str, ...]
    drop_columns: tuple[str, ...]
    phase1: list[PerStatVerdict]
    phase2: list[PositionVerdict] | None
    phase2_skip_reason: str | None = None


def phase1_should_fire_phase2(verdicts: list[PerStatVerdict]) -> bool:
    """Return True iff any POOLED verdict (year_or_pooled == "pooled") in
    the Phase-1 result set is ``SIGNAL``. Per-year cells are informational;
    they do NOT fire Phase 2 because the adoption gate operates on pooled
    statistics — a per-year SIGNAL on a single noisy year is exactly the
    sampling-variation false positive Phase 2 is meant to filter out at the
    composite level."""
    return any(v.verdict == "SIGNAL" and v.year_or_pooled == "pooled" for v in verdicts)


def _verdict_for_per_stat(
    rmse_delta: BootstrapDelta,
    *,
    effect_size_floor: float = _DEFAULT_EFFECT_SIZE_FLOOR,
) -> PerStatLabel:
    """SIGNAL iff CI strictly below 0 AND |point| >= effect_size_floor.
    REGRESSION iff CI strictly above 0 AND |point| >= effect_size_floor.
    NULL otherwise (CI brackets 0, OR effect size below noise floor, OR NaN).

    The effect_size_floor (default 0.05 fpts) is an absolute-magnitude gate
    that prevents tiny statistically-significant effects (often n>>1000 with
    near-zero point estimates) from cluttering the SIGNAL verdict. Plan 8
    measured the per-cell RMSE noise floor at ~0.08 fpts; the default 0.05
    is a conservative threshold that catches all genuine effects while
    excluding noise-floor wobble.
    """
    if (
        not np.isfinite(rmse_delta.point)
        or not np.isfinite(rmse_delta.lo_95)
        or not np.isfinite(rmse_delta.hi_95)
    ):
        return "NULL"
    if abs(rmse_delta.point) < effect_size_floor:
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
    effect_size_floor: float = _DEFAULT_EFFECT_SIZE_FLOOR,
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

    ``effect_size_floor`` (default 0.05 fpts) is the absolute |point|
    threshold for SIGNAL/REGRESSION; below-floor effects collapse to NULL
    even when statistically significant. See ``_verdict_for_per_stat``.
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
                    verdict=_verdict_for_per_stat(rmse_delta, effect_size_floor=effect_size_floor),
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
                verdict=_verdict_for_per_stat(rmse_delta, effect_size_floor=effect_size_floor),
            )
        )
    return out


def _loosened_features_schema(base: type[pa.DataFrameModel]) -> type[pa.DataFrameModel]:
    """Return a DataFrameModel subclass of ``base`` with all column-existence
    checks relaxed:

    - ``Config.strict = False`` so undeclared candidate columns survive
      validation (production schemas use ``strict="filter"``, which silently
      drops them).
    - All declared columns become ``required=False`` so a frame from which a
      column has been ``--drop``'d still validates. This is load-bearing for
      Phase 2 swap-mode runs: ``_build_factory_with_columns`` overrides the
      model's ``feature_columns`` to the candidate set, but the model still
      calls ``feature_schema.validate(features)`` on the post-drop frame.
      Without optional columns, swap-mode Phase 2 errors at validate time on
      every column the user dropped.

    Type/check constraints on columns that ARE present are preserved — only
    column-existence is loosened.

    The production schema is untouched.
    """

    class _Loosened(base):  # type: ignore[misc, valid-type]
        class Config(base.Config):  # type: ignore[name-defined, misc]
            strict = False

        @classmethod
        def to_schema(cls) -> pa.DataFrameSchema:
            schema = super().to_schema()
            updates = {name: {"required": False} for name in schema.columns}
            # update_columns returns DataFrameSchema; pandera lacks a precise
            # type stub, so cast to silence the mypy any-return check.
            return cast(pa.DataFrameSchema, schema.update_columns(updates))

    return _Loosened


def _build_factory_with_columns(
    *,
    position: Position,
    model_class: str,
    columns: tuple[str, ...],
    feature_schema: type[pa.DataFrameModel],
) -> Callable[[], Model]:
    """Return a zero-arg factory that produces an unfitted production model
    with overridden ``feature_columns`` and ``feature_schema``.

    The base factory is ``POSITION_DISPATCH[position].factories[model_class]``.
    Each call to the returned factory builds a fresh instance — no shared
    mutable state across calls or with the production factory.

    Branches on the concrete model class because each one stores feature
    metadata differently:

    - ``BaselineModel``: ``feature_columns`` and ``feature_schema`` are direct
      mutable attributes on the (non-frozen) dataclass; assign them directly.
    - ``LightGBMModel`` (and its subclasses ``LightGBMTunedModel``,
      ``LightGBMNbModel``): both fields live inside a frozen
      ``_LightGBMConfig`` at ``self._config``. Use ``dataclasses.replace`` to
      produce a new config with the overrides and reassign ``self._config``.

    Other model classes raise ``NotImplementedError``. Keeping this loud
    rather than silently no-op-ing prevents the Phase 2 probe from returning
    misleading near-zero ΔRMSE on a model whose feature set didn't actually
    change.
    """
    base_factory = POSITION_DISPATCH[position].factories[model_class]

    def _factory() -> Model:
        model = base_factory()
        if isinstance(model, BaselineModel):
            model.feature_columns = columns
            model.feature_schema = feature_schema
            return model
        if isinstance(model, LightGBMModel):
            model._config = dataclass_replace(
                model._config,
                feature_columns=columns,
                feature_schema=feature_schema,
            )
            return model
        raise NotImplementedError(
            f"feature-column override is not implemented for {type(model).__name__}; "
            "extend _build_factory_with_columns to handle this model class."
        )

    return _factory


# Synthetic class labels for the per-feature-set comparison. Underscore-prefixed
# so they cannot collide with real production model_class values.
_FEATURE_PROBE_INCUMBENT = "_baseline_features"
_FEATURE_PROBE_CANDIDATE = "_candidate_features"


def _per_year_breakdown(
    *,
    actual: np.ndarray,
    predicted_inc: np.ndarray,
    predicted_cand: np.ndarray,
    year: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Per-year (informational) breakdown rows mirroring adoption_gate's CSV shape.

    Columns: year, n_paired, rmse_delta_point, rmse_delta_lo, rmse_delta_hi,
    spearman_delta_point, spearman_delta_lo, spearman_delta_hi.

    Years with fewer than 100 paired rows (the bootstrap floor) emit a
    single NaN-filled row rather than raising — informational breakdown
    must not abort the whole probe just because one year is sparse.
    """
    years = np.unique(year)
    rows: list[dict[str, object]] = []
    for y in years:
        mask = year == y
        if mask.sum() < 100:  # _MIN_PAIRED_ROWS in adoption_gate
            rows.append(
                {
                    "year": int(y),
                    "n_paired": int(mask.sum()),
                    "rmse_delta_point": float("nan"),
                    "rmse_delta_lo": float("nan"),
                    "rmse_delta_hi": float("nan"),
                    "spearman_delta_point": float("nan"),
                    "spearman_delta_lo": float("nan"),
                    "spearman_delta_hi": float("nan"),
                }
            )
            continue
        rmse = paired_bootstrap_rmse_delta(
            (actual[mask] - predicted_inc[mask]),
            (actual[mask] - predicted_cand[mask]),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        # Per-year Spearman uses a single-group "grouping" (the year itself).
        spearman = paired_bootstrap_spearman_delta(
            predicted_inc[mask],
            predicted_cand[mask],
            actual[mask],
            year[mask],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append(
            {
                "year": int(y),
                "n_paired": rmse.n_paired_rows,
                "rmse_delta_point": rmse.point,
                "rmse_delta_lo": rmse.lo_95,
                "rmse_delta_hi": rmse.hi_95,
                "spearman_delta_point": spearman.point,
                "spearman_delta_lo": spearman.lo_95,
                "spearman_delta_hi": spearman.hi_95,
            }
        )
    return pd.DataFrame(rows)


def probe_composite(
    *,
    position: Position,
    factory_baseline: Callable[[], Model],
    factory_candidate: Callable[[], Model],
    features_baseline: pd.DataFrame,
    features_candidate: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    composite_truth_column: str,
    holdout_years: tuple[int, ...],
    ruleset: Ruleset,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PositionVerdict:
    """Walk-forward composite-fpts ΔRMSE bootstrap for one position.

    For each holdout year, fit ``factory_baseline()`` on ``features_baseline``
    rows in ``[min_season, year - 1]`` and predict on rows in ``year``;
    repeat for ``factory_candidate()`` on ``features_candidate``. The per-year
    prediction frames are concatenated and paired with ``weekly_stats``'s
    ``composite_truth_column`` on ``(gsis_id, season, week)``. The pooled
    bootstrap returns a single ``PositionVerdict`` matching the shape produced
    by ``scripts/adoption_gate.py``.

    ``factory_baseline`` and ``factory_candidate`` are zero-arg callables that
    produce an unfitted model. The model must implement
    ``fit(features, weekly_stats)`` and ``predict_distribution(features, ruleset)``
    returning a frame with at least ``gsis_id``, ``season``, ``week``, ``mean``
    columns.

    Args:
        composite_truth_column: name of the per-row truth column in
            ``weekly_stats`` (e.g., ``"fpts"`` for fantasy points under the
            production ruleset).
        ruleset: passed through to ``predict_distribution``; opaque to this
            function.
    """
    min_season_baseline = int(features_baseline["season"].min())
    min_season_candidate = int(features_candidate["season"].min())
    if min_season_baseline != min_season_candidate:
        raise ValueError(
            f"baseline and candidate features must share the same min season; "
            f"got {min_season_baseline} vs {min_season_candidate}"
        )
    min_season = min_season_baseline

    baseline_preds: list[pd.DataFrame] = []
    candidate_preds: list[pd.DataFrame] = []
    for year in holdout_years:
        train_mask_b = features_baseline["season"].between(min_season, year - 1)
        test_mask_b = features_baseline["season"] == year
        train_mask_c = features_candidate["season"].between(min_season, year - 1)
        test_mask_c = features_candidate["season"] == year

        m_baseline = factory_baseline()
        m_baseline.fit(features_baseline[train_mask_b], weekly_stats)
        baseline_preds.append(
            m_baseline.predict_distribution(features_baseline[test_mask_b], ruleset)
        )

        m_candidate = factory_candidate()
        m_candidate.fit(features_candidate[train_mask_c], weekly_stats)
        candidate_preds.append(
            m_candidate.predict_distribution(features_candidate[test_mask_c], ruleset)
        )

    base_pred_df = pd.concat(baseline_preds, ignore_index=True)
    cand_pred_df = pd.concat(candidate_preds, ignore_index=True)

    truth = weekly_stats[["gsis_id", "season", "week", composite_truth_column]].rename(
        columns={composite_truth_column: "actual"}
    )
    base_paired = base_pred_df.merge(truth, on=["gsis_id", "season", "week"], how="inner")
    cand_paired = cand_pred_df.merge(truth, on=["gsis_id", "season", "week"], how="inner")

    keys = ["gsis_id", "season", "week"]
    base_keys = base_paired[keys].sort_values(keys).reset_index(drop=True)
    cand_keys = cand_paired[keys].sort_values(keys).reset_index(drop=True)
    if not base_keys.equals(cand_keys):
        raise ValueError(
            "baseline and candidate prediction frames disagree on row coverage; "
            "this should never happen if both feature frames have identical (gsis_id, "
            "season, week) keys for the holdout years."
        )

    base_paired = base_paired.sort_values(keys).reset_index(drop=True)
    cand_paired = cand_paired.sort_values(keys).reset_index(drop=True)

    actual = base_paired["actual"].to_numpy(dtype=np.float64)
    pred_inc = base_paired["mean"].to_numpy(dtype=np.float64)
    pred_cand = cand_paired["mean"].to_numpy(dtype=np.float64)
    year = base_paired["season"].to_numpy(dtype=np.int64)

    rmse_delta = paired_bootstrap_rmse_delta(
        actual - pred_inc, actual - pred_cand, n_bootstrap=n_bootstrap, seed=seed
    )
    spearman_delta = paired_bootstrap_spearman_delta(
        pred_inc, pred_cand, actual, year, n_bootstrap=n_bootstrap, seed=seed
    )
    label, reason = verdict_for_position(rmse_delta, spearman_delta)

    breakdown = _per_year_breakdown(
        actual=actual,
        predicted_inc=pred_inc,
        predicted_cand=pred_cand,
        year=year,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return PositionVerdict(
        position=position,
        incumbent_class=_FEATURE_PROBE_INCUMBENT,
        candidate_class=_FEATURE_PROBE_CANDIDATE,
        rmse_delta=rmse_delta,
        spearman_delta=spearman_delta,
        verdict=label,
        reason=reason,
        per_year_breakdown=breakdown,
    )
