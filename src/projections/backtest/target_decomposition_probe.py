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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    paired_bootstrap_rmse_delta,
)
from projections.schemas import Ruleset, Stat

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

    Used for the section 5 risk #2 Pearson correlation diagnostic. Volume
    residual is actual_targets - predicted_targets; efficiency residual is
    actual_efficiency_ratio - predicted_efficiency on rows with targets > 0.
    """

    eval_year: int
    volume_residuals: np.ndarray
    efficiency_residuals: np.ndarray


@dataclass(frozen=True, slots=True)
class CoverageByYear:
    """Per-eval-year `targets > 0` rate on WR rows."""

    eval_year: int
    targets_positive_rate: float
    n_eval_rows: int


@dataclass(frozen=True, slots=True)
class WalkForwardOutput:
    """Bundle of all walk-forward outputs.

    `per_stat` keys: the 3 entries in `_WR_RECEIVING_DECOMPS`.
    `factor_residuals_by_year`: per-stat per-year pairs of factor residuals
        for the orthogonality-correlation diagnostic.
    `coverage_by_year`: targets > 0 rate per eval year on the eval rows.
    `train_coverage_by_year`: targets > 0 rate per eval year on the *training*
        rows for that walk-forward iteration. Same threshold applies.
    """

    per_stat: Mapping[Stat, StatResiduals]
    factor_residuals_by_year: Mapping[Stat, Sequence[FactorResidualsByYear]]
    coverage_by_year: Sequence[CoverageByYear]
    train_coverage_by_year: Sequence[CoverageByYear]
    eval_years: tuple[int, ...]


def walk_forward_residuals(
    *,
    features_by_year: Mapping[int, pd.DataFrame],
    weekly_stats: pd.DataFrame,
    feature_columns: Sequence[str],
    eval_years: Sequence[int],
    train_start: int,
) -> WalkForwardOutput:
    """Walk-forward residual collection.

    For each eval year Y in `eval_years`:
        1. Train rows = features ∩ weekly_stats inner-join on (gsis_id, season,
           week), filtered to season in [train_start, Y - 1] and position WR.
        2. Eval rows = same join filtered to season == Y, position WR.
        3. Fit 1 shared volume + 3 efficiency + 3 direct comparators on train rows.
        4. Predict per-row mu for both arms on eval rows.
        5. Append per-row tuples to per-stat residual buffers.
        6. Record per-year (volume, efficiency) factor residuals + coverage.

    Strict separation invariant: train rows for eval Y contain no row from
    season Y. Asserted at runtime for defense in depth.

    Args:
        features_by_year: Per-season WR feature DataFrames (already validated
            against WrFeaturesSchema). Each must have columns {gsis_id, season,
            week, *feature_columns}.
        weekly_stats: Canonical WeeklyStatsSchema-validated frame; will be
            filtered to position == WR internally.
        feature_columns: Ordered tuple of feature columns to use as X. Must
            match `_WR_FEATURE_COLUMNS` for an apples-to-apples comparison vs
            the production WR baseline. Boolean columns are coerced to int8.
        eval_years: Walk-forward eval years (e.g., (2021, 2022, 2023, 2024)).
        train_start: Earliest season included in any training window
            (inclusive lower bound).
    """
    weekly_stats_wr = weekly_stats[weekly_stats["position"] == "WR"].copy()

    per_stat_buffers: dict[Stat, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        stat: [] for stat in _WR_RECEIVING_DECOMPS
    }
    factor_residuals_by_year: dict[Stat, list[FactorResidualsByYear]] = {
        stat: [] for stat in _WR_RECEIVING_DECOMPS
    }
    eval_coverage: list[CoverageByYear] = []
    train_coverage: list[CoverageByYear] = []

    for eval_year in eval_years:
        # Build train + eval row sets via the same join recipe BaselineModel.fit uses.
        train_features = pd.concat(
            [features_by_year[s] for s in features_by_year if train_start <= s <= eval_year - 1],
            ignore_index=True,
        )
        eval_features = features_by_year[eval_year]

        # Inner-join features <-> weekly stats; drop NaN feature rows.
        truth_cols = [
            "gsis_id",
            "season",
            "week",
            Stat.TARGETS.value,
            Stat.RECEPTIONS.value,
            Stat.RECEIVING_YARDS.value,
            Stat.RECEIVING_TDS.value,
        ]
        train_joined = train_features.merge(
            weekly_stats_wr[truth_cols],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        eval_joined = eval_features.merge(
            weekly_stats_wr[truth_cols],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )

        # Strict separation defense in depth.
        if not train_joined.empty:
            assert int(train_joined["season"].max()) < eval_year, (
                f"Train rows for eval {eval_year} contain season >= eval year; "
                f"max train season is {int(train_joined['season'].max())}"
            )
        assert (eval_joined["season"] == eval_year).all(), (
            f"Eval rows for {eval_year} contain rows from other seasons"
        )

        # X matrix construction (mirrors BaselineModel._x_frame_with_bool_coercion).
        train_x_frame = train_joined[list(feature_columns)].copy()
        eval_x_frame = eval_joined[list(feature_columns)].copy()
        for col in train_x_frame.columns:
            if train_x_frame[col].dtype == bool:
                train_x_frame[col] = train_x_frame[col].astype(np.int8)
                eval_x_frame[col] = eval_x_frame[col].astype(np.int8)

        # Persist train medians for predict-time imputation; drop train NaN rows.
        train_medians = train_x_frame.median(skipna=True).astype(float)
        train_keep_mask = train_x_frame.notna().all(axis=1).to_numpy()
        train_x_frame = train_x_frame.loc[train_keep_mask]
        train_truth_keep = train_joined.loc[train_keep_mask]

        eval_x_frame = eval_x_frame.fillna(train_medians)
        x_train = train_x_frame.to_numpy(dtype=np.float64)
        x_eval = eval_x_frame.to_numpy(dtype=np.float64)

        # Targets and per-stat numerators (post-NaN-drop on train).
        targets_train = train_truth_keep[Stat.TARGETS.value].to_numpy(dtype=np.int64)
        targets_eval = eval_joined[Stat.TARGETS.value].to_numpy(dtype=np.int64)

        # Coverage: targets > 0 rates.
        train_coverage.append(
            CoverageByYear(
                eval_year=eval_year,
                targets_positive_rate=float((targets_train > 0).mean())
                if len(targets_train)
                else 0.0,
                n_eval_rows=len(targets_train),
            )
        )
        eval_coverage.append(
            CoverageByYear(
                eval_year=eval_year,
                targets_positive_rate=float((targets_eval > 0).mean())
                if len(targets_eval)
                else 0.0,
                n_eval_rows=len(targets_eval),
            )
        )

        # Fit shared volume sub-model.
        volume_ridge = _fit_decomposed_volume(x_train, targets_train)

        # Per-stat: fit efficiency + direct, predict on eval, record residuals.
        for stat, decomp in _WR_RECEIVING_DECOMPS.items():
            actual_eval = eval_joined[stat.value].to_numpy(dtype=np.float64)
            numerator_train = train_truth_keep[stat.value].to_numpy(dtype=np.int64)

            efficiency_ridge = _fit_decomposed_efficiency(x_train, numerator_train, targets_train)
            direct_ridge = _fit_direct(x_train, numerator_train.astype(np.float64))

            mu_decomposed = _predict_decomposed(
                volume_ridge=volume_ridge,
                efficiency_ridge=efficiency_ridge,
                x=x_eval,
                efficiency_clip_hi=decomp.efficiency_clip_hi,
            )
            mu_direct = _predict_direct(direct_ridge, x_eval)

            per_stat_buffers[stat].append((actual_eval, mu_direct, mu_decomposed))

            # Factor residuals on eval rows where targets > 0
            # (efficiency factor is undefined where targets == 0).
            mask_pos = targets_eval > 0
            volume_pred: np.ndarray = volume_ridge.predict(x_eval[mask_pos]).astype(np.float64)
            volume_resid = targets_eval[mask_pos].astype(np.float64) - volume_pred
            actual_ratio = actual_eval[mask_pos] / targets_eval[mask_pos].astype(np.float64)
            eff_pred: np.ndarray = efficiency_ridge.predict(x_eval[mask_pos]).astype(np.float64)
            predicted_ratio = np.clip(eff_pred, 0.0, decomp.efficiency_clip_hi)
            efficiency_resid = actual_ratio - predicted_ratio
            factor_residuals_by_year[stat].append(
                FactorResidualsByYear(
                    eval_year=eval_year,
                    volume_residuals=volume_resid,
                    efficiency_residuals=efficiency_resid,
                )
            )

    per_stat_residuals = {
        stat: StatResiduals(
            actual=np.concatenate([b[0] for b in buffers]),
            mu_direct=np.concatenate([b[1] for b in buffers]),
            mu_decomposed=np.concatenate([b[2] for b in buffers]),
            n_paired=int(sum(len(b[0]) for b in buffers)),
        )
        for stat, buffers in per_stat_buffers.items()
    }

    return WalkForwardOutput(
        per_stat=per_stat_residuals,
        factor_residuals_by_year=factor_residuals_by_year,
        coverage_by_year=eval_coverage,
        train_coverage_by_year=train_coverage,
        eval_years=tuple(eval_years),
    )


ProbeVerdictLabel = Literal["SIGNAL", "NULL", "REGRESSION"]


def _verdict_from_delta(delta: BootstrapDelta) -> ProbeVerdictLabel:
    """Pure CI-based per-stat verdict (no effect-size floor).

    SIGNAL iff hi_95 < 0 (decomposed strictly improves).
    REGRESSION iff lo_95 > 0 (decomposed strictly regresses).
    NULL otherwise.
    """
    if delta.hi_95 < 0:
        return "SIGNAL"
    if delta.lo_95 > 0:
        return "REGRESSION"
    return "NULL"


@dataclass(frozen=True, slots=True)
class StatProbeVerdict:
    """Per-stat probe verdict + diagnostic numbers."""

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    rmse_direct: float
    rmse_decomposed: float
    verdict: ProbeVerdictLabel
    expected_composite_fpts_delta: float
    """RMSE delta translated to expected composite-fpts contribution
    (rmse_delta.point x scoring coefficient for this stat under ESPN PPR).
    Per section 5 risk #1: surfaces probe-vs-gate magnitude calibration."""
    factor_residual_correlation_by_year: Mapping[int, float]
    """Per-eval-year Pearson rho between volume residual and efficiency residual.
    |rho| > 0.2 in any year is a documented caveat per section 5 risk #2."""


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Bundle of all per-stat verdicts + the walk-forward output for context.

    Renders to:
    - 1 summary markdown (hand-written from this struct; see Task 5).
    - 1 per-stat csv.
    - 3 per-stat markdown details.
    """

    verdicts: Mapping[Stat, StatProbeVerdict]
    walk_forward: WalkForwardOutput
    bootstrap_n: int
    seed: int


def _rmse(actual: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - mu) ** 2)))


def _stat_scoring_coefficient(stat: Stat) -> float:
    """ESPN PPR coefficient for `stat`. Mirrors
    `projections.scoring.score_distribution._scoring_coefficients`.

    Used to translate per-stat RMSE delta to expected composite-fpts impact:
    a per-stat RMSE delta of -0.1 yds with coefficient 0.1 fpt/yd implies
    the stat contributes roughly -0.01 fpts to composite-fpts RMSE (rough,
    not exact -- composite-fpts RMSE depends on cross-stat covariance).
    """
    ruleset = Ruleset.espn_ppr()
    coefficients = {
        Stat.RECEPTIONS: ruleset.reception_pts,
        Stat.RECEIVING_YARDS: 1.0 / ruleset.receiving_yds_per_pt,
        Stat.RECEIVING_TDS: ruleset.receiving_td_pts,
    }
    return coefficients[stat]


def _factor_residual_correlation(
    factor_residuals: Sequence[FactorResidualsByYear],
) -> dict[int, float]:
    """Per-eval-year Pearson rho of (volume_residual, efficiency_residual)."""
    out: dict[int, float] = {}
    for entry in factor_residuals:
        if len(entry.volume_residuals) < 2:
            out[entry.eval_year] = float("nan")
            continue
        vol = entry.volume_residuals
        eff = entry.efficiency_residuals
        std_v = float(vol.std(ddof=0))
        std_e = float(eff.std(ddof=0))
        if std_v == 0.0 or std_e == 0.0:
            out[entry.eval_year] = float("nan")
            continue
        cov_matrix: np.ndarray = np.cov(vol, eff, ddof=0)
        rho = float(cov_matrix[0, 1]) / (std_v * std_e)
        out[entry.eval_year] = rho
    return out


def render_probe_report(
    walk_forward: WalkForwardOutput,
    *,
    bootstrap_n: int = 5000,
    seed: int = 0xD3C0,
) -> ProbeReport:
    """Per-stat verdicts via paired-bootstrap on the pooled residuals.

    Args:
        walk_forward: output from `walk_forward_residuals`.
        bootstrap_n: paired-bootstrap resample count (default 5000, matches
            `feature_probe.py`).
        seed: RNG seed for the bootstrap; deterministic across runs.
    """
    verdicts: dict[Stat, StatProbeVerdict] = {}
    for stat, residuals in walk_forward.per_stat.items():
        # paired_bootstrap_rmse_delta consumes residual arrays (actual - mu);
        # compute them up front from the per-row buffers.
        residuals_direct = residuals.actual - residuals.mu_direct
        residuals_decomposed = residuals.actual - residuals.mu_decomposed
        # Sign convention: candidate - incumbent. We name "decomposed" the
        # candidate (we want it to win), so direct is incumbent.
        delta = paired_bootstrap_rmse_delta(
            residuals_incumbent=residuals_direct,
            residuals_candidate=residuals_decomposed,
            n_bootstrap=bootstrap_n,
            seed=seed,
        )
        rmse_direct = _rmse(residuals.actual, residuals.mu_direct)
        rmse_decomposed = _rmse(residuals.actual, residuals.mu_decomposed)
        coef = _stat_scoring_coefficient(stat)
        per_year_rho = _factor_residual_correlation(walk_forward.factor_residuals_by_year[stat])
        verdicts[stat] = StatProbeVerdict(
            stat=stat,
            n_paired=residuals.n_paired,
            rmse_delta=delta,
            rmse_direct=rmse_direct,
            rmse_decomposed=rmse_decomposed,
            verdict=_verdict_from_delta(delta),
            expected_composite_fpts_delta=delta.point * coef,
            factor_residual_correlation_by_year=per_year_rho,
        )

    return ProbeReport(
        verdicts=verdicts,
        walk_forward=walk_forward,
        bootstrap_n=bootstrap_n,
        seed=seed,
    )


def write_per_stat_csv(report: ProbeReport, path: Path) -> None:
    """Write a 3-row CSV: one row per decomposed stat with verdict + diagnostics."""
    rows: list[dict[str, object]] = []
    for stat, v in report.verdicts.items():
        rows.append(
            {
                "stat": stat.value,
                "n_paired": v.n_paired,
                "rmse_direct": v.rmse_direct,
                "rmse_decomposed": v.rmse_decomposed,
                "rmse_delta_point": v.rmse_delta.point,
                "rmse_delta_lo_95": v.rmse_delta.lo_95,
                "rmse_delta_hi_95": v.rmse_delta.hi_95,
                "verdict": v.verdict,
                "expected_composite_fpts_delta": v.expected_composite_fpts_delta,
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_per_stat_markdown(report: ProbeReport, stat: Stat, path: Path) -> None:
    """Render the per-stat detail markdown."""
    v = report.verdicts[stat]
    coverage = {c.eval_year: c for c in report.walk_forward.coverage_by_year}
    train_coverage = {c.eval_year: c for c in report.walk_forward.train_coverage_by_year}
    eval_years = report.walk_forward.eval_years

    lines: list[str] = []
    lines.append(f"# Target Decomposition Probe — {stat.value}")
    lines.append("")
    lines.append(f"**Verdict:** {v.verdict}")
    lines.append("")
    lines.append("## Pooled per-stat verdict")
    lines.append("")
    lines.append("| n_paired | RMSE direct | RMSE decomposed | Delta-RMSE | 95% CI | Verdict |")
    lines.append("|---:|---:|---:|---:|---|:---:|")
    lines.append(
        f"| {v.n_paired} | {v.rmse_direct:.4f} | {v.rmse_decomposed:.4f} | "
        f"{v.rmse_delta.point:+.4f} | "
        f"[{v.rmse_delta.lo_95:+.4f}, {v.rmse_delta.hi_95:+.4f}] | "
        f"**{v.verdict}** |"
    )
    lines.append("")
    lines.append(
        f"**Expected composite-fpts Delta (rough)**: "
        f"{v.expected_composite_fpts_delta:+.4f} fpts "
        f"(stat RMSE Delta x ESPN PPR coefficient {_stat_scoring_coefficient(stat):+.4f}). "
        f"Per section 5 risk #1, magnitudes < 0.005 fpts under coverage relaxation should "
        f"be treated as MARGINAL, not SIGNAL."
    )
    lines.append("")
    lines.append("## Per-eval-year coverage")
    lines.append("")
    lines.append("| Year | Eval n | Eval (targets > 0) | Train n | Train (targets > 0) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for year in eval_years:
        e = coverage[year]
        t = train_coverage[year]
        lines.append(
            f"| {year} | {e.n_eval_rows} | {e.targets_positive_rate:.3f} | "
            f"{t.n_eval_rows} | {t.targets_positive_rate:.3f} |"
        )
    lines.append("")
    lines.append("## Factor residual correlation (Pearson rho)")
    lines.append("")
    lines.append(
        "Per-eval-year Pearson rho between (predicted-volume residual, "
        "predicted-efficiency residual) on rows with targets > 0. |rho| > 0.2 in "
        "any year is a documented caveat per section 5 risk #2."
    )
    lines.append("")
    lines.append("| Year | rho |")
    lines.append("|---:|---:|")
    for year in eval_years:
        rho = v.factor_residual_correlation_by_year.get(year, float("nan"))
        lines.append(f"| {year} | {rho:+.3f} |")
    lines.append("")
    lines.append(f"_Bootstrap n_resamples = {report.bootstrap_n}, seed = {report.seed}._")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
