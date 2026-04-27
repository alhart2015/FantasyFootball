"""Plan 3e Phase 0 — calibration diagnostic CLI.

Reads the most recent data/backtest/run_<ts>/results.parquet, computes
per-(position, stat) residual diagnostics for the held-out years,
fits 2-3 alternative families per cell with AIC ranking, and writes
structured artifacts to data/diagnostics/calibration_<ts>/.

Spec: docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    unpack_per_stat_params,
)
from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import DistributionFamily, Stat

# Force a non-interactive backend so this script runs in headless environments
# (CI, cron). Calling switch_backend after all imports keeps ruff E402 happy
# without noqa and avoids importing matplotlib twice (vs. the pre-pyplot
# `matplotlib.use("Agg")` pattern).
plt.switch_backend("Agg")

StatKind = Literal["continuous", "low_count", "high_count"]
RecommendationTag = Literal["variance_bucket", "family_swap", "combined", "no_change"]


def find_latest_run_dir(backtest_root: Path) -> Path:
    """Return the most recent data/backtest/run_<ts>/ directory.

    Sorts lexicographically by directory name, which is correct because
    the timestamp format (YYYYMMDDTHHMMSSZ) sorts the same way as time.

    Raises:
        FileNotFoundError: backtest_root doesn't exist or contains no run_* subdirs.
    """
    if not backtest_root.is_dir():
        raise FileNotFoundError(f"Backtest root not found: {backtest_root}")
    candidates = sorted(
        p for p in backtest_root.iterdir() if p.is_dir() and p.name.startswith("run_")
    )
    if not candidates:
        raise FileNotFoundError(f"No run_<ts>/ subdirectories under {backtest_root}")
    return candidates[-1]


def load_per_row_results(run_dir: Path) -> pd.DataFrame:
    """Load `<run_dir>/results.parquet` produced by scripts/backtest.py.

    Returns the frame as-is (columns include identifiers, family, params,
    plus per-stat <stat>_pred / <stat>_actual columns whose names depend on
    which positions ran).

    Raises:
        FileNotFoundError: results.parquet missing under run_dir.
    """
    results_path = run_dir / "results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"results.parquet missing under {run_dir}; run "
            f"`python scripts/backtest.py --report` to generate one."
        )
    return pd.read_parquet(results_path)


def _resolve_target_stats() -> dict[str, tuple[str, ...]]:
    """Return {position_value: (stat_value, ...)} for every position in
    POSITION_DISPATCH. The factory call is cheap (constructs a dataclass
    of constants); the resulting dict is built once per script invocation.

    target_stats are identical across Model A (baseline) and Model C
    (lightgbm) by construction (Plan 5 LightGBMModel reuses each
    position's BaselineModel target stats), so the baseline factory is a
    fine source of truth here. The dispatch returns the Model Protocol
    (which intentionally does not advertise target_stats — that's a
    BaselineModel/LightGBMModel concrete-class API); cast() to read it."""
    out: dict[str, tuple[str, ...]] = {}
    for position, dispatch in POSITION_DISPATCH.items():
        model = cast(BaselineModel, dispatch.factories["baseline"]())
        out[position.value] = tuple(s.value for s in model.target_stats)
    return out


# Computed once at import time so extract_per_stat_residuals doesn't pay
# the POSITION_DISPATCH walk + 4 factory constructions on every call.
_TARGET_STATS_BY_POSITION: Final[dict[str, tuple[str, ...]]] = _resolve_target_stats()


def _classify_family(dist: Distribution) -> str:
    """Return the Plan 3d DistributionFamily.value matching the concrete class."""
    if isinstance(dist, ParametricNormal):
        return DistributionFamily.NORMAL.value
    if isinstance(dist, ParametricGamma):
        return DistributionFamily.GAMMA.value
    raise ValueError(f"Unrecognized distribution type: {type(dist).__name__}")


def _extract_assumed_params(dist: Distribution, family_name: str) -> tuple[float, float]:
    """Return (param_a, param_b) for the family. NORMAL packs std into a; GAMMA
    packs (shape, scale) into (a, b). Convention: param_b is NaN for one-param
    families."""
    if family_name == DistributionFamily.NORMAL.value:
        assert isinstance(dist, ParametricNormal)
        return float(dist.std()), float("nan")
    if family_name == DistributionFamily.GAMMA.value:
        assert isinstance(dist, ParametricGamma)
        return float(dist.shape), float(dist.scale)
    raise ValueError(f"Unhandled family: {family_name}")


def extract_per_stat_residuals(per_row: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide per-row results frame into a long-form residuals frame.

    For each row, look up the position's target stats via POSITION_DISPATCH
    and emit one output row per (position, stat) tuple where both
    `<stat>_pred` and `<stat>_actual` are present in the input. Pull the
    assumed-family parameters from the row's `params` blob via the existing
    Plan 3d codec.

    Output columns:
        position (str), stat (str), gsis_id (str), season (int), week (int),
        pred (float), actual (float), residual (= actual - pred),
        assumed_family (str: NORMAL or GAMMA),
        assumed_param_a (float: std for NORMAL, shape for GAMMA),
        assumed_param_b (float: NaN for NORMAL, scale for GAMMA)

    Notes:
        Silently skips rows where:
        - the row's position is not in POSITION_DISPATCH (e.g., K, DST)
        - the per-stat <stat>_pred or <stat>_actual column is missing
        - the per-stat <stat>_pred or <stat>_actual is NaN
        - the unpacked params blob has no entry for the position's stat
        `len(out)` may therefore be less than `len(per_row) * len(target_stats)`
        by design.
    """
    rows: list[dict[str, object]] = []
    for _idx, row in per_row.iterrows():
        position = str(row["position"])
        if position not in _TARGET_STATS_BY_POSITION:
            continue  # K / DST / unknown — skip silently, not in scope
        per_stat_dists = unpack_per_stat_params(bytes(row["params"]))
        for stat_value in _TARGET_STATS_BY_POSITION[position]:
            pred_col = f"{stat_value}_pred"
            actual_col = f"{stat_value}_actual"
            if pred_col not in per_row.columns or actual_col not in per_row.columns:
                continue
            pred = row[pred_col]
            actual = row[actual_col]
            if pd.isna(pred) or pd.isna(actual):
                continue
            stat_enum = Stat(stat_value)
            dist = per_stat_dists.get(stat_enum)
            if dist is None:
                continue
            family_name = _classify_family(dist)
            param_a, param_b = _extract_assumed_params(dist, family_name)
            rows.append(
                {
                    "position": position,
                    "stat": stat_value,
                    "gsis_id": str(row["gsis_id"]),
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "pred": float(pred),
                    "actual": float(actual),
                    "residual": float(actual) - float(pred),
                    "assumed_family": family_name,
                    "assumed_param_a": param_a,
                    "assumed_param_b": param_b,
                }
            )
    return pd.DataFrame(rows)


def compute_summary_stats(residuals: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(position, stat) aggregate diagnostics over the long-form
    residuals frame produced by extract_per_stat_residuals.

    Returns one row per (position, stat) cell with columns:
        position, stat, n, mean_pred, mean_actual,
        residual_mean, residual_std, residual_skew, residual_excess_kurtosis,
        std_tertile_low, std_tertile_mid, std_tertile_high,
        heteroscedasticity_ratio (= std_tertile_high / std_tertile_low,
            clamped denominator at 1e-9),
        coverage_p10p90, coverage_le_p90,
        ks_assumed_stat, ks_assumed_pvalue,
        assumed_family
    """
    rows: list[dict[str, object]] = []
    for (position, stat), group in residuals.groupby(["position", "stat"], sort=True):
        rows.append(_summarize_cell(position=str(position), stat=str(stat), group=group))
    return pd.DataFrame(rows)


def _summarize_cell(*, position: str, stat: str, group: pd.DataFrame) -> dict[str, object]:
    n = len(group)
    pred = group["pred"].to_numpy(dtype=np.float64)
    actual = group["actual"].to_numpy(dtype=np.float64)
    resid = group["residual"].to_numpy(dtype=np.float64)
    assumed_family = str(group["assumed_family"].iloc[0])
    param_a = group["assumed_param_a"].to_numpy(dtype=np.float64)
    param_b = group["assumed_param_b"].to_numpy(dtype=np.float64)

    # Tertile std (heteroscedasticity).
    order = np.argsort(pred)
    resid_sorted = resid[order]
    cuts = np.array_split(np.arange(n), 3)
    if len(cuts) >= 3 and all(len(c) > 1 for c in cuts):
        std_low = float(np.std(resid_sorted[cuts[0]], ddof=0))
        std_mid = float(np.std(resid_sorted[cuts[1]], ddof=0))
        std_high = float(np.std(resid_sorted[cuts[2]], ddof=0))
    else:
        std_low = std_mid = std_high = float("nan")
    hetero_ratio = std_high / max(std_low, 1e-9) if not np.isnan(std_low) else float("nan")

    # Per-row coverage of the assumed family's [p10, p90].
    coverage_p10p90, coverage_le_p90, ks_stat, ks_p = _coverage_and_ks(
        actual=actual,
        pred=pred,
        assumed_family=assumed_family,
        param_a=param_a,
        param_b=param_b,
    )

    return {
        "position": position,
        "stat": stat,
        "n": n,
        "mean_pred": float(np.mean(pred)),
        "mean_actual": float(np.mean(actual)),
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=0)),
        "residual_skew": float(scipy_stats.skew(resid)),
        "residual_excess_kurtosis": float(scipy_stats.kurtosis(resid, fisher=True)),
        "std_tertile_low": std_low,
        "std_tertile_mid": std_mid,
        "std_tertile_high": std_high,
        "heteroscedasticity_ratio": hetero_ratio,
        "coverage_p10p90": coverage_p10p90,
        "coverage_le_p90": coverage_le_p90,
        "ks_assumed_stat": ks_stat,
        "ks_assumed_pvalue": ks_p,
        "assumed_family": assumed_family,
    }


def _coverage_and_ks(
    *,
    actual: np.ndarray,
    pred: np.ndarray,
    assumed_family: str,
    param_a: np.ndarray,
    param_b: np.ndarray,
) -> tuple[float, float, float, float]:
    """Per-row CDF-transform under the assumed family. Coverage = fraction
    of `actual` within the row's per-row [p10, p90]. KS = test of u_i vs
    Uniform(0, 1) where u_i = F(actual_i; row_params); under H0 (assumed
    family is correct) u_i ~ Uniform(0, 1)."""
    if assumed_family == DistributionFamily.NORMAL.value:
        # NORMAL: row params = (loc=pred_i, scale=param_a_i = std).
        u = scipy_stats.norm.cdf(actual, loc=pred, scale=param_a)
        p10 = scipy_stats.norm.ppf(0.10, loc=pred, scale=param_a)
        p90 = scipy_stats.norm.ppf(0.90, loc=pred, scale=param_a)
    elif assumed_family == DistributionFamily.GAMMA.value:
        # GAMMA: row params = (shape=param_a, scale=param_b). Skip rows where
        # actual is non-positive (gamma support is (0, inf)).
        # We still compute u; gamma.cdf at 0 is 0, which is a valid uniform draw.
        u = scipy_stats.gamma.cdf(actual, a=param_a, scale=param_b)
        p10 = scipy_stats.gamma.ppf(0.10, a=param_a, scale=param_b)
        p90 = scipy_stats.gamma.ppf(0.90, a=param_a, scale=param_b)
    else:
        return float("nan"), float("nan"), float("nan"), float("nan")

    coverage_p10p90 = float(np.mean((actual >= p10) & (actual <= p90)))
    coverage_le_p90 = float(np.mean(actual <= p90))
    ks_result = scipy_stats.kstest(u, "uniform")
    return coverage_p10p90, coverage_le_p90, float(ks_result.statistic), float(ks_result.pvalue)


def fit_alternative_families(
    *,
    actual: np.ndarray,
    pred: np.ndarray,
    stat_kind: StatKind,
) -> dict[str, dict[str, float | bool]]:
    """MLE-fit the candidate alternative families for a (position, stat) cell.

    Returns {family_name: {"aic": float | nan, "ok": bool, "n_params": int}}.
    Failures (singular fits, non-finite log-likelihood, missing support)
    are recorded as ok=False with aic=nan; this function never raises.

    Candidate menus by stat_kind:
        "continuous":  student_t, log_normal
        "low_count":   neg_binomial
        "high_count":  neg_binomial

    Args:
        actual: Observed values for the (position, stat) cell.
        pred: Per-row point predictions for the same cell. Currently unused
            inside the helpers (alternatives are fit on `actual` directly);
            accepted in the signature so that Task 8's assemble_full_summary
            can pass both, leaving room for joint (actual, pred) fits later.
        stat_kind: Selects the candidate menu.
    """
    del pred  # reserved for future joint fits; see docstring.
    out: dict[str, dict[str, float | bool]] = {}
    if stat_kind == "continuous":
        out["student_t"] = _fit_student_t(actual)
        out["log_normal"] = _fit_log_normal(actual)
    elif stat_kind in ("low_count", "high_count"):
        out["neg_binomial"] = _fit_neg_binomial(actual)
    else:
        raise ValueError(f"Unknown stat_kind: {stat_kind!r}")
    return out


def _fit_student_t(actual: np.ndarray) -> dict[str, float | bool]:
    """MLE Student-t with location and scale; df free.

    Rejects degenerate fits where scipy collapses scale toward zero on
    (near-)constant input data — t.logpdf at the mode of a near-zero-scale
    distribution evaluates to a huge positive number, producing AIC values
    that would falsely dominate the family-swap decision rule downstream.
    """
    try:
        df, loc, scale = scipy_stats.t.fit(actual)
        # Sample-std floor on scale: a fitted scale orders of magnitude smaller
        # than the empirical std means scipy converged to a degenerate fit.
        sample_std = float(np.std(actual, ddof=0))
        if scale < max(sample_std * 1e-6, 1e-9):
            return {"aic": float("nan"), "ok": False, "n_params": 3}
        log_lik = float(np.sum(scipy_stats.t.logpdf(actual, df=df, loc=loc, scale=scale)))
        if not np.isfinite(log_lik):
            return {"aic": float("nan"), "ok": False, "n_params": 3}
        n_params = 3
        aic = 2 * n_params - 2 * log_lik
        return {"aic": float(aic), "ok": True, "n_params": n_params}
    except Exception:  # diagnostic must not abort on per-cell fit failure
        return {"aic": float("nan"), "ok": False, "n_params": 3}


def _fit_log_normal(actual: np.ndarray) -> dict[str, float | bool]:
    """MLE log-normal. Requires actual > 0; otherwise ok=False."""
    if actual.size == 0 or float(np.min(actual)) <= 0.0:
        return {"aic": float("nan"), "ok": False, "n_params": 2}
    try:
        # Fix loc=0 so we get the standard 2-param log-normal (s, scale).
        s, _loc, scale = scipy_stats.lognorm.fit(actual, floc=0)
        log_lik = float(np.sum(scipy_stats.lognorm.logpdf(actual, s, loc=0, scale=scale)))
        if not np.isfinite(log_lik):
            return {"aic": float("nan"), "ok": False, "n_params": 2}
        n_params = 2
        aic = 2 * n_params - 2 * log_lik
        return {"aic": float(aic), "ok": True, "n_params": n_params}
    except Exception:  # diagnostic must not abort on per-cell fit failure
        return {"aic": float("nan"), "ok": False, "n_params": 2}


def _fit_neg_binomial(actual: np.ndarray) -> dict[str, float | bool]:
    """MLE Negative Binomial via scipy.optimize on (n, p) parameters.

    scipy.stats.nbinom does not have a `.fit` method; we minimize the
    negative log-likelihood directly. Coerces actual to non-negative
    integers (counts); rounds and clips negatives to 0 so the helper is
    robust to noise upstream.
    """
    counts = np.clip(np.round(actual), 0, None).astype(np.int64)
    if counts.size < 2:
        return {"aic": float("nan"), "ok": False, "n_params": 2}
    mean = float(np.mean(counts))
    var = float(np.var(counts, ddof=0))
    if mean == 0.0:
        # All-zero counts: NB MLE is degenerate (any large n with p->1 explains it).
        return {"aic": float("nan"), "ok": False, "n_params": 2}
    # Method-of-moments init: var = mean + mean^2 / n  =>  n = mean^2 / (var - mean).
    # When var <= mean (NB degenerates to Poisson), the MoM is undefined; seed with a
    # large n so the NB approximates a Poisson and let the optimizer refine. We do NOT
    # short-circuit here: a NB fit on near-Poisson data is still a valid (if uninformative)
    # alternative, and downstream AIC ranking is what decides whether NB is preferred.
    if var > mean:
        n_init = mean * mean / (var - mean)
    else:
        n_init = max(mean * 100.0, 10.0)  # large n -> NB approximates Poisson(mean)
    p_init = n_init / (n_init + mean)

    def neg_log_lik(params: np.ndarray) -> float:
        n, p = params
        if n <= 0 or not 0 < p < 1:
            return float("inf")
        return -float(np.sum(scipy_stats.nbinom.logpmf(counts, n=n, p=p)))

    try:
        result = minimize(
            neg_log_lik,
            x0=np.array([n_init, p_init]),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000},
        )
        # Nelder-Mead returns success=False on near-boundary fits (e.g., NB->Poisson
        # with n->inf, p->1) even when result.fun is a sensible likelihood. We treat
        # any finite, non-inf result.fun as a usable fit; only non-finite values
        # (which indicate the optimizer wandered into infeasible territory) are rejected.
        if not np.isfinite(result.fun):
            return {"aic": float("nan"), "ok": False, "n_params": 2}
        log_lik = -float(result.fun)
        n_params = 2
        aic = 2 * n_params - 2 * log_lik
        return {"aic": float(aic), "ok": True, "n_params": n_params}
    except Exception:  # diagnostic must not abort on per-cell fit failure
        return {"aic": float("nan"), "ok": False, "n_params": 2}


_HETERO_RATIO_THRESHOLD = 1.5
_AIC_DELTA_THRESHOLD = 5.0


def compute_recommended_fix(
    *,
    heteroscedasticity_ratio: float,
    assumed_aic: float,
    alt_fits: dict[str, dict[str, float | bool]],
) -> tuple[RecommendationTag, str, float]:
    """Apply spec section 2.5's decision rule.

    Returns (recommended_fix, best_alt_family, aic_delta) where:
        recommended_fix  in {"variance_bucket", "family_swap",
                             "combined", "no_change"}
        best_alt_family  is the name of the lowest-AIC family whose fit ok=True,
                         or "none" if no alternative fit succeeded.
        aic_delta        = assumed_aic - best_alt_aic (positive = alt fits better,
                          since AIC is lower-is-better).
                          NaN if no alternative fit succeeded.
    """
    # Pick the best successful alternative.
    successful = {name: f for name, f in alt_fits.items() if f.get("ok") is True}
    if not successful:
        return "no_change", "none", float("nan")
    best_alt_family = min(successful, key=lambda n: float(successful[n]["aic"]))
    best_alt_aic = float(successful[best_alt_family]["aic"])
    aic_delta = assumed_aic - best_alt_aic

    has_hetero = (
        not np.isnan(heteroscedasticity_ratio)
        and heteroscedasticity_ratio > _HETERO_RATIO_THRESHOLD
    )
    has_better_family = aic_delta >= _AIC_DELTA_THRESHOLD

    if has_hetero and has_better_family:
        return "combined", best_alt_family, aic_delta
    if has_hetero:
        return "variance_bucket", best_alt_family, aic_delta
    if has_better_family:
        return "family_swap", best_alt_family, aic_delta
    return "no_change", best_alt_family, aic_delta


# Stat-kind classification per Stat enum value. Drives which alternative-family
# menu fit_alternative_families uses for each (position, stat) cell.
_STAT_KIND: dict[str, StatKind] = {
    "passing_yards": "continuous",
    "rushing_yards": "continuous",
    "receiving_yards": "continuous",
    "passing_tds": "low_count",
    "interceptions": "low_count",
    "rushing_tds": "low_count",
    "receiving_tds": "low_count",
    "fumbles_lost": "low_count",
    "receptions": "high_count",
}


def _assumed_aic_for_cell(
    *,
    actual: np.ndarray,
    pred: np.ndarray,
    assumed_family: str,
    param_a: np.ndarray,
    param_b: np.ndarray,
) -> float:
    """Log-likelihood under the assumed (per-row-parameterized) family,
    converted to AIC. NORMAL: 1 free param (sigma fit globally). GAMMA:
    1 free param (alpha fit globally; scale = mu / alpha is row-derived).
    """
    if assumed_family == DistributionFamily.NORMAL.value:
        log_lik = float(np.sum(scipy_stats.norm.logpdf(actual, loc=pred, scale=param_a)))
        n_params = 1
    elif assumed_family == DistributionFamily.GAMMA.value:
        log_lik = float(np.sum(scipy_stats.gamma.logpdf(actual, a=param_a, scale=param_b)))
        n_params = 1
    else:
        return float("nan")
    if not np.isfinite(log_lik):
        return float("nan")
    return 2 * n_params - 2 * log_lik


def assemble_full_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    """Combine compute_summary_stats + fit_alternative_families +
    compute_recommended_fix into the final per-(position, stat) summary frame
    matching spec section 2.3."""
    base = compute_summary_stats(residuals)
    enriched_rows: list[dict[str, object]] = []
    for _idx, row in base.iterrows():
        position = str(row["position"])
        stat = str(row["stat"])
        cell = residuals[(residuals["position"] == position) & (residuals["stat"] == stat)]
        actual = cell["actual"].to_numpy(dtype=np.float64)
        pred = cell["pred"].to_numpy(dtype=np.float64)
        param_a = cell["assumed_param_a"].to_numpy(dtype=np.float64)
        param_b = cell["assumed_param_b"].to_numpy(dtype=np.float64)
        assumed_family = str(row["assumed_family"])
        stat_kind = _STAT_KIND.get(stat, "continuous")

        alt_fits = fit_alternative_families(actual=actual, pred=pred, stat_kind=stat_kind)
        assumed_aic = _assumed_aic_for_cell(
            actual=actual,
            pred=pred,
            assumed_family=assumed_family,
            param_a=param_a,
            param_b=param_b,
        )
        recommendation, best_alt, aic_delta = compute_recommended_fix(
            heteroscedasticity_ratio=float(row["heteroscedasticity_ratio"]),
            assumed_aic=assumed_aic,
            alt_fits=alt_fits,
        )
        best_alt_aic = (
            float(alt_fits[best_alt]["aic"])
            if best_alt in alt_fits and alt_fits[best_alt]["ok"]
            else float("nan")
        )
        enriched = dict(row)
        enriched.update(
            {
                "best_alt_family": best_alt,
                "best_alt_aic": best_alt_aic,
                "assumed_aic": assumed_aic,
                "aic_delta": aic_delta,
                "recommended_fix": recommendation,
            }
        )
        enriched_rows.append(enriched)
    return pd.DataFrame(enriched_rows)


def make_residual_plot(
    *,
    residuals: np.ndarray,
    title: str,
    assumed_family: str,
    assumed_loc: float,
    assumed_scale: float,
    out_path: Path,
) -> None:
    """Histogram of residuals with the assumed-family density overlaid.

    For NORMAL: density at N(assumed_loc, assumed_scale).
    For GAMMA: density skipped (assumed params are per-row, not per-cell).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="black")
    if assumed_family == DistributionFamily.NORMAL.value and assumed_scale > 0:
        x = np.linspace(np.min(residuals), np.max(residuals), 200)
        ax.plot(
            x,
            scipy_stats.norm.pdf(x, loc=assumed_loc, scale=assumed_scale),
            color="darkred",
            linewidth=2,
            label=f"N({assumed_loc:.1f}, {assumed_scale:.1f})",
        )
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("residual (actual - pred)")
    ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def make_qq_plot(
    *,
    standardized_residuals: np.ndarray,
    title: str,
    assumed_family: str,
    out_path: Path,
) -> None:
    """Q-Q plot of standardized residuals vs the assumed-family quantiles.

    NORMAL: vs scipy.stats.norm. GAMMA: under the per-row CDF transform,
    standardized_residuals are uniform on [0, 1], so the comparison is vs
    Uniform(0, 1).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if assumed_family == DistributionFamily.NORMAL.value:
        scipy_stats.probplot(standardized_residuals, dist="norm", plot=ax)
    elif assumed_family == DistributionFamily.GAMMA.value:
        scipy_stats.probplot(standardized_residuals, dist="uniform", plot=ax)
    else:
        ax.text(0.5, 0.5, f"Q-Q not defined for {assumed_family}", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Path to a data/backtest/run_<ts>/ directory. "
            "Defaults to the lexicographically-latest run_<ts>/ directory."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Path to write diagnostic artifacts. Defaults to data/diagnostics/calibration_<ts>/.",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir or find_latest_run_dir(Path("data/backtest"))
    out_dir: Path = args.out_dir or _default_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Reading {run_dir / 'results.parquet'}")
    per_row = load_per_row_results(run_dir)
    residuals = extract_per_stat_residuals(per_row)
    if residuals.empty:
        print("No per-stat residuals extracted; nothing to do.", file=sys.stderr)
        return 1

    residuals.to_parquet(out_dir / "residuals.parquet")
    summary = assemble_full_summary(residuals)
    summary.to_parquet(out_dir / "summary.parquet")

    # Plots — one histogram + one Q-Q per (position, stat).
    for (position, stat), group in residuals.groupby(["position", "stat"]):
        title = f"{position} {stat}"
        assumed_family = str(group["assumed_family"].iloc[0])
        # NORMAL: cell-mean assumed loc/scale; OK for the histogram overlay
        # because the assumed sigma is constant across rows under the current
        # estimator. GAMMA params are per-row, so the histogram is unannotated.
        if assumed_family == DistributionFamily.NORMAL.value:
            assumed_loc = float(np.mean(group["pred"]))
            assumed_scale = float(group["assumed_param_a"].iloc[0])
        else:
            assumed_loc = 0.0
            assumed_scale = 0.0
        make_residual_plot(
            residuals=group["residual"].to_numpy(dtype=np.float64),
            title=f"{title} — residuals",
            assumed_family=assumed_family,
            assumed_loc=assumed_loc,
            assumed_scale=assumed_scale,
            out_path=out_dir / "plots" / f"{position}_{stat}_hist.png",
        )
        # Q-Q: standardize per family (NORMAL → z; GAMMA → CDF-uniform).
        std_resid = _standardized_residuals(group)
        make_qq_plot(
            standardized_residuals=std_resid,
            title=f"{title} — Q-Q",
            assumed_family=assumed_family,
            out_path=out_dir / "plots" / f"{position}_{stat}_qq.png",
        )

    print(f"\nWrote {out_dir}/residuals.parquet  ({len(residuals)} rows)")
    print(f"Wrote {out_dir}/summary.parquet    ({len(summary)} cells)")
    print(f"Wrote {len(list((out_dir / 'plots').glob('*.png')))} plots")
    print("\n=== summary ===")
    print(
        summary[
            [
                "position",
                "stat",
                "n",
                "coverage_p10p90",
                "heteroscedasticity_ratio",
                "aic_delta",
                "recommended_fix",
            ]
        ].to_string(index=False)
    )
    return 0


def _default_out_dir(run_dir: Path) -> Path:
    """Mirror the run_dir's timestamp tag under data/diagnostics/."""
    tag = run_dir.name.removeprefix("run_")
    return Path("data/diagnostics") / f"calibration_{tag}"


def _standardized_residuals(group: pd.DataFrame) -> np.ndarray:
    """For NORMAL: z = residual / std. For GAMMA: u = F_gamma(actual; per-row params)."""
    assumed_family = str(group["assumed_family"].iloc[0])
    if assumed_family == DistributionFamily.NORMAL.value:
        std = group["assumed_param_a"].to_numpy(dtype=np.float64)
        z = group["residual"].to_numpy(dtype=np.float64) / np.where(std > 0, std, np.nan)
        return np.asarray(z, dtype=np.float64)
    if assumed_family == DistributionFamily.GAMMA.value:
        actual = group["actual"].to_numpy(dtype=np.float64)
        shape = group["assumed_param_a"].to_numpy(dtype=np.float64)
        scale = group["assumed_param_b"].to_numpy(dtype=np.float64)
        u = scipy_stats.gamma.cdf(actual, a=shape, scale=scale)
        return np.asarray(u, dtype=np.float64)
    return np.full(len(group), np.nan)


if __name__ == "__main__":
    sys.exit(main())
