"""Plan 8 — adoption gate.

Paired-bootstrap CI machinery for comparing two model classes on per-row
backtest output. Pure numpy/scipy/pandas — no IO. Consumed by
scripts/adoption_gate.py (the CLI orchestrator).

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.schemas import Position

VerdictLabel = Literal["ADOPT", "MARGINAL", "DO_NOT_ADOPT"]


@dataclass(frozen=True, slots=True)
class BootstrapDelta:
    """Result of a paired bootstrap on a metric delta (candidate - incumbent).

    Sign convention is metric-specific: for RMSE, negative ``point`` means
    the candidate wins (lower error). For Spearman, positive ``point`` means
    the candidate wins (higher rank correlation).
    """

    point: float
    lo_95: float
    hi_95: float
    n_paired_rows: int
    n_bootstrap: int


@dataclass(frozen=True, slots=True)
class PositionVerdict:
    """Per-position adoption verdict bundling RMSE, Spearman, and per-year breakdown."""

    position: Position
    incumbent_class: str
    candidate_class: str
    rmse_delta: BootstrapDelta
    spearman_delta: BootstrapDelta
    verdict: VerdictLabel
    reason: str
    per_year_breakdown: pd.DataFrame


_MIN_PAIRED_ROWS = 100


def paired_bootstrap_rmse_delta(
    residuals_incumbent: np.ndarray,
    residuals_candidate: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapDelta:
    """Paired bootstrap CI on RMSE(candidate) - RMSE(incumbent).

    Both residual arrays must be aligned on the same rows (same player-week
    in the same order). The same bootstrap-sampled indices are applied to
    both arrays each draw — that's the "paired" structure.

    Args:
        residuals_incumbent: shape (n,), real-valued, ``actual - predicted``.
        residuals_candidate: shape (n,), real-valued, same row order.
        n_bootstrap: number of bootstrap resamples. Default 1000.
        seed: RNG seed for reproducibility. Default 42.

    Returns:
        BootstrapDelta with `point` = observed delta on full sample
        (no resampling), `lo_95` and `hi_95` the central 95% CI bounds
        across the bootstrap distribution.

    Raises:
        ValueError: lengths mismatch, or fewer than 100 paired rows.
    """
    inc = np.asarray(residuals_incumbent, dtype=np.float64)
    cand = np.asarray(residuals_candidate, dtype=np.float64)
    if inc.shape != cand.shape:
        raise ValueError(
            f"residuals must have the same length, got incumbent={inc.shape} "
            f"vs candidate={cand.shape}"
        )
    n = inc.shape[0]
    if n < _MIN_PAIRED_ROWS:
        raise ValueError(
            f"need at least {_MIN_PAIRED_ROWS} paired rows for a meaningful bootstrap, got {n}"
        )

    point = float(np.sqrt(np.mean(cand**2)) - np.sqrt(np.mean(inc**2)))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rmse_inc = np.sqrt(np.mean(inc[idx] ** 2))
        rmse_cand = np.sqrt(np.mean(cand[idx] ** 2))
        deltas[b] = rmse_cand - rmse_inc

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return BootstrapDelta(
        point=point,
        lo_95=float(lo),
        hi_95=float(hi),
        n_paired_rows=n,
        n_bootstrap=n_bootstrap,
    )


def verdict_for_position(
    rmse: BootstrapDelta,
    spearman: BootstrapDelta,
    *,
    spearman_floor: float = -0.02,
) -> tuple[VerdictLabel, str]:
    """Apply the §1.3-replacement rule.

    Rule (per spec §1.3):
        PASS_RMSE     := rmse.hi_95     <  0.0
        PASS_SPEARMAN := spearman.lo_95 > spearman_floor

        if  PASS_RMSE and  PASS_SPEARMAN: ADOPT
        if  PASS_RMSE and !PASS_SPEARMAN: MARGINAL
        if !PASS_RMSE and  PASS_SPEARMAN: DO_NOT_ADOPT
        if !PASS_RMSE and !PASS_SPEARMAN: DO_NOT_ADOPT

    Returns:
        (verdict_label, one-line human-readable reason).
    """
    if (
        np.isnan(rmse.point)
        or np.isnan(rmse.hi_95)
        or np.isnan(spearman.point)
        or np.isnan(spearman.lo_95)
    ):
        return ("DO_NOT_ADOPT", "degenerate prediction (NaN bootstrap statistics)")

    pass_rmse = rmse.hi_95 < 0.0
    pass_spearman = spearman.lo_95 > spearman_floor

    if pass_rmse and pass_spearman:
        return (
            "ADOPT",
            f"RMSE delta {rmse.point:+.3f} (95% CI [{rmse.lo_95:+.3f}, {rmse.hi_95:+.3f}]); "
            f"Spearman lo_95 {spearman.lo_95:+.4f} > floor {spearman_floor:+.3f}",
        )
    if pass_rmse and not pass_spearman:
        return (
            "MARGINAL",
            f"RMSE wins ({rmse.point:+.3f}) but Spearman lo_95 {spearman.lo_95:+.4f} "
            f"breaks floor {spearman_floor:+.3f}; investigate before adopting",
        )
    if not pass_rmse and pass_spearman:
        return (
            "DO_NOT_ADOPT",
            f"RMSE inconclusive: 95% CI [{rmse.lo_95:+.3f}, {rmse.hi_95:+.3f}] brackets / "
            f"exceeds zero",
        )
    return (
        "DO_NOT_ADOPT",
        f"RMSE worse ({rmse.point:+.3f}) and Spearman regresses "
        f"(lo_95 {spearman.lo_95:+.4f} < floor {spearman_floor:+.3f})",
    )


def _per_group_mean_spearman(
    predicted: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,
) -> float:
    """Spearman correlation per group, averaged unweighted across groups.

    Returns NaN if any group's correlation is undefined (constant input,
    or empty). The verdict_for_position rule downgrades NaN to DO_NOT_ADOPT.
    """
    groups = np.unique(grouping)
    if groups.size == 0:
        return float("nan")
    rhos = np.empty(groups.size, dtype=np.float64)
    for i, g in enumerate(groups):
        mask = grouping == g
        if mask.sum() < 2:
            return float("nan")
        rho = spearmanr(predicted[mask], actual[mask]).statistic
        if np.isnan(rho):
            return float("nan")
        rhos[i] = rho
    return float(rhos.mean())


def paired_bootstrap_spearman_delta(
    predicted_incumbent: np.ndarray,
    predicted_candidate: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapDelta:
    """Paired bootstrap CI on (per-group-mean Spearman) delta candidate - incumbent.

    Per-year (group) Spearman is computed within each group, then averaged
    unweighted across groups. Pooling across years would mix populations
    because the player set rotates between held-out years.

    Args:
        predicted_incumbent: shape (n,) per-row predicted composite from incumbent.
        predicted_candidate: shape (n,) per-row predicted composite from candidate.
        actual: shape (n,) per-row actual composite.
        grouping: shape (n,) integer/string per-row group key (held-out year).
        n_bootstrap: number of bootstrap resamples. Default 1000.
        seed: RNG seed. Default 42.

    Returns:
        BootstrapDelta. NaN values propagate when either model produces a
        constant prediction within a group.

    Raises:
        ValueError: arrays have inconsistent lengths or fewer than 100 rows.
    """
    inc = np.asarray(predicted_incumbent, dtype=np.float64)
    cand = np.asarray(predicted_candidate, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    grp = np.asarray(grouping)
    n = inc.shape[0]
    if not (cand.shape[0] == act.shape[0] == grp.shape[0] == n):
        raise ValueError(
            "predicted_incumbent, predicted_candidate, actual, grouping must "
            f"have the same length; got {inc.shape[0]}, {cand.shape[0]}, "
            f"{act.shape[0]}, {grp.shape[0]}"
        )
    if n < _MIN_PAIRED_ROWS:
        raise ValueError(f"need at least {_MIN_PAIRED_ROWS} paired rows, got {n}")

    point = _per_group_mean_spearman(cand, act, grp) - _per_group_mean_spearman(inc, act, grp)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_inc = _per_group_mean_spearman(inc[idx], act[idx], grp[idx])
        s_cand = _per_group_mean_spearman(cand[idx], act[idx], grp[idx])
        deltas[b] = s_cand - s_inc

    if np.isnan(point) or np.isnan(deltas).any():
        return BootstrapDelta(
            point=float("nan"),
            lo_95=float("nan"),
            hi_95=float("nan"),
            n_paired_rows=n,
            n_bootstrap=n_bootstrap,
        )

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return BootstrapDelta(
        point=point,
        lo_95=float(lo),
        hi_95=float(hi),
        n_paired_rows=n,
        n_bootstrap=n_bootstrap,
    )
