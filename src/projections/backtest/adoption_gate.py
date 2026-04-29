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
