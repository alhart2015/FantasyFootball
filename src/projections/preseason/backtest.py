"""Preseason backtest harness — walk-forward eval over target seasons.

Produces:
- A PreseasonBacktestSchema-validated CSV at reports/backtest_preseason_<model>.csv.
- A human-readable markdown report at reports/backtest_preseason_<model>.md.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §7.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_rmse_and_spearman(
    *,
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    top_n: int,
) -> tuple[float, float, int]:
    """Compute RMSE on full inner-join + Spearman on top-N actuals.

    `predicted` requires columns: `gsis_id`, `season_total_fpts_mean`.
    `actual` requires columns: `gsis_id`, `actual_season_total_fpts`.

    Returns:
        (rmse, spearman_top_n, n_players). If the inner join is empty, returns
        (NaN, NaN, 0). If fewer than 2 top-N rows, Spearman is NaN.
    """
    merged = predicted.merge(actual, on="gsis_id", how="inner")
    if merged.empty:
        return float("nan"), float("nan"), 0

    err = merged["season_total_fpts_mean"] - merged["actual_season_total_fpts"]
    rmse = float(np.sqrt((err**2).mean()))

    top_actual = merged.nlargest(top_n, "actual_season_total_fpts")
    if len(top_actual) < 2:
        spearman = float("nan")
    else:
        rho, _ = spearmanr(
            top_actual["actual_season_total_fpts"].to_numpy(),
            top_actual["season_total_fpts_mean"].to_numpy(),
        )
        spearman = float(rho)
    return rmse, spearman, len(merged)


def determine_verdict(
    *, rmse_delta_pct: float, spearman_top50: float
) -> Literal["ADOPT", "NULL", "DO_NOT_ADOPT"]:
    """Apply the per-cell verdict logic from spec §7.3.

    - ADOPT       if rmse_delta_pct < 0 AND spearman_top50 >= 0.70
    - DO_NOT_ADOPT if rmse_delta_pct >= 0 OR spearman_top50 < 0.50
    - NULL         otherwise
    """
    if rmse_delta_pct < 0 and spearman_top50 >= 0.70:
        return "ADOPT"
    if rmse_delta_pct >= 0 or spearman_top50 < 0.50:
        return "DO_NOT_ADOPT"
    return "NULL"
