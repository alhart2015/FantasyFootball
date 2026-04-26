"""Metric primitives for the walk-forward backtest harness.

Each function consumes an ``eval_df`` (the inner-join of predictions and
actuals; shape documented in tests/test_backtest/conftest.py) plus the
``target_stats`` for the position, and returns a dict[str, float] keyed
by metric name.

The harness composes these per (position, year) and folds the
dicts into the long-form metrics DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.schemas import Stat


def compute_per_stat_metrics(
    eval_df: pd.DataFrame,
    *,
    target_stats: tuple[Stat, ...],
) -> dict[str, float]:
    """Per-stat RMSE / MAE / mean_pred against suffixed columns.

    Expects columns ``{stat}_pred`` and ``{stat}_actual`` for each stat.
    Returns three keys per stat: ``{stat}_rmse``, ``{stat}_mae``,
    ``{stat}_mean_pred``.
    """
    out: dict[str, float] = {}
    for stat in target_stats:
        pred_col = f"{stat.value}_pred"
        actual_col = f"{stat.value}_actual"
        diffs = eval_df[pred_col] - eval_df[actual_col]
        out[f"{stat.value}_rmse"] = float(np.sqrt((diffs**2).mean()))
        out[f"{stat.value}_mae"] = float(diffs.abs().mean())
        out[f"{stat.value}_mean_pred"] = float(eval_df[pred_col].mean())
    return out


def compute_composite_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """RMSE / MAE on the composite mean (PPR points) prediction.

    Expects columns ``mean`` (model's composite mean) and ``actual_ppr``
    (realized PPR points). Returns ``composite_rmse`` and ``composite_mae``.
    """
    diffs = eval_df["mean"] - eval_df["actual_ppr"]
    return {
        "composite_rmse": float(np.sqrt((diffs**2).mean())),
        "composite_mae": float(diffs.abs().mean()),
    }


def compute_spearman_topN(eval_df: pd.DataFrame) -> float:  # noqa: N802 — "topN" is a domain term (top-N ranking), the capital N is intentional.
    """Spearman correlation on summed-mean season totals across players.

    Expects columns ``gsis_id``, ``mean``, ``actual_ppr``. Returns the
    Spearman rho across all players in the (position, year). NaN if
    fewer than two distinct players are present (rare but possible on
    a synthetic fixture).
    """
    pred_rank = eval_df.groupby("gsis_id")["mean"].sum().rank()
    actual_rank = eval_df.groupby("gsis_id")["actual_ppr"].sum().rank()
    common = pred_rank.index.intersection(actual_rank.index)
    if len(common) < 2:
        return float("nan")
    return float(np.corrcoef(pred_rank.loc[common], actual_rank.loc[common])[0, 1])


def compute_calibration_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Calibration coverage at the weekly level.

    Expects columns ``p10``, ``p90``, ``actual_ppr``. Returns
    ``calibration_p10p90`` (fraction of player-weeks where actual in
    [p10, p90]) and ``calibration_le_p90`` (fraction where actual <= p90).
    """
    in_p10p90 = (
        (eval_df["actual_ppr"] >= eval_df["p10"]) & (eval_df["actual_ppr"] <= eval_df["p90"])
    ).mean()
    le_p90 = (eval_df["actual_ppr"] <= eval_df["p90"]).mean()
    return {
        "calibration_p10p90": float(in_p10p90),
        "calibration_le_p90": float(le_p90),
    }


def compute_all_metrics(
    eval_df: pd.DataFrame,
    *,
    target_stats: tuple[Stat, ...],
) -> dict[str, float]:
    """Convenience wrapper: returns the union of all metric dicts."""
    out: dict[str, float] = {}
    out.update(compute_per_stat_metrics(eval_df, target_stats=target_stats))
    out.update(compute_composite_metrics(eval_df))
    out["spearman_topN"] = compute_spearman_topN(eval_df)
    out.update(compute_calibration_metrics(eval_df))
    return out
