"""Unit tests for src/projections/backtest/metrics.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from projections.backtest.metrics import (
    compute_calibration_metrics,
    compute_composite_metrics,
    compute_per_stat_metrics,
    compute_spearman_topN,
)
from projections.schemas import Stat


def test_compute_per_stat_metrics_returns_rmse_mae_mean_pred(
    fake_eval_df: pd.DataFrame,
) -> None:
    """Per-stat RMSE, MAE, and mean_pred computed against suffixed columns."""
    out = compute_per_stat_metrics(fake_eval_df, target_stats=(Stat.RECEPTIONS,))

    # receptions_pred = [4, 5, 2], receptions_actual = [3, 6, 1]
    # diffs = [1, -1, 1] -> abs mean = 1.0; rmse = sqrt(mean([1, 1, 1])) = 1.0
    assert math.isclose(out["receptions_rmse"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["receptions_mae"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["receptions_mean_pred"], (4 + 5 + 2) / 3, rel_tol=1e-9)


def test_compute_per_stat_metrics_handles_multiple_stats(
    fake_eval_df: pd.DataFrame,
) -> None:
    """Two stats produce 6 keys (3 per stat)."""
    out = compute_per_stat_metrics(
        fake_eval_df, target_stats=(Stat.RECEPTIONS, Stat.RECEIVING_YARDS)
    )
    assert set(out.keys()) == {
        "receptions_rmse",
        "receptions_mae",
        "receptions_mean_pred",
        "receiving_yards_rmse",
        "receiving_yards_mae",
        "receiving_yards_mean_pred",
    }


def test_compute_composite_metrics(fake_eval_df: pd.DataFrame) -> None:
    """Composite RMSE/MAE against the `mean` and `actual_ppr` columns."""
    out = compute_composite_metrics(fake_eval_df)
    # diffs = [12-10, 14-18, 6-4] = [2, -4, 2] -> abs mean = 8/3; rmse = sqrt((4+16+4)/3)
    assert math.isclose(out["composite_mae"], 8 / 3, rel_tol=1e-9)
    assert math.isclose(out["composite_rmse"], math.sqrt(24 / 3), rel_tol=1e-9)


def test_compute_spearman_topN_groups_by_player(fake_eval_df: pd.DataFrame) -> None:  # noqa: N802 — mirrors the function under test (top-N domain term).
    """Spearman is computed on summed-mean season totals per gsis_id."""
    # Player A: pred sum=12+14=26, actual sum=10+18=28
    # Player B: pred sum=6,        actual sum=4
    # Two players: ranks are perfectly aligned -> Spearman = 1.0
    out = compute_spearman_topN(fake_eval_df)
    assert math.isclose(out, 1.0, rel_tol=1e-9)


def test_compute_calibration_metrics(fake_eval_df: pd.DataFrame) -> None:
    """Calibration is fraction of player-weeks with actual in [p10, p90] / <= p90."""
    # Row 0: actual=10, p10=4, p90=22 -> in [4,22], <= 22
    # Row 1: actual=18, p10=6, p90=24 -> in [6,24], <= 24
    # Row 2: actual=4,  p10=1, p90=14 -> in [1,14], <= 14
    out = compute_calibration_metrics(fake_eval_df)
    assert math.isclose(out["calibration_p10p90"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["calibration_le_p90"], 1.0, rel_tol=1e-9)


def test_compute_season_calibration_metrics_known_frame() -> None:
    from projections.backtest.metrics import compute_season_calibration_metrics

    df = pd.DataFrame(
        {
            "season_p10": [10.0] * 10,
            "season_p90": [50.0] * 10,
            "actual_season_total": [
                5.0,  # below p10  -> not in p10p90, but <= p90 (5 <= 50)
                15.0,  # in p10p90, <= p90
                25.0,  # in p10p90, <= p90
                40.0,  # in p10p90, <= p90
                45.0,  # in p10p90, <= p90
                49.0,  # in p10p90, <= p90
                51.0,  # > p90, not <= p90
                60.0,  # > p90
                70.0,  # > p90
                100.0,  # > p90
            ],
        }
    )
    out = compute_season_calibration_metrics(df)
    # 5 of 10 in [10,50]: 15, 25, 40, 45, 49 -> 0.5
    assert out["season_calibration_p10p90"] == pytest.approx(0.5)
    # <= p90: 5, 15, 25, 40, 45, 49 -> 6 of 10 -> 0.6
    assert out["season_calibration_le_p90"] == pytest.approx(0.6)


def test_compute_season_calibration_metrics_empty_frame_returns_nan() -> None:
    from math import isnan

    from projections.backtest.metrics import compute_season_calibration_metrics

    df = pd.DataFrame(columns=["season_p10", "season_p90", "actual_season_total"])
    out = compute_season_calibration_metrics(df)
    assert isnan(out["season_calibration_p10p90"])
    assert isnan(out["season_calibration_le_p90"])
