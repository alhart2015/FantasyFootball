"""Structural tests for src/projections/backtest/harness.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.backtest import BacktestRun, run_backtest
from projections.schemas import Position


def test_backtest_run_dataclass_shape() -> None:
    """BacktestRun is a frozen slots dataclass with the five documented
    attributes (timestamp, metrics, naive_metrics, per_row_results,
    per_player_results)."""
    run = BacktestRun(
        timestamp=pd.Timestamp("2026-04-26", tz="UTC"),
        metrics=pd.DataFrame(columns=["position", "year", "metric", "value"]),
        naive_metrics=pd.DataFrame(columns=["position", "year", "metric", "value"]),
        per_row_results=pd.DataFrame(),
        per_player_results=pd.DataFrame(),
    )
    assert isinstance(run.metrics, pd.DataFrame)
    assert isinstance(run.naive_metrics, pd.DataFrame)
    assert run.timestamp.tzname() == "UTC"


def test_run_backtest_skeleton_returns_empty_metrics_when_no_positions() -> None:
    """Calling run_backtest with positions=[] returns an empty BacktestRun
    with the expected schema. This is the structural smoke before Phase 3
    wires real metrics."""
    out = run_backtest(positions=[], held_out_years=[])
    assert isinstance(out, BacktestRun)
    assert list(out.metrics.columns) == ["position", "year", "metric", "value"]
    assert list(out.naive_metrics.columns) == ["position", "year", "metric", "value"]
    assert out.metrics.empty


def test_run_backtest_populates_metrics_for_one_cell(
    synthetic_backtest_layout: dict[str, Path],
) -> None:
    """End-to-end: train on prior seasons of synthetic features, predict the
    held-out year, score, and emit long-form metric rows.

    The root-scope WR fixtures only cover 2024 (weeks 1-8) + 2025 (weeks 1-4),
    so we narrow train_start=2024 and held_out_years=[2025]. That's the
    minimal range that exercises a single (position, year) cell with at
    least one prior season of training and a non-empty holdout.
    """
    out = run_backtest(
        held_out_years=[2025],
        positions=[Position.WR],
        train_start=2024,
        features_root=synthetic_backtest_layout["features_root"],
        raw_root=synthetic_backtest_layout["raw_root"],
    )
    assert not out.metrics.empty
    assert set(out.metrics.columns) == {"position", "year", "metric", "value"}
    # Should produce at least one composite_rmse + one spearman_topN row.
    metric_names = set(out.metrics["metric"].unique())
    assert "composite_rmse" in metric_names
    assert "spearman_topN" in metric_names
    # naive_metrics has the same shape and is non-empty.
    assert not out.naive_metrics.empty


def test_backtest_run_includes_season_calibration_metrics_for_every_cell(
    synthetic_backtest_layout: dict[str, Path],
) -> None:
    """Every (position, year) cell contributes both season_calibration_p10p90
    and season_calibration_le_p90 rows.

    The synthetic fixtures only cover WR (2024 train + 2025 holdout), so we
    pin positions=(Position.WR,) and held_out_years=[2025]. With one cell,
    "every cell" is one cell — the assertion still verifies the per-cell
    wiring of aggregate_to_season + compute_season_calibration_metrics into
    run_backtest.
    """
    run = run_backtest(
        held_out_years=[2025],
        positions=[Position.WR],
        train_start=2024,
        features_root=synthetic_backtest_layout["features_root"],
        raw_root=synthetic_backtest_layout["raw_root"],
    )
    metrics = run.metrics
    cells = metrics.groupby(["position", "year"])["metric"].apply(set)
    for (position, year), metric_set in cells.items():
        assert "season_calibration_p10p90" in metric_set, (
            f"{position}/{year} missing season_calibration_p10p90"
        )
        assert "season_calibration_le_p90" in metric_set, (
            f"{position}/{year} missing season_calibration_le_p90"
        )


def test_backtest_run_per_player_results_is_populated(
    synthetic_backtest_layout: dict[str, Path],
) -> None:
    """per_player_results contains expected columns and at least one row per
    cell that produced predictions+actuals.

    Pinned to (WR, 2025) because the synthetic fixtures only cover WR.
    """
    run = run_backtest(
        held_out_years=[2025],
        positions=[Position.WR],
        train_start=2024,
        features_root=synthetic_backtest_layout["features_root"],
        raw_root=synthetic_backtest_layout["raw_root"],
    )
    assert not run.per_player_results.empty
    expected_columns = {
        "gsis_id",
        "season",
        "position",
        "ruleset",
        "n_weeks",
        "season_mean",
        "season_p10",
        "season_p50",
        "season_p90",
        "model_id",
        "generated_at",
        "actual_season_total",
    }
    actual_columns = set(run.per_player_results.columns)
    missing = expected_columns - actual_columns
    assert not missing, f"per_player_results missing columns: {missing}"
