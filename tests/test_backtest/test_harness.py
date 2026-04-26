"""Structural tests for src/projections/backtest/harness.py."""

from __future__ import annotations

import pandas as pd

from projections.backtest import BacktestRun, run_backtest


def test_backtest_run_dataclass_shape() -> None:
    """BacktestRun is a frozen slots dataclass with the four documented
    attributes (timestamp, metrics, naive_metrics, per_row_results)."""
    run = BacktestRun(
        timestamp=pd.Timestamp("2026-04-26", tz="UTC"),
        metrics=pd.DataFrame(columns=["position", "year", "metric", "value"]),
        naive_metrics=pd.DataFrame(columns=["position", "year", "metric", "value"]),
        per_row_results=pd.DataFrame(),
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
