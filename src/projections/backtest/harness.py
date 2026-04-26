"""Walk-forward backtest driver.

For each (position, year) in the cartesian product, train Model A on
cached features for [train_start, year-1], predict every week of `year`
from cached features, score against actuals from data/raw/weekly_stats,
and return a BacktestRun with model + naive metrics.

Plan 3c Phase 2 lands the dataclass + driver shell with empty metrics.
Phase 3 wires metrics.py + naive.py into the per-(position, year) loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from projections.schemas import Position, Ruleset

_METRICS_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """Result of a single walk-forward backtest invocation.

    Attributes:
        timestamp: UTC time the run started; used to name diagnostic
            output directories under data/backtest/run_<ts>/.
        metrics: long-form DataFrame with columns
            (position, year, metric, value) — the model's metrics across
            (position, year, metric) cells. Becomes the snapshot input.
        naive_metrics: same shape; computed alongside model metrics for
            informational reporting. Not gated.
        per_row_results: per-(position, year, week, gsis_id) row of
            actuals + model predictions for diagnosis. Plan 3c writes
            this to data/backtest/run_<ts>/results.parquet (gitignored).
    """

    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame


def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,
) -> BacktestRun:
    """Walk-forward backtest. Plan 3c Phase 2 returns an empty BacktestRun;
    Phase 3 fills in metrics + naive_metrics + per_row_results."""
    if ruleset is None:
        ruleset = Ruleset.espn_ppr()
    if positions is None:
        positions = (Position.QB, Position.RB, Position.TE, Position.WR)

    timestamp = pd.Timestamp(datetime.now(UTC))

    # Phase 2 stub: just enumerate the cartesian product to validate args.
    # Phase 3 replaces this with real per-cell training + scoring.
    _ = list(positions)
    _ = list(held_out_years)
    _ = train_start
    _ = features_root
    _ = raw_root
    _ = ruleset

    empty_metrics = pd.DataFrame(columns=list(_METRICS_COLUMNS))
    return BacktestRun(
        timestamp=timestamp,
        metrics=empty_metrics,
        naive_metrics=empty_metrics.copy(),
        per_row_results=pd.DataFrame(),
    )
