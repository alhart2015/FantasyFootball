"""Walk-forward backtest harness + snapshot-diff gating.

Plan 3c. Public surface for the harness, the metric primitives, the naive
baseline, and the snapshot-diff machinery.
"""

from __future__ import annotations

from projections.backtest.harness import BacktestRun, run_backtest
from projections.backtest.snapshot import read_snapshot, write_snapshot

__all__ = [
    "BacktestRun",
    "read_snapshot",
    "run_backtest",
    "write_snapshot",
]
