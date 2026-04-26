"""Synthetic fixtures for the backtest unit tests.

Plan 3c Phase 2 onward. Fixtures live here (not in the per-file test
modules) so test_metrics.py / test_naive.py / test_snapshot.py /
test_harness.py can share a coherent set of synthetic inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def fake_eval_df() -> pd.DataFrame:
    """A tiny synthetic eval DataFrame matching the shape produced by
    harness.run_backtest's inner-join of predictions and actuals.

    Three player-weeks for two players. receptions/receiving_yards
    columns are suffixed _pred / _actual. Composite columns: mean,
    p10, p90, actual_ppr. (Other p-quantiles and per-stat columns
    are omitted; tests only consume what they assert against.)
    """
    return pd.DataFrame(
        {
            "gsis_id": ["00-A", "00-A", "00-B"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "mean": [12.0, 14.0, 6.0],
            "p10": [4.0, 6.0, 1.0],
            "p90": [22.0, 24.0, 14.0],
            "actual_ppr": [10.0, 18.0, 4.0],
            "receptions_pred": [4.0, 5.0, 2.0],
            "receptions_actual": [3.0, 6.0, 1.0],
            "receiving_yards_pred": [55.0, 70.0, 25.0],
            "receiving_yards_actual": [40.0, 95.0, 12.0],
        }
    )
