"""Synthetic fixtures for the backtest unit tests.

Plan 3c Phase 2 onward. Fixtures live here (not in the per-file test
modules) so test_metrics.py / test_naive.py / test_snapshot.py /
test_harness.py can share a coherent set of synthetic inputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.store import write_partition


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


@pytest.fixture
def synthetic_backtest_layout(
    tmp_path: Path,
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> dict[str, Path]:
    """Stand up a tiny data/raw + data/features layout under tmp_path so
    run_backtest can be exercised against synthetic data with no network.

    The root-scope WR fixtures cover 2024 (weeks 1-8) + 2025 (weeks 1-4),
    so the integration test trains on 2024 and holds out 2025.

    Returns paths suitable for run_backtest:
        {"raw_root": ..., "features_root": ...}
    """
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    features_root = data_root / "features"

    feats_by_season = baseline_features_wr.groupby("season")
    for season, sf in feats_by_season:
        for week, wf in sf.groupby("week"):
            write_partition(
                features_root,
                "wr",
                wf.reset_index(drop=True),
                season=int(season),
                week=int(week),
            )

    ws_by_season = baseline_weekly_stats_wr.groupby("season")
    for season, sf in ws_by_season:
        write_partition(
            raw_root,
            "weekly_stats",
            sf.reset_index(drop=True),
            season=int(season),
            week=None,
        )

    return {"raw_root": raw_root, "features_root": features_root}
