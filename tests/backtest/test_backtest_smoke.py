"""Default-on smoke for the backtest harness.

Catches "I broke the harness wiring" without paying the full 16-fit
runtime of the gated test. Runs in default `pytest -v`.

Skipped automatically if the feature cache for (WR, 2024) doesn't exist
locally (fresh checkout before Phase 6 of Plan 3c, or before
`scripts/refresh_features.py` has been run).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.backtest import BacktestRun, run_backtest
from projections.schemas import Position

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


def _cache_present() -> bool:
    return (_FEATURES_ROOT / "wr" / "season=2024").exists() and (
        _RAW_ROOT / "weekly_stats" / "season=2024"
    ).exists()


@pytest.mark.skipif(
    not _cache_present(),
    reason="data/features/wr/season=2024 missing — run scripts/refresh_features.py wr",
)
def test_backtest_smoke_one_cell() -> None:
    """One (position, year) cell. Asserts the harness produces a non-empty
    result with the expected long-form schema."""
    out = run_backtest(
        held_out_years=[2024],
        positions=[Position.WR],
        train_start=2018,
        features_root=_FEATURES_ROOT,
        raw_root=_RAW_ROOT,
    )
    assert isinstance(out, BacktestRun)
    assert not out.metrics.empty
    assert set(out.metrics.columns) == {"position", "year", "metric", "value"}
    # Exactly one (position, year) cell.
    assert sorted(out.metrics["position"].unique().tolist()) == ["WR"]
    assert sorted(out.metrics["year"].unique().tolist()) == [2024]
    # Composite + Spearman + at least one per-stat row are present.
    metric_names = set(out.metrics["metric"].unique())
    assert "composite_rmse" in metric_names
    assert "spearman_topN" in metric_names

    # Plan 3d: season-calibration metrics are present and finite for the
    # smoke cell (default-on smoke catches accidental regressions in the
    # season aggregation wiring before a full --run-backtest).
    season_metrics = out.metrics[
        out.metrics["metric"].isin(["season_calibration_p10p90", "season_calibration_le_p90"])
    ]
    assert len(season_metrics) == 2, (
        f"Expected 2 season_calibration rows, got {len(season_metrics)}: "
        f"{season_metrics['metric'].tolist()}"
    )
    assert season_metrics["value"].notna().all()
    assert ((season_metrics["value"] >= 0.0) & (season_metrics["value"] <= 1.0)).all()
