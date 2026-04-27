"""End-to-end harness fold under both Model A + Model C. Plan 5 Task 15.

Single (WR, 2024) fold under ``model_classes=("baseline", "lightgbm")``;
verifies both models contributed rows for the cell and that the same
metric set is emitted for each. The season-calibration asymmetry between
the SAMPLED_SUMMARY and QUANTILE families is intentionally NOT asserted
here -- ``tests/backtest/test_backtest_smoke.py`` pins that contract.
This test focuses on the core, model-class-agnostic metrics that both
branches must emit.

Gated behind ``@pytest.mark.backtest`` (and therefore ``--run-backtest``)
because it trains both a baseline and a real LightGBM model on the full
feature cache; budget ~45-90s. Skipped automatically on a fresh checkout
where the local feature cache for (WR, 2024) doesn't exist yet -- mirrors
the guard used by ``tests/backtest/test_backtest_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.backtest.harness import run_backtest
from projections.schemas import Position

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


def _cache_present() -> bool:
    return (_FEATURES_ROOT / "wr" / "season=2024").exists() and (
        _RAW_ROOT / "weekly_stats" / "season=2024"
    ).exists()


@pytest.mark.backtest
@pytest.mark.skipif(
    not _cache_present(),
    reason="data/features/wr/season=2024 missing — run scripts/refresh_features.py wr",
)
def test_harness_runs_both_models_for_one_cell() -> None:
    """Run one (WR, 2024) fold with both model classes; assert both
    appear in the long-form metrics frame and that the core metric set
    is the same for each."""
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("baseline", "lightgbm"),
    )
    df = result.metrics
    assert set(df["model_class"].unique()) == {"baseline", "lightgbm"}
    # Core, model-class-agnostic metrics must agree across both models.
    # (season_calibration_* rows are emitted by baseline only today; see
    # tests/backtest/test_backtest_smoke.py for the pinned asymmetry.)
    core_metrics = {
        "composite_rmse",
        "composite_mae",
        "spearman_topN",
        "calibration_p10p90",
    }
    a_metrics = set(df[df["model_class"] == "baseline"]["metric"].unique())
    c_metrics = set(df[df["model_class"] == "lightgbm"]["metric"].unique())
    assert core_metrics.issubset(a_metrics)
    assert core_metrics.issubset(c_metrics)
