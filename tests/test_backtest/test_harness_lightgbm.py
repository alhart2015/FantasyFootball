"""End-to-end harness fold for Model C (LightGBM). Plan 5 Task 15.

Single (WR, 2024) fold under ``model_classes=("lightgbm",)`` only. Asserts
every row is tagged ``lightgbm`` and the core metric set is present and
finite.

Gated behind ``@pytest.mark.backtest`` (and therefore ``--run-backtest``)
because it trains a real LightGBM model on the full feature cache; budget
~30-60s per fold. Skipped automatically on a fresh checkout where the
local feature cache for (WR, 2024) doesn't exist yet -- mirrors the
guard used by ``tests/backtest/test_backtest_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
def test_harness_lightgbm_single_cell() -> None:
    """Run one (WR, 2024) fold under Model C and assert per-row metrics
    are all tagged ``lightgbm`` with finite values + the expected core
    metric set."""
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("lightgbm",),
    )
    df = result.metrics
    assert (df["model_class"] == "lightgbm").all()
    assert df["value"].notna().all()
    assert np.isfinite(df["value"].to_numpy()).all()
    expected_metrics = {
        "composite_rmse",
        "composite_mae",
        "spearman_topN",
        "calibration_p10p90",
    }
    assert expected_metrics.issubset(set(df["metric"].unique()))
