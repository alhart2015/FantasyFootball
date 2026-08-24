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
from tests.feature_cache_guard import feature_cache_skip_reason

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


#: None when this machine's cache is usable, else why to skip. Checks the schema-validated
#: read the test performs, not just that the directory exists — a cache predating a schema
#: change passes an existence check and then fails inside pandera. See the helper's docstring.
_SKIP_REASON = feature_cache_skip_reason(
    Position.WR, 2024, features_root=_FEATURES_ROOT, raw_root=_RAW_ROOT
)


@pytest.mark.backtest
@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
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
