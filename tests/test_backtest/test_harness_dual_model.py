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
