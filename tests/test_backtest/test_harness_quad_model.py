"""End-to-end harness fold under all 4 model classes — Plan 5c Phase 2.

Mirrors test_harness_triple_model.py: single (WR, 2024) fold under
model_classes=(baseline, lightgbm, lightgbm-tuned, lightgbm-nb). Verifies
all four contribute rows for the cell and that the same model-class-
agnostic metric set is emitted for each.
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
def test_harness_runs_all_four_models_for_one_cell() -> None:
    """Run one (WR, 2024) fold with all 4 model classes; assert all four
    appear in the long-form metrics frame and that the core metric set
    is the same for each."""
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"),
    )
    df = result.metrics
    assert set(df["model_class"].unique()) == {
        "baseline",
        "lightgbm",
        "lightgbm-tuned",
        "lightgbm-nb",
    }
    core_metrics = {
        "composite_rmse",
        "composite_mae",
        "spearman_topN",
        "calibration_p10p90",
    }
    for model_class in ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"):
        per_model = set(df[df["model_class"] == model_class]["metric"].unique())
        assert core_metrics.issubset(per_model), (
            f"model_class={model_class!r} missing core metrics; got {sorted(per_model)}"
        )
