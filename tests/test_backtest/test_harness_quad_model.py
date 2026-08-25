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
