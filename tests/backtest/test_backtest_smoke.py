"""Default-on smoke for the backtest harness.

Catches "I broke the harness wiring" without paying the full 16-fit
runtime of the gated test. Runs in default `pytest -v`.

Skipped automatically if the feature cache for (WR, 2024) doesn't exist
locally (fresh checkout before Phase 6 of Plan 3c, or before
`scripts/refresh_features.py` has been run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    """One (position, year) cell, both Model A (baseline) and Model C (lightgbm).

    Plan 5 Task 14: extends the smoke to invoke both model classes so
    accidental regressions in either branch surface in the default
    `pytest -v` run. Budget: ~15s baseline + ~30s lightgbm = ~45s for
    the (WR, 2024) cell. Asserts the harness produces a non-empty
    result with the expected long-form schema, both model_class rows
    present, and every metric value finite.
    """
    out = run_backtest(
        held_out_years=[2024],
        positions=[Position.WR],
        train_start=2018,
        model_classes=("baseline", "lightgbm"),
        features_root=_FEATURES_ROOT,
        raw_root=_RAW_ROOT,
    )
    assert isinstance(out, BacktestRun)
    assert not out.metrics.empty
    # Plan 5 Task 12: metrics now carry `model_class`.
    assert set(out.metrics.columns) == {"position", "year", "metric", "model_class", "value"}
    # Plan 5 Task 14: both models ran for the cell.
    assert set(out.metrics["model_class"].unique()) == {"baseline", "lightgbm"}
    # Exactly one (position, year) cell.
    assert sorted(out.metrics["position"].unique().tolist()) == ["WR"]
    assert sorted(out.metrics["year"].unique().tolist()) == [2024]

    # Every metric value is finite for both models. `notna` catches NaN;
    # `np.isfinite` additionally rejects +-inf. Per the Task 12 reviewer
    # note, this is the most direct guard against silent regressions in
    # either model branch.
    assert out.metrics["value"].notna().all()
    assert np.isfinite(out.metrics["value"].to_numpy()).all()

    # Composite + Spearman + at least one per-stat row are present
    # across both models combined.
    metric_names = set(out.metrics["metric"].unique())
    assert "composite_rmse" in metric_names
    assert "spearman_topN" in metric_names

    # Per the Task 12 reviewer's note: assert every model produces the
    # core composite metrics for the (WR, 2024) cell. Catches the case
    # where the LightGBM branch silently drops rows. The composite +
    # ranking + per-week calibration metrics are model-class-agnostic
    # (they consume per-row predicted means + p10/p90 only), so both
    # models must emit them.
    core_metrics = {
        "composite_rmse",
        "composite_mae",
        "spearman_topN",
        "calibration_p10p90",
    }
    for model_class in ("baseline", "lightgbm"):
        per_model = out.metrics[out.metrics["model_class"] == model_class]
        per_model_metrics = set(per_model["metric"].unique())
        missing = core_metrics - per_model_metrics
        assert not missing, (
            f"model_class={model_class!r} missing core metrics {missing}; "
            f"got {sorted(per_model_metrics)}"
        )

    # Plan 3d: season-calibration metrics are present and finite for the
    # smoke cell on the baseline (SAMPLED_SUMMARY) branch only -- the
    # season aggregator does not yet handle the QUANTILE family emitted
    # by LightGBM (see harness.py around line 296; widening is filed in
    # Plan 5 Task 18). Asymmetry is intentional: assert the baseline
    # branch still emits the rows so we catch wiring regressions.
    baseline_metrics = out.metrics[out.metrics["model_class"] == "baseline"]
    season_metrics = baseline_metrics[
        baseline_metrics["metric"].isin(["season_calibration_p10p90", "season_calibration_le_p90"])
    ]
    assert len(season_metrics) == 2, (
        f"Expected 2 baseline season_calibration rows, got {len(season_metrics)}: "
        f"{season_metrics['metric'].tolist()}"
    )
    assert season_metrics["value"].notna().all()
    assert ((season_metrics["value"] >= 0.0) & (season_metrics["value"] <= 1.0)).all()

    # The LightGBM branch does NOT emit season_calibration_* rows today
    # (asymmetry documented above + in Task 13's snapshot logic). Pin
    # the asymmetry so a future widening of the season aggregator is a
    # deliberate change to this test, not a silent surprise.
    lightgbm_metrics = out.metrics[out.metrics["model_class"] == "lightgbm"]
    lightgbm_season_rows = lightgbm_metrics[
        lightgbm_metrics["metric"].isin(["season_calibration_p10p90", "season_calibration_le_p90"])
    ]
    assert lightgbm_season_rows.empty, (
        "LightGBM is not expected to emit season_calibration_* rows yet; "
        "if the season aggregator was widened to QUANTILE, update this test "
        f"to assert presence + finite values. Got: {lightgbm_season_rows.to_dict('records')}"
    )
