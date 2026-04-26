"""Unit tests for src/projections/backtest/snapshot.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.backtest.snapshot import (
    GateResult,
    diff_snapshot,
    read_snapshot,
    write_snapshot,
)


def test_write_then_read_roundtrips_a_metrics_df(tmp_path: Path) -> None:
    """write_snapshot serializes a long-form metrics DataFrame; read_snapshot
    returns the same columns + values, sorted by (metric, position, year)."""
    df = pd.DataFrame(
        [
            {"position": "WR", "year": 2024, "metric": "composite_rmse", "value": 6.78},
            {"position": "QB", "year": 2021, "metric": "spearman_topN", "value": 0.928},
        ]
    )
    path = tmp_path / "snap.json"
    write_snapshot(df, path)

    out = read_snapshot(path)
    assert set(out.columns) == {"position", "year", "metric", "value"}
    assert len(out) == 2
    # Sorted by (metric, position, year): composite_rmse-WR-2024 then spearman_topN-QB-2021
    assert out["metric"].tolist() == ["composite_rmse", "spearman_topN"]


def test_write_snapshot_emits_human_readable_json(tmp_path: Path) -> None:
    """The on-disk JSON is a list of objects (not pandas-serialized) with
    a 2-space indent so PR diffs stay clean."""
    df = pd.DataFrame([{"position": "WR", "year": 2024, "metric": "composite_rmse", "value": 6.78}])
    path = tmp_path / "snap.json"
    write_snapshot(df, path)

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert parsed[0] == {
        "position": "WR",
        "year": 2024,
        "metric": "composite_rmse",
        "value": 6.78,
    }
    # Indented for readability.
    assert "\n  " in raw


def _baseline_row(metric: str, value: float) -> dict[str, Any]:
    return {"position": "WR", "year": 2024, "metric": metric, "value": value}


_DEFAULT_TOLS = {
    "rmse_relative": 0.05,
    "mae_relative": 0.05,
    "spearman_absolute": 0.02,
    "calibration_absolute": 0.03,
    "mean_pred_relative": 0.10,
}


def test_diff_passes_when_metrics_within_tolerance() -> None:
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.2)])  # +3.3%, under 5%

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert isinstance(out, GateResult)
    assert out.passed is True
    assert out.regressions == []


def test_diff_fails_on_rmse_regression_above_tolerance() -> None:
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.5)])  # +8.3% > 5%

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert out.passed is False
    assert len(out.regressions) == 1
    assert out.regressions[0].metric == "composite_rmse"
    assert out.regressions[0].direction == "worse"


def test_diff_passes_on_rmse_improvement() -> None:
    """Direction-aware: RMSE going DOWN is improvement, never a regression."""
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 4.0)])  # 33% better

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert out.passed is True


def test_diff_fails_on_spearman_drop_below_absolute_tolerance() -> None:
    baseline = pd.DataFrame([_baseline_row("spearman_topN", 0.97)])
    current = pd.DataFrame([_baseline_row("spearman_topN", 0.94)])  # -0.03 > 0.02

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert out.passed is False


def test_diff_passes_on_spearman_improvement() -> None:
    """Spearman going UP is improvement."""
    baseline = pd.DataFrame([_baseline_row("spearman_topN", 0.94)])
    current = pd.DataFrame([_baseline_row("spearman_topN", 0.99)])

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert out.passed is True


def test_diff_calibration_fails_on_drift_in_either_direction() -> None:
    """calibration_p10p90's target is 0.80; both 0.75 and 0.85 would be valid
    if baseline were at 0.80, but since baseline is the snapshot value we
    treat any move of >tolerance from snapshot as regression."""
    baseline = pd.DataFrame([_baseline_row("calibration_p10p90", 0.80)])
    current = pd.DataFrame([_baseline_row("calibration_p10p90", 0.85)])  # +0.05 > 0.03

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert out.passed is False


def test_diff_overrides_loosen_per_row_tolerance() -> None:
    """An override for the (position, year, metric) cell uses its tolerance
    instead of the default."""
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.5)])  # +8.3%

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[
            {
                "position": "WR",
                "year": 2024,
                "metric": "composite_rmse",
                "tolerance_kind": "rmse_relative",
                "tolerance_value": 0.10,
                "rationale": "fixture noise",
            }
        ],
    )
    assert out.passed is True


def test_diff_unknown_metric_kind_fails_closed() -> None:
    """A metric in the snapshot whose name doesn't match any known suffix
    raises so we never silently let a metric through ungated."""
    baseline = pd.DataFrame([_baseline_row("totally_made_up_metric", 1.0)])
    current = pd.DataFrame([_baseline_row("totally_made_up_metric", 1.5)])

    with pytest.raises(ValueError, match="unknown tolerance kind"):
        diff_snapshot(
            current=current,
            baseline=baseline,
            defaults=_DEFAULT_TOLS,
            overrides=[],
        )


def test_diff_missing_baseline_row_fails() -> None:
    """A current-run row with no baseline row to compare against fails the
    gate (the snapshot must be re-generated to add new metrics
    intentionally)."""
    baseline = pd.DataFrame(columns=["position", "year", "metric", "value"])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert out.passed is False
    assert any("missing from baseline" in r.message for r in out.regressions)
    assert any(r.direction == "missing" for r in out.regressions)
