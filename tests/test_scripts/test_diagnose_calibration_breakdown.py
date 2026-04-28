"""Tests for scripts/diagnose_calibration_breakdown.py — Plan 7 Phase 0.

The diagnostic reads per-row backtest output, decomposes the [p10, p90]
coverage gap vs Model A into count-stat vs yards-stat contributions, and
emits a CSV summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the script via the bare module name. tests/test_scripts/conftest.py
# inserts the scripts/ directory onto sys.path, matching the existing
# diagnose_calibration test pattern.
from diagnose_calibration_breakdown import (
    attribute_coverage_gap,
    compute_per_stat_coverage,
    main,
)

from projections.distributions import (
    ParametricNegativeBinomial,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import Stat


def _build_synthetic_per_row(rng: np.random.Generator) -> pd.DataFrame:
    """Build a minimal per-row frame matching scripts/backtest.py's output:
    identifiers + per-stat <stat>_actual columns + family + model_id +
    a `params` bytes blob carrying per-stat distributions (matches the
    real backtest output schema)."""
    n = 500
    yards_dist = ParametricNormal(mean=50.0, std=25.0)
    tds_dist = ParametricNegativeBinomial(mean=0.4, dispersion=2.0)
    params_blob = pack_per_stat_params(
        {Stat.RECEIVING_YARDS: yards_dist, Stat.RECEIVING_TDS: tds_dist}
    )
    rows = pd.DataFrame(
        {
            "gsis_id": [f"00-{i:07d}" for i in range(n)],
            "season": np.full(n, 2024, dtype=np.int64),
            "week": rng.integers(1, 18, size=n).astype(np.int64),
            "position": np.full(n, "WR", dtype=object),
            "team": np.full(n, "KC", dtype=object),
            "opponent": np.full(n, "DEN", dtype=object),
            "ruleset": np.full(n, "ESPN_PPR", dtype=object),
            "family": np.full(n, "MIXED", dtype=object),
            "model_id": np.full(n, "lightgbm-nb:wr:abc12345:2018-2023", dtype=object),
            "model_class": np.full(n, "lightgbm-nb", dtype=object),
            "params": [params_blob] * n,
            "receiving_yards_actual": rng.normal(50, 25, size=n),
            "receiving_tds_actual": rng.poisson(0.4, size=n).astype(np.float64),
        }
    )
    return rows


def test_compute_per_stat_coverage_returns_one_row_per_stat() -> None:
    rng = np.random.default_rng(0)
    per_row = _build_synthetic_per_row(rng)
    out = compute_per_stat_coverage(per_row, position="WR", year=2024)
    assert set(out["stat"]) == {"receiving_yards", "receiving_tds"}
    assert out["coverage_p10p90"].between(0.0, 1.0).all()


def test_compute_per_stat_coverage_matches_hand_computed() -> None:
    """For a hand-built frame where all actuals fall well inside the
    [p10, p90] band, coverage should be 1.0."""
    n = 100
    yards_dist = ParametricNormal(mean=50.0, std=25.0)
    tds_dist = ParametricNegativeBinomial(mean=0.4, dispersion=2.0)
    params_blob = pack_per_stat_params(
        {Stat.RECEIVING_YARDS: yards_dist, Stat.RECEIVING_TDS: tds_dist}
    )
    per_row = pd.DataFrame(
        {
            "season": np.full(n, 2024, dtype=np.int64),
            "position": np.full(n, "WR", dtype=object),
            "model_id": np.full(n, "lightgbm-nb:wr:x:2018-2023", dtype=object),
            "model_class": np.full(n, "lightgbm-nb", dtype=object),
            "params": [params_blob] * n,
            "receiving_yards_actual": np.full(n, 50.0),
            "receiving_tds_actual": np.zeros(n, dtype=np.float64),
        }
    )
    out = compute_per_stat_coverage(per_row, position="WR", year=2024)
    assert (out.loc[out["stat"] == "receiving_yards", "coverage_p10p90"] == 1.0).all()


def test_attribute_coverage_gap_sums_to_one_across_stat_classes() -> None:
    """count_share + yards_share = 1.0 by construction."""
    per_stat = pd.DataFrame(
        {
            "stat": ["receiving_yards", "receiving_tds", "rushing_tds"],
            "stat_class": ["yards", "count", "count"],
            "variance_contribution": [0.7, 0.2, 0.1],
            "coverage_gap_vs_a": [-0.04, -0.05, -0.06],
        }
    )
    out = attribute_coverage_gap(per_stat)
    assert out["yards_share"] + out["count_share"] == pytest.approx(1.0, abs=1e-9)


def test_attribute_coverage_gap_handles_zero_variance_safely() -> None:
    """All-zero variance contributions -> zero shares (no NaN)."""
    per_stat = pd.DataFrame(
        {
            "stat": ["x", "y"],
            "stat_class": ["yards", "count"],
            "variance_contribution": [0.0, 0.0],
            "coverage_gap_vs_a": [-0.01, -0.02],
        }
    )
    out = attribute_coverage_gap(per_stat)
    assert out["yards_share"] == 0.0
    assert out["count_share"] == 0.0


def test_decision_proceeds_phase_1_when_count_gap_dominates() -> None:
    """Counts gap meaningfully larger than yards gap => proceed_phase_1."""
    from diagnose_calibration_breakdown import _decision

    assert _decision(count_coverage_gap=0.20, yards_coverage_gap=0.00) == "proceed_phase_1"
    assert _decision(count_coverage_gap=0.16, yards_coverage_gap=0.05) == "proceed_phase_1"


def test_decision_stops_when_yards_gap_dominates() -> None:
    """Yards gap meaningfully larger than counts gap => stop_file_yards_plan."""
    from diagnose_calibration_breakdown import _decision

    assert _decision(count_coverage_gap=0.00, yards_coverage_gap=0.20) == "stop_file_yards_plan"
    assert _decision(count_coverage_gap=0.05, yards_coverage_gap=0.16) == "stop_file_yards_plan"


def test_decision_proceeds_with_followup_when_gaps_within_tolerance() -> None:
    """Gaps within +/-0.02 of each other => proceed_with_followup."""
    from diagnose_calibration_breakdown import _decision

    assert _decision(count_coverage_gap=0.10, yards_coverage_gap=0.10) == "proceed_with_followup"
    assert _decision(count_coverage_gap=0.10, yards_coverage_gap=0.11) == "proceed_with_followup"
    assert _decision(count_coverage_gap=0.00, yards_coverage_gap=0.00) == "proceed_with_followup"


def test_main_writes_csv_and_records_decision(tmp_path: Path) -> None:
    """Smoke: `main()` produces the expected CSV columns when given a fixture."""
    rng = np.random.default_rng(0)
    per_row = _build_synthetic_per_row(rng)
    in_path = tmp_path / "results.parquet"
    per_row.to_parquet(in_path)
    out_path = tmp_path / "calibration_breakdown.csv"

    main(["--per-row-parquet", str(in_path), "--output-csv", str(out_path)])

    assert out_path.exists()
    out = pd.read_csv(out_path)
    expected_cols = {
        "position",
        "year",
        "count_share",
        "yards_share",
        "count_coverage_gap",
        "yards_coverage_gap",
        "decision",
    }
    assert expected_cols.issubset(out.columns)
