"""Tests for src/projections/preseason/backtest.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.preseason.backtest import (
    compute_rmse_and_spearman,
    determine_verdict,
    walk_forward_backtest,
)
from projections.schemas import PreseasonBacktestSchema, Ruleset
from projections.store import write_partition
from tests.test_scripts.test_preseason_project_season_cli import _seed_minimal_data


def test_compute_rmse_zero_when_predicted_equals_actual() -> None:
    predicted = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "season_total_fpts_mean": [100.0, 200.0]}
    )
    actual = pd.DataFrame({"gsis_id": ["00-1", "00-2"], "actual_season_total_fpts": [100.0, 200.0]})
    rmse, spearman, n = compute_rmse_and_spearman(predicted=predicted, actual=actual, top_n=10)
    assert rmse == pytest.approx(0.0)
    assert spearman == pytest.approx(1.0)
    assert n == 2


def test_compute_rmse_known_value() -> None:
    predicted = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "season_total_fpts_mean": [100.0, 150.0]}
    )
    actual = pd.DataFrame({"gsis_id": ["00-1", "00-2"], "actual_season_total_fpts": [110.0, 140.0]})
    rmse, _, _ = compute_rmse_and_spearman(predicted=predicted, actual=actual, top_n=10)
    # sqrt(((10^2) + (10^2)) / 2) = sqrt(100) = 10
    assert rmse == pytest.approx(10.0)


def test_determine_verdict_adopt() -> None:
    assert determine_verdict(rmse_delta_pct=-5.0, spearman_top50=0.75) == "ADOPT"


def test_determine_verdict_do_not_adopt_worse_rmse() -> None:
    assert determine_verdict(rmse_delta_pct=2.0, spearman_top50=0.75) == "DO_NOT_ADOPT"


def test_determine_verdict_do_not_adopt_bad_spearman() -> None:
    assert determine_verdict(rmse_delta_pct=-5.0, spearman_top50=0.40) == "DO_NOT_ADOPT"


def test_determine_verdict_null_band() -> None:
    assert determine_verdict(rmse_delta_pct=-1.0, spearman_top50=0.60) == "NULL"


def test_walk_forward_backtest_returns_one_row_per_position_per_year(
    tmp_path: Path,
) -> None:
    """End-to-end smoke: walk_forward_backtest produces a backtest frame per cell."""
    raw_root = tmp_path / "raw"
    proj_root = tmp_path / "projections"
    _seed_minimal_data(raw_root, target_season=2024)

    # Also seed 2024 weekly_stats as actuals.
    actual_2024 = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2024,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "opponent": "BUF",
                "passing_yards": 260.0,
                "passing_tds": 2,
                "interceptions": 1,
                "attempts": 30,
                "completions": 22,
                "sacks": 2,
                "rushing_yards": 30.0,
                "rushing_tds": 0,
                "carries": 5,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        ]
    )
    write_partition(raw_root, "weekly_stats", actual_2024, season=2024, week=None)

    out = walk_forward_backtest(
        raw_root=raw_root,
        projections_root=proj_root,
        target_seasons=[2024],
        train_start=2018,
        ruleset=Ruleset.espn_ppr(),
    )
    assert len(out) >= 1
    assert "verdict" in out.columns
    out = PreseasonBacktestSchema.validate(out)


def test_write_backtest_report_produces_markdown(tmp_path: Path) -> None:
    from projections.preseason.backtest import write_backtest_report

    backtest_df = pd.DataFrame(
        {
            "target_season": pd.array([2024], dtype="int32"),
            "position": ["QB"],
            "model_class": pd.array(["naive-preseason-v1"], dtype="string[pyarrow]"),
            "ruleset": ["ESPN_PPR"],
            "rmse": pd.array([35.0], dtype="float32"),
            "rmse_naive_baseline": pd.array([35.0], dtype="float32"),
            "rmse_delta_pct": pd.array([0.0], dtype="float32"),
            "spearman_top50": pd.array([0.72], dtype="float32"),
            "n_players": pd.array([28], dtype="Int64"),
            "coverage_diff_projected_not_played": pd.array([3], dtype="Int64"),
            "coverage_diff_played_not_projected": pd.array([1], dtype="Int64"),
            "verdict": ["NULL"],
        }
    )
    out_path = tmp_path / "backtest_report.md"
    write_backtest_report(backtest_df, out_path)
    text = out_path.read_text()
    assert "Backtest Report" in text
    assert "naive-preseason-v1" in text
    assert "QB" in text
    assert "NULL" in text
