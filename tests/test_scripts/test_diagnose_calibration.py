"""Smoke + helper tests for scripts/diagnose_calibration.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_find_latest_run_dir_returns_most_recent(tmp_path: Path) -> None:
    from diagnose_calibration import find_latest_run_dir

    backtest_root = tmp_path / "backtest"
    backtest_root.mkdir()
    older = backtest_root / "run_20260101T000000Z"
    newer = backtest_root / "run_20260201T000000Z"
    older.mkdir()
    newer.mkdir()
    # Order by directory name (timestamp sorts lexicographically).
    assert find_latest_run_dir(backtest_root) == newer


def test_find_latest_run_dir_raises_when_empty(tmp_path: Path) -> None:
    from diagnose_calibration import find_latest_run_dir

    backtest_root = tmp_path / "backtest"
    backtest_root.mkdir()
    with pytest.raises(FileNotFoundError, match="No run_"):
        find_latest_run_dir(backtest_root)


def test_find_latest_run_dir_raises_when_root_missing(tmp_path: Path) -> None:
    from diagnose_calibration import find_latest_run_dir

    with pytest.raises(FileNotFoundError):
        find_latest_run_dir(tmp_path / "does-not-exist")


def _make_minimal_per_row(tmp_path: Path) -> Path:
    """Build a 4-row results.parquet with one WR row and one QB row across
    two seasons. Includes only the columns load_per_row_results requires:
    identifiers + at least one *_pred / *_actual pair + family + params."""
    df = pd.DataFrame(
        {
            "gsis_id": ["00-1", "00-1", "00-2", "00-2"],
            "season": [2023, 2024, 2023, 2024],
            "week": [1, 1, 1, 1],
            "position": ["WR", "WR", "QB", "QB"],
            "team": ["KC", "KC", "MIN", "MIN"],
            "opponent": ["MIN", "MIN", "KC", "KC"],
            "ruleset": ["PPR_DEFAULT"] * 4,
            "family": ["SAMPLED_SUMMARY"] * 4,
            "params": [b""] * 4,  # ignored by load
            "receptions_pred": [4.0, 5.0, 0.0, 0.0],
            "receptions_actual": [3.0, 6.0, 0.0, 0.0],
            "passing_yards_pred": [0.0, 0.0, 250.0, 280.0],
            "passing_yards_actual": [0.0, 0.0, 220.0, 300.0],
        }
    )
    out = tmp_path / "results.parquet"
    df.to_parquet(out)
    return out


def test_load_per_row_results_round_trips(tmp_path: Path) -> None:
    from diagnose_calibration import load_per_row_results

    path = _make_minimal_per_row(tmp_path)
    loaded = load_per_row_results(path.parent)
    assert len(loaded) == 4
    assert {"gsis_id", "season", "position", "params"} <= set(loaded.columns)


def test_load_per_row_results_missing_file_raises(tmp_path: Path) -> None:
    from diagnose_calibration import load_per_row_results

    with pytest.raises(FileNotFoundError, match=r"results\.parquet"):
        load_per_row_results(tmp_path)
