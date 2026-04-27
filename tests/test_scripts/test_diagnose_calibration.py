"""Smoke + helper tests for scripts/diagnose_calibration.py."""

from __future__ import annotations

from pathlib import Path

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
