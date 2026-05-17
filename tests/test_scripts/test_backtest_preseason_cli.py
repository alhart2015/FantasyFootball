"""Happy-path integration test for scripts/backtest_preseason.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from projections.store import write_partition
from tests.test_scripts.test_preseason_project_season_cli import _seed_minimal_data


def test_backtest_preseason_cli_happy_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw_root = tmp_path / "raw"
    proj_root = tmp_path / "projections"
    reports_root = tmp_path / "reports"
    _seed_minimal_data(raw_root, target_season=2024)

    # Also seed 2024 weekly_stats as the actuals.
    actual = pd.DataFrame(
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
    write_partition(raw_root, "weekly_stats", actual, season=2024, week=None)

    script = repo_root / "scripts" / "backtest_preseason.py"
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "naive-preseason",
            "--target-seasons",
            "2024",
            "--raw-root",
            str(raw_root),
            "--projections-root",
            str(proj_root),
            "--reports-root",
            str(reports_root),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (reports_root / "backtest_preseason_naive-preseason.csv").exists()
    assert (reports_root / "backtest_preseason_naive-preseason.md").exists()
