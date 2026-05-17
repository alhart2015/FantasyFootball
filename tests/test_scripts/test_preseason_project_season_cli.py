"""Happy-path integration test for scripts/preseason_project_season.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from projections.schemas import PreseasonProjectionSchema
from projections.store import write_partition


def _seed_minimal_data(raw_root: Path, target_season: int = 2024) -> None:
    """Seed minimal raw partitions sufficient for a 1-player projection."""
    # weekly_stats: one veteran with 1 game in 2023 (prior_1).
    weekly = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2023,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "opponent": "BUF",
                "passing_yards": 250.0,
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
    write_partition(raw_root, "weekly_stats", weekly, season=2023, week=None)

    # depth_charts for target_season.
    depth = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": target_season,
                "week": 1,
                "team": "KC",
                "position": "QB",
                "depth_team": "QB1",
                "depth_rank": 1,
            }
        ]
    )
    write_partition(raw_root, "depth_charts", depth, season=target_season, week=None)

    # draft_picks 2017 (rookie year for the veteran).
    picks = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "season": 2017,
                "round": 1,
                "pick": 10,
            }
        ]
    )
    write_partition(raw_root, "draft_picks", picks, season=2017, week=None)

    # id_map (unpartitioned).
    id_map = pd.DataFrame(
        [
            {
                "gsis_id": "00-1111111",
                "full_name": "Patrick Mahomes",
                "birth_date": pd.Timestamp("1995-09-17"),
                "team": "KC",
                "espn_id": pd.NA,
                "sleeper_id": pd.NA,
                "pfr_id": pd.NA,
            }
        ]
    )
    write_partition(raw_root, "id_map", id_map, season=None, week=None)


def test_preseason_project_season_cli_happy_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "preseason_project_season.py"
    raw_root = tmp_path / "raw"
    proj_root = tmp_path / "projections"
    reports_root = tmp_path / "reports"
    _seed_minimal_data(raw_root, target_season=2024)

    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--season",
            "2024",
            "--ruleset",
            "espn_ppr",
            "--raw-root",
            str(raw_root),
            "--projections-root",
            str(proj_root),
            "--reports-root",
            str(reports_root),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    out_path = proj_root / "preseason" / "ruleset=ESPN_PPR" / "season=2024" / "part.parquet"
    assert out_path.exists(), f"Expected projections parquet at {out_path}"
    df = pd.read_parquet(out_path)
    df = PreseasonProjectionSchema.validate(df)
    assert len(df) == 1

    assert (reports_root / "preseason_2024.csv").exists()
