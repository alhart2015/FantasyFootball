"""CLI tests for scripts.build_trajectory_override.

Mirrors tests/test_scripts/test_build_pbp_pressure_override_cli.py.
Real data is monkey-patched out — tests assert argparse + main()'s
file-write contract, not the feature math (which has its own tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from scripts.build_trajectory_override import _parse_season_range, main


def test_parse_season_range_dash() -> None:
    assert _parse_season_range("2018-2024") == range(2018, 2025)


def test_parse_season_range_single() -> None:
    assert _parse_season_range("2024") == range(2024, 2025)


def test_main_rejects_existing_output_without_force(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.parquet"
    output.write_bytes(b"placeholder")
    with pytest.raises(SystemExit) as exc:
        main(["--seasons", "2024", "--data-root", str(tmp_path), "--output", str(output)])
    assert exc.value.code != 0


def test_main_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: monkey-patch _read_concat to feed synthetic frames; assert
    main() writes a parquet with the expected schema."""
    output = tmp_path / "trajectory.parquet"

    def fake_read_concat(raw_root: Path, table: str, seasons: Sequence[int]) -> pd.DataFrame:
        if table == "weekly_stats":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "season": 2024,
                        "week": 1,
                        "position": "QB",
                        "team": "KC",
                        "opponent": "BUF",
                        "attempts": 30,
                        "completions": 20,
                        "sacks": 2,
                        "passing_yards": 250.0,
                        "passing_tds": 2,
                        "interceptions": 0,
                        "rushing_yards": 10.0,
                        "rushing_tds": 0,
                        "carries": 3,
                        "receptions": 0,
                        "receiving_yards": 0.0,
                        "receiving_tds": 0,
                        "receiving_air_yards": 0.0,
                        "targets": 0,
                        "fumbles_lost": 0,
                    }
                ]
            )
        if table == "snap_counts":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "season": 2024,
                        "week": 1,
                        "team": "KC",
                        "opponent": "BUF",
                        "position": "QB",
                        "offense_snaps": 50,
                        "offense_pct": 0.7,
                        "defense_snaps": 0,
                        "defense_pct": 0.0,
                        "st_snaps": 0,
                        "st_pct": 0.0,
                    }
                ]
            )
        if table == "depth_charts":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "season": 2024,
                        "week": 1,
                        "team": "KC",
                        "position": "QB",
                        "depth_team": "QB1",
                        "depth_rank": 1,
                    }
                ]
            )
        if table == "schedules":
            return pd.DataFrame(
                [
                    {
                        "season": 2024,
                        "week": 1,
                        "game_id": "2024_01_BUF_KC",
                        "home_team": "KC",
                        "away_team": "BUF",
                        "kickoff": pd.Timestamp("2024-09-05", tz="UTC"),
                        "spread_line": -3.5,
                        "total_line": 47.5,
                        "home_moneyline": -180,
                        "away_moneyline": 160,
                        "surface": "grass",
                        "roof": "outdoors",
                        "temp": 70,
                        "wind": 5,
                    }
                ]
            )
        if table == "draft_picks":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "draft_year": 2017,
                        "draft_round": 1,
                        "draft_overall_pick": 10,
                        "pfr_id": "MahoPa00",
                        "draft_age": 21.5,
                    }
                ]
            )
        raise FileNotFoundError(table)

    monkeypatch.setattr("scripts.build_trajectory_override._read_concat", fake_read_concat)

    rc = main(["--seasons", "2024", "--data-root", str(tmp_path), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    written = pd.read_parquet(output)
    assert set(written.columns) >= {
        "gsis_id",
        "season",
        "week",
        "age",
        "is_rookie",
        "volume_trend_l4_minus_prior_l4",
        "snap_pct_change_l4_vs_prior_l4",
        "draft_year_inferred",
    }
    assert len(written) == 1
