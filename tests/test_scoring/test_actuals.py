"""Tests for projections.scoring.actuals.actual_season_total — parity with the
legacy inline helper at scripts/compare_predictions_to_actuals.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import Ruleset
from projections.scoring.actuals import actual_season_total


def _synthetic_weekly_stats() -> pd.DataFrame:
    """Two players, two weeks each. Exercises passing + rushing + receiving stat sums."""
    return pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000001", "00-0000002", "00-0000002"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "position": ["QB", "QB", "WR", "WR"],
            "passing_yards": [300.0, 250.0, 0.0, 0.0],
            "passing_tds": [2, 1, 0, 0],
            "interceptions": [1, 0, 0, 0],
            "rushing_yards": [30.0, 15.0, 5.0, 0.0],
            "rushing_tds": [0, 0, 0, 0],
            "receptions": [0, 0, 7, 5],
            "receiving_yards": [0.0, 0.0, 90.0, 60.0],
            "receiving_tds": [0, 0, 1, 0],
            "fumbles_lost": [0, 0, 0, 0],
        }
    )


def test_actual_season_total_groups_by_gsis_id_position() -> None:
    out = actual_season_total(_synthetic_weekly_stats(), Ruleset.espn_ppr())
    assert set(out.columns) >= {"gsis_id", "position", "actual_total", "actual_n_weeks"}
    assert len(out) == 2
    qb_row = out[out["gsis_id"] == "00-0000001"].iloc[0]
    wr_row = out[out["gsis_id"] == "00-0000002"].iloc[0]
    assert qb_row["actual_n_weeks"] == 2
    assert wr_row["actual_n_weeks"] == 2
    # QB: 550 pass yd / 25 = 22, 3 pass td * 4 = 12, -2 int, 45 rush yd / 10 = 4.5 -> 36.5
    assert qb_row["actual_total"] == pytest.approx(36.5)
    # WR: 12 rec * 1 PPR = 12, 150 rec yd / 10 = 15, 1 rec td * 6 = 6, 5 rush yd / 10 = 0.5 -> 33.5
    assert wr_row["actual_total"] == pytest.approx(33.5)
