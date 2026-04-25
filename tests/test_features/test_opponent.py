"""Opponent-strength helper tests."""

from __future__ import annotations

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.schemas import Position, Ruleset


def test_opp_allowed_fppg_computes_trailing_mean_per_defense() -> None:
    """Team B plays Team A every week. Team A's WR scores 25/20/15/10 fantasy
    points across weeks 1-4 (PPR). For Team B's WRs in week 5, the trailing-4
    average of opponent-allowed WR points is (25+20+15+10)/4 = 17.5."""
    weekly_stats = pd.DataFrame(
        {
            "gsis_id": [
                "00-0000001",
                "00-0000001",
                "00-0000001",
                "00-0000001",
            ],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "position": ["WR", "WR", "WR", "WR"],
            "team": ["A", "A", "A", "A"],
            "opponent": ["B", "B", "B", "B"],
            "passing_yards": [0.0, 0.0, 0.0, 0.0],
            "passing_tds": [0, 0, 0, 0],
            "interceptions": [0, 0, 0, 0],
            "rushing_yards": [0.0, 0.0, 0.0, 0.0],
            "rushing_tds": [0, 0, 0, 0],
            "carries": [0, 0, 0, 0],
            "receptions": [10, 8, 6, 4],
            "receiving_yards": [90.0, 60.0, 30.0, 0.0],  # produces 25/20/15/10 PPR pts
            "receiving_tds": [1, 1, 1, 1],
            "receiving_air_yards": [120.0, 80.0, 40.0, 0.0],
            "targets": [12, 10, 8, 6],
            "fumbles_lost": [0, 0, 0, 0],
        }
    )
    # PPR scoring math:
    # Week 1: receptions=10 (10) + rec_yards=90 (9.0) + rec_tds=1 (6) = 25.0
    # Week 2: receptions=8 (8)  + rec_yards=60 (6.0) + rec_tds=1 (6) = 20.0
    # Week 3: receptions=6 (6)  + rec_yards=30 (3.0) + rec_tds=1 (6) = 15.0
    # Week 4: receptions=4 (4)  + rec_yards=0 (0.0)  + rec_tds=1 (6) = 10.0

    result = opp_allowed_fppg(
        weekly_stats,
        position=Position.WR,
        ruleset=Ruleset.espn_ppr(),
        n_weeks=4,
    )

    # Result is keyed (season, week, opp_team) where opp_team is the defense.
    # For team B in week 5 (i.e., team B has just allowed weeks 1-4): mean of 25,20,15,10 = 17.5.
    row = result[(result["season"] == 2024) & (result["week"] == 5) & (result["opp_team"] == "B")]
    assert len(row) == 1
    assert row.iloc[0]["opp_allowed_fppg"] == 17.5


def test_opp_allowed_fppg_filters_to_position() -> None:
    """RB stats must not contribute to opp_allowed_wr_fppg."""
    weekly_stats = pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000002"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["WR", "RB"],
            "team": ["A", "A"],
            "opponent": ["B", "B"],
            "passing_yards": [0.0, 0.0],
            "passing_tds": [0, 0],
            "interceptions": [0, 0],
            "rushing_yards": [0.0, 100.0],
            "rushing_tds": [0, 1],
            "carries": [0, 20],
            "receptions": [5, 0],
            "receiving_yards": [50.0, 0.0],
            "receiving_tds": [0, 0],
            "receiving_air_yards": [60.0, 0.0],
            "targets": [7, 0],
            "fumbles_lost": [0, 0],
        }
    )

    wr_result = opp_allowed_fppg(
        weekly_stats,
        position=Position.WR,
        ruleset=Ruleset.espn_ppr(),
        n_weeks=4,
    )
    # Team B allowed only the WR's points in week 1.
    # WR points: 5 (rec) + 5 (yds/10) + 0 (td) = 10. So team B in week 2 should see 10.
    row = wr_result[(wr_result["week"] == 2) & (wr_result["opp_team"] == "B")]
    assert len(row) == 1
    assert row.iloc[0]["opp_allowed_fppg"] == 10.0
