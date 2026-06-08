# tests/test_scripts/test_benchmark_projections.py
import benchmark_projections as bench
import pandas as pd

from projections.schemas import Ruleset


def test_actual_season_points_sums_weeks_and_scores_ppr() -> None:
    # One WR, two weeks. Wk1: 5 rec, 70 yds, 1 TD. Wk2: 3 rec, 30 yds, 0 TD.
    # PPR points = receptions*1 + yards*0.1 + tds*6 = (8) + (100*0.1=10) + (1*6=6) = 24.0
    ws = pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000001"],
            "position": ["WR", "WR"],
            "passing_yards": [0.0, 0.0],
            "passing_tds": [0, 0],
            "interceptions": [0, 0],
            "rushing_yards": [0.0, 0.0],
            "rushing_tds": [0, 0],
            "receptions": [5, 3],
            "receiving_yards": [70.0, 30.0],
            "receiving_tds": [1, 0],
            "fumbles_lost": [0, 0],
        }
    )
    out = bench.actual_season_points(ws, Ruleset.espn_ppr())
    assert list(out["gsis_id"]) == ["00-0000001"]
    assert out.iloc[0]["position"] == "WR"
    assert out.iloc[0]["actual_pts"] == 24.0


def test_our_season_points_reads_csv_mean_as_points() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0000001"],
            "position": ["WR"],
            "season_total_mean": [180.5],
            "full_name": ["A B"],
        }
    )
    out = bench.our_season_points(df)
    assert out.iloc[0]["our_pts"] == 180.5
    assert out.iloc[0]["gsis_id"] == "00-0000001"
