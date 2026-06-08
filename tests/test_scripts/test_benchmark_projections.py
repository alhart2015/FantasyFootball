# tests/test_scripts/test_benchmark_projections.py
import benchmark_projections as bench
import numpy as np
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


def test_build_benchmark_frame_joins_on_gsis_and_scores_espn() -> None:
    # ESPN row keyed by espn_id; our + actual keyed by gsis_id; id_map crosswalks.
    espn = pd.DataFrame(
        [
            {
                "espn_id": "E1",
                "full_name": "A B",
                "position": "WR",
                "espn_adp": 4.0,
                "espn_pos_rank": 2,
                "espn_actual_applied_total": 200.0,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 80,
                "receiving_yards": 1000.0,
                "receiving_tds": 5,
                "fumbles_lost": 0,
            }
        ]
    )  # ESPN proj PPR = 80 + 100 + 30 = 210.0
    ours = pd.DataFrame({"gsis_id": ["00-0000001"], "position": ["WR"], "our_pts": [195.0]})
    actuals = pd.DataFrame({"gsis_id": ["00-0000001"], "position": ["WR"], "actual_pts": [188.0]})
    id_map = pd.DataFrame(
        {
            "gsis_id": ["00-0000001"],
            "espn_id": ["E1"],
            "sleeper_id": ["S1"],
            "full_name": ["A B"],
            "position": ["WR"],
            "team": ["KC"],
        }
    )
    sleeper = pd.DataFrame({"sleeper_id": ["S1"], "sleeper_adp": [3.5]})
    frame = bench.build_benchmark_frame(espn, ours, actuals, id_map, sleeper, Ruleset.espn_ppr())
    row = frame.iloc[0]
    assert row["gsis_id"] == "00-0000001"
    assert row["espn_pts"] == 210.0
    assert row["our_pts"] == 195.0
    assert row["actual_pts"] == 188.0
    assert row["espn_adp"] == 4.0 and row["sleeper_adp"] == 3.5
    assert row["espn_pos_rank"] == 2


def test_source_metrics_drops_nan_pairs_and_computes_error() -> None:
    frame = pd.DataFrame(
        {
            "espn_pts": [100.0, 200.0, np.nan],  # 3rd row unmatched -> dropped
            "actual_pts": [110.0, 180.0, 50.0],
        }
    )
    m = bench.source_metrics(frame, "espn_pts")
    assert m["n"] == 2
    # residuals: -10, +20 -> RMSE = sqrt((100+400)/2)=sqrt(250)=15.811..., MAE=15.0
    assert round(m["rmse"], 3) == 15.811
    assert m["mae"] == 15.0


def test_top_n_by_rank_picks_smallest_rank_per_position() -> None:
    frame = pd.DataFrame(
        {
            "position": ["WR", "WR", "WR", "RB"],
            "espn_pos_rank": [1.0, 2.0, 3.0, 1.0],
            "actual_pts": [1, 2, 3, 4],
        }
    )
    top2 = bench.top_n_by_rank(frame, "espn_pos_rank", n=2)
    assert set(zip(top2["position"], top2["espn_pos_rank"], strict=False)) == {
        ("WR", 1.0),
        ("WR", 2.0),
        ("RB", 1.0),
    }
