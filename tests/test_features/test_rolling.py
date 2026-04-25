"""Rolling-window helper tests."""

from __future__ import annotations

import pandas as pd

from projections.features._rolling import (
    last_n_per_group,
    per_game_rate,
    season_to_date_mean,
)


def test_last_n_per_group_returns_only_last_n_rows_per_group() -> None:
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A", "A", "A", "B", "B"],
            "season": [2024, 2024, 2024, 2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4, 5, 1, 2],
            "value": [10, 20, 30, 40, 50, 100, 200],
        }
    )
    out = last_n_per_group(df, group_cols=["player"], sort_cols=["season", "week"], n=3)
    a_rows = out[out["player"] == "A"].sort_values("week")
    assert a_rows["week"].tolist() == [3, 4, 5]
    b_rows = out[out["player"] == "B"].sort_values("week")
    assert b_rows["week"].tolist() == [1, 2]  # only 2 rows total, returns all


def test_last_n_per_group_handles_unsorted_input() -> None:
    """Helper sorts internally so caller order doesn't matter."""
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A"],
            "season": [2024, 2024, 2024],
            "week": [3, 1, 2],
            "value": [30, 10, 20],
        }
    )
    out = last_n_per_group(df, group_cols=["player"], sort_cols=["season", "week"], n=2)
    weeks = sorted(out["week"].tolist())
    assert weeks == [2, 3]


def test_last_n_per_group_empty_input_returns_empty() -> None:
    df = pd.DataFrame({"player": [], "season": [], "week": [], "value": []})
    out = last_n_per_group(df, group_cols=["player"], sort_cols=["season", "week"], n=4)
    assert len(out) == 0


def test_per_game_rate_handles_zero_denominator() -> None:
    df = pd.DataFrame({"num": [10, 20, 0], "denom": [2, 0, 0]})
    out = per_game_rate(df, num_col="num", denom_col="denom")
    assert out.tolist() == [5.0, 0.0, 0.0]


def test_per_game_rate_handles_missing_denom_as_zero() -> None:
    df = pd.DataFrame({"num": [10, 20], "denom": [None, 5]})
    out = per_game_rate(df, num_col="num", denom_col="denom")
    assert out.tolist() == [0.0, 4.0]


def test_season_to_date_mean_running_average_within_season() -> None:
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A", "A"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = season_to_date_mean(
        df,
        group_cols=["player", "season"],
        sort_cols=["week"],
        value_col="value",
    )
    # After week 1: 10. After week 2: 15. After week 3: 20. After week 4: 25.
    assert out.sort_index().tolist() == [10.0, 15.0, 20.0, 25.0]


def test_season_to_date_mean_resets_across_seasons() -> None:
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A", "A"],
            "season": [2023, 2023, 2024, 2024],
            "week": [16, 17, 1, 2],
            "value": [100.0, 200.0, 10.0, 20.0],
        }
    )
    out = season_to_date_mean(
        df,
        group_cols=["player", "season"],
        sort_cols=["week"],
        value_col="value",
    )
    # 2023 weeks: 100, then 150. 2024 weeks: 10, then 15.
    assert out.sort_index().tolist() == [100.0, 150.0, 10.0, 15.0]
