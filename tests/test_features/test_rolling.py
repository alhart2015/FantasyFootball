"""Rolling-window helper tests."""

from __future__ import annotations

import pandas as pd

from projections.features._rolling import (
    last_n_per_group,
    per_game_rate,
    season_to_date_mean,
    trailing_n_share_in_group,
)
from projections.schemas import _PYARROW_STR


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


def _share_input_two_teams() -> pd.DataFrame:
    """Two teams, two players each, trailing-4 sums easy to verify by hand."""
    rows = []
    # Team A: player A1 = 4 + 4 + 4 + 4 = 16; player A2 = 1 + 1 + 1 + 1 = 4. Team total = 20.
    # Team B: player B1 = 5 + 5 + 5 + 5 = 20; player B2 = 5 + 5 + 5 + 5 = 20. Team total = 40.
    for week in range(1, 5):
        rows.extend(
            [
                {"gsis_id": "00-000A001", "season": 2024, "week": week, "team": "A", "value": 4},
                {"gsis_id": "00-000A002", "season": 2024, "week": week, "team": "A", "value": 1},
                {"gsis_id": "00-000B001", "season": 2024, "week": week, "team": "B", "value": 5},
                {"gsis_id": "00-000B002", "season": 2024, "week": week, "team": "B", "value": 5},
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    return df


def test_trailing_n_share_in_group_basic_shares() -> None:
    df = _share_input_two_teams()
    out = trailing_n_share_in_group(df, value_col="value", n=4)
    assert set(out.columns) == {"gsis_id", "share_l4"}
    by_id = out.set_index("gsis_id")["share_l4"]
    assert by_id["00-000A001"] == 16 / 20  # 0.8
    assert by_id["00-000A002"] == 4 / 20  # 0.2
    assert by_id["00-000B001"] == 20 / 40  # 0.5
    assert by_id["00-000B002"] == 20 / 40  # 0.5


def test_trailing_n_share_in_group_zero_team_total_yields_zero() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-000A001"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [1],
            "team": pd.array(["A"], dtype=_PYARROW_STR),
            "value": [0],
        }
    )
    out = trailing_n_share_in_group(df, value_col="value", n=4)
    assert out.loc[out["gsis_id"] == "00-000A001", "share_l4"].iloc[0] == 0.0


def test_trailing_n_share_in_group_empty_input() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array([], dtype=_PYARROW_STR),
            "season": pd.array([], dtype=int),
            "week": pd.array([], dtype=int),
            "team": pd.array([], dtype=_PYARROW_STR),
            "value": pd.array([], dtype=float),
        }
    )
    out = trailing_n_share_in_group(df, value_col="value", n=4)
    assert len(out) == 0
    assert set(out.columns) == {"gsis_id", "share_l4"}


def test_trailing_n_share_in_group_default_n_is_4() -> None:
    """Calling without n= uses n=4 (the established convention)."""
    df = _share_input_two_teams()
    explicit = trailing_n_share_in_group(df, value_col="value", n=4)
    default = trailing_n_share_in_group(df, value_col="value")
    pd.testing.assert_frame_equal(explicit, default)
