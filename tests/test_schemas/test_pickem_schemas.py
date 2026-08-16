"""Pick'em schema tests — happy path plus one rejection per constraint."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.schemas import (
    _PYARROW_STR,
    PickemPicksSchema,
    PickemSheetSchema,
    PickemSlateSchema,
)


def _sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "home_team": pd.array(["SEA", "CAR"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE", "CHI"], dtype=_PYARROW_STR),
            # SEA favored by 3.5; CAR a 2.5-point dog.
            "home_spread": [-3.5, 2.5],
        }
    )


def _slate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": pd.array(["2026_01_NE_SEA"], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE"], dtype=_PYARROW_STR),
            "sheet_home_spread": [-3.5],
            "consensus_home_spread": [-3.0],
            "home_win_prob": [0.63],
            "away_win_prob": [0.37],
            "sheet_favorite": pd.array(["SEA"], dtype=_PYARROW_STR),
            "sheet_dog": pd.array(["NE"], dtype=_PYARROW_STR),
            "dog_win_prob": [0.37],
            "dog_line_move": [-0.5],
            "free_dog": [False],
        }
    )


def _picks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": pd.array(["2026_01_NE_SEA"], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE"], dtype=_PYARROW_STR),
            "pick": pd.array(["SEA"], dtype=_PYARROW_STR),
            "pick_win_prob": [0.63],
            "is_dog_pick": [False],
            "forced": [False],
            "switch_cost": [0.0],
            "winner": pd.array([None], dtype=_PYARROW_STR),
            "correct": pd.array([None], dtype=pd.BooleanDtype()),
        }
    )


def test_sheet_schema_accepts_valid_frame() -> None:
    PickemSheetSchema.validate(_sheet())


def test_sheet_schema_rejects_unknown_team_code() -> None:
    df = _sheet()
    df["home_team"] = pd.array(["XX", "CAR"], dtype=_PYARROW_STR)
    with pytest.raises(SchemaError):
        PickemSheetSchema.validate(df)


def test_slate_schema_accepts_valid_frame() -> None:
    PickemSlateSchema.validate(_slate())


def test_slate_schema_allows_na_dog_columns_for_a_pickem_game() -> None:
    """A sheet spread of exactly 0.0 has neither favorite nor dog."""
    df = _slate()
    df["sheet_home_spread"] = [0.0]
    df["sheet_favorite"] = pd.array([None], dtype=_PYARROW_STR)
    df["sheet_dog"] = pd.array([None], dtype=_PYARROW_STR)
    df["dog_win_prob"] = pd.array([None], dtype="float64")
    df["dog_line_move"] = pd.array([None], dtype="float64")
    validated = PickemSlateSchema.validate(df)
    assert validated["sheet_dog"].isna().all()
    assert validated["dog_win_prob"].isna().all()


@pytest.mark.parametrize("bad_prob", [-0.01, 1.01])
def test_slate_schema_rejects_probability_outside_unit_interval(bad_prob: float) -> None:
    df = _slate()
    df["home_win_prob"] = [bad_prob]
    with pytest.raises(SchemaError):
        PickemSlateSchema.validate(df)


def test_picks_schema_accepts_valid_ungraded_frame() -> None:
    PickemPicksSchema.validate(_picks())


def test_picks_schema_accepts_graded_frame() -> None:
    df = _picks()
    df["winner"] = pd.array(["SEA"], dtype=_PYARROW_STR)
    df["correct"] = pd.array([True], dtype=pd.BooleanDtype())
    PickemPicksSchema.validate(df)


def test_picks_schema_rejects_negative_switch_cost() -> None:
    """Switching to the dog can only ever cost probability, never gain it."""
    df = _picks()
    df["switch_cost"] = [-0.05]
    with pytest.raises(SchemaError):
        PickemPicksSchema.validate(df)


def test_picks_schema_correct_column_holds_na_distinctly_from_false() -> None:
    """Unplayed (NA) must stay distinguishable from wrong/tied (False), which a
    plain bool dtype would silently coerce."""
    df = _picks()
    df["correct"] = pd.array([None], dtype=pd.BooleanDtype())
    validated = PickemPicksSchema.validate(df)
    assert validated["correct"].isna().all()
    assert validated["correct"].dtype == pd.BooleanDtype()
