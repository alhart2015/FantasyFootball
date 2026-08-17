"""Grading tests — wins, losses, ties, and games that have not been played."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.pickem.grade import grade_picks, record
from projections.schemas import _PYARROW_STR, PickemPicksSchema


def _picks(picks: list[str]) -> pd.DataFrame:
    n = len(picks)
    return pd.DataFrame(
        {
            "season": [2026] * n,
            "week": [1] * n,
            "game_id": pd.array([f"2026_01_A{i}_B{i}" for i in range(n)], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA", "CAR", "LAR"][:n], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE", "CHI", "SF"][:n], dtype=_PYARROW_STR),
            "pick": pd.array(picks, dtype=_PYARROW_STR),
            "pick_win_prob": [0.6] * n,
            "is_dog_pick": [False] * n,
            "forced": [False] * n,
            "switch_cost": [0.0] * n,
            "winner": pd.array([None] * n, dtype=_PYARROW_STR),
            "correct": pd.array([None] * n, dtype=pd.BooleanDtype()),
        }
    )


def _schedules(scores: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.array(
                [f"2026_01_A{i}_B{i}" for i in range(len(scores))], dtype=_PYARROW_STR
            ),
            "home_score": pd.array([s[0] for s in scores], dtype=pd.Int64Dtype()),
            "away_score": pd.array([s[1] for s in scores], dtype=pd.Int64Dtype()),
        }
    )


def test_correct_pick_of_the_home_winner() -> None:
    graded = grade_picks(_picks(["SEA"]), _schedules([(24, 17)]))
    assert graded.loc[0, "winner"] == "SEA"
    assert graded.loc[0, "correct"] is True or bool(graded.loc[0, "correct"])


def test_correct_pick_of_the_away_winner() -> None:
    graded = grade_picks(_picks(["NE"]), _schedules([(17, 24)]))
    assert graded.loc[0, "winner"] == "NE"
    assert bool(graded.loc[0, "correct"])


def test_wrong_pick_is_false_not_na() -> None:
    graded = grade_picks(_picks(["NE"]), _schedules([(24, 17)]))
    assert graded.loc[0, "winner"] == "SEA"
    assert graded.loc[0, "correct"] is not pd.NA
    assert not bool(graded.loc[0, "correct"])


def test_tie_has_no_winner_and_counts_as_incorrect() -> None:
    """13 regular-season games since 2010 ended tied. A tie is a loss here: we
    picked a team to win and it did not."""
    graded = grade_picks(_picks(["SEA"]), _schedules([(20, 20)]))
    assert pd.isna(graded.loc[0, "winner"])
    assert graded.loc[0, "correct"] is not pd.NA
    assert not bool(graded.loc[0, "correct"])


def test_unplayed_game_is_na_not_false() -> None:
    """Mid-week, an ungraded game must not read as a loss."""
    graded = grade_picks(_picks(["SEA"]), _schedules([(float("nan"), float("nan"))]))
    assert pd.isna(graded.loc[0, "winner"])
    assert pd.isna(graded.loc[0, "correct"])


def test_tie_and_unplayed_are_distinguishable() -> None:
    """Both have a NA winner; only `correct` separates them."""
    graded = grade_picks(
        _picks(["SEA", "CAR"]), _schedules([(20, 20), (float("nan"), float("nan"))])
    )
    assert pd.isna(graded.loc[0, "winner"]) and pd.isna(graded.loc[1, "winner"])
    assert not bool(graded.loc[0, "correct"])  # tie -> False
    assert pd.isna(graded.loc[1, "correct"])  # unplayed -> NA


def test_mixed_week_grades_each_row_independently() -> None:
    graded = grade_picks(
        _picks(["SEA", "CHI", "LAR"]),
        _schedules([(24, 17), (10, 31), (14, 21)]),
    )
    assert list(graded["winner"]) == ["SEA", "CHI", "SF"]
    assert [bool(c) for c in graded["correct"]] == [True, True, False]


def test_graded_frame_still_validates_and_keeps_the_pick_columns() -> None:
    graded = grade_picks(_picks(["SEA"]), _schedules([(24, 17)]))
    PickemPicksSchema.validate(graded)
    assert graded.loc[0, "pick"] == "SEA"
    assert graded.loc[0, "pick_win_prob"] == pytest.approx(0.6)


def test_grade_does_not_mutate_the_input() -> None:
    picks = _picks(["SEA"])
    grade_picks(picks, _schedules([(24, 17)]))
    assert picks["correct"].isna().all()


def test_raises_when_score_columns_are_absent() -> None:
    schedules = pd.DataFrame({"game_id": pd.array(["2026_01_A0_B0"], dtype=_PYARROW_STR)})
    with pytest.raises(ValueError, match="refresh_schedules"):
        grade_picks(_picks(["SEA"]), schedules)


# --- record ----------------------------------------------------------------


def test_record_counts_only_played_games() -> None:
    graded = grade_picks(
        _picks(["SEA", "CHI", "LAR"]),
        _schedules([(24, 17), (10, 31), (float("nan"), float("nan"))]),
    )
    assert record(graded) == (2, 2)


def test_record_counts_a_tie_as_played_but_not_correct() -> None:
    graded = grade_picks(_picks(["SEA", "CHI"]), _schedules([(20, 20), (10, 31)]))
    assert record(graded) == (1, 2)
