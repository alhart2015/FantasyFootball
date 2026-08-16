"""Slate-building tests.

Sign conventions are the highest-risk part of this pipeline — getting one
backwards inverts every pick while still producing a plausible-looking table —
so each direction is asserted explicitly rather than inferred.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.pickem.slate import build_slate
from projections.schemas import _PYARROW_STR, PickemSlateSchema


def _schedules(
    *,
    spread_line: float = 3.5,
    home_moneyline: int = -190,
    away_moneyline: int = 160,
) -> pd.DataFrame:
    """One game: NE at SEA. `spread_line=+3.5` means SEA (home) favored by 3.5."""
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": pd.array(["2026_01_NE_SEA"], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE"], dtype=_PYARROW_STR),
            "spread_line": [spread_line],
            "home_moneyline": pd.array([home_moneyline], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([away_moneyline], dtype=pd.Int64Dtype()),
        }
    )


def _sheet(home_spread: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "home_team": pd.array(["SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE"], dtype=_PYARROW_STR),
            "home_spread": [home_spread],
        }
    )


# --- who the sheet calls the underdog --------------------------------------


def test_home_favored_on_the_sheet_makes_the_away_team_the_dog() -> None:
    slate = build_slate(_sheet(-3.5), _schedules())
    assert slate.loc[0, "sheet_favorite"] == "SEA"
    assert slate.loc[0, "sheet_dog"] == "NE"


def test_away_favored_on_the_sheet_makes_the_home_team_the_dog() -> None:
    slate = build_slate(_sheet(2.5), _schedules())
    assert slate.loc[0, "sheet_favorite"] == "NE"
    assert slate.loc[0, "sheet_dog"] == "SEA"


def test_zero_sheet_spread_has_neither_favorite_nor_dog() -> None:
    slate = build_slate(_sheet(0.0), _schedules())
    assert pd.isna(slate.loc[0, "sheet_favorite"])
    assert pd.isna(slate.loc[0, "sheet_dog"])
    assert pd.isna(slate.loc[0, "dog_win_prob"])
    assert pd.isna(slate.loc[0, "dog_line_move"])
    assert not slate.loc[0, "free_dog"]


# --- consensus spread sign --------------------------------------------------


def test_consensus_spread_is_the_negation_of_spread_line() -> None:
    """`spread_line=+3.5` (home favored) must become `consensus_home_spread=-3.5`
    in standard convention. This is the conversion the whole tool hinges on."""
    slate = build_slate(_sheet(-3.5), _schedules(spread_line=3.5))
    assert slate.loc[0, "consensus_home_spread"] == -3.5


def test_consensus_spread_sign_for_an_away_favorite() -> None:
    slate = build_slate(_sheet(2.5), _schedules(spread_line=-2.5))
    assert slate.loc[0, "consensus_home_spread"] == 2.5


def test_market_probability_agrees_with_the_consensus_spread_direction() -> None:
    slate = build_slate(_sheet(-3.5), _schedules(spread_line=3.5))
    assert slate.loc[0, "consensus_home_spread"] < 0  # home favored
    assert slate.loc[0, "home_win_prob"] > slate.loc[0, "away_win_prob"]


# --- line movement ----------------------------------------------------------


def test_dog_line_move_is_positive_when_the_dog_improved() -> None:
    """Sheet has SEA as a +7 dog; the market now has them +3. The dog is four
    points stronger than the organizer thinks."""
    slate = build_slate(_sheet(7.0), _schedules(spread_line=-3.0))
    assert slate.loc[0, "sheet_dog"] == "SEA"
    assert slate.loc[0, "dog_line_move"] == pytest.approx(4.0)


def test_dog_line_move_is_negative_when_the_dog_got_worse() -> None:
    slate = build_slate(_sheet(7.0), _schedules(spread_line=-10.0))
    assert slate.loc[0, "dog_line_move"] == pytest.approx(-3.0)


def test_dog_line_move_for_an_away_dog() -> None:
    """Sheet has NE as a +6 dog (home -6); market has NE +2."""
    slate = build_slate(_sheet(-6.0), _schedules(spread_line=2.0))
    assert slate.loc[0, "sheet_dog"] == "NE"
    assert slate.loc[0, "dog_line_move"] == pytest.approx(4.0)


def test_dog_line_move_is_zero_when_the_line_did_not_move() -> None:
    slate = build_slate(_sheet(-3.5), _schedules(spread_line=3.5))
    assert slate.loc[0, "dog_line_move"] == pytest.approx(0.0)


# --- free dogs --------------------------------------------------------------


def test_free_dog_when_the_market_favors_the_sheets_underdog() -> None:
    """The whole point of the tool: the sheet calls SEA a 2-point dog, but by
    Thursday the market makes them the favorite. That fills a required dog slot
    at zero cost."""
    slate = build_slate(
        _sheet(2.0), _schedules(spread_line=1.0, home_moneyline=-130, away_moneyline=110)
    )
    assert slate.loc[0, "sheet_dog"] == "SEA"
    assert slate.loc[0, "dog_win_prob"] > 0.5
    assert bool(slate.loc[0, "free_dog"])


def test_not_a_free_dog_when_the_market_agrees_with_the_sheet() -> None:
    slate = build_slate(_sheet(-3.5), _schedules(spread_line=3.5))
    assert slate.loc[0, "dog_win_prob"] < 0.5
    assert not bool(slate.loc[0, "free_dog"])


def test_dog_win_prob_tracks_the_dog_side_not_the_home_side() -> None:
    """When the sheet's dog is the home team, `dog_win_prob` must be the HOME
    probability — the commonest way to get this backwards."""
    slate = build_slate(_sheet(2.5), _schedules())
    assert slate.loc[0, "sheet_dog"] == "SEA"
    assert slate.loc[0, "dog_win_prob"] == pytest.approx(slate.loc[0, "home_win_prob"])


# --- structure and failure modes -------------------------------------------


def test_slate_validates_against_the_schema() -> None:
    PickemSlateSchema.validate(build_slate(_sheet(-3.5), _schedules()))


def test_probabilities_sum_to_one() -> None:
    slate = build_slate(_sheet(-3.5), _schedules())
    assert slate.loc[0, "home_win_prob"] + slate.loc[0, "away_win_prob"] == pytest.approx(1.0)


def test_unmatched_sheet_row_raises_naming_the_matchup() -> None:
    sheet = _sheet(-3.5)
    sheet["away_team"] = pd.array(["KC"], dtype=_PYARROW_STR)  # KC does not play SEA
    with pytest.raises(ValueError, match=r"KC@SEA"):
        build_slate(sheet, _schedules())


def test_wrong_week_raises() -> None:
    sheet = _sheet(-3.5)
    sheet["week"] = [2]
    with pytest.raises(ValueError, match="no matching scheduled game"):
        build_slate(sheet, _schedules())


def test_scheduled_games_absent_from_the_sheet_are_simply_omitted() -> None:
    """The sheet defines the slate — the organizer may leave a game off."""
    schedules = pd.concat(
        [
            _schedules(),
            pd.DataFrame(
                {
                    "season": [2026],
                    "week": [1],
                    "game_id": pd.array(["2026_01_CHI_CAR"], dtype=_PYARROW_STR),
                    "home_team": pd.array(["CAR"], dtype=_PYARROW_STR),
                    "away_team": pd.array(["CHI"], dtype=_PYARROW_STR),
                    "spread_line": [-2.5],
                    "home_moneyline": pd.array([125], dtype=pd.Int64Dtype()),
                    "away_moneyline": pd.array([-145], dtype=pd.Int64Dtype()),
                }
            ),
        ],
        ignore_index=True,
    )
    slate = build_slate(_sheet(-3.5), schedules)
    assert list(slate["game_id"]) == ["2026_01_NE_SEA"]


def test_future_games_without_prices_elsewhere_do_not_break_the_join() -> None:
    """Win probabilities are computed after the join, so an unpriced week-8 game
    sitting in the same schedules frame must not raise."""
    schedules = pd.concat(
        [
            _schedules(),
            pd.DataFrame(
                {
                    "season": [2026],
                    "week": [8],
                    "game_id": pd.array(["2026_08_CHI_CAR"], dtype=_PYARROW_STR),
                    "home_team": pd.array(["CAR"], dtype=_PYARROW_STR),
                    "away_team": pd.array(["CHI"], dtype=_PYARROW_STR),
                    "spread_line": [float("nan")],
                    "home_moneyline": pd.array([None], dtype=pd.Int64Dtype()),
                    "away_moneyline": pd.array([None], dtype=pd.Int64Dtype()),
                }
            ),
        ],
        ignore_index=True,
    )
    slate = build_slate(_sheet(-3.5), schedules)
    assert len(slate) == 1
