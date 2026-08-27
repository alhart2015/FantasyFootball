"""The injury adjustments.

Small module, dense tests, because these constants are multiplied into every number the waiver
recommender prints and a wrong one is a plausible-looking answer rather than an error.
"""

from __future__ import annotations

import pytest

from projections.midseason.injuries import (
    WEEKLY_MULTIPLIER,
    expected_games_missed,
    is_multi_week,
    season_multiplier,
    weekly_multiplier,
)
from projections.schemas import InjuryStatus

# --- the double-discount guard, which is the whole reason this module has a flag ---------------


def test_espn_priced_statuses_are_not_discounted_twice() -> None:
    """ESPN's weekly feed already zeroes `Out` players -- of 1,475 such designations reaching
    the feed, exactly one carried a projection above five points.

    Applying our multiplier on top is a second discount on an already-discounted number, and
    the result looks perfectly plausible, which is what makes it dangerous.
    """
    for status in (InjuryStatus.OUT, InjuryStatus.DOUBTFUL, InjuryStatus.INJURY_RESERVE):
        assert weekly_multiplier(status, source_is_injury_aware=True) == 1.0, status
        assert weekly_multiplier(status, source_is_injury_aware=False) < 0.1, status


def test_questionable_is_discounted_even_against_espns_own_feed() -> None:
    """The measured asymmetry: ESPN prices `Out` and verifiably does NOT price `Questionable`.
    A Questionable player's projection is 100.4% of his own healthy-week median (n=843), so the
    14% shortfall he actually delivers is ours to apply on either source."""
    aware = weekly_multiplier(InjuryStatus.QUESTIONABLE, source_is_injury_aware=True)
    unaware = weekly_multiplier(InjuryStatus.QUESTIONABLE, source_is_injury_aware=False)
    assert aware == unaware == pytest.approx(0.86)


# --- healthy means healthy ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        InjuryStatus.ACTIVE,
        InjuryStatus.NORMAL,
        InjuryStatus.DAY_TO_DAY,
        InjuryStatus.FREE_AGENT,
        InjuryStatus.UNKNOWN,
    ],
)
def test_a_healthy_designation_costs_nothing(status: InjuryStatus) -> None:
    assert weekly_multiplier(status) == 1.0
    assert expected_games_missed(status) == 0.0
    assert season_multiplier(status, games_remaining=8) == 1.0


def test_an_unrecognised_status_costs_nothing_rather_than_guessing() -> None:
    """ESPN adds statuses. One we cannot place is a gap in our mapping, not evidence about the
    player -- inventing an absence from it would quietly move every number he touches."""
    assert weekly_multiplier(InjuryStatus.UNKNOWN) == 1.0
    assert season_multiplier(InjuryStatus.UNKNOWN, games_remaining=10) == 1.0


# --- the two horizons disagree on purpose ------------------------------------------------------


def test_questionable_barely_moves_a_season_number_but_moves_a_weekly_one() -> None:
    """The finding that reshaped the design. The same designation is nearly irrelevant over a
    rest-of-season horizon and decisive over a single week, so the tool needs both and they
    must not be collapsed into one constant."""
    weekly = 1.0 - weekly_multiplier(InjuryStatus.QUESTIONABLE)
    season = 1.0 - season_multiplier(InjuryStatus.QUESTIONABLE, games_remaining=10)
    assert weekly == pytest.approx(0.14), "a 14% cut on one week decides a close start/sit"
    assert season < 0.02, "the same tag over ten weeks is noise"
    assert weekly > season * 5


def test_injured_reserve_is_the_designation_that_moves_a_season_number() -> None:
    """And it is the only guessed constant, which is why the tool shows the write-up for it."""
    assert season_multiplier(InjuryStatus.INJURY_RESERVE, games_remaining=10) == pytest.approx(0.6)
    assert is_multi_week(InjuryStatus.INJURY_RESERVE)
    assert not is_multi_week(InjuryStatus.QUESTIONABLE)
    assert not is_multi_week(InjuryStatus.OUT), "a game status covers one week; ESPN re-reports"


# --- the edges ---------------------------------------------------------------------------------


def test_a_player_cannot_miss_more_games_than_remain() -> None:
    """IR assumes four games missed. With two left, the answer is zero, not a negative share --
    an unclamped ratio would hand the simulator a roster worth less than nothing."""
    assert season_multiplier(InjuryStatus.INJURY_RESERVE, games_remaining=2) == 0.0
    assert season_multiplier(InjuryStatus.INJURY_RESERVE, games_remaining=4) == 0.0
    assert season_multiplier(InjuryStatus.INJURY_RESERVE, games_remaining=5) == pytest.approx(0.2)


def test_no_games_left_delivers_nothing_regardless_of_health() -> None:
    """A different statement from "he is fine": there is nothing left to deliver."""
    assert season_multiplier(InjuryStatus.ACTIVE, games_remaining=0) == 0.0
    assert season_multiplier(InjuryStatus.OUT, games_remaining=-1) == 0.0


def test_every_multiplier_is_a_share() -> None:
    for status, value in WEEKLY_MULTIPLIER.items():
        assert 0.0 <= value <= 1.0, f"{status} multiplier {value} is not a share"


def test_a_one_week_designation_costs_the_same_on_both_horizons() -> None:
    """The BEHAVIOUR the two tables are supposed to share, rather than the arithmetic that
    happens to produce one from the other.

    With exactly one game left, "what share of a rest-of-season total does he deliver" and
    "what share of this week's projection" are the same question, so the two tables must give
    the same answer. An earlier version of this test asserted
    `EXPECTED_GAMES_MISSED[s] == 1 - WEEKLY_MULTIPLIER[s]`, which for two of the four statuses
    re-executed the line that defines it and for the other two compared two literals that
    happen to agree -- it could not fail for the reason its docstring gave.
    """
    for status in (InjuryStatus.QUESTIONABLE, InjuryStatus.DOUBTFUL, InjuryStatus.OUT):
        assert season_multiplier(status, games_remaining=1) == pytest.approx(
            weekly_multiplier(status)
        ), status
