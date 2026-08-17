"""Win-probability tests — American odds, devigging, and the frame-level guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.pickem.probability import (
    add_win_probs,
    american_to_implied,
    devig_pair,
)
from projections.schemas import _PYARROW_STR


def _schedules(
    *,
    home_moneyline: list[int | None],
    away_moneyline: list[int | None],
) -> pd.DataFrame:
    n = len(home_moneyline)
    return pd.DataFrame(
        {
            "game_id": pd.array([f"2026_01_A{i}_B{i}" for i in range(n)], dtype=_PYARROW_STR),
            "home_moneyline": pd.array(home_moneyline, dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array(away_moneyline, dtype=pd.Int64Dtype()),
        }
    )


# --- american_to_implied ---------------------------------------------------


def test_even_money_is_one_half() -> None:
    assert american_to_implied(100) == pytest.approx(0.5)


def test_minus_110_is_the_standard_vigged_price() -> None:
    """-110 both sides is the canonical book price: each side implies ~52.4%."""
    assert american_to_implied(-110) == pytest.approx(110 / 210)


def test_favorite_and_underdog_directions() -> None:
    assert american_to_implied(-300) > 0.5  # heavy favorite
    assert american_to_implied(300) < 0.5  # heavy underdog


def test_zero_odds_rejected() -> None:
    with pytest.raises(ValueError, match="not a valid price"):
        american_to_implied(0)


# --- devig_pair ------------------------------------------------------------


def test_devig_of_a_balanced_market_is_fifty_fifty() -> None:
    home, away = devig_pair(-110, -110)
    assert home == pytest.approx(0.5)
    assert away == pytest.approx(0.5)


def test_devigged_pair_sums_to_one() -> None:
    home, away = devig_pair(-148, 124)
    assert home + away == pytest.approx(1.0)


def test_devig_removes_the_margin_rather_than_keeping_it() -> None:
    """The raw implied probabilities sum to more than 1 — that excess is the
    book's cut. The fair pair must be strictly smaller than the raw pair."""
    raw_home = american_to_implied(-148)
    raw_away = american_to_implied(124)
    assert raw_home + raw_away > 1.0

    home, away = devig_pair(-148, 124)
    assert home < raw_home
    assert away < raw_away


def test_favorite_keeps_the_larger_share_after_devig() -> None:
    home, away = devig_pair(-185, 154)
    assert home > away
    # Raw: 185/285 = 0.64912, 100/254 = 0.39370, summing to 1.04282 (the vig).
    # Fair: 0.64912 / 1.04282 = 0.62247.
    assert home == pytest.approx(0.62247, abs=1e-5)


def test_devig_is_symmetric_under_swapping_sides() -> None:
    home, away = devig_pair(-185, 154)
    swapped_home, swapped_away = devig_pair(154, -185)
    assert swapped_home == pytest.approx(away)
    assert swapped_away == pytest.approx(home)


# --- add_win_probs ---------------------------------------------------------


def test_add_win_probs_attaches_both_columns_summing_to_one() -> None:
    df = add_win_probs(_schedules(home_moneyline=[-190, 125], away_moneyline=[160, -145]))
    assert (df["home_win_prob"] + df["away_win_prob"]).round(9).eq(1.0).all()
    # Row 0: home is the favorite. Row 1: away is.
    assert df.loc[0, "home_win_prob"] > df.loc[0, "away_win_prob"]
    assert df.loc[1, "away_win_prob"] > df.loc[1, "home_win_prob"]


def test_add_win_probs_does_not_mutate_the_input() -> None:
    original = _schedules(home_moneyline=[-190], away_moneyline=[160])
    add_win_probs(original)
    assert "home_win_prob" not in original.columns


def test_add_win_probs_raises_on_missing_moneyline_naming_the_game() -> None:
    df = _schedules(home_moneyline=[-190, None], away_moneyline=[160, -145])
    with pytest.raises(ValueError, match="2026_01_A1_B1"):
        add_win_probs(df)


def test_add_win_probs_raises_on_zero_odds() -> None:
    """`np.where` evaluates both branches, so a 0 would otherwise slip through
    the positive branch as a silent 1.0 probability."""
    df = _schedules(home_moneyline=[0], away_moneyline=[160])
    with pytest.raises(ValueError, match="not a valid price"):
        add_win_probs(df)


def test_add_win_probs_raises_when_the_moneyline_columns_are_absent() -> None:
    df = pd.DataFrame({"game_id": pd.array(["2026_01_A_B"], dtype=_PYARROW_STR)})
    with pytest.raises(ValueError, match="refresh_schedules"):
        add_win_probs(df)


def test_add_win_probs_matches_devig_pair_row_by_row() -> None:
    """The vectorized path and the scalar path must not drift apart."""
    home_ml = [-190, 125, -300, 105]
    away_ml = [160, -145, 250, -125]
    df = add_win_probs(_schedules(home_moneyline=list(home_ml), away_moneyline=list(away_ml)))
    for i, (h, a) in enumerate(zip(home_ml, away_ml, strict=True)):
        expected_home, expected_away = devig_pair(h, a)
        assert df.loc[i, "home_win_prob"] == pytest.approx(expected_home)
        assert df.loc[i, "away_win_prob"] == pytest.approx(expected_away)


def test_even_money_odds_do_not_warn_or_produce_inf() -> None:
    """A moneyline of exactly +100/-100 is a common price. Computing both
    branches of the favorite/underdog formula for every row divides by zero on
    the discarded side — silently correct, but it raises a RuntimeWarning and
    builds an inf on the way through. Regression test for that.
    """
    import warnings

    df = _schedules(home_moneyline=[100, -100, 105], away_moneyline=[-120, 100, -125])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = add_win_probs(df)

    assert np.isfinite(out["home_win_prob"]).all()
    assert np.isfinite(out["away_win_prob"]).all()
    # -100 and +100 are the same price, so an all-even market is a coin flip.
    assert out.loc[1, "home_win_prob"] == pytest.approx(0.5)


def test_add_win_probs_requires_game_id_for_its_own_error_messages() -> None:
    """Both raise paths name the offending game_id, so a frame without it would
    die on a bare KeyError instead of the diagnostic this module promises."""
    df = pd.DataFrame(
        {
            "home_moneyline": pd.array([-190], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([160], dtype=pd.Int64Dtype()),
        }
    )
    with pytest.raises(ValueError, match="game_id"):
        add_win_probs(df)
