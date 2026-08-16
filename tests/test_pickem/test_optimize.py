"""Optimizer tests, including a brute-force check that greedy really is optimal."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from projections.pickem.optimize import choose_picks, expected_correct
from projections.schemas import _PYARROW_STR, PickemPicksSchema

# 16 distinct canonical codes, so slates of up to 8 games get unique matchups.
_TEAMS = [
    "SEA",
    "CAR",
    "LAR",
    "CIN",
    "DET",
    "HOU",
    "IND",
    "JAC",
    "KC",
    "BUF",
    "MIA",
    "NYJ",
    "PHI",
    "DAL",
    "GB",
    "MIN",
]


def _slate(
    *,
    home_probs: list[float],
    sheet_home_spreads: list[float],
) -> pd.DataFrame:
    """Build a slate directly, bypassing the join.

    Home team is `_TEAMS[2i]`, away is `_TEAMS[2i+1]`, so every game has a
    distinct pair and picks are unambiguous.
    """
    n = len(home_probs)
    home = [_TEAMS[(2 * i) % len(_TEAMS)] for i in range(n)]
    away = [_TEAMS[(2 * i + 1) % len(_TEAMS)] for i in range(n)]

    dogs: list[str | None] = []
    favorites: list[str | None] = []
    dog_probs: list[float] = []
    for i, spread in enumerate(sheet_home_spreads):
        if spread < 0:  # home favored -> away is the dog
            favorites.append(home[i])
            dogs.append(away[i])
            dog_probs.append(1.0 - home_probs[i])
        elif spread > 0:  # away favored -> home is the dog
            favorites.append(away[i])
            dogs.append(home[i])
            dog_probs.append(home_probs[i])
        else:  # a true pick'em has no dog
            favorites.append(None)
            dogs.append(None)
            dog_probs.append(float("nan"))

    return pd.DataFrame(
        {
            "season": [2026] * n,
            "week": [1] * n,
            "game_id": pd.array(
                [f"2026_01_{away[i]}_{home[i]}" for i in range(n)], dtype=_PYARROW_STR
            ),
            "home_team": pd.array(home, dtype=_PYARROW_STR),
            "away_team": pd.array(away, dtype=_PYARROW_STR),
            "sheet_home_spread": sheet_home_spreads,
            "consensus_home_spread": sheet_home_spreads,
            "home_win_prob": home_probs,
            "away_win_prob": [1.0 - p for p in home_probs],
            "sheet_favorite": pd.array(favorites, dtype=_PYARROW_STR),
            "sheet_dog": pd.array(dogs, dtype=_PYARROW_STR),
            "dog_win_prob": dog_probs,
            "dog_line_move": [0.0] * n,
            "free_dog": [p > 0.5 for p in dog_probs],
        }
    )


# --- the constraint ---------------------------------------------------------


def test_all_favorites_forces_exactly_three_swaps() -> None:
    """Every home team is a big favorite on both sheet and market, so no dog is
    picked naturally and the constraint has to bite exactly three times."""
    slate = _slate(
        home_probs=[0.90, 0.85, 0.80, 0.75, 0.70, 0.65],
        sheet_home_spreads=[-10.0] * 6,
    )
    picks = choose_picks(slate)
    assert int(picks["forced"].sum()) == 3
    assert int(picks["is_dog_pick"].sum()) == 3


def test_the_three_cheapest_swaps_are_the_ones_taken() -> None:
    """Cost of switching is P(favorite) - P(dog), so the closest games are
    cheapest. Here that is the 0.65, 0.70 and 0.75 games."""
    slate = _slate(
        home_probs=[0.90, 0.85, 0.80, 0.75, 0.70, 0.65],
        sheet_home_spreads=[-10.0] * 6,
    )
    picks = choose_picks(slate)
    # Switch cost is 2*home_prob - 1 here, so the three closest games are the
    # cheapest: those with home probabilities 0.75, 0.70, 0.65.
    assert set(picks.loc[picks["forced"], "game_id"]) == set(slate.loc[3:5, "game_id"])
    # Having switched, we hold the dog side of each: 0.25, 0.30, 0.35.
    assert sorted(picks.loc[picks["forced"], "pick_win_prob"]) == pytest.approx([0.25, 0.30, 0.35])


def test_no_swaps_when_enough_dogs_are_already_the_best_pick() -> None:
    """Sheet calls the away team the favorite in four games, but the market
    likes the home side. Those dogs come for free."""
    slate = _slate(
        home_probs=[0.80, 0.75, 0.70, 0.65, 0.90, 0.95],
        sheet_home_spreads=[3.0, 3.0, 3.0, 3.0, -10.0, -10.0],
    )
    picks = choose_picks(slate)
    assert int(picks["is_dog_pick"].sum()) == 4
    assert int(picks["forced"].sum()) == 0
    assert picks["switch_cost"].eq(0.0).all()


def test_free_dogs_are_picked_but_not_marked_forced() -> None:
    """A dog the market already favors costs nothing, so calling it forced
    would overstate what the constraint took."""
    slate = _slate(
        home_probs=[0.80, 0.80, 0.80, 0.90, 0.90],
        sheet_home_spreads=[2.0, 2.0, 2.0, -10.0, -10.0],
    )
    picks = choose_picks(slate)
    free = picks.loc[picks["is_dog_pick"]]
    assert len(free) == 3
    assert not free["forced"].any()
    assert expected_correct(picks) == pytest.approx(0.80 * 3 + 0.90 * 2)


def test_min_dogs_is_configurable() -> None:
    slate = _slate(home_probs=[0.9, 0.85, 0.8, 0.75, 0.7], sheet_home_spreads=[-10.0] * 5)
    assert int(choose_picks(slate, min_dogs=0)["is_dog_pick"].sum()) == 0
    assert int(choose_picks(slate, min_dogs=5)["is_dog_pick"].sum()) == 5


def test_negative_min_dogs_rejected() -> None:
    slate = _slate(home_probs=[0.9], sheet_home_spreads=[-10.0])
    with pytest.raises(ValueError, match="non-negative"):
        choose_picks(slate, min_dogs=-1)


# --- pick'em games ----------------------------------------------------------


def test_zero_spread_games_cannot_satisfy_the_constraint() -> None:
    """A sheet spread of exactly 0 has no underdog, so those games must be
    skipped over in favour of real dogs even when they look cheaper."""
    slate = _slate(
        home_probs=[0.51, 0.52, 0.53, 0.90, 0.85, 0.80],
        sheet_home_spreads=[0.0, 0.0, 0.0, -10.0, -10.0, -10.0],
    )
    picks = choose_picks(slate)
    pickem_games = picks["game_id"].isin(slate.loc[:2, "game_id"])
    assert not picks.loc[pickem_games, "forced"].any()
    assert int(picks["forced"].sum()) == 3
    # The three real-dog games are the ones switched, despite being lopsided.
    assert set(picks.loc[picks["forced"], "game_id"]) == set(slate.loc[3:, "game_id"])


def test_too_few_eligible_dog_games_raises() -> None:
    slate = _slate(home_probs=[0.9, 0.8, 0.7], sheet_home_spreads=[-10.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"need 3 more underdog pick\(s\) but only 1"):
        choose_picks(slate)


# --- determinism and structure ---------------------------------------------


def test_ties_in_switch_cost_break_deterministically() -> None:
    """Three identical games, two swaps needed: the same two must be chosen
    every run, or week-to-week results stop being reproducible."""
    slate = _slate(home_probs=[0.7, 0.7, 0.7, 0.9], sheet_home_spreads=[-5.0] * 4)
    first = choose_picks(slate, min_dogs=2)
    for _ in range(5):
        assert list(choose_picks(slate, min_dogs=2)["pick"]) == list(first["pick"])


def test_picks_validate_and_start_ungraded() -> None:
    picks = choose_picks(_slate(home_probs=[0.9] * 5, sheet_home_spreads=[-7.0] * 5))
    PickemPicksSchema.validate(picks)
    assert picks["winner"].isna().all()
    assert picks["correct"].isna().all()


def test_pick_win_prob_matches_the_side_actually_picked() -> None:
    slate = _slate(home_probs=[0.9, 0.85, 0.6, 0.55, 0.52], sheet_home_spreads=[-7.0] * 5)
    picks = choose_picks(slate)
    for row in picks.itertuples():
        game = slate[slate["game_id"] == row.game_id].iloc[0]
        expected = game["home_win_prob"] if row.pick == row.home_team else game["away_win_prob"]
        assert row.pick_win_prob == pytest.approx(expected)


def test_switch_cost_is_zero_for_unforced_picks_and_positive_when_it_bites() -> None:
    slate = _slate(home_probs=[0.9, 0.85, 0.8, 0.75], sheet_home_spreads=[-7.0] * 4)
    picks = choose_picks(slate)
    assert picks.loc[~picks["forced"], "switch_cost"].eq(0.0).all()
    assert (picks.loc[picks["forced"], "switch_cost"] > 0).all()


# --- brute force ------------------------------------------------------------


def _brute_force_best(slate: pd.DataFrame, min_dogs: int) -> float:
    """Enumerate every home/away combination, keep the feasible ones, return the
    best achievable expected total."""
    best = float("-inf")
    n = len(slate)
    for combo in itertools.product([True, False], repeat=n):  # True = pick home
        total = 0.0
        dogs = 0
        for i, take_home in enumerate(combo):
            row = slate.iloc[i]
            team = row["home_team"] if take_home else row["away_team"]
            total += row["home_win_prob"] if take_home else row["away_win_prob"]
            if pd.notna(row["sheet_dog"]) and team == row["sheet_dog"]:
                dogs += 1
        if dogs >= min_dogs:
            best = max(best, total)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_greedy_matches_brute_force_optimum(seed: int) -> None:
    """The optimality argument in the module docstring, checked exhaustively.

    Eight games, so 256 combinations per seed. Spreads are drawn so that some
    slates contain free dogs, some contain pick'em games with no dog at all,
    and the market frequently disagrees with the sheet.
    """
    rng = np.random.default_rng(seed)
    n = 8
    home_probs = [float(p) for p in rng.uniform(0.15, 0.85, size=n)]
    spreads = [float(s) for s in rng.choice([-7.0, -3.0, 0.0, 3.0, 7.0], size=n)]
    slate = _slate(home_probs=home_probs, sheet_home_spreads=spreads)

    eligible = slate["sheet_dog"].notna().sum()
    if eligible < 3:
        pytest.skip("slate cannot satisfy the constraint; covered by its own test")

    picks = choose_picks(slate, min_dogs=3)
    assert int(picks["is_dog_pick"].sum()) >= 3
    assert expected_correct(picks) == pytest.approx(_brute_force_best(slate, 3))


@pytest.mark.parametrize("min_dogs", [0, 1, 2, 3, 4, 5])
def test_greedy_matches_brute_force_across_constraint_levels(min_dogs: int) -> None:
    """Optimality must not depend on the constraint happening to be 3."""
    slate = _slate(
        home_probs=[0.82, 0.31, 0.55, 0.68, 0.44, 0.77],
        sheet_home_spreads=[-6.0, 4.0, -1.0, -8.0, 2.0, -3.0],
    )
    picks = choose_picks(slate, min_dogs=min_dogs)
    assert expected_correct(picks) == pytest.approx(_brute_force_best(slate, min_dogs))
