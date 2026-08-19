"""Backtest tests — calibration binning and the no-edge baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.pickem.backtest import (
    baseline_week_scores,
    calibration_table,
    playable_games,
    summarize_baseline,
)
from projections.schemas import _PYARROW_STR

_HOME = ["SEA", "CAR", "LAR", "CIN", "DET", "HOU"]
_AWAY = ["NE", "CHI", "SF", "TB", "NO", "BUF"]


def _games(
    *,
    week: int = 1,
    spread_lines: list[float] | None = None,
    home_scores: list[float] | None = None,
    away_scores: list[float] | None = None,
    game_type: str = "REG",
) -> pd.DataFrame:
    n = len(_HOME)
    spreads = spread_lines if spread_lines is not None else [7.0] * n
    home_ml = [-300 if s > 0 else 250 for s in spreads]
    away_ml = [250 if s > 0 else -300 for s in spreads]
    return pd.DataFrame(
        {
            "season": [2024] * n,
            "week": [week] * n,
            "game_id": pd.array(
                [f"2024_{week:02d}_{_AWAY[i]}_{_HOME[i]}" for i in range(n)], dtype=_PYARROW_STR
            ),
            "home_team": pd.array(_HOME, dtype=_PYARROW_STR),
            "away_team": pd.array(_AWAY, dtype=_PYARROW_STR),
            "game_type": pd.array([game_type] * n, dtype=_PYARROW_STR),
            "spread_line": spreads,
            "home_moneyline": pd.array(home_ml, dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array(away_ml, dtype=pd.Int64Dtype()),
            "home_score": pd.array(
                home_scores if home_scores else [24.0] * n, dtype=pd.Int64Dtype()
            ),
            "away_score": pd.array(
                away_scores if away_scores else [17.0] * n, dtype=pd.Int64Dtype()
            ),
        }
    )


# --- playable_games ---------------------------------------------------------


def test_playable_games_excludes_playoffs() -> None:
    mixed = pd.concat([_games(), _games(week=2, game_type="WC")], ignore_index=True)
    assert set(playable_games(mixed)["week"]) == {1}


def test_playable_games_excludes_unplayed_and_unpriced() -> None:
    games = _games()
    games.loc[0, "home_score"] = pd.NA
    games.loc[1, "home_moneyline"] = pd.NA
    games.loc[2, "spread_line"] = float("nan")
    assert len(playable_games(games)) == 3


def test_playable_games_raises_without_score_columns() -> None:
    games = _games().drop(columns=["home_score", "away_score"])
    with pytest.raises(ValueError, match="refresh_schedules"):
        playable_games(games)


# --- calibration ------------------------------------------------------------


def test_calibration_recovers_a_known_rate() -> None:
    """Twenty games priced identically, of which exactly half are home wins:
    the bin's actual rate must read 50%."""
    frames = []
    for w in range(1, 11):
        home_scores = [24.0, 24.0, 24.0, 17.0, 17.0, 17.0]
        away_scores = [17.0, 17.0, 17.0, 24.0, 24.0, 24.0]
        frames.append(
            _games(
                week=w,
                spread_lines=[1.0] * 6,
                home_scores=home_scores,
                away_scores=away_scores,
            )
        )
    table = calibration_table(pd.concat(frames, ignore_index=True), n_bins=10)
    assert len(table) == 1  # all games share one price, so one bin
    assert table.loc[0, "n_games"] == 60
    assert table.loc[0, "actual_rate"] == pytest.approx(0.5)


def test_calibration_error_is_actual_minus_predicted() -> None:
    table = calibration_table(_games(), n_bins=4)
    for row in table.itertuples():
        assert row.error == pytest.approx(row.actual_rate - row.mean_predicted)


def test_calibration_counts_a_tie_as_a_home_loss() -> None:
    """The pool grades a tie as wrong, so calibration must too."""
    tied = _games(home_scores=[20.0] * 6, away_scores=[20.0] * 6)
    table = calibration_table(tied, n_bins=4)
    assert table["actual_rate"].eq(0.0).all()


def test_calibration_raises_when_nothing_is_playable() -> None:
    empty = _games(game_type="WC")
    with pytest.raises(ValueError, match="no completed"):
        calibration_table(empty)


# --- baseline ---------------------------------------------------------------


def test_baseline_produces_one_row_per_week() -> None:
    two_weeks = pd.concat([_games(week=1), _games(week=2)], ignore_index=True)
    weeks = baseline_week_scores(two_weeks)
    assert len(weeks) == 2
    assert set(weeks["week"]) == {1, 2}


def test_baseline_constrained_never_beats_unconstrained_in_expectation() -> None:
    """The constraint can only cost expected value — that is what makes the
    greedy switch a cost at all."""
    weeks = baseline_week_scores(
        pd.concat([_games(week=w) for w in range(1, 5)], ignore_index=True)
    )
    assert (weeks["expected_unconstrained"] >= weeks["expected_correct"]).all()


def test_baseline_records_the_forced_dogs() -> None:
    """All six games have the home team favored on both sheet and market, so
    three dogs must be forced."""
    weeks = baseline_week_scores(_games())
    assert weeks.loc[0, "forced_dogs"] == 3
    assert weeks.loc[0, "free_dogs"] == 0


def test_baseline_has_no_free_dogs_when_the_moneyline_agrees_with_the_spread() -> None:
    """No staleness edge exists here, so free dogs can only come from the spread
    and the moneyline disagreeing about who is favored. This fixture derives its
    moneylines from the spread sign, so they never can.

    On real data they occasionally do — a near-pick'em game can close at
    `spread_line = +1.0` (home favored) while the moneylines make the away side
    the favorite. The 2015-2025 backtest shows ~0.10 such games per week even
    with the sheet set equal to the market.
    """
    weeks = baseline_week_scores(
        pd.concat(
            [_games(week=w, spread_lines=[7.0, -3.0, 5.0, -1.0, 2.0, -6.0]) for w in range(1, 4)],
            ignore_index=True,
        )
    )
    assert weeks["free_dogs"].eq(0).all()


def test_free_dog_arises_when_the_moneyline_contradicts_the_spread() -> None:
    """The complement of the test above, on the case real data actually produces:
    home favored by half a point on the spread but priced as the underdog."""
    games = _games(spread_lines=[0.5, 7.0, 7.0, 7.0, 7.0, 7.0])
    games.loc[0, "home_moneyline"] = 115  # home is a dog on the moneyline...
    games.loc[0, "away_moneyline"] = -135  # ...despite being favored on the spread
    weeks = baseline_week_scores(games)
    assert weeks.loc[0, "free_dogs"] == 1


def test_baseline_skips_a_week_that_cannot_meet_the_constraint() -> None:
    """Every game a true pick'em: no underdog exists, so the week is dropped
    rather than scored with a fabricated number."""
    weeks = baseline_week_scores(_games(spread_lines=[0.0] * 6))
    assert weeks.empty


def test_summarize_baseline_headline_numbers() -> None:
    weeks = baseline_week_scores(
        pd.concat([_games(week=w) for w in range(1, 4)], ignore_index=True)
    )
    s = summarize_baseline(weeks)
    assert s["weeks"] == 3
    assert s["games"] == 18
    assert s["games_per_week"] == pytest.approx(6.0)
    assert 0.0 <= s["hit_rate"] <= 1.0
    assert s["constraint_cost_per_week"] >= 0


def test_summarize_baseline_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="no weeks"):
        summarize_baseline(pd.DataFrame())


def test_playable_games_requires_game_type_rather_than_skipping_the_filter() -> None:
    """Filtering only when the column happens to exist would silently admit playoff
    games, and `baseline_week_scores` would score weeks 19-22 as pool weeks."""
    games = _games().drop(columns=["game_type"])
    with pytest.raises(ValueError, match="game_type"):
        playable_games(games)


def test_playable_games_excludes_post_season_game_types() -> None:
    mixed = pd.concat(
        [_games(week=1), _games(week=19, game_type="WC"), _games(week=21, game_type="SB")],
        ignore_index=True,
    )
    assert set(playable_games(mixed)["week"]) == {1}
