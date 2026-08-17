"""Score picks against final results.

Grading fills `winner` / `correct` on the existing picks frame rather than
producing a new table, so one row carries a pick from entry through result.

Two edge cases are handled deliberately rather than by accident:

- **Ties count as incorrect.** You picked a team to win and it did not. This is
  the common pool rule, though a push is also defensible — 13 regular-season
  games since 2010 are affected, so it is worth being explicit.
- **Unplayed stays distinguishable from wrong.** `correct` is NA for a game that
  has not finished and False for one we got wrong (including a tie). A plain
  bool column would silently collapse the two and quietly inflate the loss
  column mid-week.
"""

from __future__ import annotations

import pandas as pd

from projections.pickem._validate import require_schedule_columns
from projections.schemas import _PYARROW_STR, PickemPicksSchema

_SCORE_COLUMNS = ("game_id", "home_score", "away_score")


def grade_picks(picks: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Return `picks` with `winner` and `correct` filled in from final scores."""
    validated: pd.DataFrame = PickemPicksSchema.validate(picks)
    require_schedule_columns(schedules, _SCORE_COLUMNS, needed_for="grading pick'em picks")

    merged = validated.drop(columns=["winner", "correct"]).merge(
        schedules[list(_SCORE_COLUMNS)].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    home_score = merged["home_score"]
    away_score = merged["away_score"]
    played = (home_score.notna() & away_score.notna()).to_numpy(dtype=bool)
    home_won = played & (home_score.fillna(0) > away_score.fillna(0)).to_numpy(dtype=bool)
    away_won = played & (away_score.fillna(0) > home_score.fillna(0)).to_numpy(dtype=bool)

    winner = pd.Series(pd.NA, index=merged.index, dtype=_PYARROW_STR)
    winner[home_won] = merged.loc[home_won, "home_team"]
    winner[away_won] = merged.loc[away_won, "away_team"]

    # "" for a tie or an unplayed game, so the comparison yields False rather
    # than propagating NA; unplayed rows are then reset to NA below.
    winner_for_compare = winner.fillna("").astype(str).to_numpy()
    matched = merged["pick"].astype(str).to_numpy() == winner_for_compare
    correct = pd.Series(pd.array(matched, dtype=pd.BooleanDtype()), index=merged.index)
    correct[~played] = pd.NA

    # Built from `merged`, not `validated`: the merge produced a fresh index, and
    # assigning these Series back onto a differently-indexed frame would align by
    # label and silently scramble rows.
    out = merged.drop(columns=["home_score", "away_score"])
    out["winner"] = winner
    out["correct"] = correct
    validated_out: pd.DataFrame = PickemPicksSchema.validate(out)
    return validated_out


def record(graded: pd.DataFrame) -> tuple[int, int]:
    """Return `(correct, played)` for a graded frame. Unplayed games are excluded
    from both, so a mid-week record reads honestly."""
    played = graded["correct"].notna()
    return int(graded.loc[played, "correct"].sum()), int(played.sum())
