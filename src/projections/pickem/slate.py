"""Join the organizer's sheet to consensus lines into one per-game view.

This module is the single place where the two spread conventions meet, and the
single place the conversion between them is allowed to happen:

- `nflreadpy.spread_line`: **positive means the home team is favored.**
- Everywhere in our code: **standard betting convention** — favorite negative,
  dog positive, from the named team's perspective.

So `consensus_home_spread = -spread_line`, matching the existing convention in
`projections.features._shared`.

The output frame keeps the two jobs visibly separate: `sheet_*` columns come
from the organizer and decide only *who counts as the underdog*; `*_win_prob`
columns come from the market and decide only *who is likely to win*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.pickem._validate import require_schedule_columns
from projections.pickem.probability import add_win_probs
from projections.schemas import (
    _PYARROW_STR,
    PickemSheetSchema,
    PickemSlateSchema,
)

_JOIN_KEYS = ["season", "week", "home_team", "away_team"]
_REQUIRED_SCHEDULE_COLUMNS = (
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "spread_line",
    "home_moneyline",
    "away_moneyline",
)


def build_slate(sheet: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Return a `PickemSlateSchema` frame: one row per game on the organizer's sheet.

    The sheet defines the slate. Scheduled games the organizer left off are
    simply absent from the result; a sheet row with no matching scheduled game
    raises, because that means a typo or the wrong week rather than a choice.

    Win probabilities are computed *after* the join, so unrelated future games
    with no posted price can never make this raise.
    """
    validated_sheet: pd.DataFrame = PickemSheetSchema.validate(sheet)
    require_schedule_columns(
        schedules, _REQUIRED_SCHEDULE_COLUMNS, needed_for="building the pick'em slate"
    )

    merged = validated_sheet.merge(
        schedules[list(_REQUIRED_SCHEDULE_COLUMNS)],
        on=_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    unmatched = merged["game_id"].isna()
    if bool(unmatched.any()):
        matchups = [
            f"{row.away_team}@{row.home_team} (season {row.season} week {row.week})"
            for row in merged.loc[unmatched].itertuples()
        ]
        raise ValueError(
            f"sheet row(s) with no matching scheduled game: {matchups}. "
            "Check the team codes and that the week is right."
        )

    priced = add_win_probs(merged)

    # Which side the ORGANIZER calls the underdog. Nothing about likelihood.
    home_favored = priced["home_spread"] < 0
    away_favored = priced["home_spread"] > 0
    has_dog = home_favored | away_favored  # a 0.0 spread has neither side

    favorite = pd.Series(pd.NA, index=priced.index, dtype=_PYARROW_STR)
    favorite[home_favored] = priced.loc[home_favored, "home_team"]
    favorite[away_favored] = priced.loc[away_favored, "away_team"]

    dog = pd.Series(pd.NA, index=priced.index, dtype=_PYARROW_STR)
    dog[home_favored] = priced.loc[home_favored, "away_team"]
    dog[away_favored] = priced.loc[away_favored, "home_team"]

    # Which side the MARKET thinks wins. Nothing about the organizer.
    dog_win_prob = pd.Series(np.nan, index=priced.index, dtype="float64")
    dog_win_prob[home_favored] = priced.loc[home_favored, "away_win_prob"]
    dog_win_prob[away_favored] = priced.loc[away_favored, "home_win_prob"]

    consensus_home_spread = -priced["spread_line"].astype("float64")

    # Same team, two sources. The sheet's dog spread is always positive by
    # construction, so its magnitude is |home_spread| either way.
    consensus_dog_spread = pd.Series(np.nan, index=priced.index, dtype="float64")
    consensus_dog_spread[home_favored] = -consensus_home_spread[home_favored]
    consensus_dog_spread[away_favored] = consensus_home_spread[away_favored]
    sheet_dog_spread = priced["home_spread"].abs().where(has_dog)

    out = pd.DataFrame(
        {
            "season": priced["season"],
            "week": priced["week"],
            "game_id": priced["game_id"].astype(_PYARROW_STR),
            "home_team": priced["home_team"].astype(_PYARROW_STR),
            "away_team": priced["away_team"].astype(_PYARROW_STR),
            "sheet_home_spread": priced["home_spread"].astype("float64"),
            "consensus_home_spread": consensus_home_spread,
            "home_win_prob": priced["home_win_prob"].astype("float64"),
            "away_win_prob": priced["away_win_prob"].astype("float64"),
            "sheet_favorite": favorite,
            "sheet_dog": dog,
            "dog_win_prob": dog_win_prob,
            # Positive => the market rates the dog HIGHER than the sheet does,
            # i.e. the line moved our way since Tuesday.
            "dog_line_move": sheet_dog_spread - consensus_dog_spread,
            # NaN compares False, so pick'em games are correctly not free dogs.
            "free_dog": (dog_win_prob > 0.5).to_numpy(dtype=bool),
        }
    )
    validated: pd.DataFrame = PickemSlateSchema.validate(out)
    return validated
