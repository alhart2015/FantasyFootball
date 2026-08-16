"""Choose picks: maximize expected correct picks subject to a minimum number
of underdogs.

**Objective:** maximize `sum(P(chosen team wins))` across the slate.
**Constraint:** at least `min_dogs` chosen teams are underdogs *per the
organizer's sheet* — the market's opinion has no say in what counts as a dog.

The greedy rule below is not a heuristic; it is exactly optimal here, and the
tests brute-force it against every feasible combination on small slates to keep
it honest. The argument:

1. Picks are independent across games — choosing a team in one game constrains
   nothing in another.
2. In any game there are only two options, so deviating from the higher-probability
   side can only ever mean taking the sheet's underdog.
3. Therefore the cost of any feasible solution, relative to the unconstrained
   maximum, is the sum of `P(favorite) - P(dog)` over exactly the games it
   deviates on.
4. The constraint is a simple count, so meeting it at least cost means taking
   the cheapest deviations. No search is required.

Note that "favorite" in step 3 means the higher-probability side, which is not
always the sheet's favorite — a game the market has flipped since Tuesday costs
nothing at all and is picked up for free by step 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.schemas import (
    _PYARROW_STR,
    PickemPicksSchema,
    PickemSlateSchema,
)

DEFAULT_MIN_DOGS = 3


def choose_picks(slate: pd.DataFrame, *, min_dogs: int = DEFAULT_MIN_DOGS) -> pd.DataFrame:
    """Return a `PickemPicksSchema` frame — one row per game, ungraded.

    `forced` marks a pick taken only to satisfy the underdog constraint. A dog
    that was already the higher-probability side is *not* forced: it cost
    nothing, so calling it forced would overstate what the constraint took.
    """
    if min_dogs < 0:
        raise ValueError(f"min_dogs must be non-negative, got {min_dogs}")
    validated: pd.DataFrame = PickemSlateSchema.validate(slate)

    home_prob = validated["home_win_prob"].to_numpy(dtype="float64")
    away_prob = validated["away_win_prob"].to_numpy(dtype="float64")
    home_team = validated["home_team"].astype(str).to_numpy()
    away_team = validated["away_team"].astype(str).to_numpy()
    game_id = validated["game_id"].astype(str).to_numpy()

    has_dog = validated["sheet_dog"].notna().to_numpy(dtype=bool)
    # "" is a safe stand-in for NA: no real team code is empty, so the equality
    # comparisons below can stay plain numpy without NA propagation surprises.
    dog_team = validated["sheet_dog"].fillna("").astype(str).to_numpy()
    dog_prob = validated["dog_win_prob"].to_numpy(dtype="float64")  # NaN where no dog

    # Step 1: the unconstrained best pick in each game, ignoring the constraint.
    home_better = home_prob >= away_prob
    best_team = np.where(home_better, home_team, away_team)
    best_prob = np.where(home_better, home_prob, away_prob)

    # Dogs we get without paying anything — includes every free dog.
    natural_dog = has_dog & (best_team == dog_team)
    needed = max(0, min_dogs - int(natural_dog.sum()))

    # Step 3: cost of deviating, defined only where a dog exists to switch to.
    switch_cost = best_prob - dog_prob
    eligible = has_dog & ~natural_dog

    if needed > int(eligible.sum()):
        raise ValueError(
            f"need {needed} more underdog pick(s) but only {int(eligible.sum())} game(s) "
            f"have an underdog available to switch to (slate has {len(validated)} game(s), "
            f"{int(natural_dog.sum())} already on the dog). Games with a sheet spread of "
            "exactly 0 have no underdog and cannot satisfy the constraint."
        )

    forced = np.zeros(len(validated), dtype=bool)
    if needed > 0:
        # Step 4: cheapest deviations. game_id breaks ties so runs reproduce.
        candidates = sorted(
            np.flatnonzero(eligible), key=lambda i: (float(switch_cost[i]), str(game_id[i]))
        )
        forced[candidates[:needed]] = True

    pick = np.where(forced, dog_team, best_team)
    pick_prob = np.where(forced, dog_prob, best_prob)

    out = pd.DataFrame(
        {
            "season": validated["season"],
            "week": validated["week"],
            "game_id": validated["game_id"].astype(_PYARROW_STR),
            "home_team": validated["home_team"].astype(_PYARROW_STR),
            "away_team": validated["away_team"].astype(_PYARROW_STR),
            "pick": pd.array(pick, dtype=_PYARROW_STR),
            "pick_win_prob": pick_prob,
            "is_dog_pick": has_dog & (pick == dog_team),
            "forced": forced,
            "switch_cost": np.where(forced, switch_cost, 0.0),
            "winner": pd.array([None] * len(validated), dtype=_PYARROW_STR),
            "correct": pd.array([None] * len(validated), dtype=pd.BooleanDtype()),
        }
    )
    validated_out: pd.DataFrame = PickemPicksSchema.validate(out)
    return validated_out


def expected_correct(picks: pd.DataFrame) -> float:
    """Expected number of correct picks — the objective `choose_picks` maximizes.

    Sums probabilities rather than counting outcomes, so it is the mean of the
    distribution of weekly scores, not a prediction of any single week.
    """
    return float(picks["pick_win_prob"].sum())
