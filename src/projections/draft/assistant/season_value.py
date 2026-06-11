"""Expected season points under per-player availability (spec §3.4).

Monte-Carlo a season: each week, players are available (not on bye, healthy w.p.
`p`), and the best legal lineup is filled from the available roster. Because
`per_game = season_mean_fpts / 17` is a uniform scaling, the weekly optimal lineup
is exactly `optimal_lineup_points(available_subset) / 17` -- the existing greedy
fill is reused verbatim. Weeks with no roster bye are identical in expectation, so
we MC one generic week and reuse it (the factorization is exact in expectation).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.schemas import RosterSlot

_GAMES = 17  # season projection -> per-game divisor (uniform scaling, spec §3.3)


def expected_season_points(
    roster: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    n_sims: int,
    rng: np.random.Generator,
    weeks: Iterable[int] = range(1, 18),
) -> float:
    """Expected total season points of `roster` under availability risk."""
    n = len(roster)
    if n == 0:
        return 0.0
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    # -1 sentinel = "no bye"; never a real week, so it drops out of roster_bye_weeks below.
    bye_arr = np.array([b if (b := availability.bye_week(g)) is not None else -1 for g in gsis])
    weeks = list(weeks)
    roster_bye_weeks = sorted({w for w in bye_arr.tolist() if w in weeks})

    def week_expectation(forced_out: np.ndarray) -> float:
        acc = 0.0
        for _ in range(n_sims):
            available = (rng.random(n) < p_arr) & ~forced_out
            sub = roster.iloc[np.flatnonzero(available)]
            acc += optimal_lineup_points(sub, roster_slots)
        return acc / n_sims / _GAMES

    no_force = np.zeros(n, dtype=bool)
    clean_week_value = week_expectation(no_force)
    clean_weeks = sum(1 for w in weeks if w not in roster_bye_weeks)
    total = clean_weeks * clean_week_value
    for w in roster_bye_weeks:
        total += week_expectation(bye_arr == w)
    return total
