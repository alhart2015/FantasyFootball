"""Expected season points under per-player availability (spec §3.4).

Monte-Carlo a season: each week, players are available (not on bye, healthy w.p.
`p`), and the best legal lineup is filled from the available roster. Because
`per_game = season_mean_fpts / 17` is a uniform scaling, the weekly optimal lineup
is exactly `optimal_lineup_points(available_subset) / 17` -- the existing greedy
fill is reused verbatim. Weeks with no roster bye are identical in expectation, so
we MC one generic week and reuse it (the factorization is exact in expectation).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.schemas import RosterSlot

# Healthy-season denominator: a full season projection divided into per-game points
# (uniform scaling, spec §3.3). Distinct from availability._sched_games, which is the
# era-correct *historical* schedule length used to estimate injury rates.
_GAMES = 17


def _week_value(
    roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int], available: np.ndarray
) -> float:
    """Optimal weekly lineup points from the available roster rows (UNSCALED).

    The /_GAMES per-game scaling is applied once by the caller (after averaging over
    sims), exactly as the original expected_season_points did — summing pre-scaled
    values per sim would be a different float expression and break exact-equality tests.
    """
    sub = roster.iloc[np.flatnonzero(available)]
    return optimal_lineup_points(sub, roster_slots)


def _factorized_season_value(
    roster: pd.DataFrame,
    availability: PlayerAvailability,
    weeks: Iterable[int],
    week_value_fn: Callable[[np.ndarray], float],
) -> float:
    """Sum the season via the single-week factorization (spec §3.4 of PR #60).

    `week_value_fn(forced_out)` takes a boolean mask over roster rows (True where
    the player is on bye that week) and returns E[week points | those players are
    forced out]. Every non-bye week shares one expectation; each distinct roster
    bye week is recomputed with that player forced out. Exact in expectation. Call
    order (clean week, then bye weeks ascending) is fixed so callers that advance a
    shared RNG inside week_value_fn stay reproducible.
    """
    n = len(roster)
    gsis = roster["gsis_id"].astype(str).to_numpy()
    # -1 sentinel = "no bye"; never a real week, so it drops out of roster_bye_weeks below.
    bye_arr = np.array([b if (b := availability.bye_week(g)) is not None else -1 for g in gsis])
    weeks = list(weeks)
    roster_bye_weeks = sorted({w for w in bye_arr.tolist() if w in weeks})

    clean = week_value_fn(np.zeros(n, dtype=bool))
    total = (len(weeks) - len(roster_bye_weeks)) * clean
    for w in roster_bye_weeks:
        total += week_value_fn(bye_arr == w)
    return total


def expected_season_points_crn(
    roster: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    draws: np.ndarray,
    col_of: Mapping[str, int],
    weeks: Iterable[int] = range(1, 18),
) -> float:
    """Expected season points using a shared pre-drawn availability matrix (CRN).

    `draws` is `(n_sims, universe)` uniforms; `col_of` maps gsis_id -> column.
    Every roster scored against the same `draws` shares per-player draws, so a
    marginal `V(R+c) - V(R)` cancels the common noise (spec §3.3).
    """
    n = len(roster)
    if n == 0:
        return 0.0
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    cols = np.array([col_of[g] for g in gsis])
    sub_draws = draws[:, cols]  # (n_sims, n), aligned to roster row order
    n_sims: int = sub_draws.shape[0]

    def week_value_fn(forced_out: np.ndarray) -> float:
        acc = 0.0
        for s in range(n_sims):
            available = (sub_draws[s] < p_arr) & ~forced_out
            acc += _week_value(roster, roster_slots, available)
        return acc / n_sims / _GAMES

    return _factorized_season_value(roster, availability, weeks, week_value_fn)


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

    def week_value_fn(forced_out: np.ndarray) -> float:
        acc = 0.0
        for _ in range(n_sims):
            available = (rng.random(n) < p_arr) & ~forced_out
            acc += _week_value(roster, roster_slots, available)
        return acc / n_sims / _GAMES

    return _factorized_season_value(roster, availability, weeks, week_value_fn)
