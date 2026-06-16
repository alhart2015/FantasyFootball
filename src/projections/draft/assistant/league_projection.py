"""Projected-vs-projected full-league simulation of a completed draft.

Each MC season-week draws injury (availability) + over/under performance (the variance model)
for every rostered player, sets the optimal STARTING lineup for every team, and scores higher
projected total = win. Measures roster quality under our projections (NOT projection accuracy;
no actual stats). Reuses the variance sampler + optimal-lineup fill; no re-implemented scoring.

Calendar (fixed): regular weeks 1-13, wildcard wk14, semifinal wk15, championship wks 16-17;
top-6 make the playoffs, top-2 byes.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points
from projections.draft.assistant.season_value import (
    _availability_mask,
    _bye_indices,
    _lineup_points_sampled,
)
from projections.schemas import RosterSlot

REG_WEEKS: tuple[int, ...] = tuple(range(1, 14))  # 1..13
WILDCARD_WEEK = 14
SEMIFINAL_WEEK = 15
CHAMPIONSHIP_WEEKS: tuple[int, int] = (16, 17)
ALL_WEEKS: tuple[int, ...] = tuple(range(1, 18))  # draw all 17
PLAYOFF_SIZE = 6
N_BYES = 2


def gauntlet_schedule(n_teams: int, n_weeks: int) -> list[list[tuple[int, int]]]:
    """A 1-factorization of K_{n_teams} (even n) seating slot 1 as the fixed vertex, so it plays
    slot 2 in wk1, slot 3 in wk2, ... ; the rest pair via (r+k),(r-k) mod (n_teams-1). Week w uses
    round (w-1) mod (n_teams-1). Every team plays exactly one game per week."""
    if n_teams % 2 != 0:
        raise ValueError(f"n_teams must be even; got {n_teams}")
    m = n_teams - 1  # labels 0..m-1 <-> slots 2..n_teams; special vertex = slot 1
    half = (n_teams - 2) // 2
    rounds: list[list[tuple[int, int]]] = []
    for r in range(m):
        pairs: list[tuple[int, int]] = [(1, r + 2)]
        for k in range(1, half + 1):
            pairs.append(((r + k) % m + 2, (r - k) % m + 2))
        rounds.append(pairs)
    return [rounds[(w - 1) % m] for w in range(1, n_weeks + 1)]


def team_weekly_points(
    roster: pd.DataFrame,
    availability: PlayerAvailability,
    params: VarianceParams,
    *,
    n_sims: int,
    weeks: list[int],
    roster_slots: Mapping[RosterSlot, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_sims, len(weeks)) optimal STARTING-lineup points under the variance model + availability.

    `roster` carries gsis_id, position, season_mean_fpts, is_rookie. Draws mean-preserving season
    multipliers + weekly Gamma noise (variance model), masks by injury Bernoulli + byes, and fills
    the optimal legal starting lineup per (sim, week). BENCH never scores (not a starting slot).
    """
    gsis = roster["gsis_id"].astype(str).to_numpy()
    pos = roster["position"].astype(str).to_numpy()
    means = roster["season_mean_fpts"].to_numpy(dtype=np.float64)
    rookie = roster["is_rookie"].to_numpy(dtype=bool)
    p = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    bye = _bye_indices(availability, gsis, weeks)
    nw, m = len(weeks), len(gsis)
    pts = sample_weekly_points(params, pos, means, rookie, n_sims=n_sims, n_weeks=nw, rng=rng)
    av = _availability_mask(rng.random((n_sims, nw, m)), p, bye)
    return _lineup_points_sampled(
        pts.reshape(n_sims * nw, m), av.reshape(n_sims * nw, m), pos, roster_slots
    ).reshape(n_sims, nw)
