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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.assistant.season_value import (
    _availability_mask,
    _bye_indices,
    _lineup_points_sampled,
)
from projections.draft.league_config import LeagueConfig
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


__all__ = [
    "ALL_WEEKS",
    "REG_WEEKS",
    "SeasonOutcomes",
    "SeatProjection",
    "gauntlet_schedule",
    "project_draft",
    "simulate_seasons",
    "team_weekly_points",
]


@dataclass(frozen=True)
class SeatProjection:
    reg_win_pct: float
    make_playoffs_pct: float
    bye_pct: float
    champ_pct: float
    mean_seed: float
    mean_points: float


@dataclass(frozen=True)
class SeasonOutcomes:
    """Per-simulation season results, before any aggregation. Arrays are indexed [sim] or
    [sim, slot-1]; `slots` gives the 1-based seat for each column.

    `project_draft` reduces this to per-seat rates. Kept separate (and public) because the
    aggregate hides what a caller often needs: which seat actually WON a given simulated season,
    which one lost the final, and what the roster's record was — the difference between "this
    roster had 23% title equity" and "this roster won the title in this season". An eye-test or a
    post-mortem needs the realized season, not the mean.
    """

    slots: tuple[int, ...]
    wins: np.ndarray  # (n_sims, n_slots) regular-season wins
    points_for: np.ndarray  # (n_sims, n_slots) regular-season points
    seed: np.ndarray  # (n_sims, n_slots) 1-based final seed
    champion: np.ndarray  # (n_sims,) winning slot
    runner_up: np.ndarray  # (n_sims,) losing finalist slot

    def _col(self, slot: int) -> int:
        return self.slots.index(slot)

    def made_playoffs(self, slot: int) -> np.ndarray:
        return self.seed[:, self._col(slot)] <= PLAYOFF_SIZE

    def outcome_labels(self, slot: int) -> list[str]:
        """Per-sim plain-language result for one seat."""
        seeds = self.seed[:, self._col(slot)]
        out: list[str] = []
        for i, sd in enumerate(seeds):
            if self.champion[i] == slot:
                out.append("won championship")
            elif self.runner_up[i] == slot:
                out.append("lost championship")
            elif sd <= PLAYOFF_SIZE:
                out.append("made playoffs, eliminated")
            else:
                out.append("missed playoffs")
        return out


def simulate_seasons(
    rosters: Mapping[int, list[str]],
    pool: pd.DataFrame,
    availability: PlayerAvailability,
    params: VarianceParams,
    *,
    league_config: LeagueConfig,
    n_sims: int,
    rng: np.random.Generator,
) -> SeasonOutcomes:
    """`n_sims` full projected-vs-projected seasons, returning every sim's realized result.

    Regular season (wks 1-13) via the gauntlet schedule -> records + points-for; seed by
    (wins, points_for); top-6 make playoffs, top-2 bye; wk14 wildcard (3v6,4v5), wk15 reseeded
    semis (seed1 vs lowest survivor, seed2 vs other), championship = wk16+wk17 combined.
    Ties (matchup or championship) break to the better seed / lower slot.
    """
    n_teams = league_config.n_teams
    if n_teams < PLAYOFF_SIZE:
        raise ValueError(f"projected eval needs at least {PLAYOFF_SIZE} teams; got {n_teams}")
    slots = list(range(1, n_teams + 1))
    sub = pool.set_index("gsis_id")
    weekly = {
        s: team_weekly_points(
            sub.loc[rosters[s]].reset_index(),
            availability,
            params,
            n_sims=n_sims,
            weeks=list(ALL_WEEKS),
            roster_slots=league_config.roster_slots,
            rng=rng,
        )
        for s in slots
    }

    wins = {s: np.zeros(n_sims) for s in slots}
    pf = {s: np.zeros(n_sims) for s in slots}
    schedule = gauntlet_schedule(n_teams, len(REG_WEEKS))
    for w, matchups in zip(REG_WEEKS, schedule, strict=True):
        for a, b in matchups:
            pa, pb = weekly[a][:, w - 1], weekly[b][:, w - 1]
            pf[a] += pa
            pf[b] += pb
            a_win = pa >= pb
            wins[a] += a_win
            wins[b] += ~a_win

    win_mat = np.stack([wins[s] for s in slots], axis=1)
    pf_mat = np.stack([pf[s] for s in slots], axis=1)
    # seeds[i] = slots best->worst for sim i (wins dominate; points_for breaks ties)
    seeds = np.argsort(-(win_mat * 1e6 + pf_mat), axis=1) + 1

    def wk(slot: int, sim: int, week: int) -> float:
        return float(weekly[slot][sim, week - 1])

    champ_of = np.zeros(n_sims, dtype=int)
    runner_up_of = np.zeros(n_sims, dtype=int)
    for i in range(n_sims):
        o = [int(s) for s in seeds[i]]
        seedpos = {s: r for r, s in enumerate(o)}
        s1, s2, s3, s4, s5, s6 = o[:6]
        win_a = s3 if wk(s3, i, WILDCARD_WEEK) >= wk(s6, i, WILDCARD_WEEK) else s6
        win_b = s4 if wk(s4, i, WILDCARD_WEEK) >= wk(s5, i, WILDCARD_WEEK) else s5
        hi, lo = sorted([win_a, win_b], key=lambda s: seedpos[s])  # better seed, worse seed
        f1 = s1 if wk(s1, i, SEMIFINAL_WEEK) >= wk(lo, i, SEMIFINAL_WEEK) else lo
        f2 = s2 if wk(s2, i, SEMIFINAL_WEEK) >= wk(hi, i, SEMIFINAL_WEEK) else hi
        c1, c2 = CHAMPIONSHIP_WEEKS
        f1_total = wk(f1, i, c1) + wk(f1, i, c2)
        f2_total = wk(f2, i, c1) + wk(f2, i, c2)
        f1_wins = f1_total >= f2_total
        champ_of[i] = f1 if f1_wins else f2
        runner_up_of[i] = f2 if f1_wins else f1

    # seed_mat[i, col] = the 1-based seed slots[col] finished with in sim i.
    seed_mat = np.empty((n_sims, len(slots)), dtype=int)
    for col, s in enumerate(slots):
        seed_mat[:, col] = np.argmax(seeds == s, axis=1) + 1
    return SeasonOutcomes(
        slots=tuple(slots),
        wins=win_mat,
        points_for=pf_mat,
        seed=seed_mat,
        champion=champ_of,
        runner_up=runner_up_of,
    )


def project_draft(
    rosters: Mapping[int, list[str]],
    pool: pd.DataFrame,
    availability: PlayerAvailability,
    params: VarianceParams,
    *,
    league_config: LeagueConfig,
    n_sims: int,
    rng: np.random.Generator,
) -> dict[int, SeatProjection]:
    """Per-seat projected-season metrics over `n_sims` MC seasons (projected-vs-projected).

    A thin aggregation of `simulate_seasons` — see it for the calendar and bracket rules.
    """
    res = simulate_seasons(
        rosters,
        pool,
        availability,
        params,
        league_config=league_config,
        n_sims=n_sims,
        rng=rng,
    )
    out: dict[int, SeatProjection] = {}
    for col, s in enumerate(res.slots):
        seed_of_s = res.seed[:, col]
        out[s] = SeatProjection(
            reg_win_pct=float(res.wins[:, col].mean() / len(REG_WEEKS)),
            make_playoffs_pct=float((seed_of_s <= PLAYOFF_SIZE).mean()),
            bye_pct=float((seed_of_s <= N_BYES).mean()),
            champ_pct=float((res.champion == s).mean()),
            mean_seed=float(seed_of_s.mean()),
            mean_points=float(res.points_for[:, col].mean()),
        )
    return out


def project_completed_league(
    rosters: Mapping[int, list[str]],
    pool: pd.DataFrame,
    league_config: LeagueConfig,
    *,
    season: int,
    n_sims: int = 2000,
    seed: int = 0,
    availability: PlayerAvailability | None = None,
    params: VarianceParams | None = None,
    data_root: Path = Path("data"),
) -> dict[int, SeatProjection]:
    """The shared tail of both live boards' end-of-draft eval.

    Rookie flags, store availability, fitted variance params, then `project_draft`. A caller
    supplies only its own `rosters` mapping -- the one step a snake draft (reconstruct from
    pick order) and an auction (group the purchase log by seat) genuinely do differently.
    `availability`/`params` default to the store and the fitted config; tests inject them to
    stay hermetic.
    """
    pool = attach_is_rookie(pool, season=season, data_root=data_root)
    if availability is None:
        from projections.draft.assistant.availability_loader import load_store_availability

        availability = load_store_availability(pool, season=season, data_root=data_root)
    if params is None:
        params = VarianceParams.load()
    return project_draft(
        rosters=dict(rosters),
        pool=pool,
        availability=availability,
        params=params,
        league_config=league_config,
        n_sims=n_sims,
        rng=np.random.default_rng(seed),
    )
