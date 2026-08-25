"""Projected-vs-projected full-league simulation of a completed draft.

Each MC season-week draws injury (availability) + over/under performance (the variance model)
for every rostered player, sets the optimal STARTING lineup for every team, and scores higher
projected total = win. Measures roster quality under our projections (NOT projection accuracy;
no actual stats). Reuses the variance sampler + optimal-lineup fill; no re-implemented scoring.

Calendar (fixed): regular weeks 1-13, wildcard wk14, semifinal wk15, championship wks 16-17;
top-6 make the playoffs, top-2 byes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot

#: Module-level calendar constants, kept as the DEFAULT shape only. They are the values the
#: simulator used when the calendar was hard-coded, so callers that pass no `calendar=` get
#: byte-identical behaviour. Anything that knows its league's real settings should build a
#: `LeagueCalendar` instead -- ESPN reports `matchupPeriodCount` and
#: `playoffMatchupPeriodLength` per league, and they do not always match these.
DEFAULT_CALENDAR = LeagueCalendar()

REG_WEEKS: tuple[int, ...] = DEFAULT_CALENDAR.reg_week_numbers  # 1..13
WILDCARD_WEEK = DEFAULT_CALENDAR.wildcard_week
SEMIFINAL_WEEK = DEFAULT_CALENDAR.round_week(1)
CHAMPIONSHIP_WEEKS: tuple[int, ...] = DEFAULT_CALENDAR.championship_weeks
ALL_WEEKS: tuple[int, ...] = DEFAULT_CALENDAR.all_week_numbers
PLAYOFF_SIZE = DEFAULT_CALENDAR.playoff_size
N_BYES = DEFAULT_CALENDAR.n_byes


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
    "DEFAULT_CALENDAR",
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
    #: The calendar these seasons were simulated under. Carried rather than re-derived so a
    #: result can never be read back against a different bracket than produced it -- `seed <=
    #: playoff_size` means nothing without knowing which playoff_size.
    calendar: LeagueCalendar
    wins: np.ndarray  # (n_sims, n_slots) regular-season wins
    points_for: np.ndarray  # (n_sims, n_slots) regular-season points
    seed: np.ndarray  # (n_sims, n_slots) 1-based final seed
    champion: np.ndarray  # (n_sims,) winning slot
    runner_up: np.ndarray  # (n_sims,) losing finalist slot

    def _col(self, slot: int) -> int:
        return self.slots.index(slot)

    def made_playoffs(self, slot: int) -> np.ndarray:
        return self.seed[:, self._col(slot)] <= self.calendar.playoff_size

    def outcome_labels(self, slot: int) -> list[str]:
        """Per-sim plain-language result for one seat."""
        seeds = self.seed[:, self._col(slot)]
        out: list[str] = []
        for i, sd in enumerate(seeds):
            if self.champion[i] == slot:
                out.append("won championship")
            elif self.runner_up[i] == slot:
                out.append("lost championship")
            elif sd <= self.calendar.playoff_size:
                out.append("made playoffs, eliminated")
            else:
                out.append("missed playoffs")
        return out


def resolve_bracket(
    seed_order: Sequence[int],
    week_points: Callable[[int, int], float],
    calendar: LeagueCalendar,
) -> tuple[int, int]:
    """Run one league's playoff bracket. Returns (champion slot, runner-up slot).

    `seed_order` is every slot best-seeded first; `week_points(slot, week)` gives that slot's
    points in an absolute week number.

    A single-elimination ladder, generalized from the 6-team/2-bye bracket this was hard-coded
    to. Each round the top `n_byes` seeds sit out (first round only), the rest pair
    best-against-worst, and survivors reseed by regular-season seed for the next round. For
    (6, 2) that reproduces the previous bracket exactly: wildcard 3v6 and 4v5, then seed 1
    against the lower survivor and seed 2 against the higher.

    Split out of `simulate_seasons` so it can be tested against the old implementation
    directly, on synthetic points, without reaching through a Monte-Carlo run.
    """
    seedpos = {s: r for r, s in enumerate(seed_order)}
    alive = list(seed_order[: calendar.playoff_size])
    for rnd in range(calendar.n_playoff_rounds - 1):
        week = calendar.round_week(rnd)
        byes = alive[: calendar.n_byes] if rnd == 0 else []
        playing = alive[calendar.n_byes :] if rnd == 0 else alive
        winners = [
            hi if week_points(hi, week) >= week_points(lo, week) else lo
            for hi, lo in zip(playing[: len(playing) // 2], playing[::-1], strict=False)
        ]
        alive = sorted(byes + winners, key=lambda s: seedpos[s])
    f1, f2 = alive  # better seed first
    f1_total = sum(week_points(f1, w) for w in calendar.championship_weeks)
    f2_total = sum(week_points(f2, w) for w in calendar.championship_weeks)
    # Ties break to the better seed, matching the documented rule. The old code broke them to
    # the winner of seed 1's half instead -- the better seed in every case except an upset
    # there, and unreachable in practice with float point totals.
    return (f1, f2) if f1_total >= f2_total else (f2, f1)


def simulate_seasons(
    rosters: Mapping[int, list[str]],
    pool: pd.DataFrame,
    availability: PlayerAvailability,
    params: VarianceParams,
    *,
    league_config: LeagueConfig,
    n_sims: int,
    rng: np.random.Generator,
    calendar: LeagueCalendar | None = None,
) -> SeasonOutcomes:
    """`n_sims` full projected-vs-projected seasons, returning every sim's realized result.

    The regular season runs `calendar.reg_weeks` via the gauntlet schedule -> records +
    points-for; seeding is (wins, points_for). The top `calendar.playoff_size` make the
    playoffs and the top `calendar.n_byes` skip the first round; the bracket is single
    elimination, reseeded each round, with the final spanning `calendar.final_weeks`.
    Ties (matchup or championship) break to the better seed / lower slot.

    `calendar` defaults to `DEFAULT_CALENDAR` -- 13 regular weeks, top 6, 2 byes, two-week
    final -- which is exactly the shape this function hard-coded before it was parameterised,
    so callers that pass nothing are unaffected. Pass a real one whenever the league's own
    settings are known: ESPN reports `matchupPeriodCount` and `playoffMatchupPeriodLength`
    per league and they do not always match the defaults.
    """
    calendar = calendar or DEFAULT_CALENDAR
    n_teams = league_config.n_teams
    if n_teams < calendar.playoff_size:
        raise ValueError(
            f"projected eval needs at least {calendar.playoff_size} teams; got {n_teams}"
        )
    slots = list(range(1, n_teams + 1))
    sub = pool.set_index("gsis_id")
    weekly = {
        s: team_weekly_points(
            sub.loc[rosters[s]].reset_index(),
            availability,
            params,
            n_sims=n_sims,
            weeks=list(calendar.all_week_numbers),
            roster_slots=league_config.roster_slots,
            rng=rng,
        )
        for s in slots
    }

    wins = {s: np.zeros(n_sims) for s in slots}
    pf = {s: np.zeros(n_sims) for s in slots}
    reg_weeks = calendar.reg_week_numbers
    schedule = gauntlet_schedule(n_teams, len(reg_weeks))
    for w, matchups in zip(reg_weeks, schedule, strict=True):
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
        order = [int(s) for s in seeds[i]]

        def points_in(slot: int, week: int, sim: int = i) -> float:
            # `sim` bound as a default: a closure over the loop variable would make every
            # bracket read the last simulation's points (ruff B023).
            return wk(slot, sim, week)

        champ_of[i], runner_up_of[i] = resolve_bracket(order, points_in, calendar)

    # seed_mat[i, col] = the 1-based seed slots[col] finished with in sim i.
    seed_mat = np.empty((n_sims, len(slots)), dtype=int)
    for col, s in enumerate(slots):
        seed_mat[:, col] = np.argmax(seeds == s, axis=1) + 1
    return SeasonOutcomes(
        slots=tuple(slots),
        calendar=calendar,
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
    calendar: LeagueCalendar | None = None,
) -> dict[int, SeatProjection]:
    """Per-seat projected-season metrics over `n_sims` MC seasons (projected-vs-projected).

    A thin aggregation of `simulate_seasons` — see it for the calendar and bracket rules,
    including what `calendar=None` defaults to.
    """
    res = simulate_seasons(
        rosters,
        pool,
        availability,
        params,
        league_config=league_config,
        n_sims=n_sims,
        rng=rng,
        calendar=calendar,
    )
    out: dict[int, SeatProjection] = {}
    for col, s in enumerate(res.slots):
        seed_of_s = res.seed[:, col]
        out[s] = SeatProjection(
            reg_win_pct=float(res.wins[:, col].mean() / res.calendar.reg_weeks),
            make_playoffs_pct=float((seed_of_s <= res.calendar.playoff_size).mean()),
            bye_pct=float((seed_of_s <= res.calendar.n_byes).mean()),
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
    calendar: LeagueCalendar | None = None,
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
        calendar=calendar,
    )
