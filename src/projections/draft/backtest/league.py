"""Simulate one full league season.

draft -> weekly points -> standings -> playoffs -> LeagueResult.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.backtest.draft_field import draft_mixed_field
from projections.draft.backtest.lineup import weekly_lineup_points
from projections.draft.backtest.schedule import playoff_champion, regular_season_schedule
from projections.draft.league_config import LeagueConfig


@dataclass(frozen=True)
class Calendar:
    regular_weeks: tuple[int, ...]  # e.g. tuple(range(1, 15))
    playoff_weeks: tuple[int, int, int]  # (15, 16, 17)
    playoff_size: int = 6


@dataclass(frozen=True)
class LeagueResult:
    seat: int
    strategy: str
    wins: int
    losses: int
    points_for: float
    made_playoffs: bool
    is_champion: bool


@dataclass(frozen=True)
class LeagueOutcome:
    """One league simulated once, scored two ways from the same drafted rosters.

    ``actual`` scores matchups by realized points (real fantasy); ``projected`` scores
    by the started lineup's projected points (draft quality under shared beliefs).
    """

    actual: list[LeagueResult]
    projected: list[LeagueResult]


def simulate_league(
    seed: int,
    *,
    seat_strategies: Mapping[int, DraftStrategy | None],
    strategy_labels: Mapping[int, str],
    pool: pd.DataFrame,
    config: LeagueConfig,
    proj_lookup: Mapping[tuple[str, int], float],
    actual_lookup: Mapping[tuple[str, int], float],
    calendar: Calendar,
    jitter: float,
) -> LeagueOutcome:
    """Run a full league simulation once and score it both ways.

    Draft order: seat_strategies drives the draft via draft_mixed_field (None => bot).
    The drafted rosters and the regular-season schedule are shared; the league is then
    scored independently by actual points and by projected points (each gets its own
    standings, playoff seeding, and champion). Lineups are always set by projection.
    Returns a LeagueOutcome carrying both per-seat result lists.
    """
    rng = np.random.default_rng(seed)
    rosters = draft_mixed_field(dict(seat_strategies), pool, config, rng=rng, jitter=jitter)
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"], strict=False)}

    # Build each (week, seat) roster once; both scorings start the same lineup (by
    # projection) and only differ in which field they sum, so the roster is shared.
    all_weeks = set(calendar.regular_weeks) | set(calendar.playoff_weeks)
    pts_actual: dict[int, dict[int, float]] = {wk: {} for wk in all_weeks}
    pts_proj: dict[int, dict[int, float]] = {wk: {} for wk in all_weeks}
    for wk in all_weeks:
        for s in rosters:
            roster = [
                {
                    "position": pos_by_id[g],
                    "projected": proj_lookup.get((g, wk)),
                    "actual": actual_lookup.get((g, wk)),
                }
                for g in rosters[s]
            ]
            pts_actual[wk][s] = weekly_lineup_points(roster, config.roster_slots, score_by="actual")
            pts_proj[wk][s] = weekly_lineup_points(
                roster, config.roster_slots, score_by="projected"
            )

    # One schedule (one rng draw) shared by both scorings.
    sched = regular_season_schedule(
        n_teams=config.n_teams, n_weeks=len(calendar.regular_weeks), rng=rng
    )

    return LeagueOutcome(
        actual=_standings_and_playoffs(rosters, pts_actual, sched, strategy_labels, calendar),
        projected=_standings_and_playoffs(rosters, pts_proj, sched, strategy_labels, calendar),
    )


def _standings_and_playoffs(
    rosters: Mapping[int, list[str]],
    pts: Mapping[int, Mapping[int, float]],
    sched: list[list[tuple[int, int]]],
    strategy_labels: Mapping[int, str],
    calendar: Calendar,
) -> list[LeagueResult]:
    """Score one weekly points table into per-seat W/L, standings, playoffs, champion.

    Higher weekly points wins each matchup (ties break to the lower seat index); standings
    sort by wins then points-for; the top ``playoff_size`` seeds run the bracket.
    """
    wins: dict[int, int] = {s: 0 for s in rosters}
    losses: dict[int, int] = {s: 0 for s in rosters}
    pf: dict[int, float] = {s: 0.0 for s in rosters}

    for wk, matchups in zip(calendar.regular_weeks, sched, strict=False):
        for a, b in matchups:
            pf[a] += pts[wk][a]
            pf[b] += pts[wk][b]
            if (pts[wk][a], -a) >= (pts[wk][b], -b):
                wins[a] += 1
                losses[b] += 1
            else:
                wins[b] += 1
                losses[a] += 1

    standings = sorted(rosters, key=lambda s: (wins[s], pf[s]), reverse=True)
    seeds = standings[: calendar.playoff_size]
    champ = playoff_champion(
        seeds,
        {wk: pts[wk] for wk in calendar.playoff_weeks},
        playoff_weeks=calendar.playoff_weeks,
    )

    return [
        LeagueResult(
            seat=s,
            strategy=strategy_labels[s],
            wins=wins[s],
            losses=losses[s],
            points_for=pf[s],
            made_playoffs=s in seeds,
            is_champion=(s == champ),
        )
        for s in rosters
    ]
