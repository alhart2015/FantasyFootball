"""Projected standings and matchup odds from one in-season simulation run.

Turns the pieces the earlier steps built — the real fixture list, locked played weeks,
rest-of-season projections — into the two tables a manager reads: where every team is heading,
and who wins each remaining game.

Both come out of a **single** `simulate_seasons` call. Matchup odds are not a second engine:
each simulated week already produces both teams' point totals per simulation, so
`P(A beats B in week 8)` is a read of the same simulations that produced the standings. Two
runs could disagree with each other; one cannot.

The ESPN payload speaks in **team ids** (arbitrary, e.g. 17) and the simulator speaks in
**slots** (1..n_teams, contiguous). `SlotMap` is the only place that translation happens.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from projections.draft.assistant.league_projection import LockedRecord, SeasonOutcomes
from projections.draft.league_calendar import LeagueCalendar
from projections.schemas import _PYARROW_STR, MatchupOddsSchema, ProjectedStandingsSchema


@dataclass(frozen=True)
class SlotMap:
    """Bidirectional ESPN team id <-> simulator slot.

    The simulator requires contiguous slots 1..n_teams; ESPN team ids are arbitrary integers
    (Critts runs 1..17 with a gap). Sorting the ids gives a stable, reproducible assignment —
    the same league always maps the same way, so a snapshot written this week is comparable
    with one written last week.
    """

    team_ids: tuple[int, ...]

    @classmethod
    def from_team_ids(cls, team_ids: Sequence[int]) -> SlotMap:
        return cls(team_ids=tuple(sorted(set(int(t) for t in team_ids))))

    def slot(self, team_id: int) -> int:
        return self.team_ids.index(int(team_id)) + 1

    def team_id(self, slot: int) -> int:
        return self.team_ids[slot - 1]

    def __len__(self) -> int:
        return len(self.team_ids)


def first_unplayed_week(schedule: pd.DataFrame, calendar: LeagueCalendar) -> int:
    """The earliest regular-season week with an unplayed matchup.

    Derived from results rather than taken from a `--week` argument or the wall clock, so the
    snapshot can never claim a week is done that ESPN says is not. A fully played regular
    season returns `reg_weeks + 1`, which leaves the simulator nothing to simulate — correct,
    because at that point the standings are known rather than projected.
    """
    reg = schedule[schedule["week"] <= calendar.reg_weeks]
    unplayed = reg[~reg["is_played"]]
    if unplayed.empty:
        return calendar.reg_weeks + 1
    return int(unplayed["week"].min())


def schedule_to_slots(
    schedule: pd.DataFrame, slots: SlotMap, calendar: LeagueCalendar
) -> list[list[tuple[int, int]]]:
    """Real fixture list -> the week-indexed slot pairs `simulate_seasons` consumes.

    Every regular-season week must be present: a missing one would silently shorten the season
    rather than fail, and each team's record would come out low with no indication why.
    """
    by_week: dict[int, list[tuple[int, int]]] = {w: [] for w in calendar.reg_week_numbers}
    for row in schedule.itertuples():
        week = int(row.week)
        if week not in by_week:
            continue  # playoff weeks: the bracket handles those, not the matchup loop
        by_week[week].append((slots.slot(int(row.home_team_id)), slots.slot(int(row.away_team_id))))
    missing = [w for w, games in by_week.items() if not games]
    if missing:
        raise ValueError(
            f"schedule has no matchups for regular-season week(s) {missing}; the simulated "
            "season would silently be shorter than the calendar says."
        )
    return [by_week[w] for w in calendar.reg_week_numbers]


def locked_by_slot(records: pd.DataFrame, slots: SlotMap) -> dict[int, LockedRecord]:
    """`team_records` output -> the per-slot locked results the simulator seeds its tally with."""
    return {
        slots.slot(int(row.team_id)): LockedRecord(
            wins=int(row.wins),
            losses=int(row.losses),
            ties=int(row.ties),
            points_for=float(row.points_for),
        )
        for row in records.itertuples()
    }



def rosters_to_slots(
    rosters: pd.DataFrame,
    id_map: pd.DataFrame,
    slots: SlotMap,
    pool_ids: Collection[str],
) -> tuple[dict[int, list[str]], int]:
    """ESPN rosters -> `{slot: [gsis_id]}`, plus a count of players the pool cannot project.

    **The ESPN roster has no `gsis_id`.** `parse_rosters` emits ESPN's own `player_id`, and
    nothing in `espn_league.py` produces a gsis at all -- the crosswalk lives in the id_map.
    Reading a `gsis_id` column straight off an ESPN roster therefore matches nothing, and the
    resulting failure is silent in the worst way: empty rosters make `team_weekly_points`
    return all zeros, every matchup resolves `0 >= 0` so the HOME team wins every game in
    every simulation, and the output is a fully populated standings table whose playoff and
    title percentages come entirely from how many home fixtures each team happens to have.

    Same inner join on `espn_id` that `draft/backtest/espn_weekly.py` uses.

    Players the pool cannot project (kickers, defenses, anyone without a projection) are
    dropped and counted -- that part is legitimate. But a resolution that drops *everything*
    is an id_map failure rather than a roster of kickers, so it raises: the caller cannot tell
    those two apart from a count alone, and the second one produces numbers that look fine.
    """
    if rosters.empty:
        return {slot: [] for slot in range(1, len(slots) + 1)}, 0

    cross = id_map[["espn_id", "gsis_id"]].dropna().astype({"espn_id": str})
    merged = rosters.assign(espn_id=rosters["player_id"].astype(str)).merge(
        cross, on="espn_id", how="inner"
    )

    by_slot: dict[int, list[str]] = {slot: [] for slot in range(1, len(slots) + 1)}
    kept = 0
    for row in merged.itertuples():
        gsis = str(row.gsis_id)
        if gsis not in pool_ids:
            continue
        by_slot[slots.slot(int(row.team_id))].append(gsis)
        kept += 1

    if kept == 0:
        raise ValueError(
            f"no rostered player resolved to a projectable pool entry: {len(rosters)} roster "
            f"rows, {len(merged)} matched the id_map, 0 of those are in the pool. Simulating "
            "this would score every roster at zero and hand every matchup to the home team. "
            "Check that the id_map and the VORP pool cover this season."
        )
    return by_slot, len(rosters) - kept


def build_standings(
    outcomes: SeasonOutcomes,
    slots: SlotMap,
    team_names: Mapping[int, str],
    locked: Mapping[int, LockedRecord],
    *,
    season: int,
    snapshot_week: int,
) -> pd.DataFrame:
    """Per-team projected finish, validated against `ProjectedStandingsSchema`.

    `projected_wins` is banked wins plus the simulated remainder — the simulator already
    returns the total, because it seeds its tally with the locked record.
    """
    empty = LockedRecord()
    rows: list[dict[str, object]] = []
    for slot in outcomes.slots:
        team_id = slots.team_id(slot)
        col = outcomes.slots.index(slot)
        seeds = outcomes.seed[:, col]
        record = locked.get(slot, empty)
        rows.append(
            {
                "season": season,
                "week": snapshot_week,
                "team_id": team_id,
                "team_name": str(team_names.get(team_id, f"team {team_id}")),
                "wins": record.wins,
                "losses": record.losses,
                "ties": record.ties,
                "points_for": record.points_for,
                "games_played": record.games_played,
                "projected_wins": float(outcomes.wins[:, col].mean()),
                "make_playoffs_pct": float((seeds <= outcomes.calendar.playoff_size).mean()),
                "bye_pct": float((seeds <= outcomes.calendar.n_byes).mean()),
                "champ_pct": float((outcomes.champion == slot).mean()),
                "mean_seed": float(seeds.mean()),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["projected_wins", "points_for"], ascending=False, ignore_index=True
    )
    frame["team_name"] = frame["team_name"].astype(_PYARROW_STR)
    return ProjectedStandingsSchema.validate(frame)


def build_matchup_odds(
    outcomes: SeasonOutcomes,
    schedule: pd.DataFrame,
    slots: SlotMap,
    *,
    season: int,
    snapshot_week: int,
) -> pd.DataFrame:
    """P(home wins) for every remaining regular-season matchup, from the same simulations.

    Played matchups are excluded: they have a result, not a probability.
    """
    rows: list[dict[str, object]] = []
    for row in schedule.itertuples():
        week = int(row.week)
        if row.is_played or week < snapshot_week or week > outcomes.calendar.reg_weeks:
            continue
        home_slot = slots.slot(int(row.home_team_id))
        away_slot = slots.slot(int(row.away_team_id))
        rows.append(
            {
                "season": season,
                "week": week,
                "home_team_id": int(row.home_team_id),
                "away_team_id": int(row.away_team_id),
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "home_win_pct": outcomes.head_to_head(home_slot, away_slot, week),
                "home_mean_points": float(outcomes.week_points(home_slot, week).mean()),
                "away_mean_points": float(outcomes.week_points(away_slot, week).mean()),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "season",
            "week",
            "home_team_id",
            "away_team_id",
            "home_team",
            "away_team",
            "home_win_pct",
            "home_mean_points",
            "away_mean_points",
        ],
    ).sort_values(["week", "home_team_id"], ignore_index=True)
    for column in ("home_team", "away_team"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return MatchupOddsSchema.validate(frame)
