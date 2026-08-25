"""Projected standings and matchup odds for a live ESPN league.

Pulls the league (settings, rosters, schedule, results), builds rest-of-season projections,
runs one Monte-Carlo over the real remaining fixture list with played weeks locked, and prints
where every team is heading plus the odds on each remaining game. Optionally writes the run as
a weekly snapshot so the trajectory can be read back across the season.

Usage:
    python scripts/projected_standings.py --league-id 856974 --season 2026
    python scripts/projected_standings.py --league-id 856974 --season 2026 --write-snapshot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import simulate_seasons
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.league_calendar import LeagueCalendar
from projections.ingest.espn_league import (
    EspnCredentials,
    build_league_config,
    fetch_league_payload,
    parse_rosters,
    parse_schedule,
    parse_teams,
    team_records,
)
from projections.midseason.rest_of_season import rest_of_season_pool
from projections.midseason.standings import (
    SlotMap,
    build_matchup_odds,
    build_standings,
    first_unplayed_week,
    locked_by_slot,
    schedule_to_slots,
)
from projections.schemas import _PYARROW_STR
from projections.store import write_partition


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--league-id", type=int, required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--team-id", type=int, default=None, help="Highlight this team as mine.")
    p.add_argument(
        "--credentials",
        type=Path,
        default=Path("configs/espn_credentials.json"),
        help="ESPN cookie file (gitignored).",
    )
    p.add_argument("--pool", type=Path, required=True, help="VORP parquet for this league.")
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Persist this run under data/processed/projected_standings/ for the trend line.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    creds = EspnCredentials.from_file(args.credentials)
    if creds is None:
        # `from_file` returns None for a missing file rather than raising, so an absent
        # credentials path would otherwise reach the request as a None and fail there.
        print(
            f"No ESPN credentials at {args.credentials}. It needs "
            '{"swid": ..., "espn_s2": ...} — copy both cookies from a logged-in '
            "fantasy.espn.com session. The file is gitignored.",
            file=sys.stderr,
        )
        return 1
    payload = fetch_league_payload(args.league_id, args.season, creds=creds)

    config = build_league_config(payload)
    calendar = LeagueCalendar.from_espn_settings(
        (payload.get("settings", {}) or {}).get("scheduleSettings", {}) or {}
    )
    teams = parse_teams(payload)
    schedule = parse_schedule(payload, teams)
    records = team_records(schedule)
    rosters_frame = parse_rosters(payload)
    if rosters_frame.empty:
        print(
            "No rosters yet — the draft has not happened, so there is nothing to project.",
            file=sys.stderr,
        )
        return 1

    week = first_unplayed_week(schedule, calendar)
    weeks_remaining = max(calendar.reg_weeks - week + 1, 0)
    print(
        f"{config.name} ({args.season}) — week {week} of {calendar.reg_weeks}, "
        f"{weeks_remaining} to play. {int(schedule['is_played'].sum())} matchups played."
    )

    slots = SlotMap.from_team_ids(list(teams["team_id"]))
    pool = pd.read_parquet(args.pool)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)

    # TODO(week 1): points_to_date needs per-player actuals, which live in the played weeks'
    # `rosterForMatchupPeriod` entries. Until a season is under way every value is zero and
    # the subtraction is the identity, so this is wired as an empty mapping rather than
    # guessed at -- see the spec's open question on the season-total assumption.
    ros_pool, diagnostics = rest_of_season_pool(
        pool,
        fresh_season_points={},
        points_to_date={},
        weeks_remaining=weeks_remaining,
        reg_weeks=calendar.reg_weeks,
    )
    warning = diagnostics.warning()
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)

    rosters = _rosters_by_slot(rosters_frame, slots, ros_pool)
    availability = load_store_availability(ros_pool, season=args.season, data_root=args.data_root)
    outcomes = simulate_seasons(
        rosters,
        ros_pool,
        availability,
        VarianceParams.load(),
        league_config=config,
        n_sims=args.n_sims,
        rng=np.random.default_rng(args.seed),
        calendar=calendar,
        schedule=schedule_to_slots(schedule, slots, calendar),
        locked=locked_by_slot(records, slots),
        first_unplayed_week=week,
    )

    names = dict(zip(teams["team_id"], teams["team_name"], strict=False))
    standings = build_standings(
        outcomes,
        slots,
        names,
        locked_by_slot(records, slots),
        season=args.season,
        snapshot_week=week,
    )
    odds = build_matchup_odds(outcomes, schedule, slots, season=args.season, snapshot_week=week)

    _print_standings(standings, my_team_id=args.team_id)
    _print_my_matchups(odds, my_team_id=args.team_id)

    if args.write_snapshot:
        for table, frame in (("projected_standings", standings), ("matchup_odds", odds)):
            path = write_partition(
                args.data_root / "processed", table, frame, season=args.season, week=week
            )
            print(f"Wrote {path}")
    return 0


def _rosters_by_slot(
    rosters_frame: pd.DataFrame, slots: SlotMap, pool: pd.DataFrame
) -> dict[int, list[str]]:
    """ESPN rosters -> {slot: [gsis_id]}, keeping only players the pool can project.

    A rostered player absent from the pool (a kicker, a defense, a rookie with no projection)
    contributes nothing and would raise on the index lookup, so it is dropped here rather than
    deeper in the simulator where the error would name a gsis id and nothing else.
    """
    known = set(pool["gsis_id"].astype(str))
    by_slot: dict[int, list[str]] = {slot: [] for slot in range(1, len(slots) + 1)}
    dropped = 0
    for row in rosters_frame.itertuples():
        gsis = str(getattr(row, "gsis_id", "") or "")
        if gsis not in known:
            dropped += 1
            continue
        by_slot[slots.slot(int(row.team_id))].append(gsis)
    if dropped:
        print(
            f"note: {dropped} rostered players are outside the projection pool "
            "(K/DST, or no projection) and were skipped.",
            file=sys.stderr,
        )
    return by_slot


def _print_standings(standings: pd.DataFrame, *, my_team_id: int | None) -> None:
    print()
    print(f"{'TEAM':<28}{'REC':>8}{'PROJ W':>9}{'PLAYOFF':>10}{'BYE':>8}{'TITLE':>8}")
    for row in standings.itertuples():
        mark = " <-- you" if my_team_id is not None and row.team_id == my_team_id else ""
        record = f"{row.wins}-{row.losses}" + (f"-{row.ties}" if row.ties else "")
        print(
            f"{str(row.team_name)[:27]:<28}{record:>8}{row.projected_wins:>9.1f}"
            f"{row.make_playoffs_pct:>9.1%}{row.bye_pct:>8.1%}{row.champ_pct:>8.1%}{mark}"
        )


def _print_my_matchups(odds: pd.DataFrame, *, my_team_id: int | None) -> None:
    if my_team_id is None or odds.empty:
        return
    mine = odds[(odds["home_team_id"] == my_team_id) | (odds["away_team_id"] == my_team_id)]
    if mine.empty:
        return
    print("\nYour remaining games:")
    for row in mine.itertuples():
        at_home = row.home_team_id == my_team_id
        opponent = row.away_team if at_home else row.home_team
        win_pct = row.home_win_pct if at_home else 1.0 - row.home_win_pct
        side = "vs" if at_home else "@ "
        print(f"  wk{row.week:>3}  {side} {str(opponent)[:24]:<26}{win_pct:>6.1%}")


if __name__ == "__main__":
    raise SystemExit(main())
