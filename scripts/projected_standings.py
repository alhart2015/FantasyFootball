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
    EspnLeagueError,
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
    rosters_to_slots,
    schedule_to_slots,
)
from projections.schemas import _PYARROW_STR, VorpTableSchema
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
    # `resolve` tries the environment first and then the file, and raises with a longer
    # message than anything reproduced here. Using it also keeps ESPN_SWID / ESPN_S2 working,
    # which `from_file` alone silently ignored.
    try:
        creds = EspnCredentials.resolve(args.credentials)
    except EspnLeagueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = fetch_league_payload(args.league_id, args.season, creds=creds)

    config = build_league_config(payload)
    calendar = LeagueCalendar.from_espn_settings(
        (payload.get("settings", {}) or {}).get("scheduleSettings", {}) or {}
    )
    teams = parse_teams(payload)
    if args.team_id is not None and args.team_id not in set(teams["team_id"]):
        # `write_league_snapshot` performs exactly this check. Without it a typo just produces
        # a report with no "you" marker and no matchup section, which reads like success.
        print(
            f"--team-id {args.team_id} is not a team in this league. Valid ids: "
            f"{sorted(teams['team_id'])}.",
            file=sys.stderr,
        )
        return 1

    schedule = parse_schedule(payload, teams)
    if schedule.empty:
        # Without this, `first_unplayed_week` reports the season COMPLETE (no unplayed week
        # exists), the banner reads "week 15 of 14, 0 to play" like a finished season, the ROS
        # build zeroes the whole pool without warning, and the run dies later on an unrelated
        # "no matchups for weeks 1..14".
        print(
            "ESPN returned no schedule for this league. The mMatchup view is missing or the "
            "fixtures are not published yet; without it there is nothing to project over.",
            file=sys.stderr,
        )
        return 1

    rosters_frame = parse_rosters(payload)
    if rosters_frame.empty:
        print(
            "No rosters yet — the draft has not happened, so there is nothing to project.",
            file=sys.stderr,
        )
        return 1

    week = first_unplayed_week(schedule, calendar)
    weeks_remaining = max(calendar.reg_weeks - week + 1, 0)
    # Bounded to the weeks the simulator will NOT replay. `simulate_seasons` re-runs every
    # week from `week` onward, so anything banked from those same weeks double-counts -- which
    # a partially-played week and any played playoff week both cause.
    records = team_records(schedule, through_week=week - 1)
    print(
        f"{config.name} ({args.season}) — week {week} of {calendar.reg_weeks}, "
        f"{weeks_remaining} to play. {int(schedule['is_played'].sum())} matchups played."
    )

    slots = SlotMap.from_team_ids(list(teams["team_id"]))
    pool = _load_pool(args.pool)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)

    # **The pool IS the fresh projection source.** Its `season_mean_fpts` comes from whatever
    # `external_projections` snapshot it was built against, so rebuilding it (the same
    # generate_league_vorp_table.py run the draft prep already uses) is what makes these
    # numbers current. Passing an empty mapping here instead -- as the first version did --
    # sent 100% of the pool down the no-fresh-projection fallback and made the whole
    # `ros = fresh - to_date` path dead code from its only caller.
    fresh = {
        str(gsis): float(points)
        for gsis, points in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=True)
    }

    # TODO(week 1): per-player actuals live in the played weeks' `rosterForMatchupPeriod`
    # entries and are not parsed yet. Every value is zero until a season is under way, so the
    # subtraction is currently the identity and this is left explicit rather than guessed at
    # -- see the spec's open question on whether a provider's in-season season-total includes
    # games already played.
    ros_pool, diagnostics = rest_of_season_pool(
        pool,
        fresh_season_points=fresh,
        points_to_date={},
        weeks_remaining=weeks_remaining,
    )
    warning = diagnostics.warning()
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)

    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")
    rosters, dropped = rosters_to_slots(
        rosters_frame, id_map, slots, set(ros_pool["gsis_id"].astype(str))
    )
    if dropped:
        print(
            f"note: {dropped} rostered players are outside the projection pool "
            "(K/DST, or no projection) and were skipped.",
            file=sys.stderr,
        )
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


def _load_pool(path: Path) -> pd.DataFrame:
    """Read a VORP parquet the way every other consumer does, validation included.

    `tournament_cli._load_pool` is the same three lines; this mirrors it rather than dropping
    the `VorpTableSchema.validate` that version performs.
    """
    frame = pd.read_parquet(path)
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(frame)


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
