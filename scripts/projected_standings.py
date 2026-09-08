"""Projected standings and matchup odds for a live ESPN league.

Pulls the league (settings, rosters, schedule, results), builds rest-of-season projections,
runs one Monte-Carlo over the real remaining fixture list with played weeks locked, and prints
where every team is heading plus the odds on each remaining game. Optionally writes the run as
a weekly snapshot so the trajectory can be read back across the season.

Usage:
    python scripts/projected_standings.py                    # the one configured league
    python scripts/projected_standings.py --write-snapshot
    python scripts/projected_standings.py --league-id 856974 --season 2026 \
        --team-id 17 --pool data/vorp_2026/critts_half16_snake.parquet

With no league arguments, league id, season, team and pool come from the single
`board_profile.json` under `data/leagues/` — the same file the draft board loads.
Anything typed wins over the file, and the run announces which profile it used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.league_profile import (
    add_league_arguments,
)
from projections.ingest.espn_league import (
    EspnLeagueError,
)
from projections.midseason.context import InSeasonContext, build_context
from projections.midseason.standings import ProjectionInputError, project_league_standings
from projections.midseason.swap_impact import injury_adjusted_pool
from projections.store import write_partition


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # The five league flags all default to the profile; see `resolve_league_target`.
    add_league_arguments(p, team_id_help="Highlight this team as mine.")
    p.add_argument(
        "--credentials",
        type=Path,
        default=Path("configs/espn_credentials.json"),
        help="ESPN cookie file (gitignored).",
    )
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Persist this run under data/processed/projected_standings/ for the trend line.",
    )
    return p.parse_args(argv)


def report(ctx: InSeasonContext, args: argparse.Namespace) -> int:
    """Project the league and print it. Everything comes off `ctx` — no I/O here."""
    teams = ctx.teams()
    if ctx.my_team_id is not None and ctx.my_team_id not in set(teams["team_id"]):
        # `write_league_snapshot` performs exactly this check. Without it a typo just produces
        # a report with no "you" marker and no matchup section, which reads like success.
        print(
            f"team id {ctx.my_team_id} is not a team in this league. Valid ids: "
            f"{sorted(teams['team_id'])}.",
            file=sys.stderr,
        )
        return 1

    target = ctx.target
    # **The simulator has no concept of an injury, so the pool is where one has to reach it.**
    # This ran on the raw pool until 2026-09-07, which meant every projected finish in the
    # league treated a player on IR as though he would play all seventeen games. The waiver and
    # trade tools already adjusted before simulating; only the tool whose entire output IS the
    # simulation did not, so its numbers were the least trustworthy of the three and looked the
    # most authoritative.
    #
    # **`schedule_week`, not `week`.** `project_league_standings` takes no week and
    # re-derives the schedule's own, so the discount handed to it must use that same
    # number: `season_multiplier` divides games missed by games REMAINING. With `--week 12`
    # during real week 3 the two diverge -- IR players haircut over 6 games while the
    # simulator replays 15 -- and the playoff odds still look entirely reasonable.
    adjusted_pool = injury_adjusted_pool(ctx.pool, ctx.payload, ctx.id_map, week=ctx.schedule_week)

    try:
        run = project_league_standings(
            ctx.payload,
            adjusted_pool,
            ctx.id_map,
            # Availability is fitted from history and is about MISSED GAMES generally; the
            # injury adjustment above is about THIS player's current designation. Two
            # different things, so availability still reads the unadjusted pool.
            ctx.availability(),
            ctx.variance_params(),
            season=target.season,
            n_sims=args.n_sims,
            rng=np.random.default_rng(args.seed),
        )
    except ProjectionInputError as exc:
        # Only the "this payload cannot support a projection" cases: no schedule, no rosters,
        # or a team that resolved to nothing. A bare `except ValueError` also swallowed the
        # machinery's own bugs -- a zip-strict mismatch, SlotMap on an unknown team id -- and
        # printed them as though the user had supplied bad input.
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"{run.league_name} ({target.season}) — week {run.snapshot_week} of "
        f"{run.calendar.reg_weeks}, {run.weeks_remaining} to play. "
        f"{run.n_matchups_played} matchups played."
    )
    warning = run.diagnostics.warning()
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)
    if run.n_players_dropped:
        print(
            f"note: {run.n_players_dropped} rostered players are outside the projection pool "
            "(K/DST, or no projection) and were skipped.",
            file=sys.stderr,
        )

    _print_standings(run.standings, my_team_id=ctx.my_team_id)
    _print_ties_footnote()
    _print_my_matchups(run.odds, my_team_id=ctx.my_team_id)

    if args.write_snapshot:
        for table, frame in (("projected_standings", run.standings), ("matchup_odds", run.odds)):
            path = write_partition(
                ctx.data_root / "processed",
                table,
                frame,
                season=target.season,
                week=run.snapshot_week,
            )
            print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        # Neither is required by this tool and both were optional before the refactor: it
        # runs without --team-id (losing only the "you" marker) and never read a
        # league_config.json, because `project_league_standings` derives its own from the
        # payload. Demanding either here would make a working command start failing.
        ctx = build_context(args, require_team_id=False, require_config=False)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except EspnLeagueError as exc:
        # `resolve` tries the environment first and then the file, and raises with a longer
        # message than anything reproduced here. Using it also keeps ESPN_SWID / ESPN_S2
        # working, which `from_file` alone silently ignored.
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        # The id_map is load-bearing -- rosters cannot be matched to projections without it --
        # so a missing file should say so rather than surface from inside an argument list.
        # Names the file that is actually missing. `build_context` reads the pool, the
        # id_map and the rookie history; blaming the id_map for a mistyped --pool pointed
        # the reader at the wrong file.
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1
    if ctx.target.source is not None:
        print(ctx.target.describe())
    for note in ctx.notes:
        print(f"  ! {note}", file=sys.stderr)
    return report(ctx, args)


def _print_standings(standings: pd.DataFrame, *, my_team_id: int | None) -> None:
    print()
    # The star and its footnote are unconditional, because BOTH reasons PROJ W is fractional
    # are always in play: it is a mean over simulations, and a tie counts half a win. An
    # earlier version tried to print the star only "when a tie made it differ" and keyed that
    # off `projected_wins % 1` -- which is a Monte-Carlo mean, essentially never integral, so
    # it asserted a tie rule on every run where nothing had tied.
    print(f"{'TEAM':<28}{'REC':>8}{'PROJ W*':>9}{'PLAYOFF':>10}{'BYE':>8}{'TITLE':>8}")
    for row in standings.itertuples():
        mark = " <-- you" if my_team_id is not None and row.team_id == my_team_id else ""
        record = f"{row.wins}-{row.losses}" + (f"-{row.ties}" if row.ties else "")
        print(
            f"{str(row.team_name)[:27]:<28}{record:>8}{row.projected_wins:>9.1f}"
            f"{row.make_playoffs_pct:>9.1%}{row.bye_pct:>8.1%}{row.champ_pct:>8.1%}{mark}"
        )


def _print_ties_footnote() -> None:
    """Say what PROJ W is. Unconditional, because both reasons it is fractional always are."""
    print()
    print(
        "* PROJ W is a mean over remaining-season simulations, and counts a tie as half a "
        "win the way ESPN seeds. Either is enough to make it fractional."
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
