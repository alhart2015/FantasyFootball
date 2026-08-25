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
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.ingest.espn_league import (
    EspnCredentials,
    EspnLeagueError,
    fetch_league_payload,
    parse_teams,
)
from projections.midseason.standings import ProjectionInputError, project_league_standings
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

    # Guarded like every other precondition here. The id_map is load-bearing for the whole run
    # -- rosters cannot be matched to projections without it -- so a missing file should say so
    # rather than surface as a bare FileNotFoundError from inside an argument list.
    id_map_path = args.data_root / "raw" / "id_map.parquet"
    if not id_map_path.exists():
        print(
            f"No id_map at {id_map_path}. Rosters are matched to projections through it, so "
            "the run cannot proceed without one.",
            file=sys.stderr,
        )
        return 1

    pool = attach_is_rookie(_load_pool(args.pool), season=args.season, data_root=args.data_root)
    try:
        run = project_league_standings(
            payload,
            pool,
            pd.read_parquet(id_map_path),
            load_store_availability(pool, season=args.season, data_root=args.data_root),
            VarianceParams.load(),
            season=args.season,
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
        f"{run.league_name} ({args.season}) — week {run.snapshot_week} of "
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

    _print_standings(run.standings, my_team_id=args.team_id)
    _print_ties_footnote()
    _print_my_matchups(run.odds, my_team_id=args.team_id)

    if args.write_snapshot:
        for table, frame in (("projected_standings", run.standings), ("matchup_odds", run.odds)):
            path = write_partition(
                args.data_root / "processed",
                table,
                frame,
                season=args.season,
                week=run.snapshot_week,
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
        "* PROJ W is the MEAN simulated final win total, so it is fractional whether or not "
        "anyone ties. Ties count half a win, as ESPN seeds."
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
