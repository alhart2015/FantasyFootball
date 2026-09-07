"""Is the lineup I have set the best lineup I can set this week?

    python scripts/start_sit.py                        # the one configured league
    python scripts/start_sit.py --week 5               # a week other than the next one
    python scripts/start_sit.py --weight-espn 1.0      # ESPN alone, for comparison
    python scripts/start_sit.py --fast                 # skip P(right)

**The blend is why this is worth running.** A report built on ESPN's weekly projection alone
says what the ESPN app already says. This prices every player from TWO sources — ESPN's weekly
feed and Sleeper's weekly endpoint — blends them stat by stat, and scores the result once under
*this* league's ruleset rather than either source's. The `spread` column is where they
disagree, which is where a close start/sit call actually lives.

**P(right)** is the probability the player being started outscores the one being benched, this
week. Change in expected season wins is deliberately NOT reported: paired noise there is 0.062
wins against roughly 140 season points to a win, so a 3-point weekly swap is ~0.021 wins — a
signal three times smaller than its own error bar. P(right) is the question that resolves.

Read-only. Nothing here writes to the store and nothing is sent to ESPN; setting the lineup is
still a job for the app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.league_profile import (
    add_league_arguments,
    resolve_league_target,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.backtest.espn_weekly import espn_weekly_statlines
from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import (
    EspnCredentials,
    EspnLeagueError,
    fetch_free_agents,
    fetch_league_payload,
    parse_free_agents,
    parse_rosters,
    parse_teams,
)
from projections.ingest.sleeper_weekly_projections import (
    SleeperWeeklyError,
    fetch_sleeper_weekly,
    parse_sleeper_weekly,
)
from projections.midseason.my_team import build_my_team
from projections.midseason.standings import ProjectionInputError
from projections.midseason.start_sit import StartSitRun, recommend_start_sit
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.store import read_partition


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The five league flags all default to the profile; see `resolve_league_target`.
    add_league_arguments(p, team_id_help="Set this team's lineup.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--credentials",
        type=Path,
        default=Path("configs/espn_credentials.json"),
        help="ESPN cookie file (gitignored).",
    )
    p.add_argument("--week", type=int, default=None, help="Default: the next unplayed week.")
    p.add_argument(
        "--weight-espn",
        type=float,
        default=0.5,
        help="ESPN's share of the blend. 1.0 is ESPN alone, 0.0 Sleeper alone. Default 0.5, "
        "which is a stated guess -- no weekly benchmark has measured it yet.",
    )
    p.add_argument("--n-sims", type=int, default=20_000, help="Draws behind P(right).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="Skip P(right).")
    return p.parse_args(argv)


def _fmt(value: float | None, width: int = 7, places: int = 1) -> str:
    return f"{'—':>{width}}" if value is None else f"{value:>{width}.{places}f}"


def _print_lineup(run: StartSitRun) -> None:
    espn_pct = round(run.weight_espn * 100)
    blend = f"blend: {espn_pct}% ESPN / {100 - espn_pct}% Sleeper"
    print(f"\nLINEUP — {run.team_name}, week {run.week}{blend:>38}")
    print(
        f"\n  {'player':<22}{'pos':<5}{'slot':<7}"
        f"{'espn':>7}{'slpr':>7}{'blend':>7}{'start':>7}{'spread':>8}  src"
    )
    # Repeated slots are numbered here and nowhere else: RB1/RB2 is a display convention, and
    # `label_starter_slots` deliberately returns the taxonomy rather than a rendering of it.
    seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    for slot in run.slots:
        counts[slot.value] = counts.get(slot.value, 0) + 1
    for index, slot in zip(run.starters, run.slots, strict=True):
        row = run.rows[index]
        seen[slot.value] = seen.get(slot.value, 0) + 1
        label = slot.value if counts[slot.value] == 1 else f"{slot.value}{seen[slot.value]}"
        # `blend` is the two sources; `start` is what the lineup is solved on, after the
        # injury multiplier. Separate columns because a Questionable player's `start`
        # sits BELOW both sources, which in one column reads as an arithmetic bug.
        blended = row.blend
        tag = "" if row.status.is_healthy else f"  {row.status.value}"
        print(
            f"  {row.name[:21]:<22}{row.position:<5}{label:<7}"
            f"{_fmt(blended.espn if blended else None)}"
            f"{_fmt(blended.sleeper if blended else None)}"
            f"{_fmt(blended.points if blended else None)}"
            f"{_fmt(row.points)}{_fmt(blended.spread if blended else None, 8)}"
            f"  {blended.sources if blended else 'unpriced'}{tag}"
        )
    print(f"\n  {'optimal total':<34}{run.optimal_total:>7.1f}")
    print(f"  {'currently set':<34}{run.current_total:>7.1f}")


def _print_swaps(run: StartSitRun) -> None:
    if not run.swaps:
        # A success, not a failure, and it reads as one -- the convention the waiver
        # recommender set for the same situation.
        print("\nSWAPS - none. Your lineup is already optimal.")
        return

    plural = "" if len(run.swaps) == 1 else "s"
    header = f"{len(run.swaps)} change{plural} worth making"
    print(f"\nSWAPS - {header}      {run.gain:+.1f} pts")
    for swap in run.swaps:
        print()
        for verb, row in (("START", swap.start), ("SIT", swap.sit)):
            where = row.current_slot.value if row.current_slot else "bench"
            tag = "" if row.status.is_healthy else f"   {row.status.value}"
            print(
                f"  {verb:<6}{row.name[:21]:<22}{row.position:<5}"
                f"{_fmt(row.points)}   ({where}){tag}"
            )
        odds = "" if swap.p_right is None else f"   P(right) {swap.p_right:.0%}"
        print(f"         {swap.gain:+.1f} pts{odds}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target = resolve_league_target(args, require_team_id=True)
        # Inside the try on purpose: `require_team_id` above is what normally raises, with a
        # message naming the profile, but this is the same failure and must not be a traceback.
        my_team_id = target.require_team()
        league_config_path = target.require_league_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if target.source is not None:
        print(target.describe())
    if not 0.0 <= args.weight_espn <= 1.0:
        print(f"--weight-espn must be in [0, 1], got {args.weight_espn}", file=sys.stderr)
        return 1

    try:
        creds = EspnCredentials.resolve(args.credentials)
        payload = fetch_league_payload(target.league_id, target.season, creds)
    except (EspnLeagueError, OSError) as exc:
        print(f"Cannot reach the league: {exc}", file=sys.stderr)
        return 1

    config = LeagueConfig.model_validate_json(league_config_path.read_text(encoding="utf-8"))
    pool = pd.read_parquet(target.pool)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)
    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")

    try:
        weekly_stats = read_partition(args.data_root / "raw", "weekly_stats", season=target.season)
    except FileNotFoundError:
        weekly_stats = pd.DataFrame()

    try:
        run_state = build_my_team(
            payload,
            pool,
            id_map,
            weekly_stats,
            config,
            my_team_id=my_team_id,
            season=target.season,
        )
    except ProjectionInputError as exc:
        print(f"Cannot set a lineup: {exc}", file=sys.stderr)
        return 1
    week = args.week or run_state.week

    # My own roster's weekly projections, priced by the league endpoint. The limit is sized to
    # the whole league on purpose: sharing a free-agent-sized limit silently dropped starters
    # from the projections in the waiver tool and inflated every number downstream.
    rostered_limit = config.n_teams * (sum(config.roster_slots.values()) + 2)
    try:
        mine_payload = fetch_free_agents(
            target.league_id,
            target.season,
            creds,
            scoring_period=week,
            limit=rostered_limit,
            statuses=("ONTEAM",),
        )
    except (EspnLeagueError, OSError) as exc:
        print(f"Cannot price the roster: {exc}", file=sys.stderr)
        return 1
    _, truncated = parse_free_agents(mine_payload, limit=rostered_limit)
    if truncated:
        print(f"  ! your own roster may be incompletely priced: {truncated}", file=sys.stderr)

    espn = espn_weekly_statlines(mine_payload, week=week)
    try:
        sleeper = parse_sleeper_weekly(
            fetch_sleeper_weekly(target.season, week), season=target.season, week=week
        )
    except SleeperWeeklyError as exc:
        # Not fatal, but it removes the entire reason to run this rather than open the app, so
        # it is said loudly rather than degraded into silently.
        print(f"  ! Sleeper unavailable ({exc}) — this run is ESPN alone.", file=sys.stderr)
        sleeper = parse_sleeper_weekly([], season=target.season, week=week)

    roster = parse_rosters(payload)
    roster = roster[roster["team_id"] == my_team_id]
    if roster.empty:
        teams = parse_teams(payload)
        print(f"No roster for team {my_team_id}. Teams: {sorted(teams['team_id'])}")
        return 1

    run = recommend_start_sit(
        roster,
        espn,
        sleeper,
        id_map,
        config.roster_slots,
        config.ruleset,
        team_name=run_state.team_name,
        week=week,
        weight_espn=args.weight_espn,
        params=None if args.fast else VarianceParams.load(),
        n_sims=args.n_sims,
        rng=np.random.default_rng(args.seed),
    )

    _print_lineup(run)
    _print_swaps(run)
    for note in (*run_state.notes, *run.notes):
        print(f"\n  ! {note}")
    print(
        "\n  'blend' is the two sources; 'start' is that after the injury multiplier, and\n"
        "  is what the lineup is solved on. 'spread' is how far the sources sit\n"
        "  apart — a wide spread on a close call is a reason to look, not a reason to move.\n"
        "  The 50/50 weight is a stated guess; no weekly benchmark has measured it yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
