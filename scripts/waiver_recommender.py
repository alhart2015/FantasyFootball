"""Is anyone on waivers better than someone on my team?

    python scripts/waiver_recommender.py --league-id 856974 --season 2026 --team-id 8
    python scripts/waiver_recommender.py ... --wins          # also simulate the shortlist
    python scripts/waiver_recommender.py ... --min-gain 2.0  # only real moves

Two numbers, and they answer different questions:

**LINEUP** is this week's starting-lineup gain — what adding him does to the points you would
actually start. It is cheap, and in a 16-team league it is zero for almost everybody, which is
the honest answer: a free agent who would not crack your lineup does not help you this week
however good he looks.

**Δ WINS** is what the swap is worth over the whole season, from a paired simulation. It is the
only currency in which "better this week" and "worse for the rest of the season" are the same
unit — a move that lifts your odds 20% next week and costs 2.5 expected wins is a bad move, and
only this column can say so. It costs a full Monte-Carlo season per candidate, so `--wins`
runs it on the shortlist only.

When a recommendation is driven by an injury, the beat-reporter write-up is printed under it.
The games-missed number for a player on IR is a guess (the NFL minimum); the write-up usually
is not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.backtest.espn_weekly import _statline_dict, _weekly_proj_stats
from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import (
    DEFAULT_FREE_AGENT_LIMIT,
    EspnCredentials,
    EspnLeagueError,
    fetch_free_agents,
    fetch_league_payload,
    parse_free_agents,
    parse_rosters,
    parse_teams,
)
from projections.ingest.injury_news import InjuryNote, fetch_injury_notes
from projections.midseason.injuries import is_multi_week, season_multiplier
from projections.midseason.my_team import build_my_team
from projections.midseason.standings import ProjectionInputError
from projections.midseason.waivers import (
    PAIRED_DELTA_NOISE,
    Candidate,
    SwapImpact,
    rank_free_agents,
    simulate_swaps,
)
from projections.schemas import _PYARROW_STR, InjuryStatus, Ruleset, VorpTableSchema
from projections.scoring.score import expected_points
from projections.store import read_partition


def weekly_projections(payload: dict[str, Any], week: int, ruleset: Ruleset) -> dict[str, float]:
    """ESPN's weekly projections out of a `kona_player_info` payload, keyed by ESPN id.

    **Keyed by ESPN id, not gsis.** `refresh_espn_weekly_projections` crosswalks through the
    id_map and drops anyone it cannot resolve, which is exactly the just-signed players a waiver
    tool exists to find. Rosters and free agents both arrive from ESPN, so the ESPN id is the
    key both sides already share.

    Scored under the league's own ruleset rather than read from ESPN's `appliedTotal`, so a
    free agent and a rostered player are valued the same way — and the same way the rest of
    this repo values anybody.

    A player with no projection for the week is ABSENT from the mapping rather than present at
    zero. That is what makes him unstartable downstream, which is how bye weeks work without a
    rule about bye weeks.
    """
    out: dict[str, float] = {}
    for entry in payload.get("players", []) or []:
        player = entry.get("player", {}) or {}
        raw = _weekly_proj_stats(player, week)
        if raw is None:
            continue
        out[str(player.get("id", 0))] = expected_points(_statline_dict(raw), ruleset)
    return out


def _print_header(team_name: str, week: int, n_agents: int, truncated: str | None) -> None:
    print(f"\n{team_name} — week {week}")
    print(f"{n_agents} free agents considered")
    if truncated:
        print(f"  ! {truncated}")


def _print_candidate(candidate: Candidate, note: InjuryNote | None, impact_line: str = "") -> None:
    add = f"{candidate.player} ({candidate.position})"
    where = "WAIVERS" if candidate.on_waivers else "FA"
    drop = "roster spot open" if candidate.is_free else f"drop {candidate.drop_player}"
    print(f"\n  {add:<28} {where:<8} +{candidate.lineup_gain:5.1f} lineup   {drop}")
    if not candidate.is_free:
        print(f"    {'':<28} costs {candidate.drop_cost:.0f} rest-of-season points")
    if impact_line:
        print(f"    {impact_line}")
    if candidate.injury_status is not InjuryStatus.ACTIVE:
        print(f"    ! {candidate.injury_status.value} — adjusted for this")
    if note is not None:
        if note.summary():
            print(f"    {note.summary()}")
        if note.short_comment:
            print(f'    "{note.short_comment}"')
        if is_multi_week(note.status) and note.long_comment:
            print(f"    {note.long_comment}")


def run(args: argparse.Namespace) -> int:
    creds = EspnCredentials.resolve(args.credentials)
    payload = fetch_league_payload(args.league_id, args.season, creds)
    teams = parse_teams(payload)

    my_team_id = args.team_id
    if my_team_id is None:
        print("--team-id is required. Teams in this league:")
        for _, team in teams.iterrows():
            print(f"  {int(team['team_id']):>3}  {team['team_name']}")
        return 2

    config = LeagueConfig.model_validate_json(
        (args.league_dir / "league_config.json").read_text(encoding="utf-8")
    )
    pool = pd.read_parquet(args.pool)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = attach_is_rookie(
        VorpTableSchema.validate(pool), season=args.season, data_root=args.data_root
    )
    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")

    try:
        weekly_stats = read_partition(args.data_root / "raw", "weekly_stats", season=args.season)
    except FileNotFoundError:
        weekly_stats = pd.DataFrame()

    run_state = build_my_team(
        payload,
        pool,
        id_map,
        weekly_stats,
        config,
        my_team_id=my_team_id,
        season=args.season,
    )
    week = args.week or run_state.week

    fa_payload = fetch_free_agents(
        args.league_id, args.season, creds, scoring_period=week, limit=args.free_agent_limit
    )
    free_agents, truncated = parse_free_agents(fa_payload, limit=args.free_agent_limit)
    projections = weekly_projections(fa_payload, week, config.ruleset)

    # My own roster's weekly projections come from the same call: `filterStatus` excludes them,
    # so a second request asks for the players I already have. One source, one scoring pass --
    # comparing my starter against a free agent priced any other way is not a comparison.
    mine_payload = fetch_free_agents(
        args.league_id,
        args.season,
        creds,
        scoring_period=week,
        limit=args.free_agent_limit,
        statuses=("ONTEAM",),
    )
    projections.update(weekly_projections(mine_payload, week, config.ruleset))

    remaining = _remaining_by_espn_id(run_state, id_map)
    roster = parse_rosters(payload)
    roster = roster[roster["team_id"] == my_team_id]

    candidates, open_spots = rank_free_agents(
        roster,
        free_agents,
        projections,
        remaining,
        config,
        min_gain=args.min_gain,
    )

    _print_header(run_state.team_name, week, len(free_agents), truncated)
    for note in run_state.notes:
        print(f"  ! {note}")

    if open_spots:
        # Said once, up front, because every "roster spot open" row below is claiming THESE
        # spots -- not one each. Acting on two of them when one is free overfills the roster.
        print(f"\n  {open_spots} active roster spot(s) open — an add there costs nothing.")

    if not candidates:
        print("\n  Nothing on the wire would change your starting lineup this week.")
        print("  In a 16-team league that is the usual answer, not a failure.")
        return 0

    shortlist = candidates[: args.top]
    impacts: dict[int, SwapImpact] = {}
    if args.wins:
        print(f"\n  simulating the top {len(shortlist)} ({args.n_sims} seasons each)…")
        for impact in simulate_swaps(
            payload,
            pool,
            id_map,
            load_store_availability(pool, season=args.season, data_root=args.data_root),
            VarianceParams.load(),
            shortlist,
            season=args.season,
            my_team_id=my_team_id,
            n_sims=args.n_sims,
        ):
            impacts[impact.candidate.player_id] = impact

    hurt = [
        int(c.player_id)
        for c in shortlist
        if c.injury_status is not InjuryStatus.ACTIVE and not c.injury_status.is_healthy
    ]
    notes = fetch_injury_notes(hurt) if hurt else {}

    for candidate in shortlist:
        # A different name from the loop variable above, which is a `SwapImpact`; this one is
        # optional because a candidate the pool cannot project is skipped by `simulate_swaps`.
        simulated = impacts.get(candidate.player_id)
        line = ""
        if simulated is not None:
            sign = "+" if simulated.delta_wins >= 0 else ""
            certainty = "" if abs(simulated.delta_wins) > PAIRED_DELTA_NOISE else "  (inside noise)"
            line = (
                f"{sign}{simulated.delta_wins:.2f} wins   "
                f"{simulated.delta_playoff_pct * 100:+.1f}% playoffs{certainty}"
            )
        _print_candidate(candidate, notes.get(candidate.player_id), line)

    if args.wins:
        print(
            f"\n  Δ wins is a paired simulation; anything under {PAIRED_DELTA_NOISE:.2f} is "
            "inside its own noise."
        )
    else:
        print("\n  Run with --wins to see what each move is worth over the season.")
    return 0


def _remaining_by_espn_id(run_state: Any, id_map: pd.DataFrame) -> dict[str, float]:
    """Rest-of-season points per ESPN id — the cost side of a drop.

    Injury-adjusted, because the point of this column is deciding who to let go: a player on IR
    is worth less for the rest of the season than his projection says, and that is exactly the
    situation in which you are looking for a drop candidate.
    """
    cross = id_map[["espn_id", "gsis_id"]].dropna().astype({"espn_id": str})
    cross = cross.drop_duplicates("espn_id")
    by_gsis = dict(
        zip(
            run_state.ros["gsis_id"].astype(str),
            run_state.ros["season_mean_fpts"].astype(float),
            strict=True,
        )
    )
    status_by_gsis = dict(
        zip(
            run_state.roster["gsis_id"].astype(str),
            run_state.roster["injury_status"].astype(str),
            strict=True,
        )
    )
    games_left = max(17 - (run_state.week - 1), 0)
    out: dict[str, float] = {}
    for espn_id, gsis in zip(cross["espn_id"], cross["gsis_id"].astype(str), strict=True):
        points = by_gsis.get(gsis)
        if points is None:
            continue
        status = InjuryStatus(status_by_gsis.get(gsis, InjuryStatus.ACTIVE.value))
        out[str(espn_id)] = points * season_multiplier(status, games_remaining=games_left)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--team-id", type=int, help="your team; omit to list them")
    parser.add_argument("--week", type=int, help="defaults to the first unplayed week")
    parser.add_argument("--league-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--credentials", type=Path, default=Path("configs/espn_credentials.json"))
    parser.add_argument("--free-agent-limit", type=int, default=DEFAULT_FREE_AGENT_LIMIT)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.5,
        help="ignore adds worth less than this to your starting lineup",
    )
    parser.add_argument("--top", type=int, default=5, help="how many candidates to show")
    parser.add_argument(
        "--wins", action="store_true", help="simulate the shortlist for expected-wins impact"
    )
    parser.add_argument("--n-sims", type=int, default=2000)
    args = parser.parse_args(argv)

    try:
        return run(args)
    except (ProjectionInputError, EspnLeagueError, OSError) as exc:
        print(f"Cannot recommend: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
