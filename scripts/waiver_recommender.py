"""Is anyone on waivers better than someone on my team?

    python scripts/waiver_recommender.py --league-id 856974 --season 2026 --team-id 8
    python scripts/waiver_recommender.py ... --fast          # skip the simulation
    python scripts/waiver_recommender.py ... --min-gain 2.0  # only real moves

**Δ WINS is the recommendation**, and the order the list comes back in. It is what the swap is
worth over the whole season, from a paired simulation, and it is the only currency in which
"better this week" and "worse for the rest of the season" are the same unit — a move that lifts
your odds 20% next week and costs 2.5 expected wins is a bad move, and nothing else here can
say so.

**LINEUP** is this week's starting-lineup gain. It is a filter and a sanity check, not the
answer: cheap enough to run over every free agent, zero for almost everybody in a 16-team
league, and verifiable against your own roster in a way a simulated win total is not. It
decides who gets simulated; it does not decide what to do.

`--fast` skips the simulation and leaves the lineup column alone. Use it when you want an
answer in seconds and are prepared to do the season-long arithmetic yourself.

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
from projections.midseason.swap_impact import PAIRED_DELTA_NOISE, SwapImpact, simulate_swaps
from projections.midseason.waivers import Candidate, player_id, rank_free_agents
from projections.schemas import (
    _PYARROW_STR,
    InjuryStatus,
    Ruleset,
    VorpTableSchema,
    display_str,
    parse_injury_status,
)
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


def _drop_line(candidate: Candidate) -> str:
    """What acting on this row costs you in roster space.

    Three states, not two. "Nobody has to go" and "we could not price anyone to drop" are
    opposite facts and used to print identically — a lie about the one thing this tool says is
    worth telling a reader first.
    """
    if candidate.is_free:
        return "no drop needed"
    if candidate.drop_player_id is None:
        return "NO DROP FOUND — nobody droppable could be priced"
    return f"drop {candidate.drop_player}"


def _print_candidate(
    candidate: Candidate, note: InjuryNote | None, impact: SwapImpact | None
) -> None:
    add = f"{candidate.player} ({candidate.position}, {candidate.nfl_team})"
    where = "WAIVERS" if candidate.on_waivers else "FA"
    print(f"\n  {add:<32} {where:<8} {_drop_line(candidate)}")
    if impact is None:
        print(f"    +{candidate.lineup_gain:.1f} to this week's lineup")
    elif not impact.simulated:
        # He is not in the projection pool, so there is nothing to put on the roster. Saying so
        # matters: a blank here reads as "simulated, and it came to nothing".
        print(
            f"    NOT SIMULATED — no season projection for him. "
            f"+{candidate.lineup_gain:.1f} to this week's lineup is all we can say."
        )
    else:
        certainty = "" if impact.beats_noise else "   (inside simulation noise)"
        print(
            f"    {impact.delta_wins:+.2f} wins   "
            f"{impact.delta_playoff_pct * 100:+.1f}% playoffs   "
            f"+{candidate.lineup_gain:.1f} lineup{certainty}"
        )
    if not candidate.is_free and candidate.drop_player_id is not None:
        print(f"    costs {candidate.drop_cost:.0f} rest-of-season points")
    # `is_healthy`, not `is not ACTIVE`: NORMAL, DAY_TO_DAY, FREE_AGENT and UNKNOWN all carry a
    # multiplier of 1.0, and FREE_AGENT is the expected value for much of the wire — so the old
    # test claimed an adjustment on players nothing was adjusted for.
    if not candidate.injury_status.is_healthy:
        print(f"    ! {candidate.injury_status.value} — adjusted for this")
    _print_note(note)


def _print_note(note: InjuryNote | None, *, indent: str = "    ") -> None:
    if note is None:
        return
    if note.summary():
        print(f"{indent}{note.summary()}")
    if note.short_comment:
        print(f'{indent}"{note.short_comment}"')
    if is_multi_week(note.status) and note.long_comment:
        print(f"{indent}{note.long_comment}")


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
    for run_note in run_state.notes:
        print(f"  ! {run_note}")

    if open_spots:
        # Said once, up front, because every "roster spot open" row below is claiming THESE
        # spots -- not one each. Acting on two of them when one is free overfills the roster.
        print(f"\n  {open_spots} active roster spot(s) open — an add there costs nothing.")

    if not candidates:
        print("\n  Nothing on the wire would change your starting lineup this week.")
        print("  In a 16-team league that is the usual answer, not a failure.")
        return 0

    # Lineup gain is the FILTER: it decides who is worth simulating. The order the reader
    # sees is the order stage 2 returns, because Δ wins is the recommendation.
    shortlist = candidates[: args.top]
    impacts: dict[int, SwapImpact] = {}
    if not args.fast:
        print(f"\n  simulating the top {len(shortlist)} ({args.n_sims} seasons each)…")
        simulated = simulate_swaps(
            payload,
            pool,
            id_map,
            load_store_availability(pool, season=args.season, data_root=args.data_root),
            VarianceParams.load(),
            shortlist,
            season=args.season,
            my_team_id=my_team_id,
            n_sims=args.n_sims,
            week=week,
        )
        impacts = {impact.candidate.player_id: impact for impact in simulated}
        # Re-ordered by what the tool is actually maximising. Keeping the lineup-gain order
        # made Δ wins an annotation on a points ranking rather than the objective.
        shortlist = [impact.candidate for impact in simulated]

    # My OWN injured players, not just the adds. "My starter went down, who do I pick up" is
    # the case this tool was built for, and his write-up is the thing that says how long he is
    # gone — fetching it only for the free agents answered half the question.
    mine_hurt = [
        player_id(player)
        for _, player in roster.iterrows()
        if not parse_injury_status(player.get("injury_status"))[0].is_healthy
    ]
    hurt = [int(c.player_id) for c in shortlist if not c.injury_status.is_healthy]
    notes = fetch_injury_notes([*mine_hurt, *hurt]) if (mine_hurt or hurt) else {}

    if mine_hurt:
        print("\n  on your roster:")
        for _, player in roster.iterrows():
            note = notes.get(player_id(player))
            if note is None:
                continue
            status, _ = parse_injury_status(player.get("injury_status"))
            print(f"    {display_str(player.get('player'))} — {status.value}")
            _print_note(note, indent="      ")

    for candidate in shortlist:
        _print_candidate(
            candidate, notes.get(candidate.player_id), impacts.get(candidate.player_id)
        )

    if args.fast:
        print("\n  Lineup gain only. Drop --fast to see what each move is worth in wins.")
    else:
        print(
            f"\n  Δ wins is a paired simulation; anything under {PAIRED_DELTA_NOISE:.2f} is "
            "inside its own noise. Roughly 140 season points to a win."
        )
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
        "--fast",
        action="store_true",
        help="skip the simulation and rank by this week's lineup gain alone",
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
