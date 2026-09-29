"""Is anyone on waivers better than someone on my team?

    python scripts/waiver_recommender.py                     # the one configured league
    python scripts/waiver_recommender.py --team-id 8         # somebody else's team
    python scripts/waiver_recommender.py ... --min-season-margin 15  # only real upgrades

**Two lists, one per horizon, never mixed.** "Better than someone" means better than anyone on
the roster, starter or bench -- not just "would crack this week's lineup", which is what an
earlier version asked and which scored a free agent better than my whole bench as zero.

**Per position.** Each free agent is compared with your weakest player at HIS position --
otherwise backup quarterbacks "beat" your bench running back on raw points and fill the list.

**REST OF SEASON** — free agents with more rest-of-season points than your weakest player at
their position. That player is the drop.

**THIS WEEK** — free agents projected to outscore your weakest player at their position who
is playing this week, whatever their rest of season looks like. For spotting a one-week stream.
No drop is named; the season list is where cuts are decided.

**Points, not wins.** Both lists are in projected fantasy points, sorted by how many points he
beats your player by. An earlier version simulated each season swap and ranked by expected
wins; the owner dropped it on 2026-09-29 in favour of numbers a reader can check by hand. The
simulator is still in `midseason.swap_impact`.

**LINEUP** on every row is what this week's starting lineup does if you make the move. It is a
sanity check you can verify against your own roster, not a filter: a bench upgrade reads 0.0.

When a recommendation is driven by an injury, the beat-reporter write-up is printed under it.
The games-missed number for a player on IR is a guess (the NFL minimum); the write-up usually
is not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from projections.console import force_utf8_stdio
from projections.draft.assistant.league_profile import (
    add_league_arguments,
    resolve_league_target,
)
from projections.ingest.espn_league import (
    DEFAULT_FREE_AGENT_LIMIT,
    EspnCredentials,
    EspnLeagueError,
    fetch_free_agents,
    fetch_league_payload,
    parse_free_agents,
    parse_teams,
)
from projections.ingest.injury_news import InjuryNote, fetch_injury_notes
from projections.midseason.context import InSeasonContext, assemble_context
from projections.midseason.injuries import is_multi_week
from projections.midseason.standings import ProjectionInputError
from projections.midseason.waivers import (
    Candidate,
    player_id,
    remaining_points_by_espn_id,
    season_upgrades,
    weekly_projections_by_espn_id,
    weekly_upgrades,
)
from projections.schemas import (
    InjuryStatus,
    display_str,
    parse_injury_status,
)


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
        # Deliberately vague about the cause: a player can be undroppable because the pool
        # cannot price him OR because he is on IR (dropping whom frees an IR slot, not the
        # active one). Naming only the pricing reason was false whenever it was the other.
        return "NO DROP FOUND — nobody on your roster can be dropped for him"
    return f"drop {candidate.drop_player}"


def _headline(candidate: Candidate, tail: str) -> None:
    add = f"{candidate.player} ({candidate.position}, {candidate.nfl_team})"
    where = "WAIVERS" if candidate.on_waivers else "FA"
    print(f"\n  {add:<32} {where:<8} {tail}")


def _print_candidate(candidate: Candidate, note: InjuryNote | None) -> None:
    """One row of the REST OF SEASON list."""
    _headline(candidate, _drop_line(candidate))
    print(
        f"    {candidate.season_points or 0.0:.0f} rest-of-season pts vs "
        f"{_versus(candidate, f'{candidate.beats_points:.0f}')} ({candidate.margin:+.0f})   "
        f"{candidate.lineup_gain:+.1f} to this week's lineup"
    )
    _print_status(candidate)
    _print_note(note)


def _versus(candidate: Candidate, points: str) -> str:
    """Who he is compared against. Nobody, when I have nobody playing at his position."""
    if not candidate.beats_player:
        return f"nobody at {candidate.position}"
    return f"{candidate.beats_player} {points}"


def _print_weekly(candidate: Candidate, note: InjuryNote | None) -> None:
    """One row of the THIS WEEK list. Out-projecting someone for a week is no reason to cut him."""
    season = (
        "no rest-of-season projection"
        if candidate.season_points is None
        else f"{candidate.season_points:.0f} rest-of-season pts"
    )
    _headline(candidate, season)
    print(
        f"    {candidate.projected or 0.0:.1f} pts this week vs "
        f"{_versus(candidate, f'{candidate.beats_points:.1f}')} ({candidate.margin:+.1f})   "
        f"{candidate.lineup_gain:+.1f} to this week's lineup"
    )
    _print_status(candidate)
    _print_note(note)


def _print_status(candidate: Candidate) -> None:
    # `is_healthy`, not `is not ACTIVE`: NORMAL, DAY_TO_DAY, FREE_AGENT and UNKNOWN all carry a
    # multiplier of 1.0, and FREE_AGENT is the expected value for much of the wire — so the old
    # test claimed an adjustment on players nothing was adjusted for.
    if not candidate.injury_status.is_healthy:
        print(f"    ! {candidate.injury_status.value} — adjusted for this")
    elif candidate.injury_status is InjuryStatus.UNKNOWN:
        # `UNKNOWN` counts as healthy on purpose -- an unrecognised status is a gap in our
        # mapping, not evidence about the player. But `injury_status_raw` exists precisely so
        # the gap can be reported rather than swallowed, and keying the notice off `is_healthy`
        # alone made it silent. The reader should know we saw something we could not place.
        print("    ! ESPN reported a status we do not recognise; treated as healthy")


def _print_note(note: InjuryNote | None, *, indent: str = "    ") -> None:
    if note is None:
        return
    if note.summary():
        print(f"{indent}{note.summary()}")
    if note.short_comment:
        print(f'{indent}"{note.short_comment}"')
    if is_multi_week(note.status) and note.long_comment:
        print(f"{indent}{note.long_comment}")


def report(ctx: InSeasonContext, args: argparse.Namespace) -> int:
    """Rank the wire against my roster. Everything shared comes off `ctx`."""
    target = ctx.target
    config = ctx.require_config()
    id_map = ctx.id_map
    creds = ctx.creds
    run_state = ctx.my_team()
    week = ctx.week

    fa_payload = fetch_free_agents(
        target.league_id,
        target.season,
        creds,
        scoring_period=week,
        limit=args.free_agent_limit,
    )
    free_agents, truncated = parse_free_agents(fa_payload, limit=args.free_agent_limit)
    projections = weekly_projections_by_espn_id(fa_payload, week, config.ruleset)

    # My own roster's weekly projections come from the same call: `filterStatus` excludes them,
    # so a second request asks for the players I already have. One source, one scoring pass --
    # comparing my starter against a free agent priced any other way is not a comparison.
    # Its OWN limit, sized to the whole league. Sharing --free-agent-limit meant lowering that
    # flag to speed a run up silently dropped my own starters from the projections, left holes
    # in the baseline lineup, and inflated every candidate's gain -- with no warning, because
    # the truncation check only runs on the free-agent side.
    rostered_limit = ctx.rostered_limit()
    # Memoised on the context: start/sit asks for the identical payload, and in a combined
    # report the second caller reuses this response rather than repeating the request.
    mine_payload = ctx.onteam_payload()
    _, mine_truncated = parse_free_agents(mine_payload, limit=rostered_limit)
    if mine_truncated:
        print(f"  ! your own roster may be incompletely priced: {mine_truncated}")
    projections.update(weekly_projections_by_espn_id(mine_payload, week, config.ruleset))

    # `free_agents` passed so the ADD side is injury-discounted too, not just my roster.
    remaining = remaining_points_by_espn_id(run_state, id_map, free_agents)
    roster = ctx.roster()

    season, open_spots = season_upgrades(
        roster,
        free_agents,
        projections,
        remaining,
        config,
        min_margin=args.min_season_margin,
    )
    weekly = weekly_upgrades(
        roster,
        free_agents,
        projections,
        remaining,
        config,
        min_margin=args.min_week_margin,
    )[: args.top]

    _print_header(run_state.team_name, week, len(free_agents), truncated)
    for run_note in run_state.notes:
        print(f"  ! {run_note}")

    if open_spots:
        # Said once, up front, because every "roster spot open" row below is claiming THESE
        # spots -- not one each. Acting on two of them when one is free overfills the roster.
        print(f"\n  {open_spots} active roster spot(s) open — an add there costs nothing.")

    shortlist = season[: args.top]

    # My OWN injured players, not just the adds. "My starter went down, who do I pick up" is
    # the case this tool was built for, and his write-up is the thing that says how long he is
    # gone — fetching it only for the free agents answered half the question.
    mine_hurt = [
        player_id(player)
        for _, player in roster.iterrows()
        if not parse_injury_status(player.get("injury_status"))[0].is_healthy
    ]
    hurt = sorted(
        {int(c.player_id) for c in [*shortlist, *weekly] if not c.injury_status.is_healthy}
    )
    notes = fetch_injury_notes([*mine_hurt, *hurt]) if (mine_hurt or hurt) else {}

    # Gated on there being something to SHOW, not on there being someone hurt: ESPN has no
    # write-up for plenty of designated players, and the header printed above an empty section.
    shown = [
        (player, notes[player_id(player)])
        for _, player in roster.iterrows()
        if player_id(player) in notes
    ]
    if shown:
        print("\n  on your roster:")
        for player, note in shown:
            status, _ = parse_injury_status(player.get("injury_status"))
            print(f"    {display_str(player.get('player'))} — {status.value}")
            _print_note(note, indent="      ")

    print("\n  REST OF SEASON — better than your weakest player at his position, starter or bench")
    if not shortlist:
        print("\n  Nobody on the wire out-projects anyone on your roster for the rest of the year.")
    for candidate in shortlist:
        _print_candidate(candidate, notes.get(candidate.player_id))

    print("\n  THIS WEEK — projected to outscore your weakest player at his position this week")
    if not weekly:
        print("\n  Nobody on the wire out-projects anyone on your roster this week.")
    for candidate in weekly:
        _print_weekly(candidate, notes.get(candidate.player_id))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    # The five league flags all default to the profile; see `resolve_league_target`.
    add_league_arguments(
        parser,
        team_id_help="your team; defaults to the league profile's team_id, and with "
        "neither set the run lists the league's teams instead",
    )
    parser.add_argument("--week", type=int, help="defaults to the first unplayed week")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--credentials", type=Path, default=Path("configs/espn_credentials.json"))
    parser.add_argument("--free-agent-limit", type=int, default=DEFAULT_FREE_AGENT_LIMIT)
    parser.add_argument(
        "--min-season-margin",
        type=float,
        default=5.0,
        help="rest-of-season points a free agent must beat your weakest player by",
    )
    parser.add_argument(
        "--min-week-margin",
        type=float,
        default=0.5,
        help="points this week a free agent must beat your weakest playing player by",
    )
    parser.add_argument("--top", type=int, default=5, help="how many to show on each list")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Resolve, check, then assemble. Exit code 2 for a usage problem, as before."""
    try:
        target = resolve_league_target(args)
        target.require_league_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if target.source is not None:
        print(target.describe())

    if target.team_id is None:
        # **Before the assembly, deliberately.** Loading a pool, an id_map and a decade of
        # weekly_stats in order to print a list of team ids would be absurd, and this is the
        # one thing the tool must never guess at.
        creds = EspnCredentials.resolve(args.credentials)
        payload = fetch_league_payload(target.league_id, target.season, creds)
        print("--team-id is required. Teams in this league:")
        for _, team in parse_teams(payload).iterrows():
            print(f"  {int(team['team_id']):>3}  {team['team_name']}")
        return 2

    ctx = assemble_context(target, args)
    for note in ctx.notes:
        # The config-vs-ESPN drift warning. `roster_slots` from the file sizes the
        # rostered-player request, so a drift silently thins the projections behind
        # every number below -- computing this and discarding it is worse than not
        # computing it, because it looks like the check is running.
        print(f"  ! {note}", file=sys.stderr)
    return report(ctx, args)


def main(argv: list[str] | None = None) -> int:
    # The em dashes in the output are not encodable in a Windows console's cp1252; see
    # `projections.console`.
    force_utf8_stdio()
    args = _parse_args(argv)
    try:
        return run(args)
    except (ProjectionInputError, EspnLeagueError, OSError) as exc:
        print(f"Cannot recommend: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
