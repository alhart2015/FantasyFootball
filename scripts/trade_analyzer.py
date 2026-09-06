"""Where your roster is strong, who needs what, and which trades are worth sending.

Three sections, answering the three questions separately, because the first two are useful on
their own before any specific proposal exists:

  A. YOUR SHAPE   surplus per player (what he costs you to give up, after the lineup
                  re-optimises) and need per position, against the league distribution.
  B. MARKET MAP   need by team and position for all 16 teams -- the "who should I call" answer.
  C. PROPOSALS    size-neutral trades that improve THEIR lineup on ESPN's projections and
                  YOURS on the consensus, ranked by change in expected wins.

The asymmetry in C is the point. They evaluate on the public numbers; we evaluate on ours. That
is a stated, checkable proxy for acceptability -- **the tool says a trade looks fair on ESPN's
numbers, never that anyone will accept it.**

Usage:
    python scripts/trade_analyzer.py                       # the one configured league
    python scripts/trade_analyzer.py --league-dir data/leagues/critts_2025_2026
    python scripts/trade_analyzer.py --league-id 856974 --season 2026 --team-id 17 \\
        --pool data/vorp_2026/critts_half16_snake.parquet

With no league arguments, all four are read from the single `board_profile.json` under
`data/leagues/` — the same file the draft board loads. Anything typed on the command line
wins over the file, and the run announces which profile it used, because a tool that
silently defaults to the wrong league still prints a complete, confident report.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_profile import (
    add_league_arguments,
    resolve_league_target,
)
from projections.draft.assistant.performance_variance import SEASON_GAMES, VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.league_calendar import LeagueCalendar
from projections.ingest.espn_league import (
    DEFAULT_CREDS_PATH,
    EspnCredentials,
    EspnLeagueError,
    build_league_config,
    espn_to_gsis,
    fetch_league_payload,
    parse_rosters,
    parse_schedule,
    parse_teams,
    pool_name_index,
)
from projections.midseason.roster_shape import TRADEABLE, team_shapes
from projections.midseason.standings import ProjectionInputError, first_unplayed_week
from projections.midseason.trades import WINS_NOISE_FLOOR, generate_all, simulate_trades
from projections.midseason.valuation import PlayerValue, build_values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The five league flags all default to the profile; see `resolve_league_target`.
    add_league_arguments(p)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--credentials", type=Path, default=DEFAULT_CREDS_PATH)
    p.add_argument("--max-players", type=int, default=2, help="players per side (size-neutral)")
    p.add_argument("--min-lineup-gain", type=float, default=1.0)
    p.add_argument(
        "--espn-tolerance",
        type=float,
        default=5.0,
        help="ESPN points a trade may tip my way before it reads as a lowball",
    )
    p.add_argument("--top", type=int, default=8, help="proposals to simulate and show")
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="skip stage 2; rank by lineup gain alone")
    return p.parse_args(argv)


def _latest_external(data_root: Path, season: int) -> pd.DataFrame:
    """The most recent `external_projections` snapshot for this season -- the market's view."""
    parts = sorted(
        glob.glob(
            str(
                data_root
                / "raw"
                / "external_projections"
                / f"season={season}"
                / "asof=*"
                / "*.parquet"
            )
        )
    )
    if not parts:
        raise ProjectionInputError(
            f"No external_projections for {season} under {data_root}. ESPN's own projection is "
            "the market half of every comparison here, so the run cannot proceed without it. "
            f"Refresh with: python -m projections.ingest.external_projections --season {season}"
        )
    return pd.read_parquet(parts[-1])


def _fmt(value: float, width: int = 7, places: int = 1) -> str:
    return f"{value:>{width}.{places}f}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target = resolve_league_target(args, require_team_id=True)
        # Inside the try on purpose: `require_team_id` above is what normally raises, with a
        # message naming the profile, but this is the same failure and must not be a traceback.
        my_team_id = target.require_team()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if target.source is not None:
        print(target.describe())

    try:
        creds = EspnCredentials.resolve(args.credentials)
        payload = fetch_league_payload(target.league_id, target.season, creds)
    except EspnLeagueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    config = build_league_config(payload, name=f"league {target.league_id}")
    teams = parse_teams(payload)
    # A team id read off a file deserves the check `projected_standings` gives a typed one:
    # without it, `shapes[team_id]` raises a bare KeyError partway down the report.
    if my_team_id not in set(teams["team_id"]):
        print(
            f"team id {my_team_id} is not a team in this league. Valid ids: "
            f"{sorted(teams['team_id'])}.",
            file=sys.stderr,
        )
        return 1
    names = dict(zip(teams["team_id"], teams["team_name"].astype(str), strict=True))
    rosters = parse_rosters(payload)
    if rosters.empty:
        print("No rosters yet — the draft has not happened.", file=sys.stderr)
        return 1

    pool = attach_is_rookie(
        pd.read_parquet(target.pool), season=target.season, data_root=args.data_root
    )
    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")
    try:
        external = _latest_external(args.data_root, target.season)
    except ProjectionInputError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    settings = payload.get("settings", {}) or {}
    calendar = LeagueCalendar.from_espn_settings(settings.get("scheduleSettings", {}) or {})
    week = first_unplayed_week(parse_schedule(dict(payload), teams), calendar)
    games_remaining = max(SEASON_GAMES - (week - 1), 0)
    gsis = espn_to_gsis(rosters, id_map, name_index=pool_name_index(pool))
    values = build_values(
        rosters, gsis, pool, external, config.ruleset, games_remaining=games_remaining
    )

    by_team: dict[int, list[PlayerValue]] = {int(t): [] for t in teams["team_id"]}
    unvalued = 0
    for row, g in zip(rosters.itertuples(), gsis, strict=True):
        key = None if pd.isna(g) else str(g)
        if key is not None and key in values:
            by_team[int(row.team_id)].append(values[key])
        else:
            unvalued += 1

    # Availability is what makes `surplus` see byes and injuries. Without it every bench
    # player prices at exactly 0.0 -- see `roster_shape.surplus`.
    availability = load_store_availability(pool, season=target.season, data_root=args.data_root)
    shapes = team_shapes(
        by_team,
        names,
        config.roster_slots,
        availability,
        weeks=list(range(week, calendar.reg_weeks + 1)),
    )
    mine = shapes[my_team_id]

    print(f"\n{names.get(my_team_id, my_team_id)} — week {week}, {games_remaining} games left")
    print(
        f"  {unvalued} rostered players are not valued (K/DST, or only one source has an "
        "opinion) and cannot be traded by this tool."
    )
    print(
        "  'edge' is ours minus ESPN's, both scored under this league's rules. Our consensus\n"
        "  CONTAINS ESPN, so edge is where Sleeper pulls the blend away from it — a real signal\n"
        "  about market disagreement, not an independent second opinion."
    )

    # ---- A. my shape -------------------------------------------------------------------
    print("\n=== A. YOUR ROSTER — what each player costs you to trade ===")
    print(f"  {'player':<22}{'pos':<5}{'ours':>8}{'ESPN':>8}{'edge':>8}{'surplus':>9}")
    for p in sorted(mine.players, key=lambda p: -mine.surplus.get(p.gsis_id, 0.0)):
        tag = "" if p.injury_status.is_healthy else f"  [{p.injury_raw}]"
        print(
            f"  {p.full_name[:21]:<22}{p.position:<5}{_fmt(p.ours, 8)}{_fmt(p.market, 8)}"
            f"{_fmt(p.edge, 8)}{_fmt(mine.surplus.get(p.gsis_id, 0.0), 9)}{tag}"
        )
    print("\n  low surplus = a chip you barely miss. High = do not trade him.")

    print(f"\n  your need, vs the league:  {'':<4}" + "".join(f"{pos:>9}" for pos in TRADEABLE))
    print(f"  {'you':<28}" + "".join(_fmt(mine.need[pos], 9) for pos in TRADEABLE))
    league_med = {
        pos: sorted(s.need[pos] for s in shapes.values())[len(shapes) // 2] for pos in TRADEABLE
    }
    print(f"  {'league median':<28}" + "".join(_fmt(league_med[pos], 9) for pos in TRADEABLE))

    # ---- B. market map -----------------------------------------------------------------
    print("\n=== B. MARKET MAP — need by team (lineup points a median starter would add) ===")
    print(f"  {'team':<30}" + "".join(f"{pos:>9}" for pos in TRADEABLE))
    for shape in sorted(shapes.values(), key=lambda s: -max(s.need.values())):
        mark = "  <-- you" if shape.team_id == my_team_id else ""
        print(
            f"  {shape.team_name[:29]:<30}"
            + "".join(_fmt(shape.need[pos], 9) for pos in TRADEABLE)
            + mark
        )

    # ---- C. proposals ------------------------------------------------------------------
    proposals = generate_all(
        shapes,
        my_team_id,
        config.roster_slots,
        max_players=args.max_players,
        min_lineup_gain=args.min_lineup_gain,
        espn_tolerance=args.espn_tolerance,
        top=args.top,
    )
    print("\n=== C. PROPOSALS — fair on ESPN's numbers, better on ours ===")
    if not proposals:
        print(
            "  Nothing clears both filters. In a 16-team league that is a normal answer:\n"
            "  every roster is thin, so a trade that helps you usually hurts them on their\n"
            "  own numbers. Loosen --espn-tolerance or --min-lineup-gain to see near misses."
        )
        return 0

    if not args.fast:
        proposals = simulate_trades(
            payload,
            pool,
            id_map,
            availability,
            VarianceParams.load(),
            proposals,
            season=target.season,
            my_team_id=my_team_id,
            week=week,
            n_sims=args.n_sims,
            seed=args.seed,
        )

    for i, t in enumerate(proposals, start=1):
        out = ", ".join(f"{p.full_name} ({p.position})" for p in t.send)
        inn = ", ".join(f"{p.full_name} ({p.position})" for p in t.receive)
        print(f"\n  {i}. with {t.partner_name}")
        print(f"     send    {out}")
        print(f"     receive {inn}")
        line = (
            f"     lineup {t.my_lineup_gain:+.1f} (fit {t.fit_gain:+.1f}, edge {t.edge_gain:+.1f})"
            f"  |  ESPN balance {t.espn_balance:+.1f}"
            f"  |  their lineup on ESPN {t.their_lineup_gain_market:+.1f}"
        )
        print(line)
        if t.delta_wins is not None:
            verdict = "" if t.above_noise else "   (inside simulation noise — not a difference)"
            print(
                f"     wins {t.delta_wins:+.2f}  playoff {100 * (t.delta_playoff or 0):+.1f}pp"
                f"  title {100 * (t.delta_title or 0):+.1f}pp{verdict}"
            )
        if t.is_lowball:
            print("     ! takes more ESPN value than it gives — reads as a lowball")
        hurt = [p for p in (*t.send, *t.receive) if not p.injury_status.is_healthy]
        for p in hurt:
            print(f"     ! {p.full_name} is {p.injury_raw} — both valuations are haircut for it")

    if not args.fast:
        print(
            f"\n  Anything under {WINS_NOISE_FLOOR:.2f} wins is inside the paired simulation "
            "noise (sd ~0.062 at 2,000 sims) and is not a difference."
        )
    print("\n  'Fair on ESPN's numbers' is a proxy for acceptability, not a prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
