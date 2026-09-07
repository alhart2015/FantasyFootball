"""One-shot refresh of every data source and every derived table. Run it with no arguments.

    python scripts/refresh_data.py

**What gets refreshed is not decided here.** `projections.ingest.sources.INGEST_SOURCES` is the one
registry of ingest sources, in dependency order, and this script iterates it. Adding a source means
adding a registry entry; nothing in this file changes. The library's `refresh()` drives the same
registry fail-fast for programmatic callers.

This script exists because a by-hand refresh means remembering which failures mean "something is
broken" versus "the NFL has not played that game yet". Those look identical from a traceback: a 404
on an nflverse release, a `ValueError: Season must be between 2012 and 2025`, and a genuine schema
regression all abort a naive loop at its first source and leave the rest unrun.

So it adds three things to the registry walk:

**It pre-checks the season.** Per-game sources only have rows once the season is underway (kickoff
is the Thursday after Labor Day, matching `nflreadpy.get_current_season`). Before that date they are
reported SKIPPED *with the kickoff date*, rather than attempted and caught. The market-facing
sources -- schedules, draft picks, the id map, projections -- have no such gate and always run,
which is the whole point in the preseason, when the projections are the only thing that moves.

**It isolates every source.** One failure never costs the others their run. Registry order puts
`id_map` before `snap_counts` and `schedules` before `depth_charts`, so on a clean run a
prerequisite is always in place. If a prerequisite itself fails, its dependent fails too and both
appear in the summary, adjacent and in order -- which is the readable outcome, and better than
laundering the dependent into a quiet "skipped".

**It separates "not published" from "broken".** SKIPPED is an expected, quiet outcome and exits 0.
FAILED is a real defect: it is listed with its exception and exits 1. A run that reported success
while five sources 404'd would be worse than no script at all.

Derived tables (the 9 preset VORP tables, plus one per configured league) rebuild only when the
projection snapshot they are built from actually refreshed. Rebuilding them from an unchanged
snapshot writes new file mtimes and identical numbers, which reads as "the pool is current" when
nothing was refreshed at all.

Flags, none of them required:

    --season 2026            one season (default: the configured league's, else the calendar's)
    --seasons 2021-2025      an inclusive range, or a comma list
    --with-pbp               also pull play-by-play (hundreds of MB; off by default)
    --skip-derived           ingest only, leave the VORP tables alone
    --verbose                print full tracebacks for FAILED sources
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import nflreadpy

from projections.draft.assistant.league_profile import (
    DEFAULT_PROFILE_ROOT,
    LeagueProfile,
    discover_profiles,
)
from projections.ingest.sources import (
    IngestSource,
    games_played,
    season_start_date,
    selected_sources,
)

#: The registry entry the derived VORP tables are built from. Named once so the gate in `main`
#: cannot drift from the source list.
PROJECTIONS_SOURCE = "external_projections"

# Substrings marking an upstream "this does not exist yet" rather than a defect on our side.
# `nflreadpy` raises a bare `ValueError` for a season past its own rollover and wraps the GitHub
# release 404 in a `ConnectionError`; neither carries a type we can distinguish on, so the message
# is the only available signal. Kept deliberately narrow -- see `classify_error`.
_NOT_PUBLISHED_MARKERS = (
    "404 client error",
    "season must be between",
    "not found for url",
)


class Status(StrEnum):
    """Outcome of one source. Only FAILED is a defect; SKIPPED is an expected quiet no-op."""

    OK = "OK"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StepResult:
    name: str
    status: Status
    detail: str


def classify_error(exc: BaseException) -> Status:
    """SKIPPED when upstream simply has not published the data yet, FAILED otherwise.

    Matching on message text is unpleasant but forced: the "not yet published" cases arrive as
    plain `ValueError` / `ConnectionError` from `nflreadpy`, the same types a real bug raises. The
    marker list is kept narrow and an unrecognised message is FAILED, because a defect misreported
    as "skipped" is precisely the failure this script exists to prevent.
    """
    text = f"{exc}".lower()
    return Status.SKIPPED if any(m in text for m in _NOT_PUBLISHED_MARKERS) else Status.FAILED


def run_step(name: str, thunk: Callable[[], str], *, verbose: bool = False) -> StepResult:
    """Run one source, turning any failure into a classified result rather than aborting.

    `SystemExit` is caught alongside `Exception` and is not a hypothetical: the derived steps call
    sibling scripts' `main()`, and `generate_preset_vorp_tables` raises `SystemExit` when the
    id_map is missing. `SystemExit` derives from `BaseException`, so a bare `except Exception`
    lets it through -- which would terminate the process after the full raw pull and discard every
    result collected so far, printing no summary at all. `KeyboardInterrupt` is deliberately still
    allowed to propagate: Ctrl-C must stop the run, not be recorded as one source failing.
    """
    print(f"  -> {name} ...", flush=True)
    try:
        detail = thunk()
    except (Exception, SystemExit) as exc:
        status = classify_error(exc)
        if verbose and status is Status.FAILED:
            traceback.print_exc()
        first_line = f"{type(exc).__name__}: {exc}".splitlines()[0]
        return StepResult(name, status, first_line[:200])
    return StepResult(name, Status.OK, detail)


def _source_step(source: IngestSource, data_root: Path, seasons: list[int]) -> Callable[[], str]:
    """Bind one registry entry to its run. Eager binding, so a loop variable cannot leak."""

    def _run() -> str:
        written = source.run(data_root, seasons)
        # A single write is worth naming -- `id_map` and each `external_projections` snapshot are
        # the two a reader actually goes and looks at, and "1 partition(s)" tells them nothing.
        return str(written[0]) if len(written) == 1 else f"{len(written)} partition(s)"

    return _run


def refresh_sources(
    *,
    data_root: Path,
    seasons: list[int],
    with_pbp: bool = False,
    verbose: bool = False,
    today: date | None = None,
) -> list[StepResult]:
    """Every ingest source in `projections.ingest.sources.INGEST_SOURCES`, isolated.

    The registry is the single list of what we ingest and in what order; this function adds only
    the two things a library fail-fast call cannot: per-source isolation, and reporting the
    seasons a per-game source could not be asked for. Adding a source means adding a registry
    entry -- nothing here changes.
    """
    playable = [s for s in seasons if games_played(s, today=today)]
    not_started = [s for s in seasons if s not in playable]
    kickoffs = ", ".join(f"{s} kicks off {season_start_date(s)}" for s in not_started)

    results: list[StepResult] = []
    if not_started and playable:
        # A mixed range (say `--seasons 2024-2026` today) must not let the unplayable season
        # disappear. Running the rest and reporting them OK is a clean zero-failure run that says
        # nothing about a season the user explicitly asked for.
        results.append(
            StepResult(
                "game_stats(not started)",
                Status.SKIPPED,
                f"per-game sources ran for {playable} only; {kickoffs}",
            )
        )

    for source in selected_sources(with_pbp=with_pbp):
        applicable = playable if source.needs_games_played else seasons
        if not applicable:
            results.append(
                StepResult(source.name, Status.SKIPPED, f"no games played yet; {kickoffs}")
            )
            continue
        results.append(
            run_step(source.name, _source_step(source, data_root, applicable), verbose=verbose)
        )
    return results


def _check_rc(script: str, rc: int) -> None:
    """A sibling `main()` returning non-zero must fail the step, not be reported OK.

    Both siblings currently only return 0 or raise, so this is latent -- but the whole premise of
    this script is not laundering a defect into a success, and the first non-zero return path
    either of them grows would otherwise print `OK  vorp_presets` and exit 0.
    """
    if rc != 0:
        raise RuntimeError(f"{script}.main() returned {rc}")


def _rebuild_preset_tables(data_root: Path, season: int) -> str:
    import generate_preset_vorp_tables  # sibling script; scripts/ is on sys.path

    rc = generate_preset_vorp_tables.main(["--season", str(season), "--data-root", str(data_root)])
    _check_rc("generate_preset_vorp_tables", rc)
    return "9 preset tables"


def _league_step(profile: LeagueProfile, data_root: Path, season: int) -> Callable[[], str]:
    def _run() -> str:
        import generate_league_vorp_table  # sibling script; scripts/ is on sys.path

        rc = generate_league_vorp_table.main(
            [
                "--league-config",
                str(profile.league_config_path),
                "--season",
                str(season),
                "--out",
                str(profile.vorp_path),
                "--data-root",
                str(data_root),
            ]
        )
        _check_rc("generate_league_vorp_table", rc)
        return str(profile.vorp_path)

    return _run


def refresh_derived(
    *,
    data_root: Path,
    season: int,
    projections_ok: bool,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    verbose: bool = False,
) -> list[StepResult]:
    """Rebuild the VORP tables the draft board and the in-season CLIs read.

    Gated on the projection pull for a reason: rebuilding from an unchanged snapshot writes fresh
    mtimes over identical numbers, which reads as "the pool is current" when nothing moved.
    """
    profiles, errors = discover_profiles(profile_root)
    # A malformed profile is FAILED before anything else is decided, and in particular before the
    # `projections_ok` gate below. A broken profile is broken whether or not the projections
    # refreshed, and a league whose pool quietly stopped rebuilding would keep serving last week's
    # numbers under a current-looking filename.
    results: list[StepResult] = [
        StepResult(f"league:{err.path.parent.name}", Status.FAILED, err.message) for err in errors
    ]

    if not projections_ok:
        # Name every table that did NOT rebuild, leagues included. A summary showing one generic
        # "presets skipped" line reads as though the league pools were fine.
        reason = "external_projections did not refresh; tables left as they are"
        names = ["vorp_presets", *(f"league:{p.key}" for p in profiles)]
        results.extend(StepResult(name, Status.SKIPPED, reason) for name in names)
        return results

    results.append(
        run_step("vorp_presets", lambda: _rebuild_preset_tables(data_root, season), verbose=verbose)
    )
    for profile in profiles:
        if profile.season != season:
            results.append(
                StepResult(
                    f"league:{profile.key}",
                    Status.SKIPPED,
                    f"profile season {profile.season} != refreshed season {season}",
                )
            )
            continue
        results.append(
            run_step(
                f"league:{profile.key}", _league_step(profile, data_root, season), verbose=verbose
            )
        )
    return results


def default_season(profile_root: Path = DEFAULT_PROFILE_ROOT, *, today: date | None = None) -> int:
    """The season to refresh when none was given: the configured league's, else the calendar's.

    Falls back rather than raising, so a checkout with no `data/leagues/` still runs.
    `nflreadpy.get_current_season()` is deliberately not used: it rolls over only at kickoff, and
    the season worth refreshing in August is the one about to start, not the one just finished.
    """
    profiles, _ = discover_profiles(profile_root)
    seasons = {p.season for p in profiles}
    if len(seasons) == 1:
        return seasons.pop()
    now = today or date.today()
    # From March the upcoming season is the interesting one; January and February still belong
    # to the season that just ended.
    return now.year if now.month >= 3 else now.year - 1


def parse_seasons(raw: str) -> list[int]:
    """`2026`, `2021-2025`, or `2021,2023,2026`. Sorted and de-duplicated."""
    out: list[int] = []
    for chunk in (c.strip() for c in raw.split(",")):
        if not chunk:
            continue
        if "-" in chunk:
            try:
                lo, hi = (int(part) for part in chunk.split("-", 1))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"bad season range: {chunk!r}") from exc
            if hi < lo:
                raise argparse.ArgumentTypeError(f"empty season range: {chunk!r}")
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(chunk))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"not a season: {chunk!r}") from exc
    if not out:
        raise argparse.ArgumentTypeError("no seasons given")
    return sorted(set(out))


def render_summary(results: list[StepResult]) -> str:
    """The point of the whole run: one block readable without scrolling, by an agent or a human."""
    width = max((len(r.name) for r in results), default=0)
    counts = {s: sum(1 for r in results if r.status is s) for s in Status}
    lines = ["", "=" * 78, "REFRESH SUMMARY", "=" * 78]
    lines.extend(f"  {r.status.value:<7}  {r.name:<{width}}  {r.detail}" for r in results)
    lines.append("-" * 78)
    lines.append(
        f"  {counts[Status.OK]} refreshed | {counts[Status.SKIPPED]} skipped "
        f"(nothing upstream to fetch -- expected) | {counts[Status.FAILED]} failed"
    )
    if counts[Status.FAILED]:
        lines.append("")
        lines.append("  FAILED means a real defect. Fix it; do not just re-run:")
        lines.extend(f"    - {r.name}: {r.detail}" for r in results if r.status is Status.FAILED)
        lines.append("  Re-run with --verbose for full tracebacks.")
    lines.append("=" * 78)
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refresh every data source and derived table. No arguments needed."
    )
    p.add_argument("--season", type=int, help="Single season (default: the league profile's).")
    p.add_argument(
        "--seasons", type=parse_seasons, help="Range or list, e.g. 2021-2025 or 2021,2026."
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    p.add_argument("--with-pbp", action="store_true", help="Also pull play-by-play (slow, large).")
    p.add_argument("--skip-derived", action="store_true", help="Ingest only.")
    p.add_argument("--verbose", action="store_true", help="Print tracebacks for failures.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.season is not None and args.seasons is not None:
        raise SystemExit("Pass --season or --seasons, not both.")
    if args.seasons is not None:
        seasons: list[int] = args.seasons
    elif args.season is not None:
        seasons = [args.season]
    else:
        seasons = [default_season(args.profile_root)]
    # Derived tables are per-season artifacts; on a multi-season pull the newest is the one the
    # board and the in-season CLIs actually read.
    target_season = max(seasons)

    print(f"Refreshing {seasons} under {args.data_root}; derived tables for {target_season}.")
    print(f"nflreadpy reports the current season as {nflreadpy.get_current_season()}.\n")

    results = refresh_sources(
        data_root=args.data_root,
        seasons=seasons,
        with_pbp=args.with_pbp,
        verbose=args.verbose,
    )
    # The derived tables are built from the projection snapshot, so they rebuild only if that
    # source actually refreshed. Read off the registry result by name rather than tracking it
    # separately, so the gate cannot drift from the source list.
    projections_ok = any(r.name == PROJECTIONS_SOURCE and r.status is Status.OK for r in results)
    if not args.skip_derived:
        results.extend(
            refresh_derived(
                data_root=args.data_root,
                season=target_season,
                projections_ok=projections_ok,
                profile_root=args.profile_root,
                verbose=args.verbose,
            )
        )

    print(render_summary(results))
    return 1 if any(r.status is Status.FAILED for r in results) else 0


if __name__ == "__main__":
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    raise SystemExit(main())
