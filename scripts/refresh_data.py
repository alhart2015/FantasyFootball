"""One-shot refresh of every data source and every derived table. Run it with no arguments.

    python scripts/refresh_data.py

This exists because refreshing by hand means remembering eight ingest entrypoints, the order they
depend on each other in, and -- the part that actually burns time -- which of their failures mean
"something is broken" versus "the NFL has not played that game yet". Those two look identical from
a traceback: a 404 on an nflverse release, a `ValueError: Season must be between 2012 and 2025`,
and a genuine schema regression all abort a naive loop at its first source and leave the remaining
seven unrun.

So this script does three things a bare loop does not:

**It pre-checks the season.** The per-game sources only have rows once the season is underway
(kickoff is the Thursday after Labor Day, matching `nflreadpy.get_current_season`). Before that
date they are reported SKIPPED *with the kickoff date*, rather than attempted and caught. The
market-facing sources -- schedules, draft picks, the id map, projections -- have no such gate and
always run, which is the whole point in the preseason, when the projections are the only thing
that moves.

**It isolates every source.** One failure never costs the others their run. Sources run in
dependency order (`id_map` before `snap_counts`, `schedules` before `depth_charts`), so on a clean
run a prerequisite is always in place. If a prerequisite itself fails, its dependent fails too and
both appear in the summary, adjacent and in order -- which is the readable outcome, and better than
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
    --skip-derived           refresh raw sources only, leave the VORP tables alone
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
from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.draft_picks import refresh_draft_picks
from projections.ingest.external_projections import refresh_external_projections
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import STAT_TYPES as NGS_STAT_TYPES
from projections.ingest.ngs import NgsStatType, refresh_ngs
from projections.ingest.pbp import refresh_pbp
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats

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


def season_start_date(season: int) -> date:
    """Kickoff Thursday -- the Thursday after Labor Day -- mirroring `nflreadpy`'s own rollover.

    Used to tell the reader *when* skipped game data becomes available, so the summary line is
    actionable ("comes back 2026-09-10") instead of just "no data".
    """
    labor_day = next(
        date(season, 9, day) for day in range(1, 8) if date(season, 9, day).weekday() == 0
    )
    return date(season, 9, labor_day.day + 3)


def games_played(season: int, *, today: date | None = None) -> bool:
    """Whether `season` has begun, so its per-game sources could have rows upstream.

    Deliberately not `nflreadpy.get_current_season()`: that takes no date argument, so it cannot
    be pinned to a fixed point in the calendar by a test.
    """
    return (today or date.today()) >= season_start_date(season)


def _count(paths: list[Path]) -> str:
    return f"{len(paths)} partition(s)"


def _ngs_step(data_root: Path, stat_type: NgsStatType, seasons: list[int]) -> Callable[[], str]:
    """Bind `stat_type` per iteration -- a closure over the loop variable would run every NGS
    step against whichever stat type the loop happened to end on."""

    def _run() -> str:
        return _count(refresh_ngs(data_root, stat_type=stat_type, seasons=seasons))

    return _run


def game_stat_steps(data_root: Path, seasons: list[int]) -> list[tuple[str, Callable[[], str]]]:
    """The sources whose rows only exist once games have been played, in dependency order.

    `id_map` and `schedules` are *not* here: they are published year-round and run earlier, which
    is exactly what `snap_counts` (needs the gsis <-> pfr translation) and `depth_charts` (needs
    schedules to derive season/week from the 2025+ snapshot format) depend on.
    """
    steps: list[tuple[str, Callable[[], str]]] = [
        ("weekly_stats", lambda: _count(refresh_weekly_stats(data_root, seasons=seasons))),
        ("depth_charts", lambda: _count(refresh_depth_charts(data_root, seasons=seasons))),
        ("snap_counts", lambda: _count(refresh_snap_counts(data_root, seasons=seasons))),
    ]
    steps.extend(
        (f"ngs_{stat_type}", _ngs_step(data_root, stat_type, seasons))
        for stat_type in NGS_STAT_TYPES
    )
    return steps


def run_step(name: str, thunk: Callable[[], str], *, verbose: bool = False) -> StepResult:
    """Run one source, turning any exception into a classified result rather than aborting."""
    print(f"  -> {name} ...", flush=True)
    try:
        detail = thunk()
    except Exception as exc:
        status = classify_error(exc)
        if verbose and status is Status.FAILED:
            traceback.print_exc()
        first_line = f"{type(exc).__name__}: {exc}".splitlines()[0]
        return StepResult(name, status, first_line[:200])
    return StepResult(name, Status.OK, detail)


def refresh_raw(
    *,
    data_root: Path,
    seasons: list[int],
    with_pbp: bool = False,
    verbose: bool = False,
    today: date | None = None,
) -> list[StepResult]:
    """Every raw ingest source, isolated. One result per source, in run order."""
    results: list[StepResult] = [
        run_step("id_map", lambda: str(build_id_map(data_root)), verbose=verbose),
        run_step(
            "schedules",
            lambda: _count(refresh_schedules(data_root, seasons=seasons)),
            verbose=verbose,
        ),
        run_step(
            "draft_picks",
            lambda: _count(refresh_draft_picks(data_root, seasons=seasons)),
            verbose=verbose,
        ),
    ]

    playable = [s for s in seasons if games_played(s, today=today)]
    if not playable:
        reason = "no games played yet; " + ", ".join(
            f"{s} kicks off {season_start_date(s)}" for s in seasons
        )
        names = [name for name, _ in game_stat_steps(data_root, seasons)]
        if with_pbp:
            names.append("pbp")
        results.extend(StepResult(name, Status.SKIPPED, reason) for name in names)
        return results

    for name, thunk in game_stat_steps(data_root, playable):
        results.append(run_step(name, thunk, verbose=verbose))
    if with_pbp:
        results.append(
            run_step(
                "pbp",
                lambda: _count(refresh_pbp(data_root, seasons=playable)),
                verbose=verbose,
            )
        )
    return results


def refresh_projections(*, data_root: Path, season: int, verbose: bool = False) -> StepResult:
    """The ESPN + Sleeper consensus snapshot -- the only source that moves in the preseason.

    Written under an `asof=<date>` partition, so re-running on the same day overwrites in place
    rather than accumulating snapshots.
    """
    return run_step(
        "external_projections",
        lambda: str(refresh_external_projections(data_root, season=season)),
        verbose=verbose,
    )


def _rebuild_preset_tables(data_root: Path, season: int) -> str:
    import generate_preset_vorp_tables  # sibling script; scripts/ is on sys.path

    generate_preset_vorp_tables.main(["--season", str(season), "--data-root", str(data_root)])
    return "9 preset tables"


def _league_step(profile: LeagueProfile, data_root: Path, season: int) -> Callable[[], str]:
    def _run() -> str:
        import generate_league_vorp_table  # sibling script; scripts/ is on sys.path

        generate_league_vorp_table.main(
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
    if not projections_ok:
        # Name every table that did NOT rebuild, leagues included. A summary showing one generic
        # "presets skipped" line reads as though the league pools were fine.
        reason = "external_projections did not refresh; tables left as they are"
        names = ["vorp_presets", *(f"league:{p.key}" for p in profiles)]
        return [StepResult(name, Status.SKIPPED, reason) for name in names]

    results = [
        run_step("vorp_presets", lambda: _rebuild_preset_tables(data_root, season), verbose=verbose)
    ]
    # A malformed profile is FAILED, never silently skipped: a league whose pool quietly stopped
    # rebuilding would keep serving last week's numbers under a current-looking filename.
    results.extend(
        StepResult(f"league:{err.path.parent.name}", Status.FAILED, err.message) for err in errors
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
    p.add_argument("--skip-derived", action="store_true", help="Raw sources only.")
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

    results = refresh_raw(
        data_root=args.data_root,
        seasons=seasons,
        with_pbp=args.with_pbp,
        verbose=args.verbose,
    )
    projections = refresh_projections(
        data_root=args.data_root, season=target_season, verbose=args.verbose
    )
    results.append(projections)
    if not args.skip_derived:
        results.extend(
            refresh_derived(
                data_root=args.data_root,
                season=target_season,
                projections_ok=projections.status is Status.OK,
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
