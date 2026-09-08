"""It is Sunday morning. What do I do?

    python scripts/weekly_report.py                      # the one configured league
    python scripts/weekly_report.py --skip trades        # trades is most of the runtime
    python scripts/weekly_report.py --only start-sit
    python scripts/weekly_report.py --fast               # skip the simulations

Four sections, **one league fetched once**. Every tool used to pull the same payload, derive
the current week its own way, and rescan a decade of `weekly_stats` for itself; a combined
report built that way would be four documents that happen to be printed together, and nothing
would stop them disagreeing. `InSeasonContext` is the fix, and #175 is what it looks like when
they drift — the standings simulator was blind to injuries while the other tools were not, so
the same league had two sets of playoff odds.

**Ordered by deadline, not by importance.** The lineup locks at kickoff, waivers clear
Wednesday, and standings and trades keep. The thing you have least time to act on goes first.

**The sections do not reconcile their numbers, and must not.** The waiver tool scores ESPN's
weekly line; start/sit blends ESPN with Sleeper and applies its own injury multiplier. A second
opinion is the entire reason start/sit exists, so where two sections price the same player the
report says which basis each used rather than inventing an average nobody computes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import projected_standings
import start_sit
import trade_analyzer
import waiver_recommender

from projections.draft.assistant.league_profile import (
    add_league_arguments,
    resolve_league_target,
)
from projections.ingest.espn_league import EspnLeagueError
from projections.midseason.context import InSeasonContext, assemble_context
from projections.midseason.standings import ProjectionInputError

#: (name, heading, module) per section, in DEADLINE order — see the module docstring. The
#: module carries both the `report` to call and the `_parse_args` that owns that section's
#: defaults; keeping them in one tuple means a section cannot be half-registered.
SECTIONS: tuple[tuple[str, str, Any], ...] = (
    ("start-sit", "START / SIT — your lineup locks at kickoff", start_sit),
    ("waivers", "WAIVERS — claims clear Wednesday", waiver_recommender),
    ("standings", "STANDINGS — where this is heading", projected_standings),
    ("trades", "TRADES — no deadline, slowest to act on", trade_analyzer),
)
_NAMES = tuple(name for name, _, _ in SECTIONS)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The five league flags all default to the profile; see `resolve_league_target`.
    add_league_arguments(p, team_id_help="Report on this team.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--credentials",
        type=Path,
        default=Path("configs/espn_credentials.json"),
        help="ESPN cookie file (gitignored).",
    )
    p.add_argument("--week", type=int, default=None, help="Default: the next unplayed week.")
    p.add_argument("--only", default=None, help=f"Comma-separated: {', '.join(_NAMES)}.")
    p.add_argument("--skip", default=None, help="Comma-separated sections to leave out.")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip the OPTIONAL simulations (start/sit P(right), the waiver paired swaps, "
        "trade stage 2). The standings Monte-Carlo still runs -- it is not optional, it IS "
        "that section.",
    )
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args(argv)


#: Flags this script owns. A section reading one of these gets the report's value, because the
#: reader typed it here; everything else falls back to that section's own default.
_REPORT_OWNED = frozenset({"week", "fast", "seed", "data_root", "credentials", "only", "skip"})


def section_args(module: Any, args: argparse.Namespace) -> argparse.Namespace:
    """That section's own defaults, overlaid with what the reader actually typed here.

    **Not one merged namespace.** The first cut built a single Namespace by filling absent keys
    from each tool in turn, which resolved collisions by module import order and got two of
    them wrong: `--top` is 5 for waivers and 8 for trades, so trades silently ran at 5; and
    `--n-sims` is 20,000 for start/sit's P(right) and 2,000 elsewhere, so P(right) was computed
    from a tenth of the draws the standalone tool uses. Both contradicted this report's claim
    that a section behaves here exactly as it does alone.

    So each section is handed ITS parser's defaults. A flag this script owns (`--week`,
    `--fast`, `--seed`, …) overrides, because the reader typed it; a flag it merely happens to
    share a name with does not.
    """
    section: argparse.Namespace = module._parse_args([])
    for key in _REPORT_OWNED:
        if hasattr(args, key) and hasattr(section, key):
            setattr(section, key, getattr(args, key))
    # The five league flags are consumed off `args` by `resolve_league_target` before any
    # section runs, so they are neither present nor needed here.
    return section


def _selected(args: argparse.Namespace) -> list[str] | None:
    """Which sections to run, or None when a name was not recognised."""
    chosen = list(_NAMES)
    for flag, value in (("--only", args.only), ("--skip", args.skip)):
        if value is None:
            continue
        asked = [name.strip() for name in value.split(",") if name.strip()]
        unknown = [name for name in asked if name not in _NAMES]
        if unknown:
            # An unrecognised name must not silently run everything: "--skip trade" (singular)
            # would quietly produce the 23-second report the user was trying to avoid.
            print(
                f"{flag}: unknown section(s) {', '.join(unknown)}. "
                f"Valid names: {', '.join(_NAMES)}.",
                file=sys.stderr,
            )
            return None
        chosen = (
            [n for n in chosen if n in asked]
            if flag == "--only"
            else [n for n in chosen if n not in asked]
        )
    return chosen


def _run_section(
    heading: str,
    report: Callable[[InSeasonContext, argparse.Namespace], int],
    ctx: InSeasonContext,
    args: argparse.Namespace,
) -> int:
    """Run one section. **A failure here must not take the report down.**

    The sections are genuinely independent — a stale `external_projections` breaks trades and
    says nothing about your lineup — and a report that dies at section three has told you
    nothing at all. The failure prints under its own heading so it is impossible to mistake a
    missing section for an empty one.
    """
    print(f"\n\n{'=' * 78}\n{heading}\n{'=' * 78}")
    try:
        return report(ctx, args)
    except (ProjectionInputError, EspnLeagueError, OSError, ValueError) as exc:
        print(f"  ! this section could not run: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sections = _selected(args)
    if sections is None:
        return 1
    if not sections:
        print("Nothing selected.", file=sys.stderr)
        return 1

    try:
        # Resolved once and handed to `assemble_context`: `resolve_league_target` deletes the
        # five league flags off the Namespace, so it cannot be called a second time.
        target = resolve_league_target(args, require_team_id=True)
        ctx = assemble_context(target, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (EspnLeagueError, OSError) as exc:
        print(f"Cannot reach the league: {exc}", file=sys.stderr)
        return 1

    if target.source is not None:
        print(target.describe())
    for note in ctx.notes:
        print(f"  ! {note}", file=sys.stderr)

    failures = 0
    for name, heading, module in SECTIONS:
        if name in sections:
            ran = _run_section(heading, module.report, ctx, section_args(module, args))
            failures += 1 if ran else 0

    # Non-zero when a section failed, so this is usable from a cron. The sections that DID run
    # have already printed; the exit code is about whether the document is complete.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
