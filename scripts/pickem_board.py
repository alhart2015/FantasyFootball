"""Weekly pick'em board - turn the organizer's Tuesday sheet into Thursday picks.

The league picks outright winners, but at least three picks each week must be
the underdog *according to the organizer's spread*. The organizer sets that
spread on Tuesday; picks are due before Thursday kickoff. Two days of line
movement - most importantly Wednesday's first injury report - sit in between,
and that gap is the edge this board exists to find.

The sheet decides who counts as an underdog. The market decides who wins. This
script never lets the first influence the second.

Weekly workflow:

    # Tuesday - emit a template with the real matchups already filled in
    python scripts/pickem_board.py --season 2026 --week 1 --template sheet.csv

    # ...type the organizer's spreads into the home_spread column...

    # Thursday morning - refresh lines, then pick
    python scripts/pickem_board.py --season 2026 --week 1 --sheet sheet.csv --refresh

    # Monday - grade last week
    python scripts/pickem_board.py --season 2026 --week 1 --grade
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.ingest import refresh_schedules
from projections.pickem.grade import grade_picks, record
from projections.pickem.optimize import DEFAULT_MIN_DOGS, choose_picks, expected_correct
from projections.pickem.sheet import read_sheet, write_template
from projections.pickem.slate import build_slate
from projections.pickem.store import read_picks, write_picks, write_sheet
from projections.store import read_partition

# A sheet-vs-market disagreement worth eyeballing. Two points is roughly a
# starting-lineup-level injury; below that it is mostly noise.
NOTABLE_LINE_MOVE = 2.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        help="Write a blank sheet CSV pre-filled with this week's matchups, then exit.",
    )
    parser.add_argument("--sheet", type=Path, help="The organizer's sheet CSV, spreads filled in.")
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Grade previously stored picks for this week against final scores.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-ingest schedules first, so consensus lines are current. Do this Thursday.",
    )
    parser.add_argument("--min-dogs", type=int, default=DEFAULT_MIN_DOGS)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--no-save", action="store_true", help="Print only; do not write to the store."
    )
    return parser.parse_args(argv)


def _load_schedules(data_root: Path, season: int) -> pd.DataFrame:
    return read_partition(data_root / "raw", "schedules", season=season)


def _fmt_spread(value: float) -> str:
    """Betting convention: an explicit sign on both sides, `PK` for a true pick'em."""
    if pd.isna(value):
        return "  -  "
    if value == 0:
        return "  PK "
    return f"{value:+5.1f}"


def render_board(slate: pd.DataFrame, picks: pd.DataFrame, *, min_dogs: int) -> str:
    merged = slate.merge(
        picks[["game_id", "pick", "pick_win_prob", "is_dog_pick", "forced", "switch_cost"]],
        on="game_id",
        validate="one_to_one",
    ).sort_values("pick_win_prob", ascending=False)

    lines = [
        f"Week {int(slate['week'].iloc[0])} - {len(slate)} games, {min_dogs} underdogs required",
        "",
        f"{'MATCHUP':<14}{'SHEET':>7}{'MKT':>7}{'PICK':>6}{'WIN%':>7}  FLAG",
        "-" * 60,
    ]
    for row in merged.itertuples():
        flags = []
        if row.forced:
            flags.append(f"forced dog (-{row.switch_cost:.0%})")
        elif row.is_dog_pick:
            flags.append("DOG - free")
        if row.free_dog:
            flags.append("market flipped it")
        if pd.notna(row.dog_line_move) and abs(row.dog_line_move) >= NOTABLE_LINE_MOVE:
            flags.append(f"line moved {row.dog_line_move:+.1f}")
        lines.append(
            f"{row.away_team + ' @ ' + row.home_team:<14}"
            f"{_fmt_spread(row.sheet_home_spread):>7}"
            f"{_fmt_spread(row.consensus_home_spread):>7}"
            f"{row.pick:>6}{row.pick_win_prob:>7.1%}  {', '.join(flags)}"
        )

    dogs = merged[merged["is_dog_pick"]]
    lines += [
        "-" * 60,
        f"Expected correct: {expected_correct(picks):.2f} of {len(picks)}",
        f"Underdog picks ({len(dogs)}): {', '.join(dogs['pick'].astype(str))}",
    ]

    free = merged[merged["free_dog"]]
    if not free.empty:
        lines += [
            "",
            "FREE DOGS - the sheet calls these underdogs but the market favors them.",
            "They satisfy the underdog rule at no cost. This is the edge.",
        ]
        for row in free.itertuples():
            lines.append(
                f"  {row.sheet_dog} ({row.away_team} @ {row.home_team}): "
                f"sheet {_fmt_spread(row.sheet_home_spread).strip()}, "
                f"market {_fmt_spread(row.consensus_home_spread).strip()}, "
                f"win {row.dog_win_prob:.1%}"
            )

    moved = merged[merged["dog_line_move"].abs() >= NOTABLE_LINE_MOVE].sort_values(
        "dog_line_move", ascending=False
    )
    if not moved.empty:
        lines += ["", f"LINE MOVES >= {NOTABLE_LINE_MOVE:.0f} PTS since the sheet was set:"]
        for row in moved.itertuples():
            direction = "toward" if row.dog_line_move > 0 else "away from"
            lines.append(
                f"  {row.away_team} @ {row.home_team}: {row.dog_line_move:+.1f} "
                f"{direction} {row.sheet_dog}"
            )
    return "\n".join(lines)


def _run_template(args: argparse.Namespace) -> int:
    schedules = _load_schedules(args.data_root, args.season)
    path = write_template(args.template, schedules, season=args.season, week=args.week)
    n = len(pd.read_csv(path))
    print(f"Wrote {n} matchups to {path}")
    print("Fill in home_spread from the organizer's sheet (negative = home favored), then re-run")
    print(
        f"  python scripts/pickem_board.py --season {args.season} --week {args.week} "
        f"--sheet {path} --refresh"
    )
    return 0


def _run_grade(args: argparse.Namespace) -> int:
    picks = read_picks(args.data_root, season=args.season, week=args.week)
    graded = grade_picks(picks, _load_schedules(args.data_root, args.season))
    correct, played = record(graded)

    if not args.no_save:
        write_picks(args.data_root, graded)

    print(f"Week {args.week}: {correct} of {played} correct")
    dogs = graded[graded["is_dog_pick"] & graded["correct"].notna()]
    if not dogs.empty:
        dog_correct = int(dogs["correct"].sum())
        print(f"  underdog picks: {dog_correct} of {len(dogs)} correct")
    missed = graded[graded["correct"].eq(False)]
    if not missed.empty:
        print(
            "  missed: "
            + ", ".join(f"{r.pick} ({r.pick_win_prob:.0%})" for r in missed.itertuples())
        )
    return 0


def _run_picks(args: argparse.Namespace) -> int:
    if args.refresh:
        print(f"Refreshing schedules for {args.season}...")
        refresh_schedules(args.data_root, seasons=[args.season])

    schedules = _load_schedules(args.data_root, args.season)
    sheet = read_sheet(args.sheet, season=args.season, week=args.week)
    slate = build_slate(sheet, schedules)
    picks = choose_picks(slate, min_dogs=args.min_dogs)

    scheduled = len(
        schedules[(schedules["season"] == args.season) & (schedules["week"] == args.week)]
    )
    if len(slate) != scheduled:
        print(f"Note: sheet covers {len(slate)} of {scheduled} scheduled games this week.\n")

    print(render_board(slate, picks, min_dogs=args.min_dogs))

    if not args.no_save:
        write_sheet(args.data_root, sheet)
        write_picks(args.data_root, picks)
        print(f"\nSaved to {args.data_root / 'pickem'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.template:
        return _run_template(args)
    if args.grade:
        return _run_grade(args)
    if not args.sheet:
        raise SystemExit("one of --template, --sheet or --grade is required")
    return _run_picks(args)


if __name__ == "__main__":
    raise SystemExit(main())
