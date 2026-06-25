"""CLI for the DFS projection edge study. All logic lives in projections.dfs."""

from __future__ import annotations

import argparse
from pathlib import Path

from projections.dfs.run import run_study, write_report
from projections.ingest.sleeper_weekly_projections import refresh_sleeper_weekly
from projections.schemas import Position, Ruleset
from projections.season_calendar import last_regular_week


def _seasons(arg: str) -> list[int]:
    lo, hi = (int(x) for x in arg.split("-")) if "-" in arg else (int(arg), int(arg))
    return list(range(lo, hi + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="DFS projection edge study")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared args via parent parsers so a subcommand can't silently omit one its
    # handler reads — the class of bug where `calibrate` read args.features_root
    # without declaring it (the original defect on main). `--data-root` is common
    # to all three; `--features-root` to the two that run the study.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=Path("data"))
    feat = argparse.ArgumentParser(add_help=False)
    feat.add_argument("--features-root", type=Path, default=Path("data/features"))

    ing = sub.add_parser("ingest-sleeper", parents=[common])
    ing.add_argument("--seasons", type=_seasons, required=True)

    cal = sub.add_parser("calibrate", parents=[common, feat])
    cal.add_argument("--prior-season", type=int, default=2020)

    stu = sub.add_parser("study", parents=[common, feat])
    stu.add_argument("--seasons", type=_seasons, required=True)
    stu.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    positions = [Position.QB, Position.RB, Position.WR, Position.TE]

    if args.cmd == "ingest-sleeper":
        for season in args.seasons:
            for week in range(1, last_regular_week(season) + 1):
                refresh_sleeper_weekly(args.data_root / "raw", season=season, week=week)
    elif args.cmd == "calibrate":
        # Print empirical |our-sleeper| / usage / cluster-count distributions so the
        # committed config.py constants are justified; do NOT auto-write them.
        out = run_study(
            seasons=[args.prior_season],
            positions=positions,
            data_root=args.data_root,
            features_root=args.features_root,
            ruleset=Ruleset.draftkings(),
        )
        print(
            "prior-season diagnostics:",
            out.coverage,
            out.inclusion,
            "n_clusters=",
            out.primary.n_clusters,
        )
    elif args.cmd == "study":
        out = run_study(
            seasons=args.seasons,
            positions=positions,
            data_root=args.data_root,
            features_root=args.features_root,
            ruleset=Ruleset.draftkings(),
        )
        write_report(args.out, out, seasons=args.seasons)
        print(f"verdict: {out.primary.verdict} -> {args.out}")


if __name__ == "__main__":
    main()
