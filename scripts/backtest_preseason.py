"""CLI: run preseason backtest harness and emit verdict report.

Usage:
    python scripts/backtest_preseason.py --model naive-preseason --target-seasons 2024,2025
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from projections.preseason.backtest import walk_forward_backtest, write_backtest_report
from projections.schemas import Ruleset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="naive-preseason", help="Model class name (info only)")
    parser.add_argument(
        "--target-seasons",
        required=True,
        help="Comma-separated list, e.g. 2024,2025",
    )
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument(
        "--ruleset",
        choices=["espn_ppr", "espn_half", "standard"],
        default="espn_ppr",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    args = parser.parse_args()

    ruleset_map = {
        "espn_ppr": Ruleset.espn_ppr(),
        "espn_half": Ruleset.espn_half(),
        "standard": Ruleset.standard(),
    }
    ruleset = ruleset_map[args.ruleset]
    target_seasons = [int(s) for s in args.target_seasons.split(",")]

    backtest = walk_forward_backtest(
        raw_root=args.raw_root,
        projections_root=args.projections_root,
        target_seasons=target_seasons,
        train_start=args.train_start,
        ruleset=ruleset,
    )

    csv_path = args.reports_root / f"backtest_preseason_{args.model}.csv"
    md_path = args.reports_root / f"backtest_preseason_{args.model}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(csv_path, index=False)
    write_backtest_report(backtest, md_path)
    print(f"Wrote CSV -> {csv_path}")
    print(f"Wrote markdown -> {md_path}")


if __name__ == "__main__":
    main()
