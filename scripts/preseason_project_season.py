"""CLI: produce preseason season-total projections for a target season.

Usage:
    python scripts/preseason_project_season.py --season 2026
    python scripts/preseason_project_season.py --season 2026 --ruleset espn_half
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from projections.preseason.project import project_preseason
from projections.schemas import Ruleset
from projections.store import read_partition


def _summary_csv(projections: pd.DataFrame, raw_root: Path, out_path: Path) -> None:
    """Write a human-readable top-N-per-position summary CSV."""
    id_map = read_partition(raw_root, "id_map")
    summary = projections.merge(
        id_map[["gsis_id", "full_name"]],
        on="gsis_id",
        how="left",
    )
    summary = summary[
        ["gsis_id", "full_name", "position", "team", "season_total_fpts_mean", "model_id"]
    ].sort_values("season_total_fpts_mean", ascending=False)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", type=int, required=True, help="Target preseason year (e.g., 2026)"
    )
    parser.add_argument(
        "--ruleset",
        choices=["espn_ppr", "espn_half", "standard"],
        default="espn_ppr",
    )
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    ruleset_map = {
        "espn_ppr": Ruleset.espn_ppr(),
        "espn_half": Ruleset.espn_half(),
        "standard": Ruleset.standard(),
    }
    ruleset = ruleset_map[args.ruleset]

    dropped_path = args.reports_root / f"preseason_dropped_{args.season}.csv"
    projections = project_preseason(
        raw_root=args.raw_root,
        projections_root=args.projections_root,
        target_season=args.season,
        train_start=args.train_start,
        ruleset=ruleset,
        dropped_csv_path=dropped_path,
    )

    if not args.no_summary:
        summary_path = args.reports_root / f"preseason_{args.season}.csv"
        _summary_csv(projections, args.raw_root, summary_path)
        print(f"Wrote summary -> {summary_path}")

    print(
        f"Done. {len(projections)} projections written for "
        f"season={args.season} ruleset={ruleset.name}."
    )


if __name__ == "__main__":
    main()
