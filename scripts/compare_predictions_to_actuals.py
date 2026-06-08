"""Compare the season_projection_{season}.csv (from scripts/project_season.py) against
the actual ESPN-PPR season totals derived from data/raw/weekly_stats/season=2024.

Prints one table per position: top-10 predicted players, with predicted points,
actual points, delta (pred - actual), actual rank within position.

Usage:
    python scripts/compare_predictions_to_actuals.py --season 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring import score
from projections.scoring.score import StatLine
from projections.store import read_partition


def _actual_ppr_total(ws: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Per-row PPR points, summed per (gsis_id, position) -> actual season total."""
    points: list[float] = []
    for _, row in ws.iterrows():
        line = StatLine(
            passing_yards=float(row["passing_yards"]),
            passing_tds=int(row["passing_tds"]),
            interceptions=int(row["interceptions"]),
            rushing_yards=float(row["rushing_yards"]),
            rushing_tds=int(row["rushing_tds"]),
            receptions=int(row["receptions"]),
            receiving_yards=float(row["receiving_yards"]),
            receiving_tds=int(row["receiving_tds"]),
            fumbles_lost=int(row["fumbles_lost"]),
        )
        points.append(score(line, ruleset))
    ws = ws.copy()
    ws["actual_ppr"] = points
    season_actuals = ws.groupby(["gsis_id", "position"], as_index=False).agg(
        actual_total=("actual_ppr", "sum"), actual_n_weeks=("week", "nunique")
    )
    return season_actuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    predictions_csv = args.predictions_csv or (
        Path("reports") / f"season_projection_{args.season}.csv"
    )

    print(f"Loading predictions from {predictions_csv}", flush=True)
    preds = pd.read_csv(predictions_csv)

    print(f"Loading {args.season} weekly_stats actuals", flush=True)
    ws = read_partition(args.raw_root, "weekly_stats", season=args.season)
    actuals = _actual_ppr_total(ws, Ruleset.espn_ppr())

    # Within-position actual rank (for context: "Nabers was predicted #1 WR; he was actually #N").
    actuals["actual_pos_rank"] = (
        actuals.groupby("position")["actual_total"].rank(ascending=False, method="min").astype(int)
    )

    merged = preds.merge(
        actuals.rename(columns={"position": "position_actual"}),
        on="gsis_id",
        how="left",
    )
    merged["delta"] = merged["season_total_mean"] - merged["actual_total"]
    # Sanity check on the position column: predictions and actuals should agree.
    pos_mismatch = merged[
        merged["position_actual"].notna() & (merged["position"] != merged["position_actual"])
    ]
    if len(pos_mismatch):
        print(
            f"Note: {len(pos_mismatch)} rows have position mismatch (player switched roles); "
            "using prediction's position for grouping."
        )

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.1f}".format)

    for pos in ("QB", "RB", "WR", "TE"):
        pos_df = merged[merged["position"] == pos].copy()
        pos_df = (
            pos_df.sort_values("season_total_mean", ascending=False).head(10).reset_index(drop=True)
        )
        pos_df.insert(0, "pred_rank", range(1, len(pos_df) + 1))
        view = pos_df[
            [
                "pred_rank",
                "full_name",
                "team",
                "season_total_mean",
                "actual_total",
                "actual_n_weeks",
                "actual_pos_rank",
                "delta",
            ]
        ].rename(
            columns={
                "season_total_mean": "predicted",
                "actual_total": "actual",
                "actual_n_weeks": "actual_wks",
                "actual_pos_rank": "actual_rk",
            }
        )
        print(f"\n=== {pos}: top-10 predicted vs actual (ESPN PPR, {args.season}) ===")
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
