"""Throwaway grounding diagnostic for TODO #55.

Runs the walk-forward model backtest for RB ONLY across
{baseline, lightgbm-nb, ensemble} and prints the composite accuracy +
ranking metrics per (year, model_class), so we can see whether any
already-built RB model beats the incumbent `baseline` before designing
anything. Read-only; not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.backtest import run_backtest
from projections.schemas import Position

MODEL_CLASSES = ("baseline", "lightgbm-nb", "ensemble")
METRICS = ("composite_rmse", "composite_mae", "spearman_topN")


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-only model bake-off (#55).")
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="feature cache root; point at an augmented root to A/B feature sets.",
    )
    args = parser.parse_args()
    run = run_backtest(
        positions=(Position.RB,),
        model_classes=MODEL_CLASSES,
        features_root=args.features_root,
    )
    m = run.metrics
    m = m[m["metric"].isin(METRICS)]
    pivot = m.pivot_table(
        index=["metric", "year"],
        columns="model_class",
        values="value",
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== RB walk-forward backtest (ESPN-PPR composite) ===")
    print(pivot.to_string(float_format=lambda x: f"{x:8.4f}"))

    # Pooled mean across years per metric/model.
    pooled = m.groupby(["metric", "model_class"])["value"].mean().unstack("model_class")
    print("\n=== Pooled (mean over 2021-2024) ===")
    print(pooled.to_string(float_format=lambda x: f"{x:8.4f}"))


if __name__ == "__main__":
    main()
