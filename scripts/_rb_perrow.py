"""Throwaway (#55 follow-up): emit RB baseline per-row walk-forward predictions
for a given feature-cache root, so we can stratify composite error by usage and
test whether the trajectory-trend features help the high-volume case even though
they did not on the full RB population. Read-only; not committed."""

from __future__ import annotations

import argparse
from pathlib import Path

from projections.backtest import run_backtest
from projections.schemas import Position


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run = run_backtest(
        positions=(Position.RB,),
        model_classes=("baseline",),
        features_root=a.features_root,
    )
    pr = run.per_row_results
    pr.to_parquet(a.out)
    print(f"wrote {len(pr)} rows to {a.out}; cols: {list(pr.columns)}")


if __name__ == "__main__":
    main()
