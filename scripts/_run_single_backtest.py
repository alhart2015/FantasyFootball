"""Internal helper: run a single backtest (RB+WR x baseline+lgb-nb) and
write the results to a specified run directory. Designed to be called as
a subprocess so each invocation gets a fresh Python process with current
source files (avoids import-caching across schema-revert boundaries).

Usage: python scripts/_run_single_backtest.py <output_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from projections.backtest.harness import run_backtest
from projections.schemas import Position

_TARGET_CLASSES = ("baseline", "lightgbm-nb")
_TARGET_POSITIONS = (Position.RB, Position.WR)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: _run_single_backtest.py <output_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_backtest(model_classes=_TARGET_CLASSES, positions=_TARGET_POSITIONS)
    print(f"  produced {len(run.metrics)} metric rows", flush=True)

    if not run.per_row_results.empty:
        run.per_row_results.to_parquet(out_dir / "results.parquet")
    if not run.per_player_results.empty:
        run.per_player_results.to_parquet(out_dir / "season_results.parquet")
    # Also persist metrics so the orchestrator can merge into the snapshot.
    run.metrics.to_parquet(out_dir / "metrics.parquet")
    print(f"  wrote {out_dir}/results.parquet ({len(run.per_row_results)} rows)", flush=True)


if __name__ == "__main__":
    main()
