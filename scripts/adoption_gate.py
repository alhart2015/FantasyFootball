"""Plan 8 — adoption gate CLI.

Reads a backtest run's per-row results.parquet, pairs rows on
(gsis_id, season, week) between two model classes, and emits per-position
adoption verdicts via paired-bootstrap CIs.

Usage:
    python -m scripts.adoption_gate \\
        --run data/backtest/run_<ts> \\
        --candidate <model_class> \\
        [--incumbent baseline] \\
        [--position QB|RB|TE|WR|all] \\
        [--csv-out reports/adoption_gate_<cand>_<ts>.csv] \\
        [--n-bootstrap 1000] \\
        [--seed 42]

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_run_parquet(run_dir: Path) -> pd.DataFrame:
    """Load <run_dir>/results.parquet.

    Raises:
        FileNotFoundError: results.parquet missing under run_dir.
    """
    results_path = run_dir / "results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"results.parquet missing under {run_dir}; this CLI expects per-row "
            f"backtest output produced by scripts/backtest.py."
        )
    return pd.read_parquet(results_path)


def validate_model_classes_present(df: pd.DataFrame, *, incumbent: str, candidate: str) -> None:
    """Raise ValueError if either incumbent or candidate is not in df['model_class']."""
    present = set(df["model_class"].unique())
    if candidate not in present:
        raise ValueError(
            f"candidate model_class '{candidate}' not present in run; "
            f"present classes: {sorted(present)}"
        )
    if incumbent not in present:
        raise ValueError(
            f"incumbent model_class '{incumbent}' not present in run; "
            f"present classes: {sorted(present)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 8 adoption gate.")
    parser.add_argument("--run", type=Path, required=True, help="run_<ts> directory")
    parser.add_argument("--candidate", type=str, required=True, help="candidate model_class")
    parser.add_argument(
        "--incumbent", type=str, default="baseline", help="incumbent model_class (default baseline)"
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["QB", "RB", "TE", "WR", "all"],
        default="all",
        help="position to evaluate (default all)",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="optional CSV output path")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_run_parquet(args.run)
    validate_model_classes_present(df, incumbent=args.incumbent, candidate=args.candidate)
    print(f"Loaded {len(df)} rows from {args.run / 'results.parquet'}.")
    print(f"Model classes present: {sorted(df['model_class'].unique())}")


if __name__ == "__main__":
    main()
