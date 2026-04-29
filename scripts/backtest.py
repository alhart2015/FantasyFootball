"""Plan 3c — walk-forward backtest CLI.

Three modes:
  --check (default):   run harness, diff snapshot, exit 0/1.
  --update-snapshot:   run harness, overwrite tests/backtest/model_metrics.json.
  --report:            run harness, print model + naive metrics; no gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from projections.backtest import diff_snapshot, read_snapshot, run_backtest, write_snapshot

_SNAPSHOT_PATH = Path("tests/backtest/model_metrics.json")
_TOLERANCES_PATH = Path("tests/backtest/tolerances.json")


def _write_diagnostic_outputs(run: object) -> None:
    """Write per-row + per-player diagnostic frames to
    data/backtest/run_<ts>/. The directory is gitignored. Skipped silently
    if either frame is empty (e.g., a synthetic test run with no positions)."""
    timestamp_str = (
        pd.Timestamp(run.timestamp).strftime("%Y%m%dT%H%M%SZ")  # type: ignore[attr-defined]
    )
    out_dir = Path("data/backtest") / f"run_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not run.per_row_results.empty:  # type: ignore[attr-defined]
        run.per_row_results.to_parquet(out_dir / "results.parquet")  # type: ignore[attr-defined]
    if not run.per_player_results.empty:  # type: ignore[attr-defined]
        run.per_player_results.to_parquet(out_dir / "season_results.parquet")  # type: ignore[attr-defined]


def _print_metrics_table(label: str, metrics: pd.DataFrame, naive: pd.DataFrame) -> None:
    """Print a per-(position, year) table merging model + naive metrics
    side by side."""
    if metrics.empty:
        print(f"({label}: no metrics)")
        return
    pivot = metrics.pivot_table(index=["position", "year"], columns="metric", values="value")
    print(f"\n=== {label} ===")
    print(pivot.to_string(float_format=lambda x: f"{x:8.3f}"))

    if not naive.empty:
        naive_pivot = naive.pivot_table(
            index=["position", "year"], columns="metric", values="value"
        )
        print(f"\n=== {label} — naive baseline (informational) ===")
        # Print only composite + Spearman to keep the report compact.
        compact_cols = [
            c
            for c in naive_pivot.columns
            if c
            in {
                "composite_rmse",
                "composite_mae",
                "spearman_topN",
            }
        ]
        if compact_cols:
            print(naive_pivot[compact_cols].to_string(float_format=lambda x: f"{x:8.3f}"))


def _check(run: object, tolerances: dict[str, object]) -> int:
    """Run the diff against the committed snapshot. Returns POSIX exit code."""
    if not _SNAPSHOT_PATH.exists():
        print(
            f"ERROR: {_SNAPSHOT_PATH} missing. Run with --update-snapshot first.",
            file=sys.stderr,
        )
        return 2

    baseline = read_snapshot(_SNAPSHOT_PATH)
    result = diff_snapshot(
        current=run.metrics,  # type: ignore[attr-defined]
        baseline=baseline,
        defaults=tolerances["defaults"],  # type: ignore[arg-type]
        overrides=tolerances["overrides"],  # type: ignore[arg-type]
    )
    if result.passed:
        print(f"PASS — {len(run.metrics)} metrics within tolerance.")  # type: ignore[attr-defined]
        return 0
    print(f"FAIL — {len(result.regressions)} regression(s):")
    for r in result.regressions:
        print(f"  - {r.message}")
    return 1


def _update(run: object) -> int:
    """Write the current run's metrics as the new snapshot. Print a diff
    against the prior snapshot for human review."""
    if _SNAPSHOT_PATH.exists():
        prior = read_snapshot(_SNAPSHOT_PATH)
        # Quick diff summary.
        print(f"Previous snapshot: {len(prior)} rows.")
        print(f"New snapshot:      {len(run.metrics)} rows.")  # type: ignore[attr-defined]
    write_snapshot(run.metrics, _SNAPSHOT_PATH)  # type: ignore[attr-defined]
    print(f"Wrote {_SNAPSHOT_PATH}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest CLI.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Run + diff + exit 0/1 (default).")
    mode.add_argument(
        "--update-snapshot", action="store_true", help="Run + overwrite the committed snapshot."
    )
    mode.add_argument(
        "--report", action="store_true", help="Run + print model + naive metrics; no gate."
    )
    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "ensemble",
            "both",
            "all",
        ],
        default="both",
        help=(
            "Which model class(es) to run. "
            "'both' = Model A + Model C (legacy default). "
            "'all' = Model A + Model C + Model C-tuned + Model C-NB + Ensemble (Model D)."
        ),
    )
    args = parser.parse_args()

    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = (
            "baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "ensemble",
        )
    else:
        model_classes = (args.model,)

    tolerances = json.loads(_TOLERANCES_PATH.read_text(encoding="utf-8"))
    run = run_backtest(model_classes=model_classes)

    if args.update_snapshot:
        _write_diagnostic_outputs(run)
        sys.exit(_update(run))
    if args.report:
        _write_diagnostic_outputs(run)
        _print_metrics_table("Backtest", run.metrics, run.naive_metrics)
        sys.exit(0)
    # Default: check (no diagnostic output to keep the gate hermetic).
    sys.exit(_check(run, tolerances))


if __name__ == "__main__":
    main()
