"""Plan 3e Phase 0 — calibration diagnostic CLI.

Reads the most recent data/backtest/run_<ts>/results.parquet, computes
per-(position, stat) residual diagnostics for the held-out years,
fits 2-3 alternative families per cell with AIC ranking, and writes
structured artifacts to data/diagnostics/calibration_<ts>/.

Spec: docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_latest_run_dir(backtest_root: Path) -> Path:
    """Return the most recent data/backtest/run_<ts>/ directory.

    Sorts lexicographically by directory name, which is correct because
    the timestamp format (YYYYMMDDTHHMMSSZ) sorts the same way as time.

    Raises:
        FileNotFoundError: backtest_root doesn't exist or contains no run_* subdirs.
    """
    if not backtest_root.is_dir():
        raise FileNotFoundError(f"Backtest root not found: {backtest_root}")
    candidates = sorted(
        p for p in backtest_root.iterdir() if p.is_dir() and p.name.startswith("run_")
    )
    if not candidates:
        raise FileNotFoundError(f"No run_<ts>/ subdirectories under {backtest_root}")
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Path to a data/backtest/run_<ts>/ directory. "
            "Defaults to the lexicographically-latest run_<ts>/ directory."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Path to write diagnostic artifacts. Defaults to data/diagnostics/calibration_<ts>/.",
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
