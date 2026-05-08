"""One-off: generate baseline + candidate backtest run directories needed
by adoption_gate.py --baseline-run/--candidate-run, then merge the
candidate metrics into the committed snapshot (preserving rows for
model classes we're not running).

Both backtest invocations are subprocess'd so each gets a fresh Python
process with the current source-file state — avoids import caching
across the schema-revert boundary.

Workflow:
1. Subprocess _run_single_backtest.py to run candidate backtest (current
   branch state, with weather features) and write data/backtest/run_candidate/.
2. Git-checkout main's source files to roll back the weather changes.
3. Refresh RB + WR feature caches (now without weather cols).
4. Subprocess _run_single_backtest.py for the baseline run (pre-weather
   state) and write data/backtest/run_baseline/.
5. Git-checkout HEAD's source files back to restore the weather state.
6. Refresh RB + WR feature caches (back to with weather cols).
7. Read candidate metrics (from step 1's metrics.parquet) and merge into
   the committed snapshot, preserving rows for non-target model classes.

Used by: weather features RB+WR integration (2026-05-08), Phase 5 / Task 13.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

_SNAPSHOT_PATH = Path("tests/backtest/model_metrics.json")
_TARGET_CLASSES = ("baseline", "lightgbm-nb")
_TARGET_POSITIONS_VALUES = ("RB", "WR")
_BASELINE_RUN_DIR = Path("data/backtest/run_baseline")
_CANDIDATE_RUN_DIR = Path("data/backtest/run_candidate")
_FLIPPED_FILES = (
    "src/projections/schemas.py",
    "src/projections/features/rb.py",
    "src/projections/features/wr.py",
    "src/projections/models/baseline.py",
)
_PYTHON = sys.executable


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _git_checkout(rev: str) -> None:
    _run(["git", "checkout", rev, "--", *_FLIPPED_FILES])


def _refresh_caches() -> None:
    for pos in ("rb", "wr"):
        _run([_PYTHON, "scripts/refresh_features.py", pos, "--seasons", "2018-2024"])


def _backtest(out_dir: Path) -> None:
    _run([_PYTHON, "scripts/_run_single_backtest.py", str(out_dir)])


def main() -> None:
    print("=== Phase 1: candidate run (current branch state, with weather features) ===")
    _backtest(_CANDIDATE_RUN_DIR)

    # Phases 2-4 leave the working tree with main's source files; ensure we
    # always restore HEAD on exit (success OR failure) so a crash mid-flight
    # doesn't leave the worktree in an inconsistent state.
    try:
        print("\n=== Phase 2: roll back source to main HEAD ===")
        _git_checkout("main")

        print("\n=== Phase 3: refresh feature caches (now without weather cols) ===")
        _refresh_caches()

        print("\n=== Phase 4: baseline run (main state, no weather features) ===")
        _backtest(_BASELINE_RUN_DIR)
    finally:
        print("\n=== Phase 5: restore source to HEAD (with weather features) ===")
        _git_checkout("HEAD")

    print("\n=== Phase 6: refresh feature caches (back to with weather cols) ===")
    _refresh_caches()

    # Lazy imports to avoid pulling project code before phase 5 restores it.
    from projections.backtest.snapshot import read_snapshot, write_snapshot

    print("\n=== Phase 7: merge candidate metrics into snapshot ===")
    candidate_metrics = pd.read_parquet(_CANDIDATE_RUN_DIR / "metrics.parquet")
    print(f"  candidate metrics: {len(candidate_metrics)} rows")

    prior = read_snapshot(_SNAPSHOT_PATH)
    print(f"  prior: {len(prior)} rows; classes: {sorted(prior['model_class'].unique())}")
    drop_mask = prior["position"].isin(_TARGET_POSITIONS_VALUES) & prior["model_class"].isin(
        _TARGET_CLASSES
    )
    preserved = prior[~drop_mask]
    print(
        f"  preserving {len(preserved)} rows; replacing {drop_mask.sum()} rows "
        f"((RB|WR) x ({'|'.join(_TARGET_CLASSES)}))"
    )
    merged = pd.concat([preserved, candidate_metrics], ignore_index=True)
    print(f"  merged total: {len(merged)} rows")
    write_snapshot(merged, _SNAPSHOT_PATH)
    print(f"  wrote {_SNAPSHOT_PATH}")

    print("\n=== Done ===")
    print(f"  baseline run dir:  {_BASELINE_RUN_DIR}")
    print(f"  candidate run dir: {_CANDIDATE_RUN_DIR}")
    print(f"  snapshot updated:  {_SNAPSHOT_PATH}")
    print(
        "\nNext: run adoption_gate.py --baseline-run <dir> --candidate-run <dir> --position RB,WR"
    )


if __name__ == "__main__":
    main()
