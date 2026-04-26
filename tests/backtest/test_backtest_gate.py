"""Opt-in walk-forward backtest gate.

Plan 3c Phase 5, part 3. Runs only with `pytest -m backtest --run-backtest`.
Loads the committed snapshot + tolerances, runs the full harness, calls
diff_snapshot, and asserts the gate passed.

The committed snapshot file is produced by Phase 6 of Plan 3c. Until
then this test additionally guards on file existence and skips with a
clear message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projections.backtest import diff_snapshot, read_snapshot, run_backtest

_SNAPSHOT_PATH = Path("tests/backtest/baseline_metrics.json")
_TOLERANCES_PATH = Path("tests/backtest/tolerances.json")


@pytest.mark.backtest
def test_backtest_gate_does_not_regress() -> None:
    """Run the full harness, diff against the committed snapshot, fail on
    any regression beyond per-metric-type tolerance."""
    if not _SNAPSHOT_PATH.exists():
        pytest.skip(
            f"{_SNAPSHOT_PATH} missing — Phase 6 of Plan 3c hasn't produced "
            f"the v1 snapshot yet. Run: python scripts/backtest.py --update-snapshot"
        )

    tolerances = json.loads(_TOLERANCES_PATH.read_text(encoding="utf-8"))
    baseline = read_snapshot(_SNAPSHOT_PATH)

    run = run_backtest()
    result = diff_snapshot(
        current=run.metrics,
        baseline=baseline,
        defaults=tolerances["defaults"],
        overrides=tolerances["overrides"],
    )

    if not result.passed:
        msg_lines = [
            f"Backtest gate failed with {len(result.regressions)} regression(s):",
            *[f"  - {r.message}" for r in result.regressions],
        ]
        pytest.fail("\n".join(msg_lines))
