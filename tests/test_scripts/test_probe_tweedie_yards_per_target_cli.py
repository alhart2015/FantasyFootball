"""CLI smoke for scripts/probe_tweedie_yards_per_target.py.

Mocks walk_forward_residuals to avoid real data; verifies argparse + report
writing.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import scripts.probe_tweedie_yards_per_target as probe_cli

from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.tweedie_yards_per_target_probe import (
    PerStatVerdict,
    ProbeResults,
)
from projections.schemas import Stat


def _fake_results() -> ProbeResults:
    return ProbeResults(
        actual_yards=np.array([55.0, 30.0, 72.0], dtype=np.float64),
        pred_ridge=np.array([55.1, 31.2, 70.8], dtype=np.float64),
        pred_tweedie=np.array([54.9, 30.5, 71.4], dtype=np.float64),
        year=np.array([2021, 2021, 2022], dtype=np.int64),
        coverage_per_year={2021: 0.98, 2022: 0.99},
    )


def _fake_verdict() -> PerStatVerdict:
    return PerStatVerdict(
        stat=Stat.RECEIVING_YARDS,
        n_paired=3,
        rmse_delta=BootstrapDelta(
            point=-0.5,
            lo_95=-0.8,
            hi_95=-0.1,
            n_paired_rows=3,
            n_bootstrap=1000,
        ),
        verdict="SIGNAL",
    )


def test_cli_writes_summary_and_csv(tmp_path: Path) -> None:
    """CLI invocation produces both the summary .md and the .csv report."""
    summary_path = tmp_path / "summary.md"
    csv_path = tmp_path / "deltas.csv"

    fake_features = mock.MagicMock(name="features")
    fake_weekly_stats = mock.MagicMock(name="weekly_stats")

    with (
        mock.patch.object(
            probe_cli,
            "_load_inputs",
            return_value=(fake_features, fake_weekly_stats),
        ),
        mock.patch.object(probe_cli, "walk_forward_residuals", return_value=_fake_results()),
        mock.patch.object(probe_cli, "compute_verdict", return_value=_fake_verdict()),
        mock.patch.object(
            sys,
            "argv",
            [
                "probe_tweedie_yards_per_target",
                "--summary-out",
                str(summary_path),
                "--csv-out",
                str(csv_path),
            ],
        ),
    ):
        probe_cli.main()

    assert summary_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "SIGNAL" in summary
    assert "n_paired: 3" in summary

    assert csv_path.exists()
    csv = csv_path.read_text(encoding="utf-8")
    assert "2021" in csv
    assert "2022" in csv
    assert "pooled" in csv.lower() or "all" in csv.lower()


def test_cli_rejects_unknown_year(tmp_path: Path) -> None:
    """argparse should choke on an out-of-range --eval-years value."""
    with mock.patch.object(
        sys,
        "argv",
        [
            "probe_tweedie_yards_per_target",
            "--eval-years",
            "2099",
            "--summary-out",
            str(tmp_path / "out.md"),
        ],
    ):
        with pytest.raises(SystemExit):
            probe_cli.main()
