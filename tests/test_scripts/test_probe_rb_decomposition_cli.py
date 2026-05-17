"""CLI smoke for scripts/probe_rb_decomposition.py.

Mocks walk_forward_residuals + compute_verdicts to avoid real data; verifies
argparse + report writing.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import scripts.probe_rb_decomposition as probe_cli

from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.rb_decomposition_probe import (
    PerStatVerdict,
    StatResiduals,
    WalkForwardOutput,
)
from projections.schemas import Stat


def _fake_output() -> WalkForwardOutput:
    actual = np.array([55.0, 30.0, 72.0], dtype=np.float64)
    md = np.array([55.1, 31.2, 70.8], dtype=np.float64)
    mdc = np.array([54.9, 30.5, 71.4], dtype=np.float64)
    per_stat = {
        s: StatResiduals(actual=actual, mu_direct=md, mu_decomposed=mdc, n_paired=3)
        for s in (
            Stat.RUSHING_YARDS,
            Stat.RUSHING_TDS,
            Stat.RECEPTIONS,
            Stat.RECEIVING_YARDS,
            Stat.RECEIVING_TDS,
        )
    }
    return WalkForwardOutput(
        per_stat=per_stat,
        factor_residuals_by_year={s: [] for s in per_stat},
        coverage_carries_by_year={2021: 0.92, 2022: 0.94},
        coverage_targets_by_year={2021: 0.88, 2022: 0.90},
        eval_years=(2021, 2022),
    )


def _fake_verdicts() -> list[PerStatVerdict]:
    return [
        PerStatVerdict(
            stat=stat,
            n_paired=3,
            rmse_delta=BootstrapDelta(
                point=-0.5, lo_95=-0.8, hi_95=-0.1, n_paired_rows=3, n_bootstrap=1000
            ),
            verdict="SIGNAL",
        )
        for stat in (
            Stat.RUSHING_YARDS,
            Stat.RUSHING_TDS,
            Stat.RECEPTIONS,
            Stat.RECEIVING_YARDS,
            Stat.RECEIVING_TDS,
        )
    ]


def test_cli_writes_summary_and_csv(tmp_path: Path) -> None:
    """CLI invocation produces both the summary .md and the .csv report."""
    summary_path = tmp_path / "summary.md"
    csv_path = tmp_path / "per_stat.csv"

    with (
        mock.patch.object(
            probe_cli,
            "_load_inputs",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(probe_cli, "walk_forward_residuals", return_value=_fake_output()),
        mock.patch.object(probe_cli, "compute_verdicts", return_value=_fake_verdicts()),
        mock.patch.object(
            sys,
            "argv",
            [
                "probe_rb_decomposition",
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
    # All 5 stats appear, all SIGNAL verdicts.
    for stat_value in (
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ):
        assert stat_value in summary
    assert summary.count("SIGNAL") >= 5
    # Coverage section surfaces below-threshold flag (0.88 < 0.95).
    assert "BELOW THRESHOLD" in summary

    assert csv_path.exists()
    csv = csv_path.read_text(encoding="utf-8")
    for stat_value in (
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ):
        assert stat_value in csv


def test_cli_rejects_unknown_year(tmp_path: Path) -> None:
    """argparse should choke on an out-of-range --eval-years value."""
    with mock.patch.object(
        sys,
        "argv",
        [
            "probe_rb_decomposition",
            "--eval-years",
            "2099",
            "--summary-out",
            str(tmp_path / "out.md"),
        ],
    ):
        with pytest.raises(SystemExit):
            probe_cli.main()
