# tests/test_backtest/test_feature_probe.py
"""Feature signal probe — tests."""

from __future__ import annotations

import numpy as np  # noqa: F401  # used by Task 1.2 tests appended to this file
import pandas as pd  # noqa: F401  # used by Task 1.2 tests appended to this file
import pytest

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    PositionVerdict,  # noqa: F401  # used by Task 1.2 tests appended to this file
)
from projections.backtest.feature_probe import (
    PerStatVerdict,
    ProbeReport,
    phase1_should_fire_phase2,
)
from projections.schemas import Position, Stat


def test_per_stat_verdict_is_frozen_dataclass() -> None:
    psv = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled=2024,
        n_paired=670,
        rmse_delta=BootstrapDelta(
            point=-0.42, lo_95=-0.71, hi_95=-0.13, n_paired_rows=670, n_bootstrap=1000
        ),
        r_squared_delta=0.0023,
        verdict="SIGNAL",
    )
    assert psv.position is Position.QB
    assert psv.year_or_pooled == 2024
    assert psv.verdict == "SIGNAL"
    # FrozenInstanceError subclasses AttributeError; accept either to keep the
    # test robust across Python's dataclass internals.
    with pytest.raises(Exception):  # noqa: B017
        psv.verdict = "NULL"  # type: ignore[misc]


def test_per_stat_verdict_year_or_pooled_accepts_pooled_literal() -> None:
    psv = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=2676, n_bootstrap=1000
        ),
        r_squared_delta=0.0,
        verdict="NULL",
    )
    assert psv.year_or_pooled == "pooled"


def test_probe_report_phase2_optional() -> None:
    psv = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=2676, n_bootstrap=1000
        ),
        r_squared_delta=0.0,
        verdict="NULL",
    )
    report = ProbeReport(
        candidate_name="test",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("data/features_probe/x.parquet",),
        drop_columns=(),
        phase1=[psv],
        phase2=None,
    )
    assert report.phase2 is None
    assert report.candidate_name == "test"
    assert len(report.phase1) == 1


def test_phase1_should_fire_phase2_truth_table() -> None:
    bd_null = BootstrapDelta(point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=2676, n_bootstrap=1000)
    bd_signal = BootstrapDelta(
        point=-0.5, lo_95=-0.9, hi_95=-0.1, n_paired_rows=2676, n_bootstrap=1000
    )
    bd_regression = BootstrapDelta(
        point=0.5, lo_95=0.1, hi_95=0.9, n_paired_rows=2676, n_bootstrap=1000
    )

    def _verdict(label: str, bd: BootstrapDelta) -> PerStatVerdict:
        return PerStatVerdict(
            position=Position.QB,
            stat=Stat.PASSING_YARDS,
            year_or_pooled="pooled",
            n_paired=bd.n_paired_rows,
            rmse_delta=bd,
            r_squared_delta=0.0,
            verdict=label,  # type: ignore[arg-type]
        )

    assert phase1_should_fire_phase2([]) is False
    assert phase1_should_fire_phase2([_verdict("NULL", bd_null)]) is False
    assert phase1_should_fire_phase2([_verdict("REGRESSION", bd_regression)]) is False
    assert (
        phase1_should_fire_phase2(
            [_verdict("NULL", bd_null), _verdict("REGRESSION", bd_regression)]
        )
        is False
    )
    assert phase1_should_fire_phase2([_verdict("SIGNAL", bd_signal)]) is True
    assert (
        phase1_should_fire_phase2([_verdict("NULL", bd_null), _verdict("SIGNAL", bd_signal)])
        is True
    )
