# tests/test_backtest/test_feature_probe.py
"""Feature signal probe — tests."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.feature_probe import (
    PerStatVerdict,
    ProbeReport,
    phase1_should_fire_phase2,
    probe_per_stat,
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


def test_probe_per_stat_signal_on_orthogonal_signal_column(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A candidate column orthogonal to baseline features but driving the
    target should produce SIGNAL on the relevant stat."""
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_signal"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    # 1 stat x (1 year + 1 pooled) = 2 verdicts.
    assert len(verdicts) == 2
    pooled = next(v for v in verdicts if v.year_or_pooled == "pooled")
    assert pooled.stat is Stat.PASSING_YARDS
    assert pooled.verdict == "SIGNAL", (
        f"expected SIGNAL on pooled passing_yards but got {pooled.verdict}; "
        f"rmse_delta={pooled.rmse_delta}"
    )
    # The signal column carries the bulk of the target; delta-R^2 should be substantially positive.
    assert pooled.r_squared_delta > 0.05, (
        f"expected r_squared_delta > 0.05 on a strong signal column, got {pooled.r_squared_delta}"
    )


def test_probe_per_stat_null_on_pure_noise_column(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A candidate column with no relationship to the target should produce NULL."""
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_null"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    pooled = next(v for v in verdicts if v.year_or_pooled == "pooled")
    assert pooled.verdict == "NULL", (
        f"expected NULL on pure-noise column, got {pooled.verdict}; rmse_delta={pooled.rmse_delta}"
    )


def test_probe_per_stat_null_on_redundant_column(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A candidate column collinear with an existing baseline column should
    produce NULL -- Ridge shrinks one of the correlated pair, so the candidate
    adds no marginal CV-RMSE benefit."""
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_redundant"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    pooled = next(v for v in verdicts if v.year_or_pooled == "pooled")
    assert pooled.verdict == "NULL", (
        f"expected NULL on collinear-with-baseline column, got {pooled.verdict}"
    )


def test_probe_per_stat_emits_per_year_and_pooled_rows(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_signal"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS, Stat.PASSING_TDS),
        holdout_years=(2021, 2022),
    )
    # 2 stats x (2 years + 1 pooled) = 6 rows.
    assert len(verdicts) == 6
    years = sorted({v.year_or_pooled for v in verdicts if v.year_or_pooled != "pooled"})
    assert years == [2021, 2022]
    pooled_rows = [v for v in verdicts if v.year_or_pooled == "pooled"]
    assert len(pooled_rows) == 2
    assert {v.stat for v in pooled_rows} == {Stat.PASSING_YARDS, Stat.PASSING_TDS}


def test_probe_per_stat_is_deterministic(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, weekly_stats = probe_synthetic_dataset
    kwargs = dict(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_signal"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    a = probe_per_stat(**kwargs)
    b = probe_per_stat(**kwargs)
    assert len(a) == len(b)
    for va, vb in zip(a, b, strict=True):
        assert va.rmse_delta.point == vb.rmse_delta.point
        assert va.rmse_delta.lo_95 == vb.rmse_delta.lo_95
        assert va.rmse_delta.hi_95 == vb.rmse_delta.hi_95
        assert va.verdict == vb.verdict
