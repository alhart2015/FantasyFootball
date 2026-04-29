"""Plan 8 — adoption gate tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    PositionVerdict,
    paired_bootstrap_rmse_delta,
)
from projections.schemas import Position


def test_bootstrap_delta_is_frozen_dataclass() -> None:
    bd = BootstrapDelta(point=-0.5, lo_95=-1.2, hi_95=0.1, n_paired_rows=1000, n_bootstrap=500)
    assert bd.point == -0.5
    assert bd.n_bootstrap == 500
    with pytest.raises(FrozenInstanceError):
        bd.point = 0.0  # type: ignore[misc]


def test_position_verdict_bundles_metrics() -> None:
    rmse = BootstrapDelta(point=-0.3, lo_95=-0.5, hi_95=-0.1, n_paired_rows=500, n_bootstrap=1000)
    spear = BootstrapDelta(
        point=0.01, lo_95=-0.005, hi_95=0.025, n_paired_rows=500, n_bootstrap=1000
    )
    breakdown = pd.DataFrame(
        {
            "year": [2021, 2022],
            "rmse_delta_point": [-0.4, -0.2],
            "rmse_delta_lo": [-0.7, -0.4],
            "rmse_delta_hi": [-0.1, 0.0],
            "spearman_delta_point": [0.02, 0.0],
            "spearman_delta_lo": [-0.01, -0.02],
            "spearman_delta_hi": [0.04, 0.02],
        }
    )
    pv = PositionVerdict(
        position=Position.QB,
        incumbent_class="baseline",
        candidate_class="ensemble",
        rmse_delta=rmse,
        spearman_delta=spear,
        verdict="ADOPT",
        reason="RMSE win, Spearman within floor",
        per_year_breakdown=breakdown,
    )
    assert pv.position is Position.QB
    assert pv.verdict == "ADOPT"
    assert len(pv.per_year_breakdown) == 2


def test_rmse_delta_identical_residuals_brackets_zero() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=2000)
    bd = paired_bootstrap_rmse_delta(residuals, residuals, n_bootstrap=500, seed=42)
    assert bd.point == 0.0
    assert bd.lo_95 <= 0.0 <= bd.hi_95
    assert bd.n_paired_rows == 2000
    assert bd.n_bootstrap == 500


def test_rmse_delta_candidate_strictly_better_has_negative_ci() -> None:
    rng = np.random.default_rng(0)
    incumbent_residuals = rng.normal(scale=2.0, size=3000)
    candidate_residuals = incumbent_residuals / 2.0  # half the variance
    bd = paired_bootstrap_rmse_delta(
        incumbent_residuals, candidate_residuals, n_bootstrap=500, seed=42
    )
    assert bd.point < 0.0
    assert bd.hi_95 < 0.0  # 95% CI entirely below zero


def test_rmse_delta_candidate_strictly_worse_has_positive_ci() -> None:
    rng = np.random.default_rng(0)
    incumbent_residuals = rng.normal(scale=2.0, size=3000)
    candidate_residuals = incumbent_residuals * 2.0
    bd = paired_bootstrap_rmse_delta(
        incumbent_residuals, candidate_residuals, n_bootstrap=500, seed=42
    )
    assert bd.point > 0.0
    assert bd.lo_95 > 0.0


def test_rmse_delta_deterministic_under_same_seed() -> None:
    rng = np.random.default_rng(0)
    inc = rng.normal(size=500)
    cand = inc + rng.normal(scale=0.1, size=500)
    bd1 = paired_bootstrap_rmse_delta(inc, cand, n_bootstrap=200, seed=99)
    bd2 = paired_bootstrap_rmse_delta(inc, cand, n_bootstrap=200, seed=99)
    assert bd1 == bd2


def test_rmse_delta_raises_on_too_few_rows() -> None:
    inc = np.zeros(50)
    cand = np.zeros(50)
    with pytest.raises(ValueError, match="at least 100 paired rows"):
        paired_bootstrap_rmse_delta(inc, cand)


def test_rmse_delta_raises_on_length_mismatch() -> None:
    inc = np.zeros(200)
    cand = np.zeros(199)
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_rmse_delta(inc, cand)
