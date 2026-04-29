"""Plan 8 — adoption gate tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    PositionVerdict,
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
