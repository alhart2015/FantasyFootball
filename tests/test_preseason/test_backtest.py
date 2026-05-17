"""Tests for src/projections/preseason/backtest.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.preseason.backtest import (
    compute_rmse_and_spearman,
    determine_verdict,
)


def test_compute_rmse_zero_when_predicted_equals_actual() -> None:
    predicted = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "season_total_fpts_mean": [100.0, 200.0]}
    )
    actual = pd.DataFrame({"gsis_id": ["00-1", "00-2"], "actual_season_total_fpts": [100.0, 200.0]})
    rmse, spearman, n = compute_rmse_and_spearman(predicted=predicted, actual=actual, top_n=10)
    assert rmse == pytest.approx(0.0)
    assert spearman == pytest.approx(1.0)
    assert n == 2


def test_compute_rmse_known_value() -> None:
    predicted = pd.DataFrame(
        {"gsis_id": ["00-1", "00-2"], "season_total_fpts_mean": [100.0, 150.0]}
    )
    actual = pd.DataFrame({"gsis_id": ["00-1", "00-2"], "actual_season_total_fpts": [110.0, 140.0]})
    rmse, _, _ = compute_rmse_and_spearman(predicted=predicted, actual=actual, top_n=10)
    # sqrt(((10^2) + (10^2)) / 2) = sqrt(100) = 10
    assert rmse == pytest.approx(10.0)


def test_determine_verdict_adopt() -> None:
    assert determine_verdict(rmse_delta_pct=-5.0, spearman_top50=0.75) == "ADOPT"


def test_determine_verdict_do_not_adopt_worse_rmse() -> None:
    assert determine_verdict(rmse_delta_pct=2.0, spearman_top50=0.75) == "DO_NOT_ADOPT"


def test_determine_verdict_do_not_adopt_bad_spearman() -> None:
    assert determine_verdict(rmse_delta_pct=-5.0, spearman_top50=0.40) == "DO_NOT_ADOPT"


def test_determine_verdict_null_band() -> None:
    assert determine_verdict(rmse_delta_pct=-1.0, spearman_top50=0.60) == "NULL"
