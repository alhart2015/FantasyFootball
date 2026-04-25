"""BaselineModel unit tests."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.models import wr_baseline
from projections.schemas import DistributionFamily, Position, Stat


def test_wr_baseline_factory_returns_unfitted_model() -> None:
    model = wr_baseline()
    assert model.position == Position.WR
    # Unfitted models do not yet have model_id; accessing it should error or
    # return a sentinel. We pick the explicit-error path.
    expected_targets = {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.FUMBLES_LOST,
    }
    assert set(model.target_stats) == expected_targets
    assert model.dist_families[Stat.RECEPTIONS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RECEIVING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RECEIVING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.GAMMA
    assert model.feature_columns  # non-empty; specific list verified in Task 6


def test_baseline_fit_populates_ridges_per_target_stat(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # One fitted RidgeCV per target stat.
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        # Fitted ridges expose coef_; unfitted ones don't.
        assert hasattr(model.ridges[stat], "coef_")


def test_baseline_fit_persists_feature_means(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_baseline_fit_records_train_seasons(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Fixture covers 2024 + 2025; the fit signature consumes whatever it
    # receives, so train_seasons is just min/max season seen in training.
    assert model.train_seasons == (2024, 2025)
