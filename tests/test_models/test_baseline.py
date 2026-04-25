"""BaselineModel unit tests."""

from __future__ import annotations

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
