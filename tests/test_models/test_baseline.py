"""BaselineModel unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.distributions.parametric import ParametricGamma, ParametricNormal
from projections.models import wr_baseline
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset, Stat


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


def test_baseline_fit_populates_normal_variance_params(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Normal stats: variance_params should have a positive 'std'.
    for stat in (Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS):
        params = model.variance_params[stat]
        assert "std" in params
        assert params["std"] > 0


def test_baseline_fit_populates_gamma_variance_params(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Gamma stats: variance_params should have a 'shape' in [0.01, 100].
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_method_of_moments_alpha_matches_hand_computed_value() -> None:
    """Verify _gamma_alpha_from_residuals on a hand-built example."""
    import numpy as np

    from projections.models.baseline import _gamma_alpha_from_residuals

    # mu_hat = [1, 2, 3, 4, 5] -> mean(mu_hat) = 3
    # residuals = y - mu_hat; var(residuals) = ?
    # Let var = 4.5 (arbitrary but easy). Then alpha_hat = 3^2 / 4.5 = 2.0
    mu_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Build residuals with sample variance = 4.5
    # variance is computed with ddof=0 (numpy default for .var()) -- match it.
    residuals = np.array([3.0, -3.0, 0.0, 1.5, -1.5])
    assert abs(residuals.var() - 4.5) < 1e-9
    alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=residuals)
    assert abs(alpha - 2.0) < 1e-9


def test_gamma_alpha_clipped_to_safety_range() -> None:
    """Pathological residuals should clip to [0.01, 100]."""
    import numpy as np

    from projections.models.baseline import _gamma_alpha_from_residuals

    # Almost-zero variance -> alpha blows up -> clipped to 100
    mu_hat = np.array([1.0, 1.0, 1.0])
    near_zero_resid = np.array([0.0001, 0.0, -0.0001])
    alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=near_zero_resid)
    assert alpha == 100.0

    # Massive variance with near-zero mean -> alpha ~0 -> clipped to 0.01
    mu_hat_zero = np.array([0.001, 0.001])
    huge_resid = np.array([100.0, -100.0])
    alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat_zero, residuals=huge_resid)
    assert alpha == 0.01


def test_build_stat_distributions_returns_one_per_row(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Pick a single week of test features.
    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    assert not week_features.empty
    stat_dists_per_row = model._build_stat_distributions(week_features)
    assert len(stat_dists_per_row) == len(week_features)
    for row_dists in stat_dists_per_row:
        assert set(row_dists.keys()) == set(model.target_stats)
        # Family-specific concrete types.
        assert isinstance(row_dists[Stat.RECEIVING_YARDS], ParametricNormal)
        assert isinstance(row_dists[Stat.RECEPTIONS], ParametricGamma)


def test_build_stat_distributions_clamps_gamma_mu() -> None:
    """A regression that predicts mu_hat <= 0 for a gamma stat should produce a
    ParametricGamma with finite, positive scale."""
    model = wr_baseline()
    # Hand-craft a fake fitted state with one stat configured.
    model.target_stats = (Stat.RECEPTIONS,)
    model.dist_families = {Stat.RECEPTIONS: DistributionFamily.GAMMA}
    model.feature_columns = ("dummy_feat",)
    model.feature_means = pd.Series({"dummy_feat": 0.0}, dtype=float)
    model.variance_params = {Stat.RECEPTIONS: {"shape": 2.0}}

    class _FakeRidge:
        def predict(self, x: np.ndarray) -> np.ndarray:
            # Predict negative mu (should be clamped at predict time).
            return np.full(x.shape[0], -5.0)

    model.ridges = {Stat.RECEPTIONS: _FakeRidge()}

    fake_features = pd.DataFrame({"dummy_feat": [1.0]})
    out = model._build_stat_distributions(fake_features)
    assert len(out) == 1
    rec_dist = out[0][Stat.RECEPTIONS]
    assert isinstance(rec_dist, ParametricGamma)
    # mean = shape * scale > 0 (clamped, not negative)
    assert rec_dist.mean() > 0


def test_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    # Schema validates without raising.
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    # Identity columns preserved.
    assert set(out["gsis_id"].astype(str)) == set(week_features["gsis_id"].astype(str))
    # ruleset / model_id / generated_at populated.
    assert (out["ruleset"] == "ESPN_PPR").all()
    assert out["model_id"].str.startswith("baseline:wr:").all()


def test_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    empty = baseline_features.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)


def test_predict_distribution_p10_le_p50_le_p90(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()
