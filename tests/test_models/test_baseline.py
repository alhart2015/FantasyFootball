"""BaselineModel unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from sklearn.linear_model import RidgeCV

from projections.distributions.parametric import ParametricGamma, ParametricStudentT
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
    assert model.dist_families[Stat.RECEIVING_YARDS] is DistributionFamily.STUDENT_T
    assert model.dist_families[Stat.RECEIVING_TDS] is DistributionFamily.NEGATIVE_BINOMIAL
    # Plan 3e Phase 1+ Task 2.6: WR RUSHING_YARDS reverted from STUDENT_T to
    # NORMAL after Phase 2 retrain produced a degenerate Student-t fit
    # (mean ~1 yard, df snapped to floor).
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.feature_columns  # non-empty; specific list verified in Task 6


def test_baseline_fit_populates_ridges_per_target_stat(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    # One fitted RidgeCV per target stat.
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        # Fitted ridges expose coef_; unfitted ones don't.
        assert hasattr(model.ridges[stat], "coef_")


def test_baseline_fit_persists_feature_means(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_baseline_fit_records_train_seasons(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    # Fixture covers 2024 + 2025; the fit signature consumes whatever it
    # receives, so train_seasons is just min/max season seen in training.
    assert model.train_seasons == (2024, 2025)


def test_baseline_fit_populates_student_t_variance_params(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    # Plan 3e Phase 2: WR RECEIVING_YARDS routes to STUDENT_T with (scale, df).
    # WR RUSHING_YARDS was reverted to NORMAL in Task 2.6 (degenerate fit).
    params = model.variance_params[Stat.RECEIVING_YARDS]
    assert "scale" in params
    assert "df" in params
    assert params["scale"] > 0
    assert params["df"] > 2.0


def test_baseline_fit_populates_normal_variance_params(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """Plan 3e Phase 1+ Task 2.6: WR RUSHING_YARDS reverted from STUDENT_T to
    NORMAL after Phase 2 retrain produced a degenerate Student-t fit."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    params = model.variance_params[Stat.RUSHING_YARDS]
    assert "std" in params
    assert params["std"] > 0


def test_baseline_fit_populates_gamma_variance_params(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    # Gamma stats: variance_params should have a 'shape' in [0.01, 100].
    # RECEPTIONS stays Gamma; count stats moved to NEGATIVE_BINOMIAL in Plan 3e Phase 1.
    for stat in (Stat.RECEPTIONS,):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_baseline_fit_populates_nb_variance_params(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """Plan 3e Phase 1: count stats (TDs, fumbles_lost) route to NEGATIVE_BINOMIAL,
    so their variance_params carry a `dispersion` in `_NB_DISPERSION_CLIP`."""
    from projections.models.baseline import _NB_DISPERSION_CLIP

    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    for stat in (Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "dispersion" in params
        assert _NB_DISPERSION_CLIP[0] <= params["dispersion"] <= _NB_DISPERSION_CLIP[1]


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
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    # Pick a single week of test features.
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    assert not week_features.empty
    stat_dists_per_row = model.build_stat_distributions(week_features)
    assert len(stat_dists_per_row) == len(week_features)
    for row_dists in stat_dists_per_row:
        assert set(row_dists.keys()) == set(model.target_stats)
        # Family-specific concrete types.
        assert isinstance(row_dists[Stat.RECEIVING_YARDS], ParametricStudentT)
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
    out = model.build_stat_distributions(fake_features)
    assert len(out) == 1
    rec_dist = out[0][Stat.RECEPTIONS]
    assert isinstance(rec_dist, ParametricGamma)
    # mean = shape * scale > 0 (clamped, not negative)
    assert rec_dist.mean() > 0


def test_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
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
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    empty = baseline_features_wr.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)


def test_predict_distribution_p10_le_p50_le_p90(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)

    artifact = tmp_path / "wr-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel

    loaded = BaselineModel.load(artifact)
    assert loaded.position == model.position
    assert loaded.model_id == model.model_id

    week = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_model_id_format(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "wr"
    assert len(parts[2]) == 8  # code_hash
    assert "-" in parts[3]


def test_unfitted_model_id_raises() -> None:
    model = wr_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")


def test_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """If a predict-time feature row has NaN in a column, predict should impute
    with feature_means rather than crash or propagate NaN to the output."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)

    week = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ].copy()
    # Forcibly NaN one value in a non-nullable feature column.
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_fit_handles_bool_feature_columns(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """is_home / roof_dome / designed_rusher are bool in WrFeaturesSchema and
    must be coerced to numeric for Ridge.fit()."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    # If fit succeeded, the boolean coercion worked.
    assert model.feature_means is not None
    for bool_col in ("is_home", "roof_dome", "designed_rusher"):
        assert bool_col in model.feature_means.index


def test_save_unfitted_model_raises(tmp_path: Path) -> None:
    """Calling .save() on an unfitted model must raise RuntimeError so we
    don't produce un-traceable artifacts (no model_id without train_seasons /
    code_hash)."""
    model = wr_baseline()
    try:
        model.save(tmp_path / "should-not-exist.joblib")
    except RuntimeError:
        return
    raise AssertionError("save() on an unfitted model should raise RuntimeError")


def test_load_rejects_non_baseline_artifact(tmp_path: Path) -> None:
    """BaselineModel.load() must reject a joblib artifact that doesn't deserialize
    to a BaselineModel — defense-in-depth against future GBMModel etc. being
    loaded via the wrong classmethod."""
    import joblib

    artifact = tmp_path / "not-a-model.joblib"
    joblib.dump({"not": "a model"}, artifact)

    from projections.models import BaselineModel

    try:
        BaselineModel.load(artifact)
    except TypeError:
        return
    raise AssertionError("load() on a non-BaselineModel artifact should raise TypeError")


def test_predict_distribution_writes_sampled_summary_family(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    from projections.schemas import DistributionFamily

    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["family"] == DistributionFamily.SAMPLED_SUMMARY.value).all()


def test_predict_distribution_params_round_trips_to_per_stat_dists(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    from projections.distributions import unpack_per_stat_params

    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ].head(3)
    expected_dists_per_row = model.build_stat_distributions(week_features)
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())

    for row_idx, expected_dists in enumerate(expected_dists_per_row):
        decoded = unpack_per_stat_params(out["params"].iloc[row_idx])
        assert set(decoded.keys()) == set(expected_dists.keys())
        for stat, expected_dist in expected_dists.items():
            decoded_dist = decoded[stat]
            assert type(decoded_dist) is type(expected_dist)
            assert decoded_dist.mean() == pytest.approx(expected_dist.mean())
            assert decoded_dist.std() == pytest.approx(expected_dist.std())


def test_predict_distribution_uses_per_row_seeds(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """Two rows for different (gsis_id, week) tuples must produce different
    samples even if their per-stat dists happen to coincide. We check that
    the persisted (mean, p10, p50, p90) summary is never identical across
    distinct rows of the same week (would happen under shared seed=42)."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    summary_tuples = list(zip(out["mean"], out["p10"], out["p50"], out["p90"], strict=True))
    assert len(set(summary_tuples)) == len(summary_tuples)


def test_predict_distribution_is_deterministic_across_calls(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """Two predict_distribution calls with the same fitted model and the same
    input frame must produce bit-identical mean/p10/p50/p90 columns — closes
    TODO #19's gate non-determinism check by demonstration."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out_a = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    out_b = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    for col in ("mean", "p10", "p50", "p90"):
        assert (out_a[col].to_numpy() == out_b[col].to_numpy()).all(), (
            f"Determinism violated on column {col}"
        )


def test_negative_binomial_dispersion_recovers_known_param() -> None:
    """Synthesize NB-distributed `actual` from known dispersion + per-row mean,
    fit the dispersion, expect recovery within tolerance."""
    from projections.models.baseline import _negative_binomial_dispersion_from_residuals

    rng = np.random.default_rng(42)
    n = 500
    mu_hat = rng.uniform(0.1, 1.5, n)
    true_dispersion = 3.0
    # Standard NB-2 synthesis: n_size = dispersion (scalar), p = dispersion / (dispersion + mu_hat).
    p = true_dispersion / (true_dispersion + mu_hat)
    actual = scipy_stats.nbinom.rvs(n=true_dispersion, p=p, size=n, random_state=rng).astype(
        np.float64
    )

    fitted = _negative_binomial_dispersion_from_residuals(mu_hat=mu_hat, actual=actual)
    assert fitted == pytest.approx(true_dispersion, rel=0.30)


def test_negative_binomial_dispersion_clipped_for_degenerate_input() -> None:
    """All-zero actual at mu_hat > 0 is maximally overdispersed: under the
    NB-2 ``size`` parameterization (where dispersion -> inf recovers Poisson,
    and dispersion -> 0 puts all mass at zero), the MLE is driven to the LOW
    clip. The estimator should snap there rather than returning a near-zero
    interior value -- snapping is the contract that keeps the fitted
    distribution from drifting on degenerate inputs."""
    from projections.models.baseline import (
        _NB_DISPERSION_CLIP,
        _negative_binomial_dispersion_from_residuals,
    )

    mu_hat = np.full(50, 0.1)
    actual = np.zeros(50)
    fitted = _negative_binomial_dispersion_from_residuals(mu_hat=mu_hat, actual=actual)
    assert fitted == _NB_DISPERSION_CLIP[0]


def test_student_t_params_recovers_known_params() -> None:
    """Synthesize Student-t-distributed residuals from known (scale, df),
    fit, expect recovery within tolerance."""
    from projections.models.baseline import _student_t_params_from_residuals

    rng = np.random.default_rng(42)
    n = 1000
    true_scale = 50.0
    true_df = 6.0
    residuals = scipy_stats.t.rvs(df=true_df, loc=0.0, scale=true_scale, size=n, random_state=rng)

    fitted_scale, fitted_df = _student_t_params_from_residuals(residuals=residuals)
    assert fitted_scale == pytest.approx(true_scale, rel=0.20)
    assert fitted_df == pytest.approx(true_df, rel=0.50)


def test_student_t_params_rejects_degenerate_input() -> None:
    """All-zero residuals collapse scipy's scale to ~0 — guard returns floor."""
    from projections.models.baseline import (
        _STUDENT_T_DF_FLOOR,
        _STUDENT_T_SCALE_FLOOR,
        _student_t_params_from_residuals,
    )

    fitted_scale, fitted_df = _student_t_params_from_residuals(residuals=np.zeros(50))
    assert fitted_scale >= _STUDENT_T_SCALE_FLOOR
    assert fitted_df >= _STUDENT_T_DF_FLOOR


def test_baseline_model_fit_stores_nb_dispersion_for_nb_stat(
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> None:
    """A BaselineModel configured with a NEGATIVE_BINOMIAL stat must store
    a "dispersion" entry in variance_params after fit()."""
    from projections.models.baseline import _NB_DISPERSION_CLIP, wr_baseline
    from projections.schemas import DistributionFamily, Stat

    model = wr_baseline()
    # Rewire RECEIVING_TDS to NB locally for this test (production factory
    # rewire happens in Task 1.5).
    object.__setattr__(
        model,
        "dist_families",
        {**dict(model.dist_families), Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL},
    )
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    assert "dispersion" in model.variance_params[Stat.RECEIVING_TDS]
    dispersion = model.variance_params[Stat.RECEIVING_TDS]["dispersion"]
    assert _NB_DISPERSION_CLIP[0] <= dispersion <= _NB_DISPERSION_CLIP[1]
