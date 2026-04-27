"""te_baseline() unit tests. Mirrors test_baseline.py's WR coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from projections.models import te_baseline
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset, Stat


def test_te_baseline_factory_returns_unfitted_model() -> None:
    model = te_baseline()
    assert model.position == Position.TE
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
    # Plan 3e Phase 1+ Task 2.6: TE RUSHING_YARDS reverted from STUDENT_T to
    # NORMAL after Phase 2 retrain produced a degenerate Student-t fit
    # (mean ~0.10 yard, df snapped to floor).
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.feature_columns


def test_te_baseline_factory_includes_rushing_features() -> None:
    """Phase 1 added rushing_*_per_game_l4 to TeFeaturesSchema; te_baseline's
    _TE_FEATURE_COLUMNS must reference them."""
    model = te_baseline()
    assert "rushing_attempts_per_game_l4" in model.feature_columns
    assert "rushing_yards_per_game_l4" in model.feature_columns


def test_te_baseline_fit_populates_ridges_per_target_stat(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        assert hasattr(model.ridges[stat], "coef_")


def test_te_baseline_fit_persists_feature_means(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_te_baseline_fit_records_train_seasons(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    assert model.train_seasons == (2024, 2025)


def test_te_baseline_fit_populates_student_t_variance_params(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    """Plan 3e Phase 2: TE RECEIVING_YARDS routes to STUDENT_T with (scale, df).
    TE RUSHING_YARDS was reverted to NORMAL in Task 2.6 (degenerate fit)."""
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    params = model.variance_params[Stat.RECEIVING_YARDS]
    assert "scale" in params
    assert "df" in params
    assert params["scale"] > 0
    assert params["df"] > 2.0


def test_te_baseline_fit_populates_normal_variance_params(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    """Plan 3e Phase 1+ Task 2.6: TE RUSHING_YARDS reverted from STUDENT_T to
    NORMAL after Phase 2 retrain produced a degenerate Student-t fit."""
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    params = model.variance_params[Stat.RUSHING_YARDS]
    assert "std" in params
    assert params["std"] > 0


def test_te_baseline_fit_populates_gamma_variance_params(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    # RECEPTIONS stays Gamma; count stats moved to NEGATIVE_BINOMIAL in Plan 3e Phase 1.
    for stat in (Stat.RECEPTIONS,):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_te_baseline_fit_populates_nb_variance_params(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    """Plan 3e Phase 1: TE count stats route to NEGATIVE_BINOMIAL."""
    from projections.models.baseline import _NB_DISPERSION_CLIP

    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    for stat in (Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "dispersion" in params
        assert _NB_DISPERSION_CLIP[0] <= params["dispersion"] <= _NB_DISPERSION_CLIP[1]


def test_te_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    week_features = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    assert (out["model_id"].str.startswith("baseline:te:")).all()
    assert set(out["gsis_id"].astype(str)) == set(week_features["gsis_id"].astype(str))
    assert (out["ruleset"] == "ESPN_PPR").all()
    assert (out["position"] == "TE").all()


def test_te_predict_distribution_p10_le_p50_le_p90(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    week_features = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_te_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features_te: pd.DataFrame,
    baseline_weekly_stats_te: pd.DataFrame,
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)

    artifact = tmp_path / "te-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel

    loaded = BaselineModel.load(artifact)
    assert loaded.position == Position.TE
    assert loaded.model_id == model.model_id

    week = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_te_model_id_format(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "te"
    assert len(parts[2]) == 8  # code_hash
    assert "-" in parts[3]


def test_te_unfitted_model_id_raises() -> None:
    model = te_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")


def test_te_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)

    week = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ].copy()
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_te_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    empty = baseline_features_te.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)


def test_te_baseline_taysom_hill_row_predicts_nonzero_rushing_yards_mean(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    """The Phase 1 TE rushing extension and the rushing-TE row in the fixture
    (gsis_id ending in "3") together imply a non-zero predicted rushing_yards
    mean. If the model collapses to zero, the rushing features aren't
    flowing through fit->predict correctly."""
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)

    rushing_te_id = next(
        g for g in baseline_features_te["gsis_id"].astype(str).unique() if g.endswith("3")
    )
    week = baseline_features_te[
        (baseline_features_te["season"] == 2025)
        & (baseline_features_te["week"] == 4)
        & (baseline_features_te["gsis_id"].astype(str) == rushing_te_id)
    ]
    if week.empty:
        pytest.skip("rushing-TE not present in 2025 wk4 fixture slice")

    stat_dists = model.build_stat_distributions(week)
    rushing_yd_mu = stat_dists[0][Stat.RUSHING_YARDS].mean()
    assert rushing_yd_mu > 0.5, (
        f"Taysom-Hill-shape TE rushing_yards predicted mean is {rushing_yd_mu:.3f}; "
        "expected > 0.5. Verify TE rushing features made it through fit->predict."
    )
