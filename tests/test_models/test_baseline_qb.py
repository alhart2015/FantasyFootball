"""qb_baseline() unit tests. Mirrors test_baseline.py's WR coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.models import qb_baseline
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset, Stat


def test_qb_baseline_factory_returns_unfitted_model() -> None:
    model = qb_baseline()
    assert model.position == Position.QB
    expected_targets = {
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.INTERCEPTIONS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.FUMBLES_LOST,
    }
    assert set(model.target_stats) == expected_targets
    assert model.dist_families[Stat.PASSING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.PASSING_TDS] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.dist_families[Stat.INTERCEPTIONS] is DistributionFamily.NEGATIVE_BINOMIAL
    # Plan 3e Phase 2 attempted Student-t for *_yards stats; reverted because
    # heavy tails narrowed [p10, p90] coverage. Yards stats stay NORMAL.
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.NEGATIVE_BINOMIAL
    assert model.feature_columns


def test_qb_baseline_fit_populates_ridges_per_target_stat(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        assert hasattr(model.ridges[stat], "coef_")


def test_qb_baseline_fit_persists_feature_means(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_qb_baseline_fit_records_train_seasons(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    assert model.train_seasons == (2024, 2025)


def test_qb_baseline_fit_populates_normal_variance_params(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    """QB yards stats (PASSING_YARDS, RUSHING_YARDS) stay NORMAL. Plan 3e
    Phase 2 attempted STUDENT_T for *_yards but was reverted because heavy
    tails structurally narrow [p10, p90] coverage. Plan 3e Phase 3 attempted
    per-tertile bucketing and was also reverted; variance_params shape is
    back to a scalar ``std`` per stat."""
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    for stat in (Stat.PASSING_YARDS, Stat.RUSHING_YARDS):
        params = model.variance_params[stat]
        assert "std" in params
        std = params["std"]
        assert isinstance(std, float)
        assert std > 0


def test_qb_baseline_fit_populates_nb_variance_params(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    """Plan 3e Phase 1: QB count stats route to NEGATIVE_BINOMIAL.
    Phase 3 bucketing was reverted; variance_params carries a scalar ``dispersion``."""
    from projections.distributions.parametric import _NB_DISPERSION_CLIP

    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    for stat in (Stat.PASSING_TDS, Stat.INTERCEPTIONS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "dispersion" in params
        dispersion = params["dispersion"]
        assert isinstance(dispersion, float)
        assert _NB_DISPERSION_CLIP[0] <= dispersion <= _NB_DISPERSION_CLIP[1]


def test_qb_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    week_features = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    assert (out["model_id"].str.startswith("baseline:qb:")).all()
    assert set(out["gsis_id"].astype(str)) == set(week_features["gsis_id"].astype(str))
    assert (out["ruleset"] == "ESPN_PPR").all()
    assert (out["position"] == "QB").all()


def test_qb_predict_distribution_p10_le_p50_le_p90(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    week_features = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_qb_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features_qb: pd.DataFrame,
    baseline_weekly_stats_qb: pd.DataFrame,
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)

    artifact = tmp_path / "qb-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel

    loaded = BaselineModel.load(artifact)
    assert loaded.position == Position.QB
    assert loaded.model_id == model.model_id

    week = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_qb_model_id_format(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "qb"
    assert len(parts[2]) == 8  # code_hash
    assert "-" in parts[3]


def test_qb_unfitted_model_id_raises() -> None:
    model = qb_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")


def test_qb_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)

    week = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ].copy()
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_qb_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    empty = baseline_features_qb.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)
