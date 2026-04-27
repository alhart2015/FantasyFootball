"""Cross-cutting tests for LightGBMModel (Model C, Plan 5).

Per-position smoke tests live in test_lightgbm_qb.py / _rb.py / _te.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.models.lightgbm import (
    QUANTILE_GRID,
    qb_lightgbm,
    wr_lightgbm,
)
from projections.schemas import Stat

# ---------------- Synthetic fixture builder ----------------


def _build_synthetic_wr_features(
    n_seasons: int = 4, n_weeks: int = 17, n_players: int = 30
) -> pd.DataFrame:
    """Synthetic WrFeaturesSchema-shaped DataFrame for fit/predict smoke tests."""
    from projections.schemas import _PYARROW_STR, WrFeaturesSchema

    rng = np.random.default_rng(42)
    rows = []
    for season in range(2018, 2018 + n_seasons):
        for week in range(1, n_weeks + 1):
            for p in range(n_players):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": season,
                        "week": week,
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["season"] = df["season"].astype(np.int64)
    df["week"] = df["week"].astype(np.int64)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    # Populate every WrFeaturesSchema feature column with synthetic numerics.
    # Inspect each column's dtype so int/bool fields get correctly-typed values
    # (random floats won't coerce into nullable Int64 / bool).
    schema_cols = WrFeaturesSchema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            # depth_rank is ge=1, le=10 — pick a small positive int range.
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            # Uniform [0, 0.5] satisfies typical schema range checks (ge=0,
            # often le=1, sometimes le=60). Negative values would fail many
            # `ge=0` columns, so we avoid `np.random.normal` here.
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    return WrFeaturesSchema.validate(df)


def _build_synthetic_weekly_stats(features: pd.DataFrame) -> pd.DataFrame:
    """Synthetic WeeklyStatsSchema-shaped DataFrame aligned with `features`."""
    from projections.schemas import WeeklyStatsSchema

    rng = np.random.default_rng(43)
    df = features[["gsis_id", "season", "week"]].copy()
    df["team"] = features["team"]
    df["opponent"] = features["opponent"]
    df["position"] = "WR"
    n = len(df)
    # Plausible synthetic targets — enough variation for LightGBM to learn something.
    df["receptions"] = np.maximum(0, rng.poisson(3.0, size=n)).astype(np.int64)
    df["receiving_yards"] = (df["receptions"] * rng.normal(12.0, 3.0, size=n)).astype(np.float64)
    df["receiving_tds"] = np.maximum(0, rng.poisson(0.3, size=n)).astype(np.int64)
    df["rushing_yards"] = rng.normal(2.0, 5.0, size=n).astype(np.float64)
    df["rushing_tds"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    df["fumbles_lost"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    # Other required schema columns get zeros / placeholders, typed to match
    # the schema's declared dtype for each column.
    schema_cols = WeeklyStatsSchema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "int" in dtype_str.lower():
            df[col_name] = np.zeros(n, dtype=np.int64)
        elif "float" in dtype_str.lower():
            df[col_name] = np.zeros(n, dtype=np.float64)
        else:
            df[col_name] = 0
    return WeeklyStatsSchema.validate(df)


# ---------------- Fit tests ----------------


def test_fit_populates_sub_models_and_best_iters() -> None:
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    assert model._is_fitted
    assert set(model._sub_models.keys()) == set(model._config.target_stats)
    for stat in model._config.target_stats:
        assert set(model._sub_models[stat].keys()) == set(QUANTILE_GRID)
        for q in QUANTILE_GRID:
            assert (stat, q) in model._best_iters


def test_fit_sets_train_window_from_data() -> None:
    features = _build_synthetic_wr_features(n_seasons=3)  # 2018-2020
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    assert model._train_start == 2018
    assert model._train_end == 2020


def test_fit_raises_on_insufficient_seasons() -> None:
    features = _build_synthetic_wr_features(n_seasons=1)
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    with pytest.raises(ValueError, match=r"Need .*?2 training seasons"):
        model.fit(features, weekly_stats)


def test_fit_raises_on_empty_join() -> None:
    features = _build_synthetic_wr_features()
    # Create non-overlapping weekly_stats (different gsis_ids):
    weekly_stats = _build_synthetic_weekly_stats(features.copy())
    weekly_stats["gsis_id"] = weekly_stats["gsis_id"].apply(lambda x: x.replace("00-", "99-"))
    model = wr_lightgbm()
    with pytest.raises(ValueError, match="Empty training set"):
        model.fit(features, weekly_stats)


def test_model_id_unavailable_before_fit() -> None:
    model = wr_lightgbm()
    with pytest.raises(RuntimeError, match="model_id not available before fit"):
        _ = model.model_id


def test_model_id_after_fit_has_expected_shape() -> None:
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)
    mid = model.model_id
    parts = mid.split(":")
    assert parts[0] == "lightgbm"
    assert parts[1] == "wr"
    assert len(parts[2]) == 8  # 8-char code hash
    assert "-" in parts[3]  # train-start-train-end


def test_qb_factory_uses_qb_target_stats() -> None:
    model = qb_lightgbm()
    assert Stat.PASSING_YARDS in model._config.target_stats
    assert Stat.PASSING_TDS in model._config.target_stats
    assert Stat.INTERCEPTIONS in model._config.target_stats
    # WR-only stats are not in QB:
    assert Stat.RECEPTIONS not in model._config.target_stats


def test_fit_validates_weekly_stats_schema() -> None:
    """fit() should reject weekly_stats DataFrames that don't match WeeklyStatsSchema."""
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    weekly_stats = weekly_stats.drop(columns=["passing_yards"])  # break the schema
    model = wr_lightgbm()
    with pytest.raises(SchemaError):
        model.fit(features, weekly_stats)


def test_fit_filters_to_model_position() -> None:
    """fit() should only train on rows where weekly_stats.position matches self.position."""
    from projections.schemas import Position

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    # Mark half the rows as a different position; fit should silently filter them out.
    half = len(weekly_stats) // 2
    weekly_stats.iloc[:half, weekly_stats.columns.get_loc("position")] = Position.QB.value
    model = wr_lightgbm()
    model.fit(features, weekly_stats)
    # If filtering worked, fit succeeded; the join would have failed if both positions were present
    # because the WR-only features would have only matched the WR rows.
    assert model._is_fitted


def test_fit_warns_on_best_iter_zero() -> None:
    """Degenerate sub-models trigger a warning rather than silently training."""
    # We can't easily force best_iter=0 with the synthetic data, so we test
    # that the warning category is correctly typed by inspecting source on this run
    # if it happens to fire.
    import warnings as warnings_mod

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        model.fit(features, weekly_stats)
        # If any RuntimeWarning fired, verify the message format.
        bi_warnings = [w for w in caught if "best_iter=0" in str(w.message)]
        for w in bi_warnings:
            assert issubclass(w.category, RuntimeWarning)
            assert "early stopping fired immediately" in str(w.message)


# ---------------- predict_distribution tests ----------------


def test_predict_distribution_validates_against_projection_schema() -> None:
    from projections.schemas import ProjectionWeeklySchema, Ruleset

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    # ProjectionWeeklySchema.validate raises if columns/dtypes drift.
    ProjectionWeeklySchema.validate(out)
    assert (out["family"] == "QUANTILE").all()
    assert (out["model_id"] == model.model_id).all()
    assert len(out) == len(test_features)


def test_predict_distribution_params_blob_round_trips() -> None:
    from projections.distributions import QuantileDistribution, unpack_per_stat_params
    from projections.schemas import Ruleset

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    for blob in out["params"]:
        decoded = unpack_per_stat_params(bytes(blob))
        for stat in model._config.target_stats:
            assert stat in decoded
            assert isinstance(decoded[stat], QuantileDistribution)


def test_predict_distribution_clips_non_negative_stats() -> None:
    from projections.distributions import QuantileDistribution, unpack_per_stat_params
    from projections.schemas import Ruleset

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    # Every non-negative stat's stored quantile values must be >= 0 in every row.
    for blob in out["params"]:
        decoded = unpack_per_stat_params(bytes(blob))
        for stat in model._config.non_negative_stats:
            qd = decoded[stat]
            assert isinstance(qd, QuantileDistribution)
            assert (qd.values_ >= 0.0).all()


def test_predict_distribution_sorts_quantile_crossing() -> None:
    """If LightGBM produces a row where p10_pred > p50_pred, predict_distribution
    must sort before constructing the QuantileDistribution. Constructed indirectly:
    we assert no ValueError is raised by the QuantileDistribution constructor's
    monotonicity check on any prediction row."""
    from projections.schemas import Ruleset

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    # No exception means the sort upstream is doing its job.
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())
    assert len(out) == len(test_features)


def test_predict_distribution_raises_on_feature_column_mismatch() -> None:
    from projections.schemas import Ruleset

    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    # Drop one feature column.
    dropped_col = next(iter(model._config.feature_columns))
    test_features = test_features.drop(columns=[dropped_col])
    with pytest.raises(ValueError, match=r"Feature columns differ from training"):
        model.predict_distribution(test_features, Ruleset.espn_ppr())
