"""Cross-cutting tests for LightGBMModel (Model C, Plan 5).

Per-position smoke tests live in test_lightgbm_qb.py / _rb.py / _te.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
