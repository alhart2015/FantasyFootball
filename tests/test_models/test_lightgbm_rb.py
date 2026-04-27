"""Per-position smoke for LightGBMModel — RB. Plan 5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models.lightgbm import rb_lightgbm
from projections.schemas import (
    _PYARROW_STR,
    ProjectionWeeklySchema,
    RbFeaturesSchema,
    Ruleset,
    WeeklyStatsSchema,
)


def _build_synthetic_rb_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic RbFeaturesSchema + WeeklyStatsSchema fixtures aligned by
    (gsis_id, season, week). Mirrors the dtype-aware fill pattern used in the
    WR helper in `test_lightgbm.py`: every schema column is populated using the
    column's declared dtype so pandera validation accepts the result."""
    rng = np.random.default_rng(42)
    rows = []
    for season in range(2018, 2022):  # 4 seasons
        for week in range(1, 18):
            for p in range(20):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    schema_cols = RbFeaturesSchema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    features = RbFeaturesSchema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "RB"
    n = len(ws)
    # RB-relevant target stats with plausible magnitudes. Clip yards to the
    # WeeklyStatsSchema bounds (rushing_yards/receiving_yards: ge=-50, le=400)
    # so synthetic-tail draws don't fail validation.
    ws["rushing_yards"] = np.clip(rng.normal(40.0, 30.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.4, size=n)).astype(np.int64)
    ws["receptions"] = np.maximum(0, rng.poisson(2.0, size=n)).astype(np.int64)
    ws["receiving_yards"] = np.clip(rng.normal(15.0, 10.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["receiving_tds"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    # Passing stats zeroed for RB.
    ws["passing_yards"] = np.zeros(n, dtype=np.float64)
    ws["passing_tds"] = np.zeros(n, dtype=np.int64)
    ws["interceptions"] = np.zeros(n, dtype=np.int64)
    schema_cols_ws = WeeklyStatsSchema.to_schema().columns
    for col_name, col in schema_cols_ws.items():
        if col_name in ws.columns:
            continue
        dtype_str = str(col.dtype)
        if "int" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.int64)
        elif "float" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.float64)
        else:
            ws[col_name] = 0
    return features, WeeklyStatsSchema.validate(ws)


def test_rb_lightgbm_fit_predict_smoke() -> None:
    features, weekly_stats = _build_synthetic_rb_data()
    model = rb_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(10).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    ProjectionWeeklySchema.validate(out)
    assert len(out) == 10
    assert (out["family"] == "QUANTILE").all()
    assert (out["position"] == "RB").all()
