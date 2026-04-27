"""Cross-position smoke for LightGBMModel — Plan 5.

Single fit + predict for each of the 4 positions on synthetic fixtures, driven
by `POSITION_DISPATCH[position].factories["lightgbm"]`. Catches regressions
where one position breaks while others pass — the per-position smoke files
(`test_lightgbm_qb.py` etc.) cover each position individually; this file
verifies the dispatch table itself wires every position to a working factory.

Fixture builder mirrors the dtype-aware pattern in the per-position smoke
files: pandera columns are populated by inspecting each column's declared
dtype so int / bool / bounded-float fields all validate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.models import POSITION_DISPATCH
from projections.schemas import (
    _PYARROW_STR,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    WeeklyStatsSchema,
)


def _build_synthetic_data(position: Position) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic features + WeeklyStatsSchema fixtures for `position`.

    Inspects the position's feature schema (via POSITION_DISPATCH) and fills
    every column by declared dtype. The same pattern is used in the per-
    position smoke files (`test_lightgbm_qb.py` etc.) — keeping the helpers
    parallel makes it easy to spot dtype regressions across positions.
    """
    rng = np.random.default_rng(42)
    feature_schema = POSITION_DISPATCH[position].feature_schema

    rows = []
    for season in range(2018, 2022):  # 4 seasons — fit() requires >= 2 train seasons
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

    # Populate every feature-schema column with synthetic numerics, choosing
    # values per-dtype so int/bool fields and `ge=0` floats both validate.
    schema_cols = feature_schema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            # depth_rank is ge=1, le=10 — small positive int range covers it.
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            # Uniform [0, 0.5] satisfies typical `ge=0` (often `le=1`) bounds.
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = position.value
    n = len(ws)
    # Plausible target stats. Clip yards to WeeklyStatsSchema bounds
    # (passing_yards: ge=-100, le=800; rushing_yards/receiving_yards: ge=-50,
    # le=400) so synthetic-tail draws don't fail validation.
    ws["passing_yards"] = np.clip(rng.normal(220.0, 60.0, size=n), -100.0, 800.0).astype(np.float64)
    ws["passing_tds"] = np.maximum(0, rng.poisson(1.5, size=n)).astype(np.int64)
    ws["interceptions"] = np.maximum(0, rng.poisson(0.7, size=n)).astype(np.int64)
    ws["rushing_yards"] = np.clip(rng.normal(20.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["receptions"] = np.maximum(0, rng.poisson(2.5, size=n)).astype(np.int64)
    ws["receiving_yards"] = np.clip(rng.normal(25.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["receiving_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.1, size=n)).astype(np.int64)

    # Fill remaining required schema columns by dtype.
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


@pytest.mark.parametrize("position", [Position.QB, Position.RB, Position.TE, Position.WR])
def test_lightgbm_fit_predict_smoke(position: Position) -> None:
    """Each position's LightGBMModel can fit and predict via POSITION_DISPATCH."""
    features, weekly_stats = _build_synthetic_data(position)
    factory = POSITION_DISPATCH[position].factories["lightgbm"]
    model = factory()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    ProjectionWeeklySchema.validate(out)
    assert len(out) == 5
    assert (out["family"] == "QUANTILE").all()
    assert (out["position"] == position.value).all()
