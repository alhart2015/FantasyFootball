"""Cross-position smoke for EnsembleModel — Plan 6 Phase 2.

Single fit + predict for each of the 4 positions on synthetic fixtures, driven
by `POSITION_DISPATCH[position].factories["ensemble"]`. Verifies the dispatch
table itself wires every position to a working ensemble factory and that the
mixture round-trips through the codec.

Phase 2 uses static weights = 0.5; Phase 3 replaces this with optimized
weights, but the smoke here cares about wiring + round-trip, not weight values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projections.distributions import MixtureDistribution, unpack_per_stat_params
from projections.models import POSITION_DISPATCH, EnsembleModel
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    WeeklyStatsSchema,
)


def _build_synthetic_data(position: Position) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirrors test_lightgbm_nb_smoke._build_synthetic_data exactly so the
    ensemble's two child models train on identical fixtures to their direct
    smokes."""
    rng = np.random.default_rng(42)
    feature_schema = POSITION_DISPATCH[position].feature_schema

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

    schema_cols = feature_schema.to_schema().columns
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
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = position.value
    n = len(ws)
    ws["passing_yards"] = np.clip(rng.normal(220.0, 60.0, size=n), -100.0, 800.0).astype(np.float64)
    ws["passing_tds"] = np.maximum(0, rng.poisson(1.5, size=n)).astype(np.int64)
    ws["interceptions"] = np.maximum(0, rng.poisson(0.7, size=n)).astype(np.int64)
    ws["rushing_yards"] = np.clip(rng.normal(20.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["receptions"] = np.maximum(0, rng.poisson(2.5, size=n)).astype(np.int64)
    ws["receiving_yards"] = np.clip(rng.normal(25.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["receiving_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.1, size=n)).astype(np.int64)

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
def test_ensemble_fit_predict_smoke(position: Position) -> None:
    """Each position's EnsembleModel fits two children, predicts MIXED rows,
    and round-trips MixtureDistribution per stat through the codec."""
    features, weekly_stats = _build_synthetic_data(position)
    factory = POSITION_DISPATCH[position].factories["ensemble"]
    model = factory()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    ProjectionWeeklySchema.validate(out)
    assert len(out) == 5
    assert (out["family"] == DistributionFamily.MIXED.value).all()
    assert (out["position"] == position.value).all()
    assert (out["model_id"].str.startswith("ensemble:")).all()

    # Round-trip the params blob: every per-stat distribution should be a
    # MixtureDistribution wrapping the two child distributions.
    first_row = out.iloc[0]
    decoded = unpack_per_stat_params(bytes(first_row["params"]))
    assert len(decoded) > 0
    for stat, dist in decoded.items():
        assert isinstance(dist, MixtureDistribution), (
            f"stat {stat} not MixtureDistribution: got {type(dist).__name__}"
        )
        # Phase 2 uses static 0.5 weights.
        assert dist.weight == pytest.approx(0.5, abs=1e-12), (
            f"Phase 2 expects static 0.5 weights; got {dist.weight} for stat {stat}"
        )


@pytest.mark.parametrize("position", [Position.QB, Position.RB, Position.TE, Position.WR])
def test_ensemble_model_id_format(position: Position) -> None:
    """model_id format: 'ensemble:<pos>:<8-hex>:<train_start>-<train_end>'."""
    features, weekly_stats = _build_synthetic_data(position)
    factory = POSITION_DISPATCH[position].factories["ensemble"]
    model = factory()
    with pytest.raises(RuntimeError, match="model_id"):
        _ = model.model_id  # not yet fitted

    model.fit(features, weekly_stats)
    parts = model.model_id.split(":")
    assert parts[0] == "ensemble"
    assert parts[1] == position.value.lower()
    assert len(parts[2]) == 8  # 8-char hex
    assert "-" in parts[3]  # train range


def test_ensemble_save_load_round_trip(tmp_path: Path) -> None:
    """save() then load() yields a model that predicts identically (modulo
    the generated_at timestamp)."""
    features, weekly_stats = _build_synthetic_data(Position.QB)
    factory = POSITION_DISPATCH[Position.QB].factories["ensemble"]
    model = factory()
    model.fit(features, weekly_stats)
    test_features = features[features["season"] == 2021].head(5).copy()
    pred_before = model.predict_distribution(test_features, Ruleset.espn_ppr())

    save_path = tmp_path / "ensemble.joblib"
    model.save(save_path)

    loaded = EnsembleModel.load(save_path)
    pred_after = loaded.predict_distribution(test_features, Ruleset.espn_ppr())

    pd.testing.assert_frame_equal(
        pred_before.drop(columns=["generated_at"]).reset_index(drop=True),
        pred_after.drop(columns=["generated_at"]).reset_index(drop=True),
    )
