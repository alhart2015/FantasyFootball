"""Cross-cutting tests for LightGBMTunedModel (Model C-tuned, Plan 5b).

Per-position smokes live in test_lightgbm_tuned_smoke.py (Phase 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from projections.models.lightgbm import LGBM_DEFAULTS, wr_lightgbm
from projections.models.lightgbm_tuned import (
    LightGBMTunedModel,
    _load_tuned_params,
    qb_lightgbm_tuned,
    rb_lightgbm_tuned,
    te_lightgbm_tuned,
    wr_lightgbm_tuned,
)
from projections.schemas import Ruleset

# ---------------- Synthetic fixture (re-using the test_lightgbm.py shape) ----------------


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

    schema_cols = WrFeaturesSchema.to_schema().columns
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
    return WrFeaturesSchema.validate(df)


def _build_synthetic_wr_weekly_stats(features: pd.DataFrame) -> pd.DataFrame:
    """WeeklyStatsSchema-shaped truth for the synthetic WR features.

    Note: ``WeeklyStatsSchema`` declares ``*_yards`` columns as float64 (sacks
    can produce negative yardage so they're not integer-pure). Integer counters
    (receptions/targets/tds/etc.) are int64. We populate each column with a
    dtype that matches the schema so ``WeeklyStatsSchema.validate`` accepts it.
    """
    from projections.schemas import WeeklyStatsSchema

    rng = np.random.default_rng(43)
    n = len(features)
    df = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    df["position"] = "WR"
    df["receptions"] = rng.integers(0, 8, size=n).astype(np.int64)
    df["targets"] = (df["receptions"] + rng.integers(0, 4, size=n)).astype(np.int64)
    df["receiving_yards"] = rng.integers(0, 100, size=n).astype(np.float64)
    df["receiving_tds"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["receiving_air_yards"] = rng.integers(0, 80, size=n).astype(np.float64)
    df["carries"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["rushing_yards"] = rng.integers(0, 20, size=n).astype(np.float64)
    df["rushing_tds"] = rng.integers(0, 1, size=n).astype(np.int64)
    df["passing_yards"] = np.zeros(n, dtype=np.float64)
    df["passing_tds"] = np.zeros(n, dtype=np.int64)
    df["interceptions"] = np.zeros(n, dtype=np.int64)
    df["attempts"] = np.zeros(n, dtype=np.int64)
    df["completions"] = np.zeros(n, dtype=np.int64)
    df["sacks"] = np.zeros(n, dtype=np.int64)
    df["fumbles_lost"] = rng.integers(0, 1, size=n).astype(np.int64)
    return WeeklyStatsSchema.validate(df)


# ---------------- Tests ----------------


def test_factories_construct() -> None:
    for factory in (qb_lightgbm_tuned, rb_lightgbm_tuned, te_lightgbm_tuned, wr_lightgbm_tuned):
        m = factory()
        assert isinstance(m, LightGBMTunedModel)


def test_seeded_json_matches_lgbm_defaults() -> None:
    """Phase 0's seeded JSON contains LGBM_DEFAULTS values verbatim."""
    from projections.models.lightgbm_tuned import _TUNED_AXES, _TUNED_PARAMS_PATH

    raw = json.loads(_TUNED_PARAMS_PATH.read_text())
    for pos_key in ("qb", "rb", "te", "wr"):
        for stat_key, axis_map in raw[pos_key].items():
            assert set(axis_map.keys()) == _TUNED_AXES, (
                f"{pos_key}/{stat_key}: axes {sorted(axis_map.keys())} "
                f"differ from {sorted(_TUNED_AXES)}"
            )
            for axis_name, value in axis_map.items():
                assert value == LGBM_DEFAULTS[axis_name], (
                    f"{pos_key}/{stat_key}/{axis_name}: seeded JSON value "
                    f"{value} differs from LGBM_DEFAULTS value "
                    f"{LGBM_DEFAULTS[axis_name]}"
                )


def test_predictions_bit_exact_vs_untuned_under_seeded_json() -> None:
    """Phase 0 exit criterion: with seeded JSON, tuned predictions equal untuned."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    untuned = wr_lightgbm()
    tuned = wr_lightgbm_tuned()

    untuned.fit(features, weekly)
    tuned.fit(features, weekly)

    pred_untuned = untuned.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    pred_tuned = tuned.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    # Compare the numeric scoring columns. model_id, generated_at, and family
    # differ by design (different prefix / fresh timestamp / same family but
    # checked separately).
    for col in ("mean", "p10", "p50", "p90"):
        np.testing.assert_array_equal(
            pred_untuned[col].to_numpy(),
            pred_tuned[col].to_numpy(),
            err_msg=f"column {col!r} differs between untuned and tuned",
        )


def test_model_id_uses_tuned_prefix() -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    tuned = wr_lightgbm_tuned()
    tuned.fit(features, weekly)
    assert tuned.model_id.startswith("lightgbm-tuned:wr:")


def test_code_hash_differs_from_untuned() -> None:
    """The tuned subclass hashes a different file set, so model_id differs."""
    untuned = wr_lightgbm()
    tuned = wr_lightgbm_tuned()
    assert untuned.code_hash != tuned.code_hash


def test_save_load_round_trip(tmp_path: Path) -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    original = wr_lightgbm_tuned()
    original.fit(features, weekly)
    artifact = tmp_path / "tuned.joblib"
    original.save(artifact)

    loaded = LightGBMTunedModel.load(artifact)
    assert loaded.model_id == original.model_id

    pred_orig = original.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    pred_loaded = loaded.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    for col in ("mean", "p10", "p50", "p90"):
        np.testing.assert_array_equal(pred_orig[col].to_numpy(), pred_loaded[col].to_numpy())


def test_missing_position_in_json_raises(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"qb": {}}))  # missing rb/te/wr
    _load_tuned_params.cache_clear()
    with pytest.raises(ValueError, match="top-level keys must be"):
        _load_tuned_params(bad_json)


def test_unknown_axis_in_json_raises(tmp_path: Path) -> None:
    payload: dict[str, Any] = {pos: {} for pos in ("qb", "rb", "te", "wr")}
    payload["qb"]["passing_yards"] = {"learning_rate": 0.05, "bogus_axis": 1.0}
    bad_json = tmp_path / "bad_axis.json"
    bad_json.write_text(json.dumps(payload))
    _load_tuned_params.cache_clear()
    with pytest.raises(ValueError, match="unknown tuned-axis keys"):
        _load_tuned_params(bad_json)


def test_missing_stat_entry_raises(tmp_path: Path) -> None:
    """A (position, stat) gap in the JSON surfaces as KeyError at fit time."""
    payload: dict[str, dict[str, dict[str, float]]] = {pos: {} for pos in ("qb", "rb", "te", "wr")}
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(json.dumps(payload))
    _load_tuned_params.cache_clear()

    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    from projections.models.lightgbm import (
        _WR_FEATURE_COLUMNS,
        _WR_NON_NEGATIVE,
        _WR_TARGET_STATS,
        _filter_features,
        _LightGBMConfig,
    )
    from projections.schemas import Position, WrFeaturesSchema

    sparse_tuned = LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        ),
        tuned_params_path=sparse_json,
    )
    with pytest.raises(KeyError, match="missing entry for position=wr"):
        sparse_tuned.fit(features, weekly)


def test_load_tuned_params_missing_file_raises(tmp_path: Path) -> None:
    _load_tuned_params.cache_clear()
    with pytest.raises(FileNotFoundError):
        _load_tuned_params(tmp_path / "does_not_exist.json")
