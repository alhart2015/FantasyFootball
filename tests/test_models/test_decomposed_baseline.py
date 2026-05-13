"""Tests for src/projections/models/decomposed_baseline.py.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from projections.models.decomposed_baseline import DecomposedBaselineModel, DecompositionSpec
from projections.schemas import Stat


def test_decomposition_spec_is_frozen() -> None:
    spec = DecompositionSpec(
        volume_stat=Stat.TARGETS,
        efficiency_label="catch_rate",
        efficiency_clip_hi=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.efficiency_clip_hi = 2.0  # type: ignore[misc]


def test_decomposition_spec_equality_by_value() -> None:
    a = DecompositionSpec(Stat.TARGETS, "catch_rate", 1.0)
    b = DecompositionSpec(Stat.TARGETS, "catch_rate", 1.0)
    c = DecompositionSpec(Stat.TARGETS, "yards_per_target", float("inf"))
    assert a == b
    assert a != c


# Synthetic gsis_ids: 4 players, 7-digit suffix matching schema \d{2}-\d{7}
_PLAYER_IDS = [f"00-{1_000_000 + pid:07d}" for pid in range(4)]


def _synthetic_wr_fit_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic WR features + weekly_stats spanning 2018-2020 for fit testing.

    Designed to be cheap and small: 3 seasons x 3 weeks x 4 players = 36 rows.
    Feature values are plausible but synthetic. Volume (targets) varies enough
    to give the volume ridge a signal; catch_rate varies independently.
    """
    rows: list[dict[str, object]] = []
    weekly_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed=0xD3C0)
    for season in (2018, 2019, 2020):
        for week in (1, 2, 3):
            for pid in range(4):
                gsis = _PLAYER_IDS[pid]
                team = ("KC", "BUF", "DAL", "PHI")[pid]
                opp = ("LV", "NE", "NYG", "WAS")[pid]
                targets_per_game_l4 = float(rng.uniform(3.0, 12.0))
                features_row: dict[str, object] = {
                    "gsis_id": gsis,
                    "season": season,
                    "week": week,
                    "team": team,
                    "opponent": opp,
                    "depth_rank": float(rng.integers(1, 4)),
                    "targets_per_game_l4": targets_per_game_l4,
                    "targets_per_game_std": float(rng.uniform(0.5, 3.0)),
                }
                rows.append(features_row)
                # Truth: targets ~ Poisson(targets_per_game_l4); receptions ~
                # Binomial(targets, 0.65); receiving_yards ~ targets x Normal(11, 3);
                # receiving_tds ~ Bernoulli(targets x 0.05).
                tgt = int(rng.poisson(targets_per_game_l4))
                rec = int(rng.binomial(max(tgt, 0), 0.65))
                yds = float(max(rec, 0) * rng.normal(11.0, 3.0))
                tds = int(rng.random() < min(tgt * 0.05, 1.0))
                weekly_rows.append(
                    {
                        "gsis_id": gsis,
                        "season": season,
                        "week": week,
                        "team": team,
                        "opponent": opp,
                        "position": "WR",
                        # QB stats (zero for WRs — required by WeeklyStatsSchema)
                        "passing_yards": 0.0,
                        "passing_tds": 0,
                        "interceptions": 0,
                        "attempts": 0,
                        "completions": 0,
                        "sacks": 0,
                        # Rushing
                        "rushing_yards": 0.0,
                        "rushing_tds": 0,
                        "carries": 0,
                        # Receiving
                        "targets": tgt,
                        "receptions": rec,
                        "receiving_yards": max(yds, -50.0),
                        "receiving_tds": tds,
                        "receiving_air_yards": float(max(tgt, 0) * 8),
                        "fumbles_lost": 0,
                    }
                )
    features = pd.DataFrame(rows)
    weekly_stats = pd.DataFrame(weekly_rows)
    return features, weekly_stats


# Column-level defaults for WR feature columns missing from the synthetic
# fixture. All values must be non-NaN (BaselineModel.fit drops rows with any
# NaN feature) and must satisfy WrFeaturesSchema range constraints.
_WR_COLUMN_DEFAULTS: dict[str, object] = {
    "target_share_l4": 0.15,
    "air_yards_share_l4": 0.15,
    "receptions_per_game_l4": 3.0,
    "receiving_yards_per_game_l4": 35.0,
    "receiving_tds_per_game_l4": 0.2,
    "rushing_attempts_per_game_l4": 0.2,
    "rushing_yards_per_game_l4": 1.5,
    "designed_rusher": False,
    "snap_pct_l4": 0.7,
    "avg_separation_std": 1.5,
    "avg_intended_air_yards_std": 8.0,
    "percent_share_intended_air_yards_std": 0.15,
    "avg_yac_above_expectation_std": 0.0,
    "implied_team_total": 24.0,
    "spread": 0.0,
    "is_home": False,
    "roof_dome": False,
    "opp_allowed_wr_fppg_l4": 30.0,
    "age": 25.0,
    "is_rookie": 0.0,
    "volume_trend_l4_minus_prior_l4": 0.0,
    "snap_pct_change_l4_vs_prior_l4": 0.0,
    "wind_speed_mph": 5.0,
    "is_high_wind": 0.0,
    "temperature_f": 65.0,
    "is_grass_surface": 0.0,
}


def test_decomposed_baseline_constructs_with_empty_decomposed_stats() -> None:
    """An empty ``decomposed_stats`` mapping should make the model behaviorally
    indistinguishable from BaselineModel on the train side (only direct ridges
    fit). Verifies the opt-in arch's empty-config case.
    """
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WrFeaturesSchema

    model = DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={},
    )
    assert model.decomposed_stats == {}
    assert model.volume_ridges == {}
    assert model.efficiency_ridges == {}


def _wr_decomp_model_receptions_only() -> DecomposedBaselineModel:
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WrFeaturesSchema

    return DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={
            Stat.RECEPTIONS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="catch_rate",
                efficiency_clip_hi=1.0,
            ),
        },
    )


def _fit_model(model: DecomposedBaselineModel) -> DecomposedBaselineModel:
    """Shared helper: build synthetic data, backfill missing columns, validate,
    and call fit. Returns the fitted model."""
    from projections.schemas import WeeklyStatsSchema

    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    return model


def test_fit_populates_direct_ridges_for_all_target_stats() -> None:
    """All target_stats (including decomposed ones) get a direct-comparator
    ridge. Decomposed stats get those plus volume + efficiency sub-models.
    See spec SS3.1.3 'fit both arms' rationale.
    """
    model = _fit_model(_wr_decomp_model_receptions_only())
    assert set(model.ridges) == set(model.target_stats)


def test_fit_populates_decomposition_sub_models_for_receptions() -> None:
    """Decomposed stats get a shared volume RidgeCV and a per-stat efficiency
    RidgeCV; both are stored in ``volume_ridges`` / ``efficiency_ridges`` keyed
    by ``volume_stat`` and the composite stat respectively. Residual stds are
    persisted in ``volume_variance`` / ``efficiency_variance``.
    """
    model = _fit_model(_wr_decomp_model_receptions_only())

    assert Stat.TARGETS in model.volume_ridges
    assert isinstance(model.volume_ridges[Stat.TARGETS], RidgeCV)
    assert Stat.RECEPTIONS in model.efficiency_ridges
    assert isinstance(model.efficiency_ridges[Stat.RECEPTIONS], RidgeCV)
    assert model.volume_variance[Stat.TARGETS] > 0.0
    assert model.efficiency_variance[Stat.RECEPTIONS] > 0.0


def test_fit_model_id_uses_decomposed_baseline_prefix() -> None:
    model = _fit_model(_wr_decomp_model_receptions_only())
    assert model.model_id.startswith("decomposed-baseline:wr:")
