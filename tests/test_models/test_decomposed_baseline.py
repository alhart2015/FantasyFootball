"""Tests for src/projections/models/decomposed_baseline.py.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from projections.distributions import (
    FrozenSampledDistribution,
    QuantileDistribution,
    unpack_per_stat_params,
)
from projections.distributions.parametric import (
    ParametricNegativeBinomial,
    ParametricNormal,
)
from projections.models.decomposed_baseline import DecomposedBaselineModel, DecompositionSpec
from projections.schemas import DistributionFamily, Ruleset, Stat, WeeklyStatsSchema


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


def test_fit_with_empty_decomposed_stats_produces_no_volume_or_efficiency_ridges() -> None:
    """Empty decomposed_stats: fit() populates direct ridges (parent path) but
    leaves volume_ridges / efficiency_ridges empty. Behaviorally indistinguishable
    from BaselineModel on the train side."""
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WeeklyStatsSchema, WrFeaturesSchema

    model = DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={},
    )
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    assert set(model.ridges) == set(_WR_TARGET_STATS)
    assert model.volume_ridges == {}
    assert model.efficiency_ridges == {}
    assert model.volume_variance == {}
    assert model.efficiency_variance == {}
    # model_id still uses decomposed-baseline prefix (class-level identity is
    # decoupled from the runtime decomposed_stats config).
    assert model.model_id.startswith("decomposed-baseline:wr:")


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


def test_fit_raises_when_no_positive_volume_rows() -> None:
    """When all training rows have volume_stat == 0, the efficiency factor
    cannot be fitted and a clear ValueError is raised. Guards spec §3.1.3
    structurally."""
    from projections.schemas import WeeklyStatsSchema

    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    # Zero out all targets so the efficiency mask is always False.
    weekly_stats["targets"] = 0
    # Derived stats also drop to 0 since targets=0 forces all receiving stats
    # to zero in any plausible fixture; tighten the fixture to match.
    weekly_stats["receptions"] = 0
    weekly_stats["receiving_yards"] = 0.0
    weekly_stats["receiving_tds"] = 0
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    with pytest.raises(ValueError, match="no training rows with"):
        model.fit(features, weekly_stats)


def test_build_stat_distributions_emits_frozen_sampled_for_decomposed_stat() -> None:
    """build_stat_distributions emits FrozenSampledDistribution for the
    decomposed stat (carrying the per-row composed sample array) and
    parametric distributions for non-decomposed stats (unchanged path).
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    per_row = model.build_stat_distributions(features)
    assert len(per_row) == len(features)

    row0 = per_row[0]
    # Receptions is decomposed → FrozenSampledDistribution carrying 10_000 samples.
    rec_dist_row0 = row0[Stat.RECEPTIONS]
    assert isinstance(rec_dist_row0, FrozenSampledDistribution)
    assert len(rec_dist_row0.samples) == 10_000
    # Non-decomposed stats keep parametric forms per _WR_DIST_FAMILIES.
    assert isinstance(row0[Stat.RECEIVING_YARDS], ParametricNormal)
    assert isinstance(row0[Stat.RECEIVING_TDS], ParametricNegativeBinomial)
    assert isinstance(row0[Stat.RUSHING_YARDS], ParametricNormal)
    assert isinstance(row0[Stat.RUSHING_TDS], ParametricNegativeBinomial)
    assert isinstance(row0[Stat.FUMBLES_LOST], ParametricNegativeBinomial)


def _synthetic_wr_fit_inputs_low_eff_variance() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variant of _synthetic_wr_fit_inputs where yards_per_target has very low
    variance (Normal(10.0, 0.1) instead of Normal(11.0, 3.0)).

    Used by the cross-stat coherence test to ensure efficiency variance is
    small relative to volume variance, so the shared volume draw dominates the
    composed products and correlation rho > 0.5 is analytically guaranteed.
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
                tgt = int(rng.poisson(targets_per_game_l4))
                rec = int(rng.binomial(max(tgt, 0), 0.65))
                # Near-constant yards_per_target (low efficiency variance) so the
                # shared volume draw dominates and corr(rec_composed, yds_composed) > 0.5.
                yds = float(max(tgt, 0) * rng.normal(10.0, 0.1))
                tds = int(rng.random() < min(tgt * 0.05, 1.0))
                weekly_rows.append(
                    {
                        "gsis_id": gsis,
                        "season": season,
                        "week": week,
                        "team": team,
                        "opponent": opp,
                        "position": "WR",
                        "passing_yards": 0.0,
                        "passing_tds": 0,
                        "interceptions": 0,
                        "attempts": 0,
                        "completions": 0,
                        "sacks": 0,
                        "rushing_yards": 0.0,
                        "rushing_tds": 0,
                        "carries": 0,
                        "targets": tgt,
                        "receptions": rec,
                        "receiving_yards": max(yds, -50.0),
                        "receiving_tds": tds,
                        "receiving_air_yards": float(max(tgt, 0) * 8),
                        "fumbles_lost": 0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weekly_rows)


def test_within_row_cross_stat_coherence_two_stat_synthetic_config() -> None:
    """Architectural guarantee: two decomposed stats sharing the same
    volume_stat produce FrozenSampledDistribution instances with strongly
    correlated samples within a row (Pearson rho > 0.5).

    Uses a fixture with near-constant yards_per_target (efficiency variance
    << volume variance) so the shared volume draw dominates the composition and
    the cross-stat correlation is analytically guaranteed: with sigma_eff_yds ~
    0.10 and sigma_vol ~ 2.5, the theoretical rho is ~0.55.

    v1 production decomposes only receptions, but this test exercises the
    coherence code path that v2 will exercise in production. If this test
    fails, the cross-stat coherence guarantee is broken.
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
        decomposed_stats={
            Stat.RECEPTIONS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="catch_rate",
                efficiency_clip_hi=1.0,
            ),
            Stat.RECEIVING_YARDS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="yards_per_target",
                efficiency_clip_hi=float("inf"),
            ),
        },
    )
    features, weekly_stats = _synthetic_wr_fit_inputs_low_eff_variance()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    per_row = model.build_stat_distributions(features)
    row0 = per_row[0]
    rec_dist = row0[Stat.RECEPTIONS]
    yds_dist = row0[Stat.RECEIVING_YARDS]
    assert isinstance(rec_dist, FrozenSampledDistribution)
    assert isinstance(yds_dist, FrozenSampledDistribution)
    # Both compose against the same per-row volume draw → strong positive
    # element-wise correlation (volume variance >> efficiency variance ensures
    # the shared draw dominates).
    rho = float(np.corrcoef(rec_dist.samples, yds_dist.samples)[0, 1])
    assert rho > 0.5, f"expected within-row Pearson rho > 0.5, got {rho:.3f}"


def test_build_stat_distributions_is_deterministic_per_row() -> None:
    """Same model state + same features → same sample arrays. Per-row seed
    derivation via derive_row_seed makes this byte-stable.
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    per_row_a = model.build_stat_distributions(features)
    per_row_b = model.build_stat_distributions(features)
    for i in range(len(features)):
        rec_a = per_row_a[i][Stat.RECEPTIONS]
        rec_b = per_row_b[i][Stat.RECEPTIONS]
        assert isinstance(rec_a, FrozenSampledDistribution)
        assert isinstance(rec_b, FrozenSampledDistribution)
        assert np.array_equal(rec_a.samples, rec_b.samples)


def test_predict_distribution_round_trip_through_quantile_codec() -> None:
    """predict_distribution validates against ProjectionWeeklySchema; the
    params blob decodes back with the decomposed stat as a QuantileDistribution
    (the persisted form), non-decomposed stats as their parametric form.
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    ruleset = Ruleset.espn_ppr()
    pred = model.predict_distribution(features, ruleset=ruleset)
    # Schema validation already happens inside predict_distribution; if we get
    # here without raising, the frame is schema-conformant.
    assert len(pred) == len(features)
    assert (pred["family"] == DistributionFamily.SAMPLED_SUMMARY.value).all()

    # Decode the first row's params blob.
    blob = pred.iloc[0]["params"]
    decoded = unpack_per_stat_params(blob)
    assert isinstance(decoded[Stat.RECEPTIONS], QuantileDistribution)
    assert isinstance(decoded[Stat.RECEIVING_YARDS], ParametricNormal)
    assert isinstance(decoded[Stat.RECEIVING_TDS], ParametricNegativeBinomial)


def test_predict_distribution_mean_p10_p90_finite() -> None:
    """The summarized per-row mean/p10/p50/p90 should be finite and ordered
    p10 < p50 < p90 for nontrivial predictions."""
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = _WR_COLUMN_DEFAULTS.get(col, 0.0)
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    for col in ("mean", "p10", "p50", "p90"):
        assert pred[col].notna().all()
        assert np.isfinite(pred[col]).all()
    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()


def test_persistable_dists_handles_all_zero_samples_gracefully() -> None:
    """If a per-row composed sample array is all zero (e.g., zero predicted
    volume on a deep WR4), the quantile values are all zero. QuantileDistribution
    accepts non-decreasing values (equal is allowed); this test guards against
    a regression where _PERSISTED_QUANTILES is changed to require strictly
    increasing values.
    """
    from projections.models.decomposed_baseline import (
        _PERSISTED_QUANTILES,
        DecomposedBaselineModel,
    )

    zero_samples = np.zeros(10_000, dtype=np.float64)
    frozen = FrozenSampledDistribution(samples=zero_samples)

    # Minimal model just to expose _persistable_dists_for_packing as bound.
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
        decomposed_stats={
            Stat.RECEPTIONS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="catch_rate",
                efficiency_clip_hi=1.0,
            ),
        },
    )
    out = model._persistable_dists_for_packing({Stat.RECEPTIONS: frozen})
    quant = out[Stat.RECEPTIONS]
    assert isinstance(quant, QuantileDistribution)
    assert np.array_equal(quant.values_, np.zeros_like(_PERSISTED_QUANTILES))
