"""Tests for the wr_ensemble_decomposed factory.

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md.

Task 1 covers factory wiring + registry. Fit/predict tests (Task 2) live in
the same file and share the synthetic-data helper below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projections.distributions import (
    MixtureDistribution,
    QuantileDistribution,
    unpack_per_stat_params,
)
from projections.distributions.parametric import ParametricNegativeBinomial
from projections.models import (
    POSITION_DISPATCH,
    DecomposedBaselineModel,
    EnsembleModel,
    LightGBMNbModel,
    wr_ensemble_decomposed,
)
from projections.models.ensemble import _EnsembleConfig
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    Stat,
    WeeklyStatsSchema,
)


def test_wr_ensemble_decomposed_returns_ensemble_with_decomposed_child_a() -> None:
    """Factory returns an EnsembleModel whose child A factory yields a
    DecomposedBaselineModel and child B factory yields an LightGBMNbModel.
    """
    model = wr_ensemble_decomposed()
    assert isinstance(model, EnsembleModel)
    assert model.position == Position.WR

    # Children are constructed by the factories on demand; instantiate to
    # verify type (not the same as the lazily-fit children inside fit()).
    config: _EnsembleConfig = model._config
    child_a = config.child_a_factory()
    child_b = config.child_b_factory()
    assert isinstance(child_a, DecomposedBaselineModel), (
        f"child A should be DecomposedBaselineModel, got {type(child_a).__name__}"
    )
    assert isinstance(child_b, LightGBMNbModel), (
        f"child B should be LightGBMNbModel, got {type(child_b).__name__}"
    )


def test_wr_ensemble_decomposed_registered_in_factories() -> None:
    """_WR_FACTORIES has an 'ensemble-decomposed' entry resolving to a freshly
    instantiated EnsembleModel with the decomposed child A wiring.
    """
    wr_dispatch = POSITION_DISPATCH[Position.WR]
    assert "ensemble-decomposed" in wr_dispatch.factories, (
        f"_WR_FACTORIES missing 'ensemble-decomposed'; available: {sorted(wr_dispatch.factories)}"
    )
    model = wr_dispatch.factories["ensemble-decomposed"]()
    assert isinstance(model, EnsembleModel)
    child_a = model._config.child_a_factory()
    assert isinstance(child_a, DecomposedBaselineModel)


def test_wr_ensemble_decomposed_in_models_all() -> None:
    """wr_ensemble_decomposed is exported via projections.models.__all__."""
    import projections.models as models_pkg

    assert "wr_ensemble_decomposed" in models_pkg.__all__, (
        f"wr_ensemble_decomposed missing from __all__; got {sorted(models_pkg.__all__)}"
    )


def _synthetic_wr_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic WR features + weekly_stats with positive `targets` per row,
    sized for ensemble fit (>=3 seasons, enough rows per season for lgb-nb).

    Mirrors test_ensemble_model_smoke._build_synthetic_data's size pattern
    (4 seasons x 17 weeks x 20 players = 1360 rows) but populates `targets`
    correlated with a feature so DecomposedBaselineModel's volume sub-model
    has signal. `receptions = Binomial(targets, 0.65)`; receiving yards/tds
    derived similarly so the truth columns are coherent.
    """
    rng = np.random.default_rng(seed=20260515)
    feature_schema = POSITION_DISPATCH[Position.WR].feature_schema

    rows: list[dict[str, object]] = []
    for season in range(2018, 2022):  # 4 seasons
        for week in range(1, 18):
            for p in range(20):
                rows.append(
                    {
                        "gsis_id": f"00-{1_000_000 + p:07d}",
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
        elif col_name == "age":
            df[col_name] = rng.uniform(22.0, 30.0, size=len(df)).astype(np.float64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "WR"
    n = len(ws)
    # Targets correlated with a feature; lambda in [3, 12] to mirror real WR
    # target rates. Decomposed-baseline's mask = (targets > 0) holds.
    target_lambda = rng.uniform(3.0, 12.0, size=n)
    targets = np.maximum(1, rng.poisson(target_lambda)).astype(np.int64)
    receptions = np.array([int(rng.binomial(t, 0.65)) for t in targets], dtype=np.int64)
    ws["targets"] = targets
    ws["receptions"] = receptions
    ws["receiving_yards"] = np.maximum(0.0, receptions * rng.normal(11.0, 3.0, size=n)).astype(
        np.float64
    )
    ws["receiving_tds"] = np.where(
        rng.uniform(0, 1, size=n) < np.minimum(targets * 0.05, 1.0), 1, 0
    ).astype(np.int64)
    ws["rushing_yards"] = np.clip(rng.normal(2.0, 4.0, size=n), 0.0, 100.0).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    # Zero-fill QB stats (required by WeeklyStatsSchema).
    ws["passing_yards"] = 0.0
    ws["passing_tds"] = np.int64(0)
    ws["interceptions"] = np.int64(0)

    # Zero-fill any remaining required columns.
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


def test_wr_ensemble_decomposed_fit_predict_round_trip(tmp_path: Path) -> None:
    """End-to-end fit + predict on synthetic WR data.

    Validates the production code path:
      DecomposedBaselineModel (child A) emits QuantileDistribution for
      RECEPTIONS at persist time; LightGBMNbModel (child B) emits
      QuantileDistribution for RECEPTIONS too (RECEPTIONS is not in
      COUNT_STATS_FOR_NB; only TDs / FUMBLES_LOST / INTERCEPTIONS are
      NB-routed). EnsembleModel wraps each per-stat dist from child A
      with the matching dist from child B inside a MixtureDistribution.
      The codec's MIXTURE branch packs each component via _pack_single —
      including the QuantileDistribution branch for RECEPTIONS.

    Additionally checks RECEIVING_TDS, which is the genuine
    `MixtureDistribution(parametric, ParametricNegativeBinomial)` cell —
    decomposed-baseline emits a ParametricNegativeBinomial for
    RECEIVING_TDS (not decomposed; falls through to BaselineModel's
    `_WR_DIST_FAMILIES[RECEIVING_TDS] = NEGATIVE_BINOMIAL`), and lgb-nb
    emits ParametricNegativeBinomial for RECEIVING_TDS (in
    COUNT_STATS_FOR_NB).
    """
    features, weekly_stats = _synthetic_wr_inputs()
    model = wr_ensemble_decomposed()
    model._config.weights_dir = tmp_path
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    ProjectionWeeklySchema.validate(out)
    assert len(out) == 5
    assert (out["family"] == DistributionFamily.MIXED.value).all()
    assert (out["position"] == "WR").all()
    assert (out["model_id"].str.startswith("ensemble:wr:")).all()

    # Round-trip a row's params blob: per-stat dists must be Mixture, and
    # for RECEPTIONS specifically, component_a must be QuantileDistribution
    # (the persistable form of the decomposed FrozenSampledDistribution).
    decoded = unpack_per_stat_params(bytes(out.iloc[0]["params"]))
    assert Stat.RECEPTIONS in decoded
    rec_dist = decoded[Stat.RECEPTIONS]
    assert isinstance(rec_dist, MixtureDistribution), (
        f"RECEPTIONS dist should be MixtureDistribution, got {type(rec_dist).__name__}"
    )
    assert isinstance(rec_dist.component_a, QuantileDistribution), (
        f"RECEPTIONS component_a should be QuantileDistribution (from "
        f"decomposed-baseline child), got {type(rec_dist.component_a).__name__}"
    )
    # Component B for RECEPTIONS is also QuantileDistribution: RECEPTIONS is
    # not in COUNT_STATS_FOR_NB (only TDs / fumbles / interceptions are
    # NB-routed); receptions falls through to lgb-nb's quantile-regression
    # branch in LightGBMNbModel.predict_distribution.
    assert isinstance(rec_dist.component_b, QuantileDistribution), (
        f"RECEPTIONS component_b should be QuantileDistribution (lgb-nb's "
        f"non-NB branch), got {type(rec_dist.component_b).__name__}"
    )

    # RECEIVING_TDS exercises the MixtureDistribution(parametric, NB) cell
    # the spec's §3.4 risk targets: decomposed-baseline falls through to
    # BaselineModel's ParametricNegativeBinomial for RECEIVING_TDS, and
    # lgb-nb emits ParametricNegativeBinomial for RECEIVING_TDS (in
    # COUNT_STATS_FOR_NB).
    assert Stat.RECEIVING_TDS in decoded
    rtd_dist = decoded[Stat.RECEIVING_TDS]
    assert isinstance(rtd_dist, MixtureDistribution)
    assert isinstance(rtd_dist.component_a, ParametricNegativeBinomial), (
        f"RECEIVING_TDS component_a should be ParametricNegativeBinomial, "
        f"got {type(rtd_dist.component_a).__name__}"
    )
    assert isinstance(rtd_dist.component_b, ParametricNegativeBinomial), (
        f"RECEIVING_TDS component_b should be ParametricNegativeBinomial, "
        f"got {type(rtd_dist.component_b).__name__}"
    )


def test_wr_ensemble_decomposed_code_hash_differs_from_wr_ensemble(
    tmp_path: Path,
) -> None:
    """The two ensembles must have distinct code_hash + model_id post-fit
    because their child A code_hashes differ. This separates their
    data/ensemble_weights/ artifacts cleanly.
    """
    from projections.models import wr_ensemble
    from projections.models.ensemble import _weights_artifact_path

    features, weekly_stats = _synthetic_wr_inputs()

    ensemble_standard = wr_ensemble()
    ensemble_standard._config.weights_dir = tmp_path / "standard"
    ensemble_standard.fit(features, weekly_stats)

    ensemble_decomposed = wr_ensemble_decomposed()
    ensemble_decomposed._config.weights_dir = tmp_path / "decomposed"
    ensemble_decomposed.fit(features, weekly_stats)

    assert ensemble_standard.code_hash != ensemble_decomposed.code_hash, (
        "ensembles with different child A factories must have different "
        f"code_hash; got identical hash {ensemble_standard.code_hash}"
    )
    assert ensemble_standard.model_id != ensemble_decomposed.model_id

    # Sanity: weight artifacts land at distinct paths (model_id contains hash).
    standard_path = _weights_artifact_path(
        ensemble_standard._config.weights_dir, ensemble_standard.model_id
    )
    decomposed_path = _weights_artifact_path(
        ensemble_decomposed._config.weights_dir, ensemble_decomposed.model_id
    )
    assert standard_path != decomposed_path
    assert standard_path.exists()
    assert decomposed_path.exists()


def test_mixture_of_quantile_and_negative_binomial_tail_quantiles_are_finite() -> None:
    """Direct construction of MixtureDistribution(QuantileDistribution,
    ParametricNegativeBinomial). Verifies the mixture's quantile lookups
    return finite values across q in {0.10, 0.50, 0.90, 0.99}.

    This is the code path that decomposed-baseline-as-child-A introduces
    to production. If this fails, the fix scope is mixture._bracket_for_components
    (likely needs to query QuantileDistribution components at q in [0.05, 0.95]
    rather than wider tails since the persisted knots cap at q=0.95).
    """
    quantile_dist = QuantileDistribution(
        quantiles=np.arange(0.05, 0.96, 0.05),
        values=np.linspace(0.0, 10.0, 19),
    )
    nb_dist = ParametricNegativeBinomial(mean=5.0, dispersion=2.0)
    mix = MixtureDistribution(component_a=quantile_dist, component_b=nb_dist, weight=0.5)

    for q in (0.10, 0.50, 0.90, 0.99):
        value = mix.quantile(q)
        assert np.isfinite(value), (
            f"MixtureDistribution(Quantile, NB).quantile({q}) returned non-finite value: {value}"
        )

    # Mean + std are computed via sampling under MixtureDistribution;
    # both should be finite.
    assert np.isfinite(mix.mean())
    assert np.isfinite(mix.std())


def test_fit_weight_for_stat_handles_quantile_distribution_components() -> None:
    """_fit_weight_for_stat must produce a finite per-stat weight when
    components_a is a list of QuantileDistribution and components_b is a
    list of ParametricNegativeBinomial. Exercises _bracket_for_components +
    _quantile_with_bracket against a Quantile component, which is the
    inner-loop machinery during EnsembleModel.fit Stage 3.
    """
    from projections.distributions.base import Distribution
    from projections.models.ensemble import _fit_weight_for_stat

    rng = np.random.default_rng(seed=1234)
    n_rows = 20
    components_a: list[Distribution] = []
    components_b: list[Distribution] = []
    for _ in range(n_rows):
        values = np.sort(rng.uniform(0.0, 10.0, size=19))
        components_a.append(
            QuantileDistribution(
                quantiles=np.arange(0.05, 0.96, 0.05),
                values=values,
            )
        )
        components_b.append(
            ParametricNegativeBinomial(
                mean=float(rng.uniform(2.0, 8.0)),
                dispersion=float(rng.uniform(1.0, 3.0)),
            )
        )
    actuals = rng.integers(0, 15, size=n_rows).astype(np.float64)

    weight = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.001 <= weight <= 0.999, f"_fit_weight_for_stat returned out-of-bounds weight: {weight}"
    assert np.isfinite(weight)
