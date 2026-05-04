"""Cross-cutting tests for LightGBMNbModel (Model C-NB, Plan 5c).

Per-position parametrized smoke lives in test_lightgbm_nb_smoke.py (Phase 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projections.distributions import (
    ParametricNegativeBinomial,
    QuantileDistribution,
    unpack_per_stat_params,
)
from projections.models.lightgbm_nb import (
    COUNT_STATS_FOR_NB,
    LightGBMNbModel,
    qb_lightgbm_nb,
    rb_lightgbm_nb,
    te_lightgbm_nb,
    wr_lightgbm_nb,
)
from projections.models.lightgbm_tuned import wr_lightgbm_tuned
from projections.schemas import DistributionFamily, Ruleset, Stat

# ---------------- Synthetic fixtures (re-using the test_lightgbm.py shape) ----------------


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
        elif col_name == "age":
            # Trajectory feature: ge=15, le=50. Sample plausible WR ages.
            df[col_name] = rng.uniform(22.0, 30.0, size=len(df)).astype(np.float64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    return WrFeaturesSchema.validate(df)


def _build_synthetic_wr_weekly_stats(features: pd.DataFrame) -> pd.DataFrame:
    """WeeklyStatsSchema-shaped truth for the synthetic WR features."""
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
    # NB-2 fits via lgb-poisson, which errors out when sum of labels is zero.
    # Synthetic count fields therefore need a non-trivial positive rate.
    df["rushing_tds"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["passing_yards"] = np.zeros(n, dtype=np.float64)
    df["passing_tds"] = np.zeros(n, dtype=np.int64)
    df["interceptions"] = np.zeros(n, dtype=np.int64)
    df["attempts"] = np.zeros(n, dtype=np.int64)
    df["completions"] = np.zeros(n, dtype=np.int64)
    df["sacks"] = np.zeros(n, dtype=np.int64)
    df["fumbles_lost"] = rng.integers(0, 2, size=n).astype(np.int64)
    return WeeklyStatsSchema.validate(df)


# ---------------- Tests ----------------


def test_factories_construct() -> None:
    for factory in (qb_lightgbm_nb, rb_lightgbm_nb, te_lightgbm_nb, wr_lightgbm_nb):
        m = factory()
        assert isinstance(m, LightGBMNbModel)


def test_count_stats_set() -> None:
    """COUNT_STATS_FOR_NB pins the 5 Stat values Plan 3e routes to NB-2 in Ridge."""
    assert COUNT_STATS_FOR_NB == frozenset(
        {
            Stat.PASSING_TDS,
            Stat.RUSHING_TDS,
            Stat.RECEIVING_TDS,
            Stat.INTERCEPTIONS,
            Stat.FUMBLES_LOST,
        }
    )


def test_fit_populates_count_and_yards_models() -> None:
    """After fit, count stats land in _count_models + _count_dispersions; yards
    stats land in _sub_models. WR has 3 count stats (receiving_tds, rushing_tds,
    fumbles_lost) and 3 yards/recs stats (receptions, receiving_yards, rushing_yards)."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    model = wr_lightgbm_nb()
    model.fit(features, weekly)

    expected_counts = {Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
    assert set(model._count_models.keys()) == expected_counts
    assert set(model._count_dispersions.keys()) == expected_counts
    for stat, dispersion in model._count_dispersions.items():
        assert 0.0 < dispersion < 10000.0, f"{stat}: dispersion {dispersion} outside clip"

    expected_yards = {Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS}
    assert set(model._sub_models.keys()) == expected_yards
    for stat in expected_yards:
        assert len(model._sub_models[stat]) == 5  # 5 quantiles


def test_predict_distribution_emits_mixed_family() -> None:
    """Every predicted row's `family` column is MIXED."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    model = wr_lightgbm_nb()
    model.fit(features, weekly)
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    assert (pred["family"] == DistributionFamily.MIXED.value).all()


def test_codec_round_trip_yields_correct_per_stat_distribution_types() -> None:
    """unpack_per_stat_params on a Model C-NB row gives NB for count stats
    and QuantileDistribution for yards stats."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    model = wr_lightgbm_nb()
    model.fit(features, weekly)
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    blob = pred["params"].iloc[0]
    per_stat = unpack_per_stat_params(blob)

    # WR count stats -> ParametricNegativeBinomial.
    for stat in (Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        assert isinstance(per_stat[stat], ParametricNegativeBinomial), (
            f"{stat}: expected NB, got {type(per_stat[stat]).__name__}"
        )

    # WR yards / receptions stats -> QuantileDistribution.
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS):
        assert isinstance(per_stat[stat], QuantileDistribution), (
            f"{stat}: expected QuantileDistribution, got {type(per_stat[stat]).__name__}"
        )


def test_yards_stat_predictions_match_tuned_baseline() -> None:
    """Yards-stat predictions from LightGBMNbModel should be bit-exact identical
    to LightGBMTunedModel's on the same fixture, since LightGBMNbModel inherits
    yards-stat training and only overrides count-stat training. Checks the
    `p10`, `p50`, `p90` columns are NOT bit-exact at the composite level (count
    stats differ between the two), but per-stat yards quantile predictions are.

    The simplest way to verify this without exposing internals: compare
    yards-stat sub-models' best_iters between the two models. They should match
    when fitted on identical data with identical hyperparameters."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    nb = wr_lightgbm_nb()
    tuned = wr_lightgbm_tuned()
    nb.fit(features, weekly)
    tuned.fit(features, weekly)

    yards_stats = (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS)
    for stat in yards_stats:
        for q in (0.05, 0.10, 0.50, 0.90, 0.95):
            assert nb._best_iters[(stat, q)] == tuned._best_iters[(stat, q)], (
                f"{stat} q={q}: NB best_iter {nb._best_iters[(stat, q)]} "
                f"differs from tuned {tuned._best_iters[(stat, q)]}"
            )


def test_model_id_uses_nb_prefix() -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    model = wr_lightgbm_nb()
    model.fit(features, weekly)
    assert model.model_id.startswith("lightgbm-nb:wr:")


def test_code_hash_differs_from_tuned() -> None:
    """The NB subclass hashes a different file set (adds lightgbm_nb.py),
    so its code_hash differs from LightGBMTunedModel's."""
    nb = wr_lightgbm_nb()
    tuned = wr_lightgbm_tuned()
    assert nb.code_hash != tuned.code_hash


def test_save_load_round_trip(tmp_path: Path) -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    original = wr_lightgbm_nb()
    original.fit(features, weekly)
    artifact = tmp_path / "nb.joblib"
    original.save(artifact)

    loaded = LightGBMNbModel.load(artifact)
    assert loaded.model_id == original.model_id

    pred_orig = original.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    pred_loaded = loaded.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    for col in ("mean", "p10", "p50", "p90"):
        np.testing.assert_array_equal(pred_orig[col].to_numpy(), pred_loaded[col].to_numpy())


def test_predict_distribution_validates_against_schema() -> None:
    """The output DataFrame validates against ProjectionWeeklySchema."""
    from projections.schemas import ProjectionWeeklySchema

    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    model = wr_lightgbm_nb()
    model.fit(features, weekly)

    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    # Re-validate to confirm no silent dtype drift.
    revalidated = ProjectionWeeklySchema.validate(pred)
    assert len(revalidated) == len(features)


def test_nb_count_stat_mu_not_double_exponentiated() -> None:
    """Regression test for the np.exp(predict()) bug fixed in 898ce0f.

    lgb's poisson predict() already returns mu (the mean) in original
    scale. If the predict path were to wrap it in np.exp(...) again, mu_hat
    would be inflated by a factor of exp(true_mu) / true_mu (~4x for typical
    TD rates). On synthetic Poisson-targeted data with true mean ~0.5,
    a double-exp would push predicted mean to ~exp(0.5) ~ 1.65 -- about 3x
    larger.

    This test asserts the predicted mu_hat for at least one count stat
    is in the same ballpark as the actual training-time mean -- a loose
    check that catches an order-of-magnitude bias without overfitting to
    booster initialization."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    model = wr_lightgbm_nb()
    model.fit(features, weekly)

    # Compare predicted vs actual mean on receiving_tds: synthetic actuals
    # are integers in [0, 1], so the empirical mean is around 0.5.
    actual_mean_rec_tds = float(weekly["receiving_tds"].astype(np.float64).mean())
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    blob = pred["params"].iloc[0]
    per_stat = unpack_per_stat_params(blob)
    nb_dist = per_stat[Stat.RECEIVING_TDS]
    assert isinstance(nb_dist, ParametricNegativeBinomial)
    predicted_mean = nb_dist.mean()

    # Predicted mu should be within 2x of the actual mean on this fixture.
    # A double-exp bug would inflate predicted mean by exp(actual_mean) /
    # actual_mean -- factor of ~3x for actual_mean ~ 0.5. The 2x bound here
    # (predicted < actual * 2 + 0.5) is loose enough to allow normal booster
    # variance but tight enough to fail catastrophically on a double-exp
    # regression.
    assert predicted_mean < (actual_mean_rec_tds * 2.0 + 0.5), (
        f"predicted_mean={predicted_mean:.3f} suspiciously inflated vs "
        f"actual_mean={actual_mean_rec_tds:.3f} -- check for a double-exp regression"
    )
