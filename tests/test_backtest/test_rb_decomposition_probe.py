"""Tests for src/projections/backtest/rb_decomposition_probe.py.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.backtest.rb_decomposition_probe import (
    _RB_DECOMPS,
    _RIDGE_ALPHAS,
    CoverageByYear,
    FactorResidualsByYear,
    StatResiduals,
    WalkForwardOutput,
    _fit_decomposed_efficiency,
    _fit_decomposed_volume,
    _fit_direct,
    _predict_decomposed,
    _predict_direct,
    _StatDecomp,
    walk_forward_residuals,
)
from projections.schemas import Stat


def test_rb_decomps_registry_has_five_stats_across_two_volume_axes() -> None:
    """Five composed stats: 2 rushing (carries axis) + 3 receiving (targets axis)."""
    assert set(_RB_DECOMPS.keys()) == {
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    # Rushing axis.
    assert _RB_DECOMPS[Stat.RUSHING_YARDS].volume_stat is Stat.CARRIES
    assert _RB_DECOMPS[Stat.RUSHING_TDS].volume_stat is Stat.CARRIES
    # Receiving axis.
    assert _RB_DECOMPS[Stat.RECEPTIONS].volume_stat is Stat.TARGETS
    assert _RB_DECOMPS[Stat.RECEIVING_YARDS].volume_stat is Stat.TARGETS
    assert _RB_DECOMPS[Stat.RECEIVING_TDS].volume_stat is Stat.TARGETS

    # Clip semantics: rate factors -> 1.0; unbounded efficiency -> +inf.
    assert _RB_DECOMPS[Stat.RUSHING_YARDS].efficiency_clip_hi == float("inf")
    assert _RB_DECOMPS[Stat.RUSHING_TDS].efficiency_clip_hi == 1.0
    assert _RB_DECOMPS[Stat.RECEPTIONS].efficiency_clip_hi == 1.0
    assert _RB_DECOMPS[Stat.RECEIVING_YARDS].efficiency_clip_hi == float("inf")
    assert _RB_DECOMPS[Stat.RECEIVING_TDS].efficiency_clip_hi == 1.0

    # numerator_stat == the key for every entry (composition invariant).
    for stat, decomp in _RB_DECOMPS.items():
        assert isinstance(decomp, _StatDecomp)
        assert decomp.numerator_stat is stat


def test_fit_direct_fits_ridgecv_on_synthetic_linear_relationship() -> None:
    """Direct fit recovers a clean linear slope within tolerance."""
    rng = np.random.default_rng(seed=2026)
    x = rng.uniform(0.0, 5.0, size=(200, 3)).astype(np.float64)
    # true: y = 2 * x[:, 0] + 1 * x[:, 1] - 0.5 * x[:, 2] + noise
    y = (2.0 * x[:, 0] + 1.0 * x[:, 1] - 0.5 * x[:, 2] + rng.normal(0, 0.2, size=200)).astype(
        np.float64
    )

    ridge = _fit_direct(x, y)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    assert abs(ridge.coef_[0] - 2.0) < 0.1
    assert abs(ridge.coef_[1] - 1.0) < 0.1


def test_fit_decomposed_volume_fits_on_unfiltered_rows() -> None:
    """Volume sub-model trains on ALL rows including volume == 0
    (zero-volume rows are legitimate observations of low-volume players).
    """
    rng = np.random.default_rng(seed=2027)
    x = rng.uniform(0.0, 1.0, size=(150, 2)).astype(np.float64)
    # carries integer with ~30% zeros
    carries = np.where(
        rng.uniform(0, 1, size=150) < 0.3,
        0,
        rng.poisson(10, size=150),
    ).astype(np.int64)

    ridge = _fit_decomposed_volume(x, carries)

    assert isinstance(ridge, RidgeCV)
    pred = ridge.predict(x)
    assert pred.shape == (150,)


def test_fit_decomposed_efficiency_fits_only_on_positive_volume_rows() -> None:
    """Efficiency sub-model trains only on rows where volume > 0."""
    rng = np.random.default_rng(seed=2028)
    x = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    volume = np.where(
        rng.uniform(0, 1, size=100) < 0.5,
        0,
        rng.poisson(8, size=100),
    ).astype(np.int64)
    # numerator ~ volume * (4 + 2 * x[:, 0])
    rate = 4.0 + 2.0 * x[:, 0]
    numerator = (volume.astype(np.float64) * rate).astype(np.int64)

    ridge = _fit_decomposed_efficiency(x, numerator, volume)

    assert isinstance(ridge, RidgeCV)
    # The slope on x[:, 0] should be ~2.0 (the rate's coefficient).
    assert abs(ridge.coef_[0] - 2.0) < 0.5


def test_fit_decomposed_efficiency_raises_if_no_positive_volume() -> None:
    """If every training row has volume == 0, the efficiency fit cannot
    proceed (division-by-zero in the ratio).
    """
    import pytest

    x = np.zeros((10, 3), dtype=np.float64)
    volume = np.zeros(10, dtype=np.int64)
    numerator = np.zeros(10, dtype=np.int64)

    with pytest.raises(ValueError, match=r"no training rows with volume > 0"):
        _fit_decomposed_efficiency(x, numerator, volume)


def test_predict_direct_passes_through_no_clipping() -> None:
    """Direct prediction does NOT clip (mirrors BaselineModel; downstream
    Distribution constructor handles family floors).
    """
    rng = np.random.default_rng(seed=2029)
    x_train = rng.uniform(0.0, 1.0, size=(50, 2)).astype(np.float64)
    # Force negative predictions: train y = -1 + ...
    y_train = (-1.0 + rng.normal(0, 0.05, size=50)).astype(np.float64)
    ridge = _fit_direct(x_train, y_train)

    x_eval = rng.uniform(0.0, 1.0, size=(5, 2)).astype(np.float64)
    pred = _predict_direct(ridge, x_eval)

    assert pred.shape == (5,)
    # Predictions are around -1 — no clip applied.
    assert (pred < 0).all()


def test_predict_decomposed_applies_two_sided_clip_on_rate_efficiency() -> None:
    """Decomposed prediction clips efficiency to [0, clip_hi]; volume to [0, +inf).

    Two-sided clip engages on rate factors (clip_hi = 1.0); only low-side on
    yards_per_volume (clip_hi = +inf).
    """
    rng = np.random.default_rng(seed=2030)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # Volume Ridge fit on a synthetic carries response.
    volume_y = (10.0 + 5.0 * x_train[:, 0] + rng.normal(0, 0.5, size=100)).astype(np.float64)
    volume_ridge = _fit_direct(x_train, volume_y)
    # Efficiency Ridge that will predict > 1 on high-x_eval (forces clip_hi=1.0).
    rate_y = (0.5 + 1.5 * x_train[:, 0] + rng.normal(0, 0.05, size=100)).astype(np.float64)
    rate_ridge = _fit_direct(x_train, rate_y)

    x_eval = np.array([[1.0, 0.5]], dtype=np.float64)  # rate prediction ~ 2.0 unclipped.

    # Rate factor with clip_hi=1.0: predicted efficiency caps at 1.0.
    pred_clipped = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=rate_ridge,
        x=x_eval,
        efficiency_clip_hi=1.0,
    )
    # Volume at x=1.0 is ~15; efficiency clipped to 1.0 -> result ~ 15.0
    assert 10.0 < pred_clipped[0] < 20.0

    # Same data, clip_hi=+inf: predicted efficiency is unclipped.
    pred_unclipped = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=rate_ridge,
        x=x_eval,
        efficiency_clip_hi=float("inf"),
    )
    # Now efficiency ~ 2.0 -> result ~ 30.
    assert pred_unclipped[0] > pred_clipped[0]


def test_predict_decomposed_clips_negative_volume_to_zero() -> None:
    """Volume predictions below 0 are clipped (can't have negative carries)."""
    rng = np.random.default_rng(seed=2031)
    x_train = rng.uniform(0.0, 1.0, size=(50, 2)).astype(np.float64)
    # Force negative volume predictions on the eval-low-x side.
    volume_y = (-5.0 + 10.0 * x_train[:, 0] + rng.normal(0, 0.5, size=50)).astype(np.float64)
    volume_ridge = _fit_direct(x_train, volume_y)
    eff_y = (5.0 + 2.0 * x_train[:, 0] + rng.normal(0, 0.1, size=50)).astype(np.float64)
    eff_ridge = _fit_direct(x_train, eff_y)

    x_eval = np.array([[0.0, 0.5]], dtype=np.float64)  # volume ~ -5 unclipped.

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=eff_ridge,
        x=x_eval,
        efficiency_clip_hi=float("inf"),
    )
    # Volume clipped to 0 -> result is 0.
    assert pred[0] == 0.0


def _synthetic_rb_inputs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic RB (features, weekly_stats) pair for walk-forward
    integration testing.

    4 seasons x 4 weeks x 8 players = 128 rows. Features uniform random;
    truth uses two correlated volume axes (carries, targets) and matched
    efficiency factors so both arms have signal to extract.
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import _PYARROW_STR, Position, WeeklyStatsSchema

    rng = np.random.default_rng(seed=seed)
    rows: list[dict[str, object]] = []
    for season in range(2018, 2022):
        for week in range(1, 5):
            for p in range(8):
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

    feature_schema = POSITION_DISPATCH[Position.RB].feature_schema
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
            df[col_name] = rng.uniform(0.0, 1.0, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "RB"
    n = len(ws)
    carries_lambda = rng.uniform(8.0, 16.0, size=n)
    carries = np.maximum(1, rng.poisson(carries_lambda)).astype(np.int64)
    targets_lambda = rng.uniform(2.0, 6.0, size=n)
    targets = np.maximum(1, rng.poisson(targets_lambda)).astype(np.int64)
    # Yards per carry ~ N(4.3, 1.0); per-row mean from carries_lambda for signal.
    ypc = np.maximum(0.0, 4.3 + 0.05 * carries_lambda + rng.normal(0, 0.5, size=n))
    rushing_yards = np.clip(carries.astype(np.float64) * ypc, -50.0, 400.0)
    # TD rate per carry — small, ~0.03; bumped a bit by goal-line indicator (proxy
    # via carries_lambda).
    td_rate_carry = np.clip(0.02 + 0.005 * (carries_lambda - 12.0), 0.0, 0.1)
    rushing_tds = rng.binomial(carries, td_rate_carry).astype(np.int64)
    # Catch_rate ~ 0.7 for RBs; small variability.
    catch_rate = np.clip(0.7 + 0.05 * rng.normal(0, 1.0, size=n), 0.1, 1.0)
    receptions = rng.binomial(targets, catch_rate).astype(np.int64)
    # Yards per target ~ 6 yards (short routes for RBs).
    ypt = np.maximum(0.0, 6.0 + 0.1 * targets_lambda + rng.normal(0, 0.5, size=n))
    receiving_yards = np.clip(targets.astype(np.float64) * ypt, -50.0, 400.0).astype(np.float64)
    td_rate_target = np.clip(0.04 + rng.normal(0, 0.02, size=n), 0.0, 0.2)
    receiving_tds = rng.binomial(targets, td_rate_target).astype(np.int64)

    ws["carries"] = carries
    ws["targets"] = targets
    ws["rushing_yards"] = rushing_yards
    ws["rushing_tds"] = rushing_tds
    ws["receptions"] = receptions
    ws["receiving_yards"] = receiving_yards
    ws["receiving_tds"] = receiving_tds
    ws["fumbles_lost"] = np.zeros(n, dtype=np.int64)
    ws["passing_yards"] = 0.0
    ws["passing_tds"] = np.int64(0)
    ws["interceptions"] = np.int64(0)

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


def test_walk_forward_residuals_populates_all_five_stat_buffers() -> None:
    """Every entry in _RB_DECOMPS gets a StatResiduals; both arms populated."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    assert isinstance(output, WalkForwardOutput)
    assert set(output.per_stat.keys()) == set(_RB_DECOMPS.keys())
    for residuals in output.per_stat.values():
        assert isinstance(residuals, StatResiduals)
        assert residuals.actual.shape == residuals.mu_direct.shape
        assert residuals.actual.shape == residuals.mu_decomposed.shape
        assert residuals.n_paired == residuals.actual.shape[0]
        assert residuals.n_paired > 0
        # Decomposed predictions: rate-factor stats >= 0 (volume clip floors).
        assert (residuals.mu_decomposed >= 0).all()


def test_walk_forward_residuals_tracks_two_coverage_axes() -> None:
    """Coverage tracked separately for the carries and targets axes."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    assert set(output.coverage_carries_by_year.keys()) == {2020, 2021}
    assert set(output.coverage_targets_by_year.keys()) == {2020, 2021}
    for year in {2020, 2021}:
        assert 0.0 <= output.coverage_carries_by_year[year] <= 1.0
        assert 0.0 <= output.coverage_targets_by_year[year] <= 1.0


def test_walk_forward_residuals_arms_differ_on_some_rows() -> None:
    """The decomposed and direct arms must NOT produce identical predictions
    everywhere — a no-op fall-through would be a silent bug.
    """
    features, weekly_stats = _synthetic_rb_inputs(seed=12345)
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2021,))

    for stat, residuals in output.per_stat.items():
        n_different = int(np.sum(np.abs(residuals.mu_direct - residuals.mu_decomposed) > 1e-6))
        assert n_different > 0, (
            f"direct and decomposed arms produced identical predictions for {stat.value}"
        )


def test_walk_forward_residuals_skips_eval_year_with_no_train_data() -> None:
    """When eval_year is the earliest season, no train data exists — skip cleanly."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2018,))

    # 2018 is the earliest season; train_seasons is empty -> coverage dict empty.
    assert output.coverage_carries_by_year == {}
    assert output.coverage_targets_by_year == {}
    for residuals in output.per_stat.values():
        assert residuals.n_paired == 0


def test_walk_forward_residuals_emits_factor_residuals_per_stat_per_year() -> None:
    """Per-stat per-year (volume_residual, efficiency_residual) for orthogonality check."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    for stat in _RB_DECOMPS:
        assert stat in output.factor_residuals_by_year
        per_year_list = output.factor_residuals_by_year[stat]
        assert len(per_year_list) == 2  # one entry per eval year
        for entry in per_year_list:
            assert isinstance(entry, FactorResidualsByYear)
            assert entry.eval_year in {2020, 2021}
            assert entry.volume_residuals.shape == entry.efficiency_residuals.shape


# Mark CoverageByYear as used (imported for export symmetry with the canonical
# sibling API; production walk-forward uses dict[int, float] coverage dicts).
_COVERAGE_BY_YEAR_IMPORTED: type = CoverageByYear
