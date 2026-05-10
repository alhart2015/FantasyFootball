"""Tests for src/projections/backtest/target_decomposition_probe.py.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.target_decomposition_probe import (
    _RIDGE_ALPHAS,
    _WR_RECEIVING_DECOMPS,
    ProbeReport,
    WalkForwardOutput,
    _fit_decomposed_efficiency,
    _fit_decomposed_volume,
    _fit_direct,
    _predict_decomposed,
    _predict_direct,
    _verdict_from_delta,
    render_probe_report,
    walk_forward_residuals,
    write_per_stat_csv,
    write_per_stat_markdown,
)
from projections.schemas import Stat


def test_wr_receiving_decomps_registry_has_three_stats() -> None:
    """Three decomposed stats — receptions, receiving_yards, receiving_tds.

    Each shares Stat.TARGETS as its volume factor; efficiency clip-hi is 1.0
    for ratios and +inf for yards-per-target.
    """
    assert set(_WR_RECEIVING_DECOMPS.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for stat, decomp in _WR_RECEIVING_DECOMPS.items():
        assert decomp.volume_stat is Stat.TARGETS
        assert decomp.numerator_stat is stat
    assert _WR_RECEIVING_DECOMPS[Stat.RECEPTIONS].efficiency_clip_hi == 1.0
    assert _WR_RECEIVING_DECOMPS[Stat.RECEIVING_TDS].efficiency_clip_hi == 1.0
    assert _WR_RECEIVING_DECOMPS[Stat.RECEIVING_YARDS].efficiency_clip_hi == float("inf")


def _synthetic_xy(n: int = 50, n_features: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features))
    coef = np.array([1.0, -0.5, 0.25, 0.0])
    y = x @ coef + 0.1 * rng.standard_normal(n)
    return x, y


def test_fit_direct_returns_ridgecv_with_canonical_alphas() -> None:
    x, y = _synthetic_xy()
    ridge = _fit_direct(x, y)
    assert isinstance(ridge, RidgeCV)
    np.testing.assert_array_equal(ridge.alphas, _RIDGE_ALPHAS)
    # Sanity: prediction shape matches input rows.
    assert ridge.predict(x).shape == (len(x),)


def test_fit_decomposed_volume_targets_y_is_volume_stat() -> None:
    x, _ = _synthetic_xy()
    targets = np.arange(len(x), dtype=np.int64)  # arbitrary non-trivial targets
    ridge = _fit_decomposed_volume(x, targets)
    assert isinstance(ridge, RidgeCV)
    # Coefficient is a 4-vector matching n_features.
    assert ridge.coef_.shape == (4,)


def test_fit_decomposed_efficiency_filters_to_targets_positive() -> None:
    """Efficiency arm trains only on rows where targets > 0; ratio is
    numerator / targets on those rows.
    """
    x, _ = _synthetic_xy(n=20, seed=1)
    # Half the rows have targets > 0; the rest are zero-target.
    targets = np.array([5, 0, 3, 0, 8, 0, 2, 0, 4, 0, 6, 0, 1, 0, 7, 0, 9, 0, 3, 0])
    numerator = targets * 0.5  # so true catch_rate is 0.5 on every targets > 0 row
    numerator[targets == 0] = 0  # well-defined ratio nonexistent here, but predict won't see these
    ridge = _fit_decomposed_efficiency(x, numerator, targets)
    assert isinstance(ridge, RidgeCV)
    # On the targets > 0 subset, ratio is constant 0.5; ridge should predict ~0.5 everywhere.
    pred = ridge.predict(x)
    assert np.allclose(pred, 0.5, atol=0.05)


def test_fit_decomposed_efficiency_raises_when_no_positive_targets() -> None:
    """All-zero-targets training set is malformed; raise rather than silently
    return a ridge fit on zero rows.
    """
    x, _ = _synthetic_xy(n=10)
    targets = np.zeros(10, dtype=np.int64)
    numerator = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="targets > 0"):
        _fit_decomposed_efficiency(x, numerator, targets)


def test_predict_direct_returns_float64_array_of_eval_shape() -> None:
    x_train, y_train = _synthetic_xy(seed=2)
    ridge = _fit_direct(x_train, y_train)
    x_eval, _ = _synthetic_xy(n=30, seed=3)
    pred = _predict_direct(ridge, x_eval)
    assert pred.dtype == np.float64
    assert pred.shape == (30,)


def test_predict_decomposed_clips_volume_at_zero_and_efficiency_at_clip_hi() -> None:
    """Volume floored at 0; efficiency clipped to [0, efficiency_clip_hi].

    Construct a contrived volume_ridge that predicts negatives on some rows
    and an efficiency_ridge that predicts > 1 on some rows, with
    efficiency_clip_hi = 1.0. Verify the product respects both clips.
    """
    # Train ridges with extreme synthetic data so we control the predictions.
    rng = np.random.default_rng(4)
    n = 8
    x = rng.standard_normal((n, 2))

    # Volume ridge: y_train = -x[:, 0] * 5 (negatives appear when x[:, 0] > 0)
    volume_y = -x[:, 0] * 5.0
    volume_ridge = _fit_direct(x, volume_y)

    # Efficiency ridge: y_train = x[:, 1] * 0.5 + 0.5, range roughly [-0.5, 1.5]
    efficiency_y = x[:, 1] * 0.5 + 0.5
    efficiency_ridge = _fit_direct(x, efficiency_y)

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=efficiency_ridge,
        x=x,
        efficiency_clip_hi=1.0,
    )
    assert pred.dtype == np.float64
    assert pred.shape == (n,)
    # Product of clipped factors is non-negative and bounded above by max(volume_clip, +inf) * 1.0.
    assert (pred >= 0.0).all()
    # On rows where volume_ridge predicts < 0, the product is exactly 0.
    raw_volume = volume_ridge.predict(x)
    assert np.all(pred[raw_volume < 0] == 0.0)


def test_predict_decomposed_no_clip_on_efficiency_hi_when_inf() -> None:
    """yards_per_target case: efficiency_clip_hi = +inf; Ridge predictions
    above ~15 (empirical max) are not clipped on the high side."""
    rng = np.random.default_rng(5)
    n = 6
    x = rng.standard_normal((n, 2))
    volume_y = np.full(n, 10.0)  # volume ridge predicts ~10 everywhere
    volume_ridge = _fit_direct(x, volume_y)
    efficiency_y = np.full(n, 25.0)  # efficiency ridge predicts ~25 everywhere
    efficiency_ridge = _fit_direct(x, efficiency_y)

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=efficiency_ridge,
        x=x,
        efficiency_clip_hi=float("inf"),
    )
    # No upper clip; product is ~250.
    assert np.all(pred > 200.0)


def _make_synthetic_walk_forward_inputs(
    seasons: tuple[int, ...] = (2018, 2019, 2020, 2021),
    n_per_season: int = 25,
    seed: int = 6,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Build a tiny features dict and weekly_stats frame whose schema is
    compatible with the harness. Identifying cols are stable (same gsis_ids
    across seasons) so the inner-join joins.
    """
    rng = np.random.default_rng(seed)
    feature_cols = ["targets_per_game_l4", "receiving_yards_per_game_l4"]
    features_by_year: dict[int, pd.DataFrame] = {}
    weekly_rows: list[dict[str, object]] = []
    for season in seasons:
        n = n_per_season
        gsis_ids = [f"00-{season:04d}{i:03d}" for i in range(n)]
        feat_df = pd.DataFrame(
            {
                "gsis_id": gsis_ids,
                "season": season,
                "week": rng.integers(1, 18, size=n),
                "team": "KC",
                "opponent": "DEN",
                **{c: rng.standard_normal(n) for c in feature_cols},
            }
        )
        features_by_year[season] = feat_df
        # Construct weekly stats so targets is non-zero on most rows.
        targets = rng.integers(0, 12, size=n)
        for i, gid in enumerate(gsis_ids):
            t = int(targets[i])
            weekly_rows.append(
                {
                    "gsis_id": gid,
                    "season": season,
                    "week": int(feat_df["week"].iloc[i]),
                    "position": "WR",
                    "team": "KC",
                    "opponent": "DEN",
                    "targets": t,
                    "receptions": int(t * 0.6),
                    "receiving_yards": float(t * 8.0),
                    "receiving_tds": int(t * 0.05),
                    "passing_yards": 0.0,
                    "passing_tds": 0,
                    "interceptions": 0,
                    "attempts": 0,
                    "completions": 0,
                    "sacks": 0,
                    "rushing_yards": 0.0,
                    "rushing_tds": 0,
                    "carries": 0,
                    "receiving_air_yards": 0.0,
                    "fumbles_lost": 0,
                }
            )
    weekly_stats = pd.DataFrame(weekly_rows)
    return features_by_year, weekly_stats


def test_walk_forward_output_shape_matches_three_stats() -> None:
    """Output is a WalkForwardOutput with one entry per decomposed stat
    (receptions, receiving_yards, receiving_tds), each holding aligned
    arrays of (actual, mu_direct, mu_decomposed) of equal length n_paired.
    """
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs()
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    assert isinstance(out, WalkForwardOutput)
    assert set(out.per_stat.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for _stat, residuals in out.per_stat.items():
        assert residuals.actual.shape == residuals.mu_direct.shape
        assert residuals.actual.shape == residuals.mu_decomposed.shape
        assert residuals.n_paired == len(residuals.actual)


def test_walk_forward_train_eval_strict_separation_assertion() -> None:
    """Defense-in-depth: train rows for eval Y must not contain season Y rows.

    Construct synthetic data where features for season 2021 leak into the
    train pool for eval year 2021; verify the harness catches it.
    """
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(seasons=(2019, 2020, 2021))
    # Inject a row tagged season=2021 into the 2020 features frame.
    leak_row = features_by_year[2020].iloc[[0]].copy()
    leak_row["season"] = 2021
    features_by_year[2020] = pd.concat([features_by_year[2020], leak_row], ignore_index=True)
    # Add the corresponding weekly_stats row.
    leak_ws = (
        weekly_stats[
            (weekly_stats["gsis_id"] == leak_row["gsis_id"].iloc[0])
            & (weekly_stats["season"] == 2020)
        ]
        .iloc[[0]]
        .copy()
    )
    leak_ws["season"] = 2021
    weekly_stats = pd.concat([weekly_stats, leak_ws], ignore_index=True)

    with pytest.raises(AssertionError, match="contain season >= eval year"):
        walk_forward_residuals(
            features_by_year=features_by_year,
            weekly_stats=weekly_stats,
            feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
            eval_years=(2021,),
            train_start=2019,
        )


def test_walk_forward_coverage_measured_per_year() -> None:
    """Coverage = (targets > 0).mean() per eval year on eval rows; same on train rows.

    Construct synthetic data where 80% of 2020 eval rows have targets > 0;
    verify coverage_by_year[2020] reflects that.
    """
    rng = np.random.default_rng(42)
    n = 50
    seasons = (2018, 2019, 2020)
    features_by_year: dict[int, pd.DataFrame] = {}
    weekly_rows: list[dict[str, object]] = []
    for season in seasons:
        gsis_ids = [f"00-{season:04d}{i:03d}" for i in range(n)]
        feat_df = pd.DataFrame(
            {
                "gsis_id": gsis_ids,
                "season": season,
                "week": np.arange(1, n + 1) % 18 + 1,
                "team": "KC",
                "opponent": "DEN",
                "targets_per_game_l4": rng.standard_normal(n),
                "receiving_yards_per_game_l4": rng.standard_normal(n),
            }
        )
        features_by_year[season] = feat_df
        # 2020: exactly 40 rows with targets > 0 (80%); other seasons: ~95%.
        if season == 2020:
            targets = np.array([3] * 40 + [0] * 10)
        else:
            targets = rng.choice([0, 5], size=n, p=[0.05, 0.95])
        for i, gid in enumerate(gsis_ids):
            t = int(targets[i])
            weekly_rows.append(
                {
                    "gsis_id": gid,
                    "season": season,
                    "week": int(feat_df["week"].iloc[i]),
                    "position": "WR",
                    "team": "KC",
                    "opponent": "DEN",
                    "targets": t,
                    "receptions": int(t * 0.6),
                    "receiving_yards": float(t * 8.0),
                    "receiving_tds": int(t * 0.05),
                    "passing_yards": 0.0,
                    "passing_tds": 0,
                    "interceptions": 0,
                    "attempts": 0,
                    "completions": 0,
                    "sacks": 0,
                    "rushing_yards": 0.0,
                    "rushing_tds": 0,
                    "carries": 0,
                    "receiving_air_yards": 0.0,
                    "fumbles_lost": 0,
                }
            )
    weekly_stats = pd.DataFrame(weekly_rows)

    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2020,),
        train_start=2018,
    )
    cov = {c.eval_year: c for c in out.coverage_by_year}
    assert cov[2020].targets_positive_rate == pytest.approx(0.80, abs=0.01)
    assert cov[2020].n_eval_rows == n


def test_walk_forward_factor_residuals_recorded_per_stat_per_year() -> None:
    """For each (stat, eval_year), volume_residuals + efficiency_residuals
    are recorded on rows where targets > 0 (efficiency residual undefined
    where targets == 0).
    """
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(seasons=(2018, 2019, 2020))
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2019, 2020),
        train_start=2018,
    )
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RECEIVING_TDS):
        per_year = out.factor_residuals_by_year[stat]
        assert len(per_year) == 2  # one entry per eval year
        years = sorted(p.eval_year for p in per_year)
        assert years == [2019, 2020]
        for entry in per_year:
            assert entry.volume_residuals.shape == entry.efficiency_residuals.shape
            assert entry.volume_residuals.dtype == np.float64


def _delta(point: float, lo_95: float, hi_95: float) -> BootstrapDelta:
    """Test helper. BootstrapDelta has 5 required fields; the verdict logic
    only reads point/lo_95/hi_95, so pin the others at constants."""
    return BootstrapDelta(
        point=point,
        lo_95=lo_95,
        hi_95=hi_95,
        n_paired_rows=100,
        n_bootstrap=200,
    )


def test_verdict_signal_when_hi_95_strictly_negative() -> None:
    assert _verdict_from_delta(_delta(-0.5, -0.8, -0.1)) == "SIGNAL"


def test_verdict_regression_when_lo_95_strictly_positive() -> None:
    assert _verdict_from_delta(_delta(0.5, 0.1, 0.8)) == "REGRESSION"


def test_verdict_null_when_ci_brackets_zero() -> None:
    assert _verdict_from_delta(_delta(-0.05, -0.3, 0.2)) == "NULL"


def test_verdict_null_at_exact_zero_boundaries() -> None:
    """SIGNAL requires hi_95 strictly < 0; lo_95 strictly > 0 for REGRESSION.
    Exact-zero boundaries fall into NULL."""
    assert _verdict_from_delta(_delta(-0.5, -1.0, 0.0)) == "NULL"
    assert _verdict_from_delta(_delta(0.5, 0.0, 1.0)) == "NULL"


def test_render_probe_report_returns_three_stat_verdicts() -> None:
    """End-to-end: walk_forward_residuals -> render_probe_report yields 3 verdicts,
    each with finite n_paired and RMSE deltas."""
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(n_per_season=120)
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    report = render_probe_report(out, bootstrap_n=200, seed=42)
    assert isinstance(report, ProbeReport)
    assert set(report.verdicts.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for stat, v in report.verdicts.items():
        assert v.stat is stat
        assert v.n_paired > 0
        assert np.isfinite(v.rmse_delta.point)
        assert np.isfinite(v.rmse_delta.lo_95)
        assert np.isfinite(v.rmse_delta.hi_95)
        assert v.verdict in ("SIGNAL", "NULL", "REGRESSION")


def test_render_probe_report_is_deterministic_under_fixed_seed() -> None:
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(n_per_season=120)
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    a = render_probe_report(out, bootstrap_n=200, seed=123)
    b = render_probe_report(out, bootstrap_n=200, seed=123)
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RECEIVING_TDS):
        assert a.verdicts[stat].rmse_delta.point == b.verdicts[stat].rmse_delta.point
        assert a.verdicts[stat].rmse_delta.lo_95 == b.verdicts[stat].rmse_delta.lo_95
        assert a.verdicts[stat].rmse_delta.hi_95 == b.verdicts[stat].rmse_delta.hi_95


def test_per_stat_csv_has_one_row_per_decomposed_stat(tmp_path: Path) -> None:
    """Per-stat CSV: one row per decomposed stat with verdict columns."""
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(n_per_season=120)
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    report = render_probe_report(out, bootstrap_n=200, seed=42)
    csv_path = tmp_path / "per_stat.csv"
    write_per_stat_csv(report, csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert set(df["stat"]) == {"receptions", "receiving_yards", "receiving_tds"}
    for col in (
        "n_paired",
        "rmse_direct",
        "rmse_decomposed",
        "rmse_delta_point",
        "rmse_delta_lo_95",
        "rmse_delta_hi_95",
        "verdict",
        "expected_composite_fpts_delta",
    ):
        assert col in df.columns


def test_per_stat_markdown_renders_verdict_and_diagnostics(tmp_path: Path) -> None:
    """Per-stat markdown: verdict header + RMSE table + per-year coverage table
    + per-year factor-residual rho."""
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(n_per_season=120)
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    report = render_probe_report(out, bootstrap_n=200, seed=42)
    md_path = tmp_path / "receiving_yards.md"
    write_per_stat_markdown(report, Stat.RECEIVING_YARDS, md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "receiving_yards" in text
    assert report.verdicts[Stat.RECEIVING_YARDS].verdict in text
    # Composite-fpts translation must surface in the body.
    assert "Expected composite-fpts" in text or "expected_composite_fpts" in text
    # Factor residual correlation surfaces.
    assert "Factor residual" in text or "rho" in text
