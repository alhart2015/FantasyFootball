"""Smoke + helper tests for scripts/diagnose_calibration.py."""

from __future__ import annotations

from pathlib import Path

import msgpack
import numpy as np
import pandas as pd
import pytest


def test_find_latest_run_dir_returns_most_recent(tmp_path: Path) -> None:
    from diagnose_calibration import find_latest_run_dir

    backtest_root = tmp_path / "backtest"
    backtest_root.mkdir()
    older = backtest_root / "run_20260101T000000Z"
    newer = backtest_root / "run_20260201T000000Z"
    older.mkdir()
    newer.mkdir()
    # Order by directory name (timestamp sorts lexicographically).
    assert find_latest_run_dir(backtest_root) == newer


def test_find_latest_run_dir_raises_when_empty(tmp_path: Path) -> None:
    from diagnose_calibration import find_latest_run_dir

    backtest_root = tmp_path / "backtest"
    backtest_root.mkdir()
    with pytest.raises(FileNotFoundError, match="No run_"):
        find_latest_run_dir(backtest_root)


def test_find_latest_run_dir_raises_when_root_missing(tmp_path: Path) -> None:
    from diagnose_calibration import find_latest_run_dir

    with pytest.raises(FileNotFoundError):
        find_latest_run_dir(tmp_path / "does-not-exist")


def _make_minimal_per_row(tmp_path: Path) -> Path:
    """Build a 4-row results.parquet with one WR row and one QB row across
    two seasons. Includes only the columns load_per_row_results requires:
    identifiers + at least one *_pred / *_actual pair + family + params."""
    df = pd.DataFrame(
        {
            "gsis_id": ["00-1", "00-1", "00-2", "00-2"],
            "season": [2023, 2024, 2023, 2024],
            "week": [1, 1, 1, 1],
            "position": ["WR", "WR", "QB", "QB"],
            "team": ["KC", "KC", "MIN", "MIN"],
            "opponent": ["MIN", "MIN", "KC", "KC"],
            "ruleset": ["PPR_DEFAULT"] * 4,
            "family": ["SAMPLED_SUMMARY"] * 4,
            "params": [b""] * 4,  # ignored by load
            "receptions_pred": [4.0, 5.0, 0.0, 0.0],
            "receptions_actual": [3.0, 6.0, 0.0, 0.0],
            "passing_yards_pred": [0.0, 0.0, 250.0, 280.0],
            "passing_yards_actual": [0.0, 0.0, 220.0, 300.0],
        }
    )
    out = tmp_path / "results.parquet"
    df.to_parquet(out)
    return out


def test_load_per_row_results_round_trips(tmp_path: Path) -> None:
    from diagnose_calibration import load_per_row_results

    path = _make_minimal_per_row(tmp_path)
    loaded = load_per_row_results(path.parent)
    assert len(loaded) == 4
    assert {"gsis_id", "season", "position", "params"} <= set(loaded.columns)


def test_load_per_row_results_missing_file_raises(tmp_path: Path) -> None:
    from diagnose_calibration import load_per_row_results

    with pytest.raises(FileNotFoundError, match=r"results\.parquet"):
        load_per_row_results(tmp_path)


def _build_params_blob_normal(stat_name: str, mean: float, std: float) -> bytes:
    """Hand-rolled msgpack matching the Plan 3d codec's NORMAL family schema.
    Avoids depending on pack_per_stat_params; the diagnostic only needs to
    decode, not encode."""
    payload = {
        "schema_version": 1,
        "stats": {stat_name: {"family": "NORMAL", "mean": mean, "std": std}},
    }
    return bytes(msgpack.packb(payload, use_bin_type=True))


def test_extract_per_stat_residuals_long_form() -> None:
    from diagnose_calibration import extract_per_stat_residuals

    per_row = pd.DataFrame(
        {
            "gsis_id": ["00-1", "00-1"],
            "season": [2024, 2024],
            "week": [1, 2],
            "position": ["QB", "QB"],
            "params": [
                _build_params_blob_normal("passing_yards", 250.0, 70.0),
                _build_params_blob_normal("passing_yards", 280.0, 70.0),
            ],
            "passing_yards_pred": [250.0, 280.0],
            "passing_yards_actual": [220.0, 300.0],
        }
    )
    out = extract_per_stat_residuals(per_row)
    # Only QB:passing_yards (the only stat present in per_row + params).
    assert len(out) == 2
    assert set(out.columns) >= {
        "position",
        "stat",
        "gsis_id",
        "season",
        "week",
        "pred",
        "actual",
        "residual",
        "assumed_family",
        "assumed_param_a",
        "assumed_param_b",
    }
    assert out["residual"].tolist() == [-30.0, 20.0]
    assert out["assumed_family"].iloc[0] == "NORMAL"
    assert out["assumed_param_a"].iloc[0] == 70.0  # std for NORMAL
    assert pd.isna(out["assumed_param_b"].iloc[0])  # NORMAL has only one param


def _build_params_blob_gamma(stat_name: str, shape: float, scale: float) -> bytes:
    """Hand-rolled msgpack matching the Plan 3d codec's GAMMA family schema.
    Mirrors _build_params_blob_normal's pattern; the diagnostic only needs to
    decode, not encode."""
    payload = {
        "schema_version": 1,
        "stats": {stat_name: {"family": "GAMMA", "shape": shape, "scale": scale}},
    }
    return bytes(msgpack.packb(payload, use_bin_type=True))


def test_extract_per_stat_residuals_gamma_family() -> None:
    """Cover the GAMMA branch of _classify_family + _extract_assumed_params.
    WR:receptions is GAMMA-family per src/projections/models/baseline.py."""
    from diagnose_calibration import extract_per_stat_residuals

    per_row = pd.DataFrame(
        {
            "gsis_id": ["00-1"],
            "season": [2024],
            "week": [1],
            "position": ["WR"],
            "params": [_build_params_blob_gamma("receptions", 4.0, 1.25)],
            "receptions_pred": [5.0],
            "receptions_actual": [6.0],
        }
    )
    out = extract_per_stat_residuals(per_row)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["position"] == "WR"
    assert row["stat"] == "receptions"
    assert row["assumed_family"] == "GAMMA"
    assert row["assumed_param_a"] == 4.0  # shape for GAMMA
    assert row["assumed_param_b"] == 1.25  # scale for GAMMA


def test_compute_summary_stats_normal_family() -> None:
    from diagnose_calibration import compute_summary_stats

    # Build a synthetic per-stat residuals frame: one (QB, passing_yards) cell,
    # 300 rows, residuals drawn from N(0, 70). With std=70 in assumed_param_a,
    # the assumed [p10, p90] should cover ~80% of standardized residuals.
    rng = np.random.default_rng(42)
    n = 300
    pred = rng.normal(250, 30, n)
    residual = rng.normal(0, 70, n)
    # actual = pred + residual so that (actual - pred) ~ N(0, 70) matches the
    # assumed family params; otherwise coverage_p10p90 measures noise, not fit.
    actual = pred + residual
    residuals = pd.DataFrame(
        {
            "position": ["QB"] * n,
            "stat": ["passing_yards"] * n,
            "gsis_id": [f"00-{i}" for i in range(n)],
            "season": [2024] * n,
            "week": list(range(1, n + 1)),
            "pred": pred,
            "actual": actual,
            "residual": residual,
            "assumed_family": ["NORMAL"] * n,
            "assumed_param_a": [70.0] * n,
            "assumed_param_b": [float("nan")] * n,
        }
    )
    out = compute_summary_stats(residuals)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["position"] == "QB"
    assert row["stat"] == "passing_yards"
    assert row["n"] == n
    assert abs(row["residual_mean"]) < 10  # near-zero
    assert 60 < row["residual_std"] < 80  # near 70
    # Coverage of assumed [p10, p90] under N(0, 70) for residuals drawn from
    # N(0, 70) should be approximately 0.80.
    assert 0.70 < row["coverage_p10p90"] < 0.90
    # Heteroscedasticity ratio: residuals are homoscedastic, so ratio ~= 1.
    assert 0.5 < row["heteroscedasticity_ratio"] < 2.0
    # KS p-value should be high (residuals match assumed family).
    assert row["ks_assumed_pvalue"] > 0.05


def test_fit_alternative_families_continuous_returns_student_t_and_lognormal() -> None:
    from diagnose_calibration import fit_alternative_families

    rng = np.random.default_rng(0)
    actual = rng.normal(250, 70, 500)
    pred = np.full(500, 250.0)
    out = fit_alternative_families(actual=actual, pred=pred, stat_kind="continuous")
    # Both alternatives attempted.
    assert "student_t" in out
    assert "log_normal" in out  # may have ok=False if min(actual) <= 0
    # Student-t fit succeeded and produced finite AIC.
    assert out["student_t"]["ok"] is True
    assert np.isfinite(out["student_t"]["aic"])


def test_fit_alternative_families_low_mean_count_returns_neg_binom() -> None:
    from diagnose_calibration import fit_alternative_families

    rng = np.random.default_rng(1)
    # Low-mean integer counts (e.g., receiving_tds-shaped).
    actual = rng.poisson(0.3, 500).astype(np.float64)
    pred = np.full(500, 0.3)
    out = fit_alternative_families(actual=actual, pred=pred, stat_kind="low_count")
    assert "neg_binomial" in out
    assert out["neg_binomial"]["ok"] is True
    assert np.isfinite(out["neg_binomial"]["aic"])


def test_fit_alternative_families_handles_failure_gracefully() -> None:
    from diagnose_calibration import fit_alternative_families

    # Degenerate input: all zeros. Log-normal MLE undefined; Student-t
    # may fit with degenerate scale. Either way: no exceptions.
    actual = np.zeros(50)
    pred = np.zeros(50)
    out = fit_alternative_families(actual=actual, pred=pred, stat_kind="continuous")
    # log_normal must report ok=False (skipped because min(actual) <= 0).
    assert out["log_normal"]["ok"] is False
    assert np.isnan(out["log_normal"]["aic"])


def test_fit_alternative_families_rejects_degenerate_student_t() -> None:
    """All-zero input collapses scipy.stats.t.fit's scale toward 0; the
    resulting t.logpdf at the mode is a huge positive number, NOT -inf, so
    the np.isfinite guard alone wouldn't catch the degenerate fit. The
    sample-std floor on scale is what rejects it."""
    from diagnose_calibration import fit_alternative_families

    out = fit_alternative_families(
        actual=np.zeros(50),
        pred=np.zeros(50),
        stat_kind="continuous",
    )
    assert out["student_t"]["ok"] is False
    assert np.isnan(out["student_t"]["aic"])


def test_compute_recommended_fix_variance_bucket() -> None:
    from diagnose_calibration import compute_recommended_fix

    out = compute_recommended_fix(
        heteroscedasticity_ratio=2.0,
        assumed_aic=1000.0,
        alt_fits={"student_t": {"aic": 998.0, "ok": True, "n_params": 3}},
    )
    # aic_delta = assumed_aic - best_alt_aic = 1000 - 998 = +2.0 (positive = alt
    # fits better since AIC is lower-is-better). The plan's test asserted -2.0,
    # which contradicts the documented formula and AIC convention; corrected here.
    assert out == ("variance_bucket", "student_t", 2.0)


def test_compute_recommended_fix_family_swap() -> None:
    from diagnose_calibration import compute_recommended_fix

    out = compute_recommended_fix(
        heteroscedasticity_ratio=1.1,
        assumed_aic=1000.0,
        alt_fits={"student_t": {"aic": 980.0, "ok": True, "n_params": 3}},
    )
    assert out[0] == "family_swap"
    # Family name is returned via best_alt_family (out[1]), not interpolated
    # into the recommendation tag.
    assert out[1] == "student_t"


def test_compute_recommended_fix_combined() -> None:
    from diagnose_calibration import compute_recommended_fix

    out = compute_recommended_fix(
        heteroscedasticity_ratio=2.0,
        assumed_aic=1000.0,
        alt_fits={"student_t": {"aic": 980.0, "ok": True, "n_params": 3}},
    )
    assert out[0] == "combined"
    assert out[1] == "student_t"


def test_compute_recommended_fix_no_change() -> None:
    from diagnose_calibration import compute_recommended_fix

    out = compute_recommended_fix(
        heteroscedasticity_ratio=1.1,
        assumed_aic=1000.0,
        alt_fits={"student_t": {"aic": 999.0, "ok": True, "n_params": 3}},
    )
    assert out[0] == "no_change"


def test_compute_recommended_fix_handles_no_successful_alt() -> None:
    from diagnose_calibration import compute_recommended_fix

    out = compute_recommended_fix(
        heteroscedasticity_ratio=1.1,
        assumed_aic=1000.0,
        alt_fits={"student_t": {"aic": float("nan"), "ok": False, "n_params": 3}},
    )
    # Element-wise comparison: tuple `==` with NaN fails because two distinct
    # `float("nan")` instances compare unequal (the plan's `out == (..., nan)`
    # assertion is unsatisfiable). Assert structurally instead.
    assert out[0] == "no_change"
    assert out[1] == "none"
    assert np.isnan(out[2])


def test_assemble_full_summary_columns_and_one_row_per_cell() -> None:
    from diagnose_calibration import assemble_full_summary

    # Minimal residuals: one (QB, passing_yards) cell, ~120 rows (>> 3 to allow
    # tertile binning). Use NORMAL family for cheap fixture construction.
    rng = np.random.default_rng(7)
    n = 120
    residuals = pd.DataFrame(
        {
            "position": ["QB"] * n,
            "stat": ["passing_yards"] * n,
            "gsis_id": [f"00-{i}" for i in range(n)],
            "season": [2024] * n,
            "week": list(range(1, n + 1)),
            "pred": rng.normal(250, 30, n),
            "actual": rng.normal(250, 70, n),
            "residual": rng.normal(0, 70, n),
            "assumed_family": ["NORMAL"] * n,
            "assumed_param_a": [70.0] * n,
            "assumed_param_b": [float("nan")] * n,
        }
    )
    out = assemble_full_summary(residuals)
    assert len(out) == 1
    expected_cols = {
        "position",
        "stat",
        "n",
        "mean_pred",
        "mean_actual",
        "residual_mean",
        "residual_std",
        "residual_skew",
        "residual_excess_kurtosis",
        "std_tertile_low",
        "std_tertile_mid",
        "std_tertile_high",
        "heteroscedasticity_ratio",
        "coverage_p10p90",
        "coverage_le_p90",
        "ks_assumed_stat",
        "ks_assumed_pvalue",
        "best_alt_family",
        "best_alt_aic",
        "assumed_aic",
        "aic_delta",
        "recommended_fix",
    }
    assert expected_cols <= set(out.columns)
    # Recommended fix must be one of the four well-formed values.
    rec = out["recommended_fix"].iloc[0]
    assert rec in {"variance_bucket", "no_change", "combined", "family_swap"}
