"""Plan 8 — adoption gate CLI tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.adoption_gate import (
    evaluate_position,
    format_position_report,
    load_run_parquet,
    pair_rows,
    validate_model_classes_present,
)

from projections.schemas import Position


def _make_synthetic_run(tmp_path: Path, model_classes: list[str], n_per_class: int = 800) -> Path:
    """Build a tiny per-row results.parquet with the specified model_classes."""
    rng = np.random.default_rng(0)
    rows = []
    positions = ["QB", "RB", "TE", "WR"]
    seasons = [2021, 2022, 2023, 2024]
    rows_per_pos = n_per_class // (len(positions) * len(seasons))
    for cls in model_classes:
        for pos in positions:
            for season in seasons:
                for w in range(rows_per_pos):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos[0]}-{w:04d}",
                            "season": season,
                            "week": (w % 17) + 1,
                            "position": pos,
                            "model_class": cls,
                            "mean": float(rng.normal(loc=10.0, scale=5.0)),
                            "actual_ppr": float(rng.normal(loc=10.0, scale=6.0)),
                        }
                    )
    df = pd.DataFrame(rows)
    run_dir = tmp_path / "run_synthetic"
    run_dir.mkdir()
    df.to_parquet(run_dir / "results.parquet")
    return run_dir


def test_load_run_parquet_returns_frame_with_expected_columns(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"])
    df = load_run_parquet(run_dir)
    assert "model_class" in df.columns
    assert set(df["model_class"].unique()) == {"baseline", "ensemble"}


def test_load_run_parquet_raises_on_missing_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"results\.parquet"):
        load_run_parquet(run_dir)


def test_validate_model_classes_present_succeeds_when_both_present(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"])
    df = load_run_parquet(run_dir)
    validate_model_classes_present(df, incumbent="baseline", candidate="ensemble")  # no raise


def test_validate_model_classes_present_raises_on_missing_candidate(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline"])
    df = load_run_parquet(run_dir)
    with pytest.raises(ValueError, match=r"candidate.*not present"):
        validate_model_classes_present(df, incumbent="baseline", candidate="ensemble")


def test_validate_model_classes_present_raises_on_missing_incumbent(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["ensemble"])
    df = load_run_parquet(run_dir)
    with pytest.raises(ValueError, match=r"incumbent.*not present"):
        validate_model_classes_present(df, incumbent="baseline", candidate="ensemble")


def test_pair_rows_returns_aligned_arrays_for_matched_keys(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=400)
    df = load_run_parquet(run_dir)
    pos_df = df[df["position"] == "QB"]
    inc_pred, cand_pred, actual, grouping, n_dropped = pair_rows(
        pos_df, incumbent="baseline", candidate="ensemble"
    )
    assert len(inc_pred) == len(cand_pred) == len(actual) == len(grouping)
    assert len(inc_pred) > 0
    assert n_dropped == 0


def test_pair_rows_drops_unmatched_rows_with_count() -> None:
    rows = [
        # Both classes for these keys → paired.
        {
            "gsis_id": "A",
            "season": 2021,
            "week": 1,
            "position": "QB",
            "model_class": "baseline",
            "mean": 10.0,
            "actual_ppr": 12.0,
        },
        {
            "gsis_id": "A",
            "season": 2021,
            "week": 1,
            "position": "QB",
            "model_class": "ensemble",
            "mean": 11.0,
            "actual_ppr": 12.0,
        },
        # Only baseline → dropped.
        {
            "gsis_id": "B",
            "season": 2021,
            "week": 1,
            "position": "QB",
            "model_class": "baseline",
            "mean": 10.0,
            "actual_ppr": 12.0,
        },
        # Only ensemble → dropped.
        {
            "gsis_id": "C",
            "season": 2021,
            "week": 1,
            "position": "QB",
            "model_class": "ensemble",
            "mean": 11.0,
            "actual_ppr": 12.0,
        },
    ]
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="dropped 2 unpaired rows"):
        inc, _cand, _actual, _grouping, n_dropped = pair_rows(
            df, incumbent="baseline", candidate="ensemble"
        )
    assert len(inc) == 1
    assert n_dropped == 2  # B (baseline-only) + C (ensemble-only)


def test_evaluate_position_emits_position_verdict(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    df = load_run_parquet(run_dir)
    pv = evaluate_position(
        df,
        position=Position.QB,
        incumbent="baseline",
        candidate="ensemble",
        n_bootstrap=200,
        seed=42,
    )
    assert pv.position is Position.QB
    assert pv.verdict in {"ADOPT", "MARGINAL", "DO_NOT_ADOPT"}
    assert pv.rmse_delta.n_bootstrap == 200
    assert len(pv.per_year_breakdown) == 4  # 4 held-out years


def test_format_position_report_contains_verdict_and_breakdown(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    df = load_run_parquet(run_dir)
    pv = evaluate_position(
        df,
        position=Position.QB,
        incumbent="baseline",
        candidate="ensemble",
        n_bootstrap=200,
        seed=42,
    )
    text = format_position_report(pv)
    assert "QB" in text
    assert pv.verdict in text
    assert "RMSE" in text
    assert "Spearman" in text
    # Per-year breakdown table includes year numbers.
    assert "2021" in text and "2024" in text


def test_cli_smoke_exits_zero_with_verdict_in_stdout(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.adoption_gate",
            "--run",
            str(run_dir),
            "--candidate",
            "ensemble",
            "--n-bootstrap",
            "200",
            "--position",
            "QB",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "QB" in result.stdout
    assert any(v in result.stdout for v in ("ADOPT", "MARGINAL", "DO_NOT_ADOPT"))


def test_cli_missing_candidate_exits_nonzero(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline"], n_per_class=2000)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.adoption_gate",
            "--run",
            str(run_dir),
            "--candidate",
            "ensemble",
            "--n-bootstrap",
            "200",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ensemble" in result.stderr or "ensemble" in result.stdout


def test_cli_writes_csv_when_csv_out_provided(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    csv_path = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.adoption_gate",
            "--run",
            str(run_dir),
            "--candidate",
            "ensemble",
            "--n-bootstrap",
            "200",
            "--position",
            "QB",
            "--csv-out",
            str(csv_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert csv_path.is_file()
    csv = pd.read_csv(csv_path)
    expected_cols = {
        "position",
        "incumbent",
        "candidate",
        "year",
        "metric",
        "point",
        "lo_95",
        "hi_95",
        "n_paired",
        "verdict",
        "reason",
    }
    assert expected_cols <= set(csv.columns)
    # Two pooled rows (rmse, spearman) + 4 years x 2 metrics for QB.
    assert (csv["position"] == "QB").sum() == 10


def test_cli_position_filter_only_runs_one_position(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.adoption_gate",
            "--run",
            str(run_dir),
            "--candidate",
            "ensemble",
            "--n-bootstrap",
            "200",
            "--position",
            "RB",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "### RB" in result.stdout
    assert "### QB" not in result.stdout
    assert "### TE" not in result.stdout
    assert "### WR" not in result.stdout


# ---------------------------------------------------------------------------
# Plan 9 Phase 5 — dual-run mode tests.
#
# Plan 8's CLI assumed model-class-vs-model-class within ONE backtest run.
# Plan 9 (and every future feature-class plan) needs feature-set-vs-feature-set
# across TWO runs. Both runs share the same `model_class` value, so pair_rows
# can't pair on it; the dual-run loader synthesizes `_baseline_run` /
# `_candidate_run` model_class labels per file so the existing pair logic
# applies unchanged.
# ---------------------------------------------------------------------------


def _make_dual_run_pair(
    tmp_path: Path,
    *,
    n_per_position: int = 200,
) -> tuple[Path, Path]:
    """Build a (baseline_dir, candidate_dir) pair with identical (gsis_id,
    season, week, position) coverage and a single shared `model_class` value.

    Both runs hold ``mean`` and ``actual_ppr`` columns (the names ``pair_rows``
    consumes via the ``_inc`` / ``_cand`` suffixes). The candidate predictions
    are nudged off the baseline by a small per-row delta so the bootstrap CI
    is not degenerate.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    positions = ["QB", "RB", "TE", "WR"]
    seasons = [2021, 2022, 2023, 2024]
    rows_per_pos = max(1, n_per_position // len(seasons))
    for pos in positions:
        for season in seasons:
            for w in range(rows_per_pos):
                rows.append(
                    {
                        "gsis_id": f"00-{pos[0]}-{w:04d}",
                        "season": season,
                        "week": (w % 17) + 1,
                        "position": pos,
                        "model_class": "baseline",
                        "mean": float(rng.normal(loc=10.0, scale=5.0)),
                        "actual_ppr": float(rng.normal(loc=10.0, scale=6.0)),
                    }
                )
    baseline = pd.DataFrame(rows)
    candidate = baseline.copy()
    # Perturb candidate predictions so paired residuals differ across rows.
    candidate["mean"] = candidate["mean"] + rng.normal(loc=0.0, scale=0.5, size=len(candidate))

    baseline_dir = tmp_path / "run_baseline"
    candidate_dir = tmp_path / "run_candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline.to_parquet(baseline_dir / "results.parquet")
    candidate.to_parquet(candidate_dir / "results.parquet")
    return baseline_dir, candidate_dir


def test_dual_run_mode_loads_two_runs_and_emits_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--baseline-run`` + ``--candidate-run`` produces a verdict equivalent
    to the single-run path when both runs hold the same ``model_class`` but
    different feature sets (predictions)."""
    baseline_dir, candidate_dir = _make_dual_run_pair(tmp_path, n_per_position=400)
    csv_path = tmp_path / "out.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "adoption_gate",
            "--baseline-run",
            str(baseline_dir),
            "--candidate-run",
            str(candidate_dir),
            "--csv-out",
            str(csv_path),
            "--n-bootstrap",
            "100",
            "--seed",
            "42",
            "--position",
            "QB",
        ],
    )
    from scripts.adoption_gate import main

    main()
    captured = capsys.readouterr()
    assert "### QB" in captured.out
    assert any(v in captured.out for v in ("ADOPT", "MARGINAL", "DO_NOT_ADOPT"))
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert "QB" in df["position"].tolist()
    # The synthesized model_class labels propagate into the CSV so downstream
    # consumers can tell a dual-run verdict from a single-run one.
    assert "_baseline_run" in df["incumbent"].tolist()
    assert "_candidate_run" in df["candidate"].tolist()


def test_dual_run_mutually_exclusive_with_single_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing both ``--run`` and ``--baseline-run`` must fail loudly."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "adoption_gate",
            "--run",
            str(tmp_path / "x"),
            "--baseline-run",
            str(tmp_path / "y"),
            "--candidate-run",
            str(tmp_path / "z"),
        ],
    )
    from scripts.adoption_gate import main

    with pytest.raises(SystemExit):
        main()


def test_dual_run_requires_both_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--baseline-run`` without ``--candidate-run`` must fail."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "adoption_gate",
            "--baseline-run",
            str(tmp_path / "x"),
        ],
    )
    from scripts.adoption_gate import main

    with pytest.raises(SystemExit):
        main()


def test_dual_run_pair_rows_treats_runs_as_incumbent_candidate(
    tmp_path: Path,
) -> None:
    """The dual-run loader assigns ``model_class=_baseline_run`` /
    ``_candidate_run`` per file, regardless of the underlying values."""
    from scripts.adoption_gate import load_dual_run_paired

    rows_baseline = pd.DataFrame(
        {
            "gsis_id": ["00-0034857", "00-0036322"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["QB", "WR"],
            "model_class": ["something", "anything"],  # ignored by the loader
            "mean": [10.0, 8.0],
            "actual_ppr": [11.0, 9.0],
        }
    )
    rows_candidate = rows_baseline.copy()
    rows_candidate["mean"] = [10.5, 8.5]

    baseline_dir = tmp_path / "b"
    candidate_dir = tmp_path / "c"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    rows_baseline.to_parquet(baseline_dir / "results.parquet")
    rows_candidate.to_parquet(candidate_dir / "results.parquet")

    df = load_dual_run_paired(baseline_dir, candidate_dir)
    assert set(df["model_class"].unique()) == {"_baseline_run", "_candidate_run"}
    assert len(df) == 4  # 2 rows x 2 runs


def test_dual_run_row_coverage_mismatch_raises(
    tmp_path: Path,
) -> None:
    """If the two runs have different (gsis_id, season, week, position)
    coverage, the loader raises a clear ``ValueError``."""
    from scripts.adoption_gate import load_dual_run_paired

    rows_baseline = pd.DataFrame(
        {
            "gsis_id": ["00-0034857", "00-0036322"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["QB", "WR"],
            "model_class": ["baseline", "baseline"],
            "mean": [10.0, 8.0],
            "actual_ppr": [11.0, 9.0],
        }
    )
    rows_candidate = rows_baseline.iloc[:1].copy()  # missing 1 row

    baseline_dir = tmp_path / "b"
    candidate_dir = tmp_path / "c"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    rows_baseline.to_parquet(baseline_dir / "results.parquet")
    rows_candidate.to_parquet(candidate_dir / "results.parquet")

    with pytest.raises(ValueError, match="row coverage"):
        load_dual_run_paired(baseline_dir, candidate_dir)


def test_dual_run_loader_propagates_baseline_actuals_into_pair(
    tmp_path: Path,
) -> None:
    """End-to-end loader → pair_rows: the synthesized labels survive the merge
    and the resulting ``actual_ppr_inc`` array equals the baseline file's
    ``actual_ppr`` column (the ``_inc`` suffix consistently maps back to the
    baseline run, never the candidate run)."""
    from scripts.adoption_gate import load_dual_run_paired

    baseline_dir, candidate_dir = _make_dual_run_pair(tmp_path, n_per_position=400)
    df = load_dual_run_paired(baseline_dir, candidate_dir)
    qb_df = df[df["position"] == "QB"]
    inc_pred, cand_pred, actual, _grouping, n_dropped = pair_rows(
        qb_df, incumbent="_baseline_run", candidate="_candidate_run"
    )
    assert n_dropped == 0
    assert len(inc_pred) == len(cand_pred) == len(actual)
    # The candidate predictions were perturbed from the baseline, so the two
    # arrays must not be elementwise identical (sanity check on the loader).
    assert not np.allclose(inc_pred, cand_pred)
