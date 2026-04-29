"""Plan 8 — adoption gate CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.adoption_gate import (
    evaluate_position,
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
