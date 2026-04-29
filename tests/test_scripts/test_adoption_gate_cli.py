"""Plan 8 — adoption gate CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.adoption_gate import load_run_parquet, validate_model_classes_present


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
