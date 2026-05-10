"""Unit tests for scripts/probe_target_decomposition.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.probe_target_decomposition import _build_arg_parser


def test_arg_parser_defaults_match_spec() -> None:
    """Defaults: --eval-years 2021 2022 2023 2024 --train-start 2018
    --bootstrap-n 5000 --coverage-threshold 0.95."""
    parser = _build_arg_parser()
    args = parser.parse_args(["--output-dir", "reports/probe"])
    assert args.output_dir == Path("reports/probe")
    assert args.eval_years == [2021, 2022, 2023, 2024]
    assert args.train_start == 2018
    assert args.bootstrap_n == 5000
    assert args.coverage_threshold == 0.95
    assert args.seed == 0xD3C0


def test_arg_parser_overrides() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "out",
            "--eval-years",
            "2022",
            "2023",
            "--train-start",
            "2019",
            "--bootstrap-n",
            "1000",
            "--coverage-threshold",
            "0.90",
            "--seed",
            "42",
        ]
    )
    assert args.eval_years == [2022, 2023]
    assert args.train_start == 2019
    assert args.bootstrap_n == 1000
    assert args.coverage_threshold == 0.90
    assert args.seed == 42


def test_main_writes_expected_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: monkey-patch the loader to return synthetic data; verify
    main() writes the 4 expected output files to --output-dir."""
    from scripts.probe_target_decomposition import main

    # Synthetic loader returns (features_by_year, weekly_stats, feature_columns).
    rng = np.random.default_rng(7)
    seasons = (2018, 2019, 2020, 2021)
    # n=150 keeps the train and eval windows above paired_bootstrap_rmse_delta's
    # 100-row floor on every eval year (single-year eval here = 150 paired rows).
    n = 150
    features_by_year: dict[int, pd.DataFrame] = {}
    weekly_rows: list[dict[str, object]] = []
    feature_cols = ["targets_per_game_l4", "receiving_yards_per_game_l4"]
    for season in seasons:
        ids = [f"00-{season:04d}{i:03d}" for i in range(n)]
        weeks = rng.integers(1, 18, size=n)
        features_by_year[season] = pd.DataFrame(
            {
                "gsis_id": ids,
                "season": season,
                "week": weeks,
                "team": "KC",
                "opponent": "DEN",
                **{c: rng.standard_normal(n) for c in feature_cols},
            }
        )
        for i, gid in enumerate(ids):
            t = int(rng.integers(0, 10))
            weekly_rows.append(
                {
                    "gsis_id": gid,
                    "season": season,
                    "week": int(weeks[i]),
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

    import argparse as _argparse

    def fake_load(
        args: _argparse.Namespace,
    ) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, list[str]]:
        return features_by_year, weekly_stats, list(feature_cols)

    monkeypatch.setattr("scripts.probe_target_decomposition._load_inputs", fake_load)

    out_dir = tmp_path / "reports"
    rc = main(
        [
            "--output-dir",
            str(out_dir),
            "--eval-years",
            "2021",
            "--train-start",
            "2018",
            "--bootstrap-n",
            "200",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    assert (out_dir / "feature_probe_target_decomposition_per_stat.csv").exists()
    for stat_name in ("receptions", "receiving_yards", "receiving_tds"):
        md = out_dir / f"feature_probe_target_decomposition_{stat_name}.md"
        assert md.exists()
        text = md.read_text(encoding="utf-8")
        assert stat_name in text
