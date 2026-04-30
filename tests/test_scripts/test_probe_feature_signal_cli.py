# tests/test_scripts/test_probe_feature_signal_cli.py
"""Feature signal probe CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.probe_feature_signal import parse_args


def test_parse_args_minimum_required_args() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "test_candidate",
            "--override",
            "/tmp/overrides.parquet",
        ]
    )
    assert args.candidate_name == "test_candidate"
    assert args.override == [Path("/tmp/overrides.parquet")]
    assert args.drop == []
    assert args.model == "baseline"
    assert args.position == ["QB", "RB", "WR", "TE"]
    assert args.seasons == (2018, 2024)
    assert args.holdout_years == (2021, 2024)
    assert args.n_bootstrap == 1000
    assert args.seed == 42
    assert args.csv_out is None
    assert args.composite is True


def test_parse_args_drop_only_is_valid() -> None:
    """Ablation mode — drop a column from baseline, no override."""
    args = parse_args(
        [
            "--candidate-name",
            "ablation_drop_one",
            "--drop",
            "opp_allowed_qb_fppg_l4",
        ]
    )
    assert args.override == []
    assert args.drop == ["opp_allowed_qb_fppg_l4"]


def test_parse_args_rejects_no_override_and_no_drop() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--candidate-name", "test"])


def test_parse_args_repeat_override() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "multi",
            "--override",
            "/tmp/a.parquet",
            "--override",
            "/tmp/b.parquet",
        ]
    )
    assert args.override == [Path("/tmp/a.parquet"), Path("/tmp/b.parquet")]


def test_parse_args_drop_comma_separated() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "col_a,col_b,col_c",
        ]
    )
    assert args.drop == ["col_a", "col_b", "col_c"]


def test_parse_args_repeat_position() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--position",
            "QB",
            "--position",
            "WR",
        ]
    )
    assert args.position == ["QB", "WR"]


def test_parse_args_seasons_range() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--seasons",
            "2019-2023",
            "--holdout-years",
            "2022-2023",
        ]
    )
    assert args.seasons == (2019, 2023)
    assert args.holdout_years == (2022, 2023)


def test_parse_args_no_composite_flag() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--no-composite",
        ]
    )
    assert args.composite is False


def test_parse_args_model_choice() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--model",
            "lightgbm-nb",
        ]
    )
    assert args.model == "lightgbm-nb"


def test_parse_args_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--candidate-name",
                "x",
                "--drop",
                "y",
                "--model",
                "bogus",
            ]
        )
