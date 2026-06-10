"""Smoke tests for the tournament CLI (both modes, end-to-end)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.tournament_cli import run
from projections.schemas import _PYARROW_STR


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    n = 24
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    pool.to_parquet(vorp_path)

    cfg = {
        "name": "test",
        "n_teams": 4,
        "roster_slots": {"RB": 1, "WR": 1, "FLEX": 1, "BENCH": 2},
        "ruleset": "espn_ppr",
    }
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(json.dumps(cfg))
    return vorp_path, cfg_path


def test_compare_mode_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vorp_path, cfg_path = _write_inputs(tmp_path)
    code = run(
        [
            "--vorp-table",
            str(vorp_path),
            "--league-config",
            str(cfg_path),
            "--my-slot",
            "2",
            "--seeds",
            "10",
            "--seed",
            "0",
            "compare",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "now_or_never" in out and "raw_vorp" in out
    assert "Winner:" in out  # the winner/no-separation line is always printed


def test_tune_sigma_mode_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vorp_path, cfg_path = _write_inputs(tmp_path)
    code = run(
        [
            "--vorp-table",
            str(vorp_path),
            "--league-config",
            str(cfg_path),
            "--my-slot",
            "2",
            "--seeds",
            "8",
            "--seed",
            "0",
            "tune-sigma",
            "--sigma-grid",
            "2,4",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "recommended" in out.lower()


def test_tune_sigma_tolerates_trailing_comma(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vorp_path, cfg_path = _write_inputs(tmp_path)
    code = run(
        [
            "--vorp-table",
            str(vorp_path),
            "--league-config",
            str(cfg_path),
            "--my-slot",
            "2",
            "--seeds",
            "8",
            "--seed",
            "0",
            "tune-sigma",
            "--sigma-grid",
            "2, 4,",  # whitespace + trailing comma must not crash
        ]
    )
    assert code == 0
    assert "recommended" in capsys.readouterr().out.lower()


def test_tune_sigma_rejects_non_positive_grid_value(tmp_path: Path) -> None:
    # A 0 (or negative) sigma must fail at the CLI parse layer, not mid-tune_sigma
    # loop inside LogisticSurvival after wasted simulation (code-review #2).
    vorp_path, cfg_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="must all be > 0"):
        run(
            [
                "--vorp-table",
                str(vorp_path),
                "--league-config",
                str(cfg_path),
                "--my-slot",
                "2",
                "--seeds",
                "8",
                "tune-sigma",
                "--sigma-grid",
                "2,0,4",
            ]
        )


def test_missing_vorp_table_fails_loud(tmp_path: Path) -> None:
    _, cfg_path = _write_inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        run(
            [
                "--vorp-table",
                str(tmp_path / "does_not_exist.parquet"),
                "--league-config",
                str(cfg_path),
                "--my-slot",
                "2",
                "--seeds",
                "8",
                "compare",
            ]
        )
