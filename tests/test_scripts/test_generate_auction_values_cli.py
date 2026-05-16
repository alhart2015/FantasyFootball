"""End-to-end integration test for `scripts/generate_auction_values.py`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR, AuctionValuesSchema, Position, RosterSlot


@pytest.fixture
def cli_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build a minimal LeagueConfig JSON + VORP parquet for an end-to-end CLI run."""
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "tiny_test",
                "n_teams": 2,
                "budget": 100,
                "min_bid": 1,
                "roster_slots": {
                    RosterSlot.QB.value: 1,
                    RosterSlot.RB.value: 1,
                    RosterSlot.BENCH.value: 1,
                },
                "ruleset": "standard",
            }
        )
    )
    rows = [
        {
            "gsis_id": "00-1000001",
            "position": Position.QB.value,
            "season_mean_fpts": 320.0,
            "vorp": 100.0,
        },
        {
            "gsis_id": "00-1000002",
            "position": Position.QB.value,
            "season_mean_fpts": 280.0,
            "vorp": 60.0,
        },
        {
            "gsis_id": "00-2000001",
            "position": Position.RB.value,
            "season_mean_fpts": 260.0,
            "vorp": 80.0,
        },
        {
            "gsis_id": "00-2000002",
            "position": Position.RB.value,
            "season_mean_fpts": 220.0,
            "vorp": 40.0,
        },
        {
            "gsis_id": "00-2000003",
            "position": Position.RB.value,
            "season_mean_fpts": 180.0,
            "vorp": 5.0,
        },
        {
            "gsis_id": "00-2000004",
            "position": Position.RB.value,
            "season_mean_fpts": 120.0,
            "vorp": -20.0,
        },
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    vorp_path = tmp_path / "vorp.parquet"
    df.to_parquet(vorp_path, index=False)
    return {
        "config": cfg_path,
        "vorp": vorp_path,
        "out_csv": tmp_path / "auction_values.csv",
        "out_parquet": tmp_path / "auction_values.parquet",
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_auction_values.py"
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )


def test_cli_csv_output_sum_invariant(cli_inputs: dict[str, Path]) -> None:
    proc = _run_cli(
        "--season",
        "2026",
        "--league-config",
        str(cli_inputs["config"]),
        "--vorp-input",
        str(cli_inputs["vorp"]),
        "--out",
        str(cli_inputs["out_csv"]),
    )
    assert proc.returncode == 0, proc.stderr
    out = pd.read_csv(cli_inputs["out_csv"])
    # 2 teams x 3 roster slots = 6 in-pool players (4 RB + 2 QB)
    assert int(out["auction_dollars"].sum()) == 2 * 100  # n_teams * budget
    assert int(out["in_pool"].sum()) == 6


def test_cli_parquet_output_schema(cli_inputs: dict[str, Path]) -> None:
    _run_cli(
        "--season",
        "2026",
        "--league-config",
        str(cli_inputs["config"]),
        "--vorp-input",
        str(cli_inputs["vorp"]),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    out = pd.read_parquet(cli_inputs["out_parquet"])
    # round-trip through the canonical schema
    AuctionValuesSchema.validate(out)


def test_cli_errors_on_missing_vorp_input(cli_inputs: dict[str, Path], tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_cli(
            "--season",
            "2026",
            "--league-config",
            str(cli_inputs["config"]),
            "--vorp-input",
            str(missing),
            "--out",
            str(cli_inputs["out_csv"]),
        )
    stderr = exc_info.value.stderr.lower()
    assert "no such file" in stderr or "does_not_exist" in stderr
