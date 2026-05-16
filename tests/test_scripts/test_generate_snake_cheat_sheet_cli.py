"""Integration tests for scripts/generate_snake_cheat_sheet.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from projections.schemas import _PYARROW_STR, IdMapSchema, Position, VorpTableSchema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "generate_snake_cheat_sheet.py"


_POSITION_ID_PREFIX: dict[Position, int] = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
}


def _write_synthetic_vorp(path: Path) -> pd.DataFrame:
    """Write a small synthetic VorpTableSchema parquet at `path`. Returns the frame."""
    rows: list[dict[str, object]] = []
    base = 300.0
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        count = {Position.QB: 16, Position.RB: 30, Position.WR: 30, Position.TE: 12}[pos]
        replacement = base - (count - 1)
        for i in range(count):
            mean_fpts = base - i
            rows.append(
                {
                    "gsis_id": f"00-{_POSITION_ID_PREFIX[pos]}{i:06d}",
                    "position": pos.value,
                    "season_mean_fpts": mean_fpts,
                    "vorp": mean_fpts - replacement,
                    "replacement_fpts": replacement,
                }
            )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df = VorpTableSchema.validate(df)
    df.to_parquet(path)
    return df


def _position_for_gsis_id(gsis_id: str) -> str:
    """Derive Position.value from the synthetic gsis_id prefix (`00-{P}{i:06d}`)."""
    prefix_char = gsis_id[3]
    mapping = {"1": Position.QB, "2": Position.RB, "3": Position.WR, "4": Position.TE}
    return mapping[prefix_char].value


def _write_synthetic_id_map(path: Path, gsis_ids: list[str]) -> None:
    """Write a minimal IdMapSchema-valid parquet covering every gsis_id."""
    df = pd.DataFrame(
        {
            "gsis_id": pd.Series(gsis_ids, dtype=_PYARROW_STR),
            "espn_id": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
            "sleeper_id": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
            "pfr_id": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
            "full_name": pd.Series([f"Player {gid}" for gid in gsis_ids], dtype=_PYARROW_STR),
            "position": pd.Series(
                [_position_for_gsis_id(gid) for gid in gsis_ids],
                dtype=_PYARROW_STR,
            ),
            "team": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
        }
    )
    IdMapSchema.validate(df).to_parquet(path)


def _write_league_config(path: Path) -> None:
    cfg = {
        "name": "test_ppr",
        "n_teams": 4,
        "budget": 100,
        "min_bid": 1,
        "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 1},
        "ruleset": "espn_ppr",
    }
    path.write_text(json.dumps(cfg))


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI with PYTHONPATH=src so subprocess imports work."""
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_end_to_end_happy_path(tmp_path: Path) -> None:
    """§5.3 #23 — script produces a valid output CSV from synthetic inputs."""
    vorp_path = tmp_path / "vorp.parquet"
    vorp = _write_synthetic_vorp(vorp_path)
    id_map_path = tmp_path / "id_map.parquet"
    _write_synthetic_id_map(id_map_path, list(vorp["gsis_id"]))
    cfg_path = tmp_path / "league.json"
    _write_league_config(cfg_path)
    out_path = tmp_path / "cheat_sheet.csv"

    result = _run_cli(
        [
            "--season",
            "2026",
            "--league-config",
            str(cfg_path),
            "--vorp-input",
            str(vorp_path),
            "--id-map",
            str(id_map_path),
            "--out",
            str(out_path),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert out_path.exists()

    df = pd.read_csv(out_path)
    expected_cols = {
        "gsis_id",
        "position",
        "display_name",
        "positional_rank",
        "season_mean_fpts",
        "vorp",
        "replacement_fpts",
        "is_in_pool",
        "tier",
    }
    assert set(df.columns) >= expected_cols
    assert len(df) == len(vorp)
    assert (df["display_name"] != "—").any()
