"""Integration tests for scripts/generate_snake_cheat_sheet.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from projections.draft.snake_cheat_sheet import DISPLAY_NAME_FALLBACK
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
    assert (df["display_name"] != DISPLAY_NAME_FALLBACK).any()


def test_cli_missing_id_map_logs_warning_and_falls_back(tmp_path: Path) -> None:
    """§5.3 #24 — --id-map points at a non-existent file → warning + '—' names + exit 0."""
    vorp_path = tmp_path / "vorp.parquet"
    _write_synthetic_vorp(vorp_path)
    cfg_path = tmp_path / "league.json"
    _write_league_config(cfg_path)
    out_path = tmp_path / "cheat_sheet.csv"
    missing_id_map = tmp_path / "nope.parquet"  # doesn't exist

    result = _run_cli(
        [
            "--season",
            "2026",
            "--league-config",
            str(cfg_path),
            "--vorp-input",
            str(vorp_path),
            "--id-map",
            str(missing_id_map),
            "--out",
            str(out_path),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "id_map parquet not found" in result.stderr
    df = pd.read_csv(out_path)
    assert (df["display_name"] == DISPLAY_NAME_FALLBACK).all()


def test_cli_tiers_per_position_flag_propagates(tmp_path: Path) -> None:
    """§5.3 #25 — --tiers-per-position 3 caps output tiers at 3 per position."""
    vorp_path = tmp_path / "vorp.parquet"
    _write_synthetic_vorp(vorp_path)
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
            str(tmp_path / "id_map_missing.parquet"),
            "--tiers-per-position",
            "3",
            "--out",
            str(out_path),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    df = pd.read_csv(out_path)
    in_pool = df[df["is_in_pool"]]
    for pos_value in in_pool["position"].unique():
        sub = in_pool[in_pool["position"] == pos_value]
        # tier column comes back as float64 from CSV when there are NaNs;
        # but in_pool rows have integer tiers, so max() is well-defined.
        assert sub["tier"].max() <= 3, f"position {pos_value} has tier > 3"


def test_cheat_sheet_cli_carries_adp_delta(tmp_path: Path) -> None:
    """A consensus-fed VORP parquet (with consensus_adp) -> cheat sheet with
    consensus_adp + adp_delta columns populated."""
    from projections.schemas import _PYARROW_STR, SnakeCheatSheetSchema

    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-1000000", "00-1000001", "00-2000000", "00-2000001"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["QB", "QB", "RB", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.array([320.0, 300.0, 260.0, 250.0], dtype="float64"),
            "vorp": pd.array([80.0, 60.0, 30.0, 20.0], dtype="float64"),
            "replacement_fpts": pd.array([240.0, 240.0, 230.0, 230.0], dtype="float64"),
            "consensus_adp": pd.array([8.0, 3.0, 12.0, 20.0], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    vorp.to_parquet(vorp_path, index=False)

    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "n_teams": 2,
                "budget": 100,
                "min_bid": 1,
                "roster_slots": {"QB": 1, "RB": 1},
                "ruleset": "espn_ppr",
            }
        )
    )
    out_path = tmp_path / "sheet.parquet"
    proc = _run_cli(
        [
            "--season",
            "2026",
            "--league-config",
            str(cfg_path),
            "--vorp-input",
            str(vorp_path),
            "--out",
            str(out_path),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    sheet = pd.read_parquet(out_path)
    SnakeCheatSheetSchema.validate(sheet)
    assert sheet["consensus_adp"].notna().any()
    assert sheet["adp_delta"].notna().any()
    by_gsis = sheet.set_index("gsis_id")
    # QB: 00-1000000 has the better VORP (rank 1) but the LATER ADP (rank 2) -> value, +1.
    assert by_gsis.loc["00-1000000", "adp_delta"] == 1
    # 00-1000001 worse VORP (rank 2) but EARLIER ADP (rank 1) -> reach, -1.
    assert by_gsis.loc["00-1000001", "adp_delta"] == -1
