"""End-to-end smoke test for the draft-assistant CLI core."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.cli import generate_recommendation, run
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RecommendationSchema, RosterSlot, Ruleset


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json())

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"league_config": str(cfg_path), "my_slot": 7, "picks": []}))

    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 252.0],
            "vorp": [50.0, 52.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([5.0, 7.0], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    vorp.to_parquet(vorp_path, index=False)

    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "espn_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "sleeper_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "pfr_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "full_name": pd.array(["RB One", "WR One"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "team": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
        }
    )
    id_path = tmp_path / "id_map.parquet"
    id_map.to_parquet(id_path, index=False)
    return state_path, vorp_path, id_path


def test_generate_recommendation(tmp_path: Path) -> None:
    state_path, vorp_path, id_path = _setup(tmp_path)
    rec = generate_recommendation(
        state_path=state_path,
        vorp_path=vorp_path,
        id_map_path=id_path,
        strategy_name="now_or_never",
        sigma=None,
    )
    RecommendationSchema.validate(rec)
    assert set(rec["gsis_id"]) == {"00-0000010", "00-0000020"}


def test_run_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_path, vorp_path, id_path = _setup(tmp_path)
    code = run(
        [
            "--state",
            str(state_path),
            "--vorp-table",
            str(vorp_path),
            "--id-map",
            str(id_path),
            "--strategy",
            "raw_vorp",
            "--top",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "WR One" in out and "RB One" in out
