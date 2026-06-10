"""Tests for DraftState + load_draft_state."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState, load_draft_state
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset


def _write_config(tmp_path: Path) -> Path:
    from projections.draft.league_config import LeagueConfig

    cfg = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    p = tmp_path / "cfg.json"
    p.write_text(cfg.model_dump_json())
    return p


def _id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000007", "00-0000008", "00-0000018"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR", "QB"], dtype=_PYARROW_STR),
            "full_name": pd.array(["A", "B", "C"], dtype=_PYARROW_STR),
        }
    )


def _state_file(tmp_path: Path, cfg_path: Path, picks: list[str]) -> Path:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"league_config": str(cfg_path), "my_slot": 7, "picks": picks}))
    return p


_OPP_PICKS = [f"00-000000{i}" for i in range(1, 7)]  # 6 unique opponent-filler ids


def test_my_roster_from_id_map(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    # Slot 7 of 12 owns pick 7 → "00-0000007" (RB) is mine; pick 8 is an opponent's.
    state_path = _state_file(tmp_path, cfg, [*_OPP_PICKS, "00-0000007", "00-0000008"])
    state, league = load_draft_state(state_path, _id_map())
    assert isinstance(state, DraftState)
    assert state.current_pick == 9
    assert state.drafted_ids == frozenset(set(_OPP_PICKS) | {"00-0000007", "00-0000008"})
    assert state.my_roster == (Position.RB,)  # only my pick #7
    assert league.n_teams == 12


def test_missing_id_map_entry_for_my_pick_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    # My pick (#7) references an id not in id_map.
    state_path = _state_file(tmp_path, cfg, [*_OPP_PICKS, "00-0000099"])
    with pytest.raises(ValueError, match="00-0000099"):
        load_draft_state(state_path, _id_map())


def test_duplicate_pick_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    state_path = _state_file(tmp_path, cfg, ["00-0000007", "00-0000007"])
    with pytest.raises(ValueError, match="duplicate"):
        load_draft_state(state_path, _id_map())


def test_bad_slot_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"league_config": str(cfg), "my_slot": 99, "picks": []}))
    with pytest.raises(ValueError, match="my_slot"):
        load_draft_state(p, _id_map())
