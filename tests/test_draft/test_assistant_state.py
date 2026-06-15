"""Tests for DraftState + load_draft_state."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState, load_draft_state
from projections.schemas import _PYARROW_STR, GsisId, Position, RosterSlot, Ruleset


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


# _OPP_PICKS must be six UNIQUE ids: the original ["00-0000018"] * 6 fixture was
# six identical ids that the duplicate-pick guard correctly rejects, so unique
# opponent fillers are required here to keep the happy-path tests valid.
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


def test_missing_picks_key_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"league_config": str(cfg), "my_slot": 1}))
    with pytest.raises(ValueError, match="picks"):
        load_draft_state(p, _id_map())


def test_picks_non_list_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"league_config": str(cfg), "my_slot": 1, "picks": "not-a-list"}))
    with pytest.raises(ValueError, match="picks"):
        load_draft_state(p, _id_map())


def test_non_object_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps(["league_config", "my_slot", "picks"]))
    with pytest.raises(ValueError, match="object"):
        load_draft_state(p, _id_map())


def test_my_pick_ids_picks_out_my_snake_slots() -> None:
    # 4 teams, my_slot=1 → my picks are #1, #8, #9 (snake).
    picks = tuple(
        GsisId(g)
        for g in [
            "00-0000001",  # #1 mine
            "00-0000002",
            "00-0000003",
            "00-0000004",
            "00-0000005",
            "00-0000006",
            "00-0000007",
            "00-0000008",  # #8 mine
            "00-0000009",  # #9 mine
        ]
    )
    state = DraftState(
        my_slot=1,
        n_teams=4,
        rounds=5,
        picks=picks,
        my_roster=(Position.RB, Position.WR, Position.RB),
    )
    assert state.my_pick_ids == ("00-0000001", "00-0000008", "00-0000009")


def test_my_pick_ids_empty_when_no_picks() -> None:
    state = DraftState(my_slot=1, n_teams=4, rounds=5, picks=(), my_roster=())
    assert state.my_pick_ids == ()


def test_build_draft_state_matches_load_draft_state(tmp_path: Path) -> None:
    from projections.draft.assistant.state import build_draft_state
    from projections.draft.league_config import LeagueConfig

    cfg_path = _write_config(tmp_path)
    league = LeagueConfig.model_validate_json(cfg_path.read_text())
    picks = [*_OPP_PICKS, "00-0000007", "00-0000008"]
    state_path = _state_file(tmp_path, cfg_path, picks)

    from_file, _ = load_draft_state(state_path, _id_map())
    in_memory = build_draft_state(picks, my_slot=7, league=league, id_map=_id_map())
    assert in_memory == from_file


def test_build_draft_state_bad_slot_raises() -> None:
    from projections.draft.assistant.state import build_draft_state
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    league = LeagueConfig(
        name="t", n_teams=12,
        roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 1}, ruleset=Ruleset.espn_ppr(),
    )
    with pytest.raises(ValueError, match="my_slot"):
        build_draft_state([], my_slot=99, league=league, id_map=_id_map())
