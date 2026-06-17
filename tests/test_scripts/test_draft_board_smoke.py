"""Headless smoke for the Streamlit draft board: it imports and runs without raising,
renders the best-available picker (not the old search box), and records picks via the
shared confirm flow + the opponent ADP shortcut."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _smoke_session(picks: list[str] | None = None, my_slot: int = 1, n_teams: int = 12):  # type: ignore[no-untyped-def]
    from projections.draft.assistant.live import LiveDraftSession
    from projections.draft.assistant.strategy import RawVorpStrategy
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset, validate_gsis_id

    ids = [f"00-000{i:04d}" for i in range(1, 73)]
    positions = ["RB", "WR", "QB", "TE"] * 18
    names = [f"Player {i}" for i in range(1, 73)]
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * 72, dtype=_PYARROW_STR),
        }
    )
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0 - i for i in range(72)],
            "vorp": [150.0 - i for i in range(72)],
            "replacement_fpts": [100.0] * 72,
            "consensus_adp": pd.array([float(i + 1) for i in range(72)], dtype=pd.Float64Dtype()),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
        }
    )
    league = LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    return LiveDraftSession(
        league=league,
        my_slot=my_slot,
        id_map=id_map,
        pool=pool,
        strategy=RawVorpStrategy(),
        strategy_name="raw_vorp",
        mode="copilot",
        adp_jitter=0.0,
        picks=[validate_gsis_id(p) for p in (picks or [])],
    )


def test_draft_board_loads_without_session() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py").run()
    assert not at.exception
    assert any("Start" in str(getattr(el, "value", "")) for el in at.info)


def test_board_shows_best_available_and_drops_search_box() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = _smoke_session(my_slot=1)
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = None
    at.run()
    assert not at.exception
    # Best-available position dropdown is present with All + the skill positions.
    assert any(set(sb.options) >= {"All", "QB", "RB", "WR", "TE"} for sb in at.selectbox)
    # The old top "Record a pick — search a player" box is gone.
    labels = [str(getattr(ti, "label", "")) for ti in at.text_input]
    assert not any("Record a pick" in lbl for lbl in labels)


def test_board_confirm_records_staged_pick(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = _smoke_session(my_slot=1)  # pick 1 is mine
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = str(tmp_path / "auto.json")
    at.session_state["pending_pick"] = "00-0000003"  # a QB in the fixture, present in id_map
    at.run()
    assert not at.exception
    at.button(key="confirm_pending").click().run()
    assert not at.exception
    assert at.session_state["session"].picks == ["00-0000003"]


def test_board_opponent_adp_shortcut_records(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = _smoke_session(my_slot=2)  # pick 1 is an opponent's
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = str(tmp_path / "auto.json")
    at.run()
    assert not at.exception
    at.button(key="confirm_adp").click().run()
    assert not at.exception
    assert len(at.session_state["session"].picks) == 1


def test_board_results_panel_runs_projected_eval(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from projections.draft.assistant.availability import PlayerAvailability
    from projections.schemas import validate_gsis_id

    sess = _smoke_session(my_slot=1, n_teams=6)  # 6 x 11 = 66 <= 72-player fixture pool
    full = list(sess.pool["gsis_id"].astype(str))[: sess.league.n_teams * sess.league.roster_size]
    sess.picks = [validate_gsis_id(g) for g in full]
    assert sess.is_complete
    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = sess
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = str(tmp_path / "auto.json")
    # inject constant availability so the eval needs no store
    at.session_state["_eval_availability"] = PlayerAvailability(
        p={g: 1.0 for g in sess.pool["gsis_id"].astype(str)}, bye={}
    )
    at.run()
    assert not at.exception
    at.button(key="run_projected_eval").click().run()
    assert not at.exception
    # the panel rendered a championship metric for the hero seat
    assert any("Championship" in str(getattr(m, "label", "")) for m in at.metric)
