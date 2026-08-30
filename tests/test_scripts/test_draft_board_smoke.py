"""Headless smoke for the Streamlit draft board: it imports and runs without raising,
renders the best-available picker (not the old search box), and records picks via the
shared confirm flow + the opponent ADP shortcut."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# AppTest's 3s default is a wall-clock budget for the whole script run; under `pytest -n auto`
# the CPU contention alone blows it (the long-standing parallel flake on this file). Generous,
# not slow: the run finishes in well under a second when it has a core to itself.
_TIMEOUT = 60
# AppTest resolves a relative script path against the *caller's* directory, not the repo
# root, so a bare "scripts/draft_board.py" looks for tests/test_scripts/scripts/... and
# raises FileNotFoundError. Spell the path out from this file.
_BOARD = str(Path(__file__).resolve().parents[2] / "scripts" / "draft_board.py")
_PROFILE_ROOT_ENV = "FF_BOARD_PROFILE_ROOT"


@pytest.fixture(autouse=True)
def _no_configured_leagues(tmp_path_factory, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Point league discovery at an empty directory for every test in this file.

    `data/leagues/` is gitignored user data. Left to the real default, whether the board comes
    up on a preset or on the developer's own league would depend on that machine's untracked
    files — these tests would pass here and exercise a different code path on a fresh clone.
    Tests that want a profile opt in by re-pointing the env var.
    """
    monkeypatch.setenv(_PROFILE_ROOT_ENV, str(tmp_path_factory.mktemp("no_leagues")))


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

    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT).run()
    assert not at.exception
    assert any("Start" in str(getattr(el, "value", "")) for el in at.info)


def test_board_shows_best_available_and_drops_search_box() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT)
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

    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT)
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

    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT)
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
    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT)
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


def _write_board_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write a VORP parquet + id_map parquet big enough for a 16-team draft."""
    from projections.schemas import _PYARROW_STR

    n = 240
    ids = [f"00-000{i:04d}" for i in range(1, n + 1)]
    positions = ["RB", "WR", "QB", "TE"] * (n // 4)
    names = [f"Player {i}" for i in range(1, n + 1)]
    id_map_path = tmp_path / "id_map.parquet"
    pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            # IdMapSchema requires the other id flavours; the board reads them off disk
            # through the real schema, unlike `_smoke_session` which builds a frame in memory.
            "espn_id": pd.array([None] * n, dtype=_PYARROW_STR),
            "sleeper_id": pd.array([None] * n, dtype=_PYARROW_STR),
            "pfr_id": pd.array([None] * n, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * n, dtype=_PYARROW_STR),
        }
    ).to_parquet(id_map_path)
    vorp_path = tmp_path / "pool.parquet"
    pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0 - i for i in range(n)],
            "vorp": [150.0 - i for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
        }
    ).to_parquet(vorp_path)
    return vorp_path, id_map_path


def _start_board(at, vorp: Path, id_map: Path, league_json: Path):  # type: ignore[no-untyped-def]
    """Fill the sidebar's custom-league inputs by label and press Start."""
    at.run()
    for ti in at.text_input:
        label = str(getattr(ti, "label", ""))
        if label.startswith("VORP parquet"):
            ti.set_value(str(vorp))
        elif label.startswith("league_config JSON"):
            ti.set_value(str(league_json))
        elif label.startswith("id_map"):
            ti.set_value(str(id_map))
    # Widget values only reach the script on the *next* run; clicking Start in the same
    # pass would read the defaults and quietly start a preset draft instead.
    at.run()
    for b in at.button:
        if "Start" in str(getattr(b, "label", "")):
            return b.click().run()
    raise AssertionError("Start button not found")


def test_custom_league_config_overrides_the_preset_roster(tmp_path: Path) -> None:
    """The preset for Half-PPR/16 starts 3 WR over 13 rounds; this league starts 2 over 12.

    `league.roster_slots` decides roster eligibility for every recommendation, so a preset
    silently substituted for the real config recommends a third receiver that cannot start
    and runs the draft one round too long. Pin that the file wins.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from projections.draft.assistant.presets import get_preset
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    preset = get_preset("half", 16).league_config
    assert preset.roster_slots[RosterSlot.WR] == 3 and preset.roster_size == 13

    league = LeagueConfig(
        name="critts-shaped",
        n_teams=16,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
            RosterSlot.IR: 2,
        },
        ruleset=Ruleset.espn_half(),
    )
    league_json = tmp_path / "league_config.json"
    league_json.write_text(league.model_dump_json(indent=2))

    vorp, id_map = _write_board_inputs(tmp_path)
    at = _start_board(
        AppTest.from_file(_BOARD, default_timeout=_TIMEOUT), vorp, id_map, league_json
    )

    assert not at.exception
    sess = at.session_state["session"]
    assert sess.league.roster_slots[RosterSlot.WR] == 2
    assert sess.league.roster_size == 12  # IR is not drafted
    assert sess.league.name == "critts-shaped"


def test_custom_league_config_refuses_a_team_count_mismatch(tmp_path: Path) -> None:
    """The Teams dropdown bounds the slot picker and picks the preset table. A config that
    disagrees with it would draft the wrong number of rounds against an unvalidated slot."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    league = LeagueConfig(
        name="ten-team",
        n_teams=10,  # dropdown defaults to 16
        roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.BENCH: 5},
        ruleset=Ruleset.espn_half(),
    )
    league_json = tmp_path / "league_config.json"
    league_json.write_text(league.model_dump_json(indent=2))

    vorp, id_map = _write_board_inputs(tmp_path)
    at = _start_board(
        AppTest.from_file(_BOARD, default_timeout=_TIMEOUT), vorp, id_map, league_json
    )

    assert not at.exception
    assert "session" not in at.session_state
    assert any("n_teams=10" in str(getattr(e, "value", "")) for e in at.error)


def _write_profile_root(tmp_path: Path) -> Path:
    """A league directory holding a real `board_profile.json` and everything it points at."""
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    vorp, id_map = _write_board_inputs(tmp_path)
    league = LeagueConfig(
        name="Critts-shaped",
        n_teams=16,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
            RosterSlot.IR: 2,
        },
        ruleset=Ruleset.espn_half(),
    )
    league_json = tmp_path / "league_config.json"
    league_json.write_text(league.model_dump_json(indent=2))

    root = tmp_path / "leagues"
    league_dir = root / "critts_2025_2026"
    league_dir.mkdir(parents=True)
    (league_dir / "board_profile.json").write_text(
        json.dumps(
            {
                "name": "Critts 2026",
                "league_config": str(league_json),
                "vorp_table": str(vorp),
                "id_map": str(id_map),
                "my_slot": 8,
                "season": 2026,
                "strategy": "raw_vorp",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_board_starts_the_configured_league_with_no_sidebar_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of a profile: press Start, touch nothing, and be drafting the right league.

    Pins all four settings that were previously wrong or blank by default — roster shape,
    seat, strategy and pool — because a board that is right about three of them and wrong
    about the fourth still recommends the wrong player and looks entirely normal doing it.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from projections.schemas import RosterSlot

    monkeypatch.setenv(_PROFILE_ROOT_ENV, str(_write_profile_root(tmp_path)))
    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT).run()
    assert not at.exception

    for b in at.button:
        if "Start" in str(getattr(b, "label", "")):
            at = b.click().run()
            break
    assert not at.exception

    sess = at.session_state["session"]
    assert sess.league.roster_slots[RosterSlot.WR] == 2  # not the preset's 3
    assert sess.league.roster_size == 12  # not the preset's 13
    assert sess.my_slot == 8  # not the old default of 1
    assert sess.strategy_name == "raw_vorp"  # not BOARD_STRATEGIES[0] == "now_or_never"
    assert sess.league.name == "Critts-shaped"


def test_board_offers_the_generic_preset_alongside_a_configured_league(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock drafts in other formats must stay reachable, and the real league must be first."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(_PROFILE_ROOT_ENV, str(_write_profile_root(tmp_path)))
    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT).run()

    league_box = next(sb for sb in at.selectbox if str(getattr(sb, "label", "")) == "League")
    # AppTest surfaces the *formatted* options, which is what the user actually reads.
    assert league_box.value == "critts_2025_2026"  # the configured league is the default
    assert league_box.options[0].startswith("Critts 2026")
    assert "16 teams, 12 rounds, slot 8" in league_box.options[0]
    assert league_box.options[-1] == "Generic preset (mock drafts)"
    # Scoring/Teams are the config's to state, so they are not offered while it is selected.
    labels = [str(getattr(sb, "label", "")) for sb in at.selectbox]
    assert "Scoring" not in labels and "Teams" not in labels


def test_board_warns_about_a_broken_profile_rather_than_ignoring_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently skipped profile drops the board to a preset that looks fine and is not
    this league — the failure the whole feature exists to prevent."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = _write_profile_root(tmp_path)
    broken = root / "typo_league"
    broken.mkdir()
    (broken / "board_profile.json").write_text("{not json", encoding="utf-8")

    monkeypatch.setenv(_PROFILE_ROOT_ENV, str(root))
    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT).run()

    assert not at.exception
    assert any("typo_league" in str(getattr(w, "value", "")) for w in at.warning)
    # ...and the good profile is still the default.
    league_box = next(sb for sb in at.selectbox if str(getattr(sb, "label", "")) == "League")
    assert league_box.value == "critts_2025_2026"
