"""Headless smoke for the Streamlit auction board: it imports and runs without raising,
prices a staged player, records a sale through the confirm flow, and undoes one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_N = 96
_IDS = [f"00-000{i:04d}" for i in range(1, _N + 1)]
_POSITIONS = ["RB", "WR", "QB", "TE"] * (_N // 4)


def _smoke_session(my_seat: int = 1, n_teams: int = 8):  # type: ignore[no-untyped-def]
    from projections.draft.assistant.auction.live import (
        BOARD_BID_MODELS,
        DEFAULT_BID_MODEL,
        LiveAuctionSession,
    )
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset

    names = [f"Player {i}" for i in range(1, _N + 1)]
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(_IDS, dtype=_PYARROW_STR),
            "position": pd.array(_POSITIONS, dtype=_PYARROW_STR),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * _N, dtype=_PYARROW_STR),
        }
    )
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(_IDS, dtype=_PYARROW_STR),
            "position": pd.array(_POSITIONS, dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0 - i for i in range(_N)],
            "vorp": [150.0 - i for i in range(_N)],
            "replacement_fpts": [100.0] * _N,
            "consensus_adp": pd.array([float(i + 1) for i in range(_N)], dtype=pd.Float64Dtype()),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
            "espn_auction_dollars": pd.array(
                [max(1, 80 - 2 * i) for i in range(_N)], dtype=pd.Int64Dtype()
            ),
        }
    )
    league = LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=200,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 4,
        },
        ruleset=Ruleset.espn_half(),
    )
    return LiveAuctionSession(
        league=league,
        my_seat=my_seat,
        id_map=id_map,
        pool=pool,
        strategy=BOARD_BID_MODELS[DEFAULT_BID_MODEL],
        strategy_name=DEFAULT_BID_MODEL,
    )


# AppTest's 3s default is a wall-clock budget for the whole script run, and these render a
# priced board; under `pytest -n auto` the CPU contention alone blows it. Generous, not slow:
# the run finishes in well under a second when it has a core to itself.
_TIMEOUT = 60
# AppTest resolves a relative script path against the *caller's* directory, not the repo
# root, so a bare "scripts/auction_board.py" looks for tests/test_scripts/scripts/... and
# raises FileNotFoundError. Spell the path out from this file.
_BOARD = str(Path(__file__).resolve().parents[2] / "scripts" / "auction_board.py")


def _app(sess=None, tmp_path: Path | None = None, pending: str | None = None):  # type: ignore[no-untyped-def]
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_BOARD, default_timeout=_TIMEOUT)
    if sess is not None:
        at.session_state["session"] = sess
        at.session_state["session_token"] = "tok"
        at.session_state["autosave_path"] = str(tmp_path / "auto.json") if tmp_path else None
        at.session_state["pending_player"] = pending
    return at


def _winner_box(at: Any) -> Any:
    """The 'Winning team' selectbox, found by label so the test does not pin the widget key."""
    return next(sb for sb in at.selectbox if "Winning team" in str(getattr(sb, "label", "")))


def test_auction_board_loads_without_session() -> None:
    pytest.importorskip("streamlit")
    at = _app().run()
    assert not at.exception
    assert any("Start" in str(getattr(el, "value", "")) for el in at.info)


def test_board_renders_bid_board_and_budgets() -> None:
    pytest.importorskip("streamlit")
    at = _app(_smoke_session()).run()
    assert not at.exception
    assert any(set(sb.options) >= {"All", "QB", "RB", "WR", "TE"} for sb in at.selectbox)
    assert any("Budget left" in str(getattr(m, "label", "")) for m in at.metric)


def test_staged_player_shows_a_max_bid_and_records_the_sale(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    at = _app(_smoke_session(), tmp_path, pending=_IDS[0]).run()
    assert not at.exception
    assert any("Bid up to $" in str(getattr(md, "value", "")) for md in at.markdown)
    # The winner is no longer pre-selected, so recording a sale now takes an explicit choice.
    _winner_box(at).set_value(1).run()
    at.number_input[0].set_value(37).run()
    next(b for b in at.button if "→" in str(getattr(b, "label", ""))).click().run()
    assert not at.exception
    sess = at.session_state["session"]
    assert [(str(p.gsis_id), p.seat, p.price) for p in sess.purchases] == [(_IDS[0], 1, 37)]
    assert at.session_state["pending_player"] is None
    assert (tmp_path / "auto.json").exists()  # autosaved


def test_price_does_not_carry_over_to_a_newly_staged_player(tmp_path: Path) -> None:
    """A price typed for one player must not survive into the next player staged for the
    same lot. The widget key was the lot number alone, so Streamlit retained the value and
    the confirm button offered to record a different, cheaper player at the old price."""
    pytest.importorskip("streamlit")
    sess = _smoke_session()
    at = _app(sess, tmp_path, pending=_IDS[0]).run()
    assert not at.exception
    _winner_box(at).set_value(1).run()
    at.number_input[0].set_value(52).run()
    assert int(at.number_input[0].value) == 52

    at.session_state["pending_player"] = _IDS[3]  # a different, cheaper player, same lot
    at.run()
    assert not at.exception
    _winner_box(at).set_value(1).run()
    advice = sess.advise(_IDS[3])
    expected = max(sess.league.min_bid, min(advice.market_value, sess.feasible_max(1)))
    assert int(at.number_input[0].value) == expected
    assert not any("$52" in str(getattr(b, "label", "")) for b in at.button)


def test_winner_must_be_chosen_and_full_teams_are_not_offered(tmp_path: Path) -> None:
    """No seat is pre-selected (in a 12-team room the hero wins ~1 lot in 12, so any default
    is wrong most of the time and confirm is one click), and a seat with no open roster slot
    is not offerable at all -- record_purchase would reject it."""
    pytest.importorskip("streamlit")
    at = _app(_smoke_session(), tmp_path, pending=_IDS[0]).run()
    assert not at.exception
    assert _winner_box(at).value is None
    assert not any("→" in str(getattr(b, "label", "")) for b in at.button)

    full = _smoke_session()
    gids = iter(_IDS)
    while full.open_slots(2) > 0:
        full.record_purchase(next(gids), 2, 1)
    at2 = _app(full, tmp_path, pending=_IDS[-1]).run()
    assert not at2.exception
    assert len(_winner_box(at2).options) == full.league.n_teams - 1


def test_board_undo_removes_the_last_sale(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    sess = _smoke_session()
    sess.record_purchase(_IDS[0], 2, 25)
    at = _app(sess, tmp_path).run()
    assert not at.exception
    at.button(key="undo_last").click().run()
    assert not at.exception
    assert at.session_state["session"].purchases == []


def test_board_nomination_panel_names_a_nominee() -> None:
    pytest.importorskip("streamlit")
    at = _app(_smoke_session()).run()
    assert not at.exception
    assert any("Nominate:" in str(getattr(md, "value", "")) for md in at.markdown)


def test_board_results_panel_runs_projected_eval(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from projections.draft.assistant.availability import PlayerAvailability

    # 6 teams is the projected eval's minimum (top-6 playoff bracket); 6 x 11 = 66 of 96 players.
    sess = _smoke_session(n_teams=6)
    seat = 1
    for gid in _IDS:
        if sess.is_complete:
            break
        if sess.open_slots(seat) == 0:
            seat += 1
        sess.record_purchase(gid, seat, 1)
    assert sess.is_complete
    at = _app(sess, tmp_path)
    at.session_state["_eval_availability"] = PlayerAvailability(
        p=dict.fromkeys(sess.pool["gsis_id"].astype(str), 1.0), bye={}
    )
    at.run()
    assert not at.exception
    at.button(key="run_projected_eval").click().run()
    assert not at.exception
    assert any("Championship" in str(getattr(m, "label", "")) for m in at.metric)
