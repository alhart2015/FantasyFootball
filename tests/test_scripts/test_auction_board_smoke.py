"""Headless smoke for the Streamlit auction board: it imports and runs without raising,
prices a staged player, records a sale through the confirm flow, and undoes one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts.auction_board import _available_ids, _option_label

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


def _winner_box_named(at: Any, label: str) -> Any:
    """A selectbox found by label, so tests do not pin Streamlit widget keys."""
    return next(sb for sb in at.selectbox if label in str(getattr(sb, "label", "")))


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


def test_projection_cache_key_distinguishes_the_buyer() -> None:
    """The projected-eval cache must key on (player, seat, price), not on players alone.

    An auction's rosters are (player, seat) pairs -- unlike a snake draft, where pick order
    alone fixes the seat and a player-only key is sound. Undo a sale, re-award the same
    player to a different team, and a player-only key is byte-identical while the rosters
    differ, so the board would serve the previous league's win%/champ% numbers."""
    sess = _smoke_session()
    sess.record_purchase(_IDS[0], 2, 25)
    before = sess.state_key
    sess.undo()
    sess.record_purchase(_IDS[0], 3, 25)  # same player, same price, different team
    assert sess.state_key != before
    assert tuple(p[0] for p in sess.state_key) == tuple(p[0] for p in before)  # players agree


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


def _minimal(sess=None, tmp_path: Path | None = None, named: bool = True):  # type: ignore[no-untyped-def]
    """An app in minimal mode, teams named by default so the gate is out of the way."""
    if sess is not None and named:
        sess.team_names = tuple(f"Squad {seat}" for seat in sess.seats)
    at = _app(sess, tmp_path)
    at.session_state["view_mode"] = "Minimal (live draft)"
    return at


def test_minimal_mode_gates_on_team_names_then_lets_them_through() -> None:
    """Naming is step one: the confirm sentence has to read a real name, not 'Team 6'."""
    pytest.importorskip("streamlit")
    at = _minimal(_smoke_session(), named=False).run()
    assert not at.exception
    assert any("Name the teams" in str(getattr(h, "value", "")) for h in at.subheader)
    # the lot UI is not reachable until naming is done
    assert not any("Nominated player" in str(getattr(sb, "label", "")) for sb in at.selectbox)

    at2 = _minimal(_smoke_session()).run()
    assert not at2.exception
    assert not any("Name the teams" in str(getattr(h, "value", "")) for h in at2.subheader)


def test_minimal_mode_names_reach_the_session(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    sess = _smoke_session()
    at = _minimal(sess, tmp_path, named=False).run()
    assert not at.exception
    at.text_input(key="tn_2").set_value("Will's Team").run()
    at.text_input(key="tn_3").set_value("   ").run()  # blank falls back, never an empty label
    at.button(key="FormSubmitter:team_names_form-Save names & start").click().run()
    assert not at.exception
    live = at.session_state["session"]
    assert live.team_label(2) == "Will's Team"
    assert live.team_label(3) == "Team 3"
    assert (tmp_path / "auto.json").exists()  # names survive a resume, so the gate stays shut


def test_minimal_mode_prices_a_nominee_and_records_the_sale(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    sess = _smoke_session()
    at = _minimal(sess, tmp_path).run()
    assert not at.exception
    nom = next(sb for sb in at.selectbox if "Nominated player" in str(getattr(sb, "label", "")))
    assert nom.value is None  # nothing staged until you type
    # AppTest reports the *formatted* option, so map back through the same helpers the view
    # uses rather than assuming options are ids.
    staged = _available_ids(sess)[0]
    nom.set_value(_option_label(sess, staged)).run()
    assert not at.exception
    assert any("BID UP TO $" in str(getattr(md, "value", "")) for md in at.markdown)
    # the resolved name is on screen: the only cross-check that the box picked the right man
    assert any(sess.name(staged) in str(getattr(md, "value", "")) for md in at.markdown)

    _winner_box_named(at, "Won by").set_value(1).run()
    at.number_input[0].set_value(23).run()
    next(b for b in at.button if "→" in str(getattr(b, "label", ""))).click().run()
    assert not at.exception
    got = at.session_state["session"].purchases
    # gsis_id included: the label->id resolution is the only logic minimal mode adds, so a
    # (seat, price)-only assertion would pass even if the wrong player were recorded.
    assert [(str(p.gsis_id), p.seat, p.price) for p in got] == [(staged, 1, 23)]


def test_available_ids_keeps_players_whose_labels_collide() -> None:
    """Two available players rendering the same label must both stay pickable.

    The options were a dict keyed on the rendered label, so a collision silently dropped one
    player and made the survivor answer for both -- recording a purchase for someone nobody
    nominated, and leaving the other permanently un-nominatable from this view. `attach_names`
    fills an unresolved name with "—", so every unnamed player at a position collided.
    """
    sess = _smoke_session()
    twin_a, twin_b = _IDS[0], _IDS[4]
    assert sess.position_of(twin_a) is sess.position_of(twin_b)
    sess.player_names = dict(sess.player_names) | {twin_a: "Same Name", twin_b: "Same Name"}
    assert _option_label(sess, twin_a) == _option_label(sess, twin_b)  # labels do collide
    ids = _available_ids(sess)
    assert twin_a in ids and twin_b in ids  # ...and both players survive regardless


def test_minimal_mode_hides_what_yahoo_already_shows(tmp_path: Path) -> None:
    """The point of the mode: no sold log, no budget table, no 40-row bid board."""
    pytest.importorskip("streamlit")
    sess = _smoke_session()
    sess.record_purchase(_IDS[0], 2, 30)
    at = _minimal(sess, tmp_path).run()
    assert not at.exception
    rendered = " ".join(str(getattr(m, "value", "")) for m in at.markdown)
    assert "Sold" not in rendered
    assert "Budgets" not in rendered
    assert "Bid board" not in rendered
    assert not any("Position" in str(getattr(sb, "label", "")) for sb in at.selectbox)
    # Structural, not substring: the headings above were the only thing the string checks
    # caught, so a re-added dataframe or metric row would have slipped straight through.
    assert len(at.dataframe) == 1, "only the nomination shortlist renders a table"
    assert len(at.metric) == 0, "no status-bar metrics before a player is staged"


def test_skip_escape_is_reversible_and_keeps_the_you_marker(tmp_path: Path) -> None:
    """The escape must not become a one-way trip into the state the gate exists to prevent.

    It used to write "Team 1"…"Team N" -- truthy, so the gate's re-entry test never fired
    again, there is no rename affordance anywhere in minimal mode, and it overwrote the
    operator's own seat so `team_label` stopped returning "You" in BOTH views and in the
    persisted autosave.
    """
    pytest.importorskip("streamlit")
    sess = _smoke_session(my_seat=1)
    at = _minimal(sess, tmp_path, named=False).run()
    assert not at.exception
    at.button(key="skip_names").click().run()
    assert not at.exception
    live = at.session_state["session"]
    assert live.team_label(1) == "You", "the escape overwrote the hero marker"
    assert live.team_label(2) == "Team 2"
    # the lot UI is reachable...
    assert any("Nominated player" in str(getattr(sb, "label", "")) for sb in at.selectbox)
    # ...and naming is still reachable, so the escape is not one-way
    at.button(key="reopen_naming").click().run()
    assert not at.exception
    assert any("Name the teams" in str(getattr(h, "value", "")) for h in at.subheader)


def test_blanking_your_own_seat_restores_the_you_marker(tmp_path: Path) -> None:
    """A blank fell back to `Team {i+1}`, which wrote "Team 1" over the hero's own label."""
    pytest.importorskip("streamlit")
    sess = _smoke_session(my_seat=1)
    at = _minimal(sess, tmp_path, named=False).run()
    at.text_input(key="tn_1").set_value("").run()
    at.button(key="FormSubmitter:team_names_form-Save names & start").click().run()
    assert not at.exception
    assert at.session_state["session"].team_label(1) == "You"


def test_completed_auction_skips_the_naming_gate_but_keeps_undo(tmp_path: Path) -> None:
    """A finished auction should not be asked to name teams, and a mis-recorded FINAL lot is
    exactly the one that needs undo -- previously reachable only by switching views."""
    pytest.importorskip("streamlit")
    sess = _smoke_session(n_teams=6)
    gid = iter(_IDS)
    seat = 1
    while not sess.is_complete:
        if sess.open_slots(seat) == 0:
            seat += 1
        sess.record_purchase(next(gid), seat, 1)
    at = _minimal(sess, tmp_path, named=False).run()  # complete AND unnamed
    assert not at.exception
    assert not any("Name the teams" in str(getattr(h, "value", "")) for h in at.subheader)
    assert any("Auction complete" in str(getattr(h, "value", "")) for h in at.subheader)
    before = len(at.session_state["session"].purchases)
    at.button(key="m_undo").click().run()
    assert not at.exception
    assert len(at.session_state["session"].purchases) == before - 1
