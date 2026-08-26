"""The standings view model.

Where the density belongs: these are pure functions, so every rule below is checked by calling
one, with no app, request, or rendered HTML to assert substrings against.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask, render_template

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.standings import StandingsRun, project_league_standings
from projections.web.views.standings_view import (
    StandingsPage,
    build_standings_page,
    empty_standings_page,
)
from tests.test_midseason.conftest import TEAM_IDS, espn_payload, id_map, vorp_pool

_MY_TEAM = 17


def _run(*, played_weeks: int = 2) -> StandingsRun:
    pool = vorp_pool()
    return project_league_standings(
        espn_payload(played_weeks=played_weeks),
        pool,
        id_map(),
        PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
        VarianceParams.load(),
        season=2026,
        n_sims=60,
        rng=np.random.default_rng(4),
    )


def _page(*, played_weeks: int = 2, my_team_id: int | None = _MY_TEAM) -> StandingsPage:
    return build_standings_page(_run(played_weeks=played_weeks), season=2026, my_team_id=my_team_id)


def test_every_team_gets_a_row_with_a_cell_per_column() -> None:
    page = _page()
    assert len(page.rows) == len(TEAM_IDS)
    assert all(len(row.cells) == len(page.columns) for row in page.rows)


def test_the_rank_column_follows_the_order_the_domain_chose() -> None:
    """`build_standings` already sorts by projected wins then projected points. Re-sorting here
    would be a second place deciding the order, and two places is two chances to disagree."""
    page = _page()
    ranks = [row.cells[0].text for row in page.rows]
    assert ranks == [str(i) for i in range(1, len(TEAM_IDS) + 1)]


def test_my_row_is_marked_and_only_mine() -> None:
    page = _page()
    mine = [row for row in page.rows if row.is_mine]
    assert len(mine) == 1
    assert mine[0].team_id == _MY_TEAM


def test_no_row_is_marked_when_there_is_no_my_team() -> None:
    """The highlight is simply absent rather than defaulting to a team — a dashboard that
    silently calls someone else's row "you" is worse than one with no highlight."""
    assert not any(row.is_mine for row in _page(my_team_id=None).rows)


def test_a_tieless_record_omits_the_zero() -> None:
    """ "6-1-0" reads as though a tie were expected and missing."""
    page = _page()
    records = [row.cells[2].text for row in page.rows]
    assert all(record.count("-") == 1 for record in records), records


def test_a_record_with_ties_shows_them() -> None:
    run = _run()
    frame = run.standings.copy()
    frame.loc[0, "ties"] = 1
    page = build_standings_page(
        StandingsRun(**{**run.__dict__, "standings": frame}), season=2026, my_team_id=_MY_TEAM
    )
    assert page.rows[0].cells[2].text.count("-") == 2


# --- the colour scale ------------------------------------------------------------------------


def test_a_column_where_every_team_is_tied_gets_no_colour() -> None:
    """The case that matters before the season starts, when every probability is identical.
    Painting every cell at the midpoint implies a spread that does not exist; the honest
    rendering is no colour at all."""
    run = _run()
    frame = run.standings.copy()
    frame["champ_pct"] = 1.0 / len(frame)
    page = build_standings_page(
        StandingsRun(**{**run.__dict__, "standings": frame}), season=2026, my_team_id=None
    )
    champ_index = next(i for i, c in enumerate(page.columns) if c.key == "champ_pct")
    assert all(row.cells[champ_index].intensity is None for row in page.rows)


def test_intensity_spans_minus_one_to_one_across_a_column() -> None:
    page = _page()
    pf_index = next(i for i, c in enumerate(page.columns) if c.key == "projected_points_for")
    values = [row.cells[pf_index].intensity for row in page.rows]
    assert all(v is not None for v in values)
    spread = [v for v in values if v is not None]
    assert min(spread) == pytest.approx(-1.0)
    assert max(spread) == pytest.approx(1.0)


def test_a_neutral_column_never_gets_colour() -> None:
    """Team name and rank have no direction. Colouring them would imply one."""
    page = _page()
    for index, column in enumerate(page.columns):
        if column.sense == "neutral":
            assert all(row.cells[index].intensity is None for row in page.rows), column.key


# --- my remaining games ----------------------------------------------------------------------


def test_my_games_state_the_probability_from_my_side() -> None:
    """`home_win_pct` on a row where I am AWAY is my chance of losing. A table that forgets to
    flip it reports the exact opposite of what it claims."""
    run = _run()
    page = build_standings_page(run, season=2026, my_team_id=_MY_TEAM)
    assert page.my_games

    odds = run.odds
    for game in page.my_games:
        row = odds[
            (odds["week"] == game.week)
            & ((odds["home_team_id"] == _MY_TEAM) | (odds["away_team_id"] == _MY_TEAM))
        ].iloc[0]
        expected = (
            float(row["home_win_pct"])
            if int(row["home_team_id"]) == _MY_TEAM
            else 1.0 - float(row["home_win_pct"])
        )
        assert game.win_pct == pytest.approx(expected)
        assert game.at_home == (int(row["home_team_id"]) == _MY_TEAM)


def test_my_games_are_empty_without_a_my_team() -> None:
    assert _page(my_team_id=None).my_games == ()


def test_a_finished_season_has_no_remaining_games() -> None:
    assert _page(played_weeks=4).my_games == ()


# --- empty and warning states ------------------------------------------------------------------


def test_the_empty_page_carries_its_reason() -> None:
    """An empty table reads as "everyone is 0-0". The reason has to be on the page."""
    page = empty_standings_page("The draft has not happened.", season=2026)
    assert page.is_empty
    assert page.message == "The draft has not happened."
    assert page.rows == ()
    assert page.columns, "the header still describes what would be shown"


def test_a_populated_page_is_not_empty() -> None:
    assert not _page().is_empty


def test_dropped_players_are_surfaced_as_a_note() -> None:
    run = _run()
    page = build_standings_page(
        StandingsRun(**{**run.__dict__, "n_players_dropped": 3}), season=2026, my_team_id=None
    )
    assert any("3 rostered players" in note for note in page.notes)


def test_a_clean_run_has_no_notes() -> None:
    page = _page()
    assert page.notes == ()


# --- the rendered page ------------------------------------------------------------------------


def test_the_template_renders_a_full_standings_table(app: Flask) -> None:
    """The table cannot be seen against real data until Week 1, so this is the proof it renders
    at all: a synthetic mid-season league through the view model and the real template.

    Rendered through the app fixture rather than a bare Flask instance because the shared nav
    resolves routes with `url_for`, which needs the blueprints registered.
    """
    page = _page()
    with app.test_request_context():
        html = render_template("standings.html", page=page)

    for column in page.columns:
        assert f">{column.label}<" in html, f"missing header {column.label}"
    assert 'class="user-team"' in html, "my row is highlighted"
    assert "--pos:" in html or "--neg:" in html, "the colour scale reaches the markup"
    assert 'role="region"' in html and 'tabindex="0"' in html, "scroll region is reachable"
    for row in page.rows:
        assert f">{row.cells[1].text}<" in html, "every team name is on the page"


def test_the_empty_state_renders_its_reason_and_no_table(app: Flask) -> None:
    page = empty_standings_page("The draft has not happened.", season=2026)
    with app.test_request_context():
        html = render_template("standings.html", page=page)
    assert "The draft has not happened." in html
    assert "<tbody>" not in html, "no table at all, rather than an empty one reading as 0-0"
