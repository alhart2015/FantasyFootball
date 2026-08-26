"""The My Team view model.

The page this backs cannot be looked at until Week 1 — there is no 2026 weekly-stats
partition and the draft has not happened — so these tests are the only thing standing between
"it assembles correctly" and "it looks plausible".
"""

from __future__ import annotations

import pandas as pd
import pytest
from flask import Flask, render_template

from projections.schemas import _PYARROW_STR
from projections.web.views.team_view import TeamPage, build_team_page, empty_team_page

_STARTER_SLOTS = ("QB", "RB", "RB", "WR", "WR", "TE")


def _roster(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """`parse_rosters` shape plus the `gsis_id` the route resolves through the id_map."""
    frame = pd.DataFrame(
        {
            "gsis_id": pd.Series([r[0] for r in rows], dtype=_PYARROW_STR),
            "player": pd.Series([r[1] for r in rows], dtype=_PYARROW_STR),
            "pos": pd.Series([r[2] for r in rows], dtype=_PYARROW_STR),
            "lineup_slot": pd.Series([r[3] for r in rows], dtype=_PYARROW_STR),
        }
    )
    return frame


def _ytd(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """`actual_season_total` shape."""
    return pd.DataFrame(
        {
            "gsis_id": pd.Series([r[0] for r in rows], dtype=_PYARROW_STR),
            "position": pd.Series([r[1] for r in rows], dtype=_PYARROW_STR),
            "actual_total": pd.Series([r[2] for r in rows], dtype="float64"),
        }
    )


def _ros(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.Series([r[0] for r in rows], dtype=_PYARROW_STR),
            "position": pd.Series([r[1] for r in rows], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.Series([r[2] for r in rows], dtype="float64"),
        }
    )


def _cell(page: TeamPage, row_index: int, key: str) -> str:
    column_index = next(i for i, c in enumerate(page.columns) if c.key == key)
    return page.rows[row_index].cells[column_index].text


def _page(**overrides: object) -> TeamPage:
    roster = _roster(
        [
            ("00-0000001", "Star RB", "RB", "RB"),
            ("00-0000002", "Bench RB", "RB", "BENCH"),
            ("00-0000003", "Star WR", "WR", "WR"),
        ]
    )
    # A league pool wider than the roster, so ranks mean "in the league", not "on my team".
    ytd = _ytd(
        [
            ("00-0000001", "RB", 140.0),
            ("00-0000002", "RB", 90.0),
            ("00-0000003", "WR", 120.0),
            ("00-0009001", "RB", 200.0),
            ("00-0009002", "RB", 160.0),
            ("00-0009003", "WR", 130.0),
        ]
    )
    ros = _ros(
        [
            ("00-0000001", "RB", 150.0),
            ("00-0000002", "RB", 100.0),
            ("00-0000003", "WR", 110.0),
            ("00-0009001", "RB", 210.0),
            ("00-0009002", "RB", 170.0),
            ("00-0009003", "WR", 190.0),
        ]
    )
    kwargs: dict[str, object] = {
        "roster": roster,
        "ytd": ytd,
        "ros": ros,
        "team_name": "Silence of the Lamb",
        "season": 2026,
        "week": 6,
    }
    kwargs.update(overrides)
    return build_team_page(**kwargs)  # type: ignore[arg-type]


def test_every_rostered_player_gets_a_row() -> None:
    page = _page()
    assert len(page.rows) == 3
    assert all(len(row.cells) == len(page.columns) for row in page.rows)


def test_ranks_are_league_wide_not_roster_wide() -> None:
    """The point of the rank column. "RB 3" means third-best running back in the league; a
    rank computed over a 13-man roster looks identical and means nothing."""
    page = _page()
    star_rb = next(i for i, r in enumerate(page.rows) if r.gsis_id == "00-0000001")
    # 200 and 160 outscore him, so he is RB3 across the pool -- not RB1 of the two on my team.
    assert _cell(page, star_rb, "ytd_rank") == "3"


def test_rank_is_within_position_not_overall() -> None:
    page = _page()
    star_wr = next(i for i, r in enumerate(page.rows) if r.gsis_id == "00-0000003")
    # 130 outscores his 120 among WRs; the four RBs above him are irrelevant.
    assert _cell(page, star_wr, "ytd_rank") == "2"


def test_starters_come_before_the_bench() -> None:
    page = _page()
    assert [row.is_starter for row in page.rows] == [True, True, False]


def test_within_a_group_the_order_is_by_projection() -> None:
    """A bench player projected above a starter is the most actionable thing this page can
    show, so the ordering has to make it visible rather than follow ESPN's slot order."""
    page = _page()
    starters = [row for row in page.rows if row.is_starter]
    ros_values = [float(_cell(page, page.rows.index(r), "ros_points")) for r in starters]
    assert ros_values == sorted(ros_values, reverse=True)


def test_the_two_totals_measure_different_things_on_purpose() -> None:
    """`roster_ytd` spans the WHOLE roster; `starter_ros` counts starters only.

    Not an inconsistency. `is_starter` reads today's slot, while YTD points span the season, so
    a starters-only YTD would credit a player benched for weeks 1-5 with all of his points --
    neither what the starting lineups scored nor the team's points-for, which the standings
    page reports as PF. Rest-of-season is forward-looking, so attributing it to the current
    starters IS meaningful.
    """
    page = _page()
    assert page.roster_ytd == pytest.approx(140.0 + 90.0 + 120.0), "bench included"
    assert page.starter_ros == pytest.approx(150.0 + 110.0), "bench excluded"


# --- the empty and partial states ---------------------------------------------------------


def test_a_player_with_no_stats_shows_a_dash_not_a_zero() -> None:
    """He has not scored zero — he has not played. A "0.0" in that cell is a false statement
    about a player who was never active."""
    page = _page(ytd=_ytd([("00-0009001", "RB", 200.0)]))
    star_rb = next(i for i, r in enumerate(page.rows) if r.gsis_id == "00-0000001")
    assert _cell(page, star_rb, "ytd_points") == "—"
    assert _cell(page, star_rb, "ytd_rank") == "—"


def test_an_empty_ytd_frame_is_explained_rather_than_left_blank() -> None:
    """The preseason state. A roster of em dashes looks like a broken page unless something
    says why."""
    page = _page(ytd=_ytd([]))
    assert any("Week 1" in note for note in page.notes)
    assert all(_cell(page, i, "ytd_points") == "—" for i in range(len(page.rows)))


def test_unprojected_players_are_named() -> None:
    """Kickers and defenses are rostered and unprojectable. Naming them stops the reader
    wondering whether the page is broken."""
    roster = _roster(
        [("00-0000001", "Star RB", "RB", "RB"), ("00-0000009", "Some Kicker", "K", "K")]
    )
    page = _page(roster=roster)
    assert any("Some Kicker" in note for note in page.notes)


def test_an_unprojected_player_sorts_last_within_his_group() -> None:
    """His cell holds an em dash, not a number, and a naive float() would either crash or
    float him to the top."""
    roster = _roster(
        [
            ("00-0000009", "Some Kicker", "K", "K"),
            ("00-0000001", "Star RB", "RB", "RB"),
        ]
    )
    page = _page(roster=roster)
    assert page.rows[-1].gsis_id == "00-0000009"


def test_a_clean_full_page_has_no_notes() -> None:
    assert _page().notes == ()


def test_the_empty_page_carries_its_reason() -> None:
    page = empty_team_page("No team selected.", season=2026)
    assert page.is_empty
    assert page.message == "No team selected."
    assert page.rows == ()


# --- the rendered page ----------------------------------------------------------------------


def test_the_template_renders_the_roster(app: Flask) -> None:
    page = _page()
    with app.test_request_context():
        html = render_template("team.html", page=page)
    for column in page.columns:
        assert f">{column.label}<" in html, f"missing header {column.label}"
    assert "Star RB" in html and "Bench RB" in html
    assert "Silence of the Lamb" in html


def test_the_template_renders_the_empty_state(app: Flask) -> None:
    with app.test_request_context():
        html = render_template("team.html", page=empty_team_page("Nothing yet.", season=2026))
    assert "Nothing yet." in html
    assert "<tbody>" not in html


def test_the_team_table_carries_the_colour_scale() -> None:
    """`sense` on TEAM_COLUMNS was dead configuration: every cell was built with the default
    intensity and the template emitted nothing, so `test_rank_columns_are_lower_better`
    asserted a property with no consumer. The scale is the affordance the spec used to justify
    choosing Flask over Streamlit, so it should actually be there."""
    page = _page()
    ros_index = next(i for i, c in enumerate(page.columns) if c.key == "ros_points")
    values = [row.cells[ros_index].intensity for row in page.rows]
    assert any(v is not None for v in values), "a directional column must carry intensity"


def test_a_rank_column_reads_rank_one_as_good() -> None:
    """Rank is lower-better. Colouring it like points would paint the best player at each
    position as the worst."""
    page = _page()
    rank_index = next(i for i, c in enumerate(page.columns) if c.key == "ros_rank")
    ranks = [
        (float(row.cells[rank_index].text), row.cells[rank_index].intensity)
        for row in page.rows
        if row.cells[rank_index].text != "—"
    ]
    best = min(ranks, key=lambda pair: pair[0])
    worst = max(ranks, key=lambda pair: pair[0])
    assert best[1] is not None and worst[1] is not None
    assert best[1] > worst[1], "the best rank must read as the most positive"


def test_a_neutral_column_carries_no_intensity() -> None:
    page = _page()
    for index, column in enumerate(page.columns):
        if column.sense == "neutral":
            assert all(row.cells[index].intensity is None for row in page.rows), column.key


def test_the_sort_reads_the_value_not_the_formatted_string() -> None:
    """At precision=1, 150.04 and 149.96 both render "150.0" and would tie if the sort parsed
    the cell text back out."""
    roster = _roster([("00-0000001", "A", "RB", "RB"), ("00-0000002", "B", "RB", "RB")])
    ros = _ros([("00-0000001", "RB", 149.96), ("00-0000002", "RB", 150.04)])
    page = _page(roster=roster, ros=ros)
    assert page.rows[0].gsis_id == "00-0000002", "the genuinely larger value sorts first"
