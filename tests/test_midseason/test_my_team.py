"""`build_my_team` — the My Team pipeline.

Every one of these tests was impossible before this function existed. The assembly lived
inline in `routes/team.py`, reachable only through an HTTP request that first made a live ESPN
call, and pass 1 of the review found eight defects in it — the single densest cluster on the
branch. That is not a coincidence: it was the only code there without a test.

It then spent a pass living in `web/views/team_view.py`, which was the wrong home for a
different reason: a presenter that also parses ESPN payloads has two halves, and the seam
between them is where the next defects were. It sits here now, beside
`project_league_standings`, which is the same layer for the other page.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from projections.midseason.my_team import MyTeamRun, build_my_team
from projections.midseason.standings import ProjectionInputError
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.web.views.team_view import TeamPage, build_team_page
from tests.test_midseason.conftest import (
    MY_TEAM_ID,
    POSITIONS,
    TEAM_IDS,
    espn_payload,
    gsis_id,
    league_config,
)
from tests.test_midseason.conftest import id_map as league_id_map


def _pool(extra_positions: dict[str, str] | None = None) -> pd.DataFrame:
    """The league VORP pool. Deliberately does NOT contain kickers or defenses, exactly like
    the real one — that absence is what the roster filter used to swallow."""
    rows: list[dict[str, object]] = []
    for rank, team_id in enumerate(TEAM_IDS):
        for i, pos in enumerate(POSITIONS):
            mean = 240.0 - 20.0 * rank - 2.0 * i
            rows.append(
                {
                    "gsis_id": gsis_id(team_id, i),
                    "full_name": f"Player {team_id}-{i}",
                    "position": pos,
                    "season_mean_fpts": mean,
                    "vorp": mean - 80.0,
                    "replacement_fpts": 80.0,
                    "is_rookie": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    frame["position"] = frame["position"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(frame)


#: Columns `actual_season_total` reads. Declared so a zero-week frame is still correctly
#: shaped -- `pd.DataFrame([])` has no columns, which tests the fixture rather than the code.
_WEEKLY_COLUMNS = (
    "gsis_id", "season", "week", "position", "team", "opponent",
    "passing_yards", "passing_tds", "interceptions", "attempts", "completions", "sacks",
    "rushing_yards", "rushing_tds", "carries", "receptions", "receiving_yards",
    "receiving_tds", "fumbles_lost", "two_pt_conversions", "return_tds",
)  # fmt: skip


def _weekly_stats(weeks: int) -> pd.DataFrame:
    """`actual_season_total` input shape, for `weeks` played weeks."""
    rows: list[dict[str, Any]] = []
    for week in range(1, weeks + 1):
        for team_id in TEAM_IDS:
            for i, pos in enumerate(POSITIONS):
                rows.append(
                    {
                        "gsis_id": gsis_id(team_id, i),
                        "season": 2026,
                        "week": week,
                        "position": pos,
                        "team": "KC",
                        "opponent": "DEN",
                        "passing_yards": 0.0,
                        "passing_tds": 0,
                        "interceptions": 0,
                        "attempts": 0,
                        "completions": 0,
                        "sacks": 0,
                        "rushing_yards": 50.0,
                        "rushing_tds": 0,
                        "carries": 10,
                        "receptions": 2,
                        "receiving_yards": 20.0,
                        "receiving_tds": 0,
                        "fumbles_lost": 0,
                        "two_pt_conversions": 0,
                        "return_tds": 0,
                    }
                )
    frame = pd.DataFrame(rows, columns=list(_WEEKLY_COLUMNS))
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    return frame


def _run(*, played_weeks: int = 2, my_team_id: int = MY_TEAM_ID, **overrides: Any) -> MyTeamRun:
    kwargs: dict[str, Any] = {
        "payload": espn_payload(played_weeks=played_weeks),
        "pool": _pool(),
        "id_map": league_id_map(),
        "weekly_stats": _weekly_stats(played_weeks),
        "league_config": league_config(len(TEAM_IDS)),
        "my_team_id": my_team_id,
        "season": 2026,
    }
    kwargs.update(overrides)
    return build_my_team(**kwargs)


def _assemble(**overrides: Any) -> TeamPage:
    """The pipeline and the presenter together, which is what the route does.

    Several of these tests are about a number that only exists once the page is rendered
    (`starter_ros`, a cell showing an em dash), so they need both halves. The ones that are
    purely about the run assert on `_run(...)` directly.
    """
    return build_team_page(_run(**overrides), season=2026)


# --- the critical defect ---------------------------------------------------------------------


def test_ros_is_a_rest_of_season_figure_not_the_full_season_pool() -> None:
    """Two reviewers found this independently.

    The route passed the raw VORP pool -- whose `season_mean_fpts` is a FULL-SEASON figure --
    straight through, under a column labelled "Projected points for the rest of the season".
    `rest_of_season_pool` was never called anywhere under `web/`. At week 10 the column showed
    roughly double the truth, `ros_rank` inherited it, the header total overstated, and the two
    pages disagreed about the same player because standings runs the real path.

    The tell: as the season runs out there is less of it left, so the same player's remaining
    projection must fall.
    """
    early = _assemble(played_weeks=1)
    late = _assemble(played_weeks=3)
    assert late.starter_ros < early.starter_ros, (
        "rest-of-season points must shrink as the season runs out; if they do not, the raw "
        "full-season pool is being displayed under a rest-of-season label"
    )


def test_preseason_ros_equals_the_pool_because_nothing_has_been_played() -> None:
    """The reason the fixtures never caught it: with a full season remaining and nothing
    scored, the adjusted and unadjusted figures agree exactly."""
    page = _assemble(played_weeks=0, weekly_stats=_weekly_stats(0))
    pool = _pool().set_index("gsis_id")
    ros_index = next(i for i, c in enumerate(page.columns) if c.key == "ros_points")
    for row in page.rows:
        expected = float(pool.at[row.gsis_id, "season_mean_fpts"])
        assert float(row.cells[ros_index].text) == pytest.approx(expected, rel=1e-6)


# --- the roster is the whole roster ------------------------------------------------------------


def test_players_the_pool_cannot_project_still_appear() -> None:
    """The route filtered the roster through pool membership, so kickers and defenses -- which
    the pool never contains -- were deleted before the view saw them. A 13-man roster rendered
    as 11 with no message, and the note written to name them was unreachable in production."""
    payload = espn_payload(played_weeks=2)
    my_team = next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)
    my_team["roster"]["entries"].append(
        {
            "lineupSlotId": 17,  # K
            "playerId": 999_001,
            "playerPoolEntry": {
                "player": {
                    "id": 999_001,
                    "fullName": "Some Kicker",
                    "defaultPositionId": 5,
                    "proTeamId": 1,
                }
            },
        }
    )
    id_map = pd.concat(
        [
            league_id_map(),
            pd.DataFrame(
                {
                    "espn_id": pd.Series(["999001"], dtype=_PYARROW_STR),
                    "gsis_id": pd.Series(["00-9990001"], dtype=_PYARROW_STR),
                }
            ),
        ],
        ignore_index=True,
    )
    page = _assemble(payload=payload, id_map=id_map)

    assert any("Some Kicker" in row.cells[1].text for row in page.rows), (
        "an unprojectable player must still be listed, with an em dash rather than vanishing"
    )
    assert any("Some Kicker" in note for note in page.notes), "and be named as unprojected"


def test_only_my_players_appear() -> None:
    page = _assemble()
    mine = {gsis_id(MY_TEAM_ID, i) for i in range(len(POSITIONS))}
    assert {row.gsis_id for row in page.rows} == mine


# --- errors the page must survive ---------------------------------------------------------------


def test_another_teams_unresolvable_roster_does_not_break_my_page() -> None:
    """The route used `rosters_to_slots`, which raises when ANY team in the league has no
    projectable players -- a per-slot check that exists for the simulator. So a brand-new
    manager's roster elsewhere in the league took down MY page with a 500."""
    id_map = league_id_map()
    other = TEAM_IDS[1]
    drop = [str(100_000 + other * 100 + i) for i in range(len(POSITIONS))]
    thinned = id_map[~id_map["espn_id"].astype(str).isin(drop)]

    page = _assemble(id_map=thinned)
    assert page.rows, "my roster still renders"


def test_an_unknown_team_id_is_reported_not_raised() -> None:
    """A typo'd --team-id used to reach `tuple.index` and raise a bare ValueError."""
    with pytest.raises(ProjectionInputError, match="not a team"):
        _assemble(my_team_id=9999)


# --- the week ------------------------------------------------------------------------------------


def test_the_week_comes_from_the_schedule_not_the_stats_partition() -> None:
    """The route derived the week as `max(weekly_stats.week) + 1`, which is a statement about
    nfl_data_py's ingest lag rather than about the league -- so the two pages showed different
    weeks for the same league on the same day. The canonical answer is `first_unplayed_week`,
    which the standings page already uses.

    Here the schedule says three weeks are played while the stats partition holds only one.
    """
    page = _assemble(played_weeks=3, weekly_stats=_weekly_stats(1))
    assert page.week == 4, "the schedule is the authority on which week it is"


def test_a_stats_partition_behind_the_schedule_is_called_out() -> None:
    """The dangerous half of the same split: a stale-but-present partition yields a PARTIAL
    season total under a header claiming a later week, and nothing said so. The old note only
    fired when the partition was completely empty."""
    page = _assemble(played_weeks=3, weekly_stats=_weekly_stats(1))
    assert any("behind" in note.lower() or "stale" in note.lower() for note in page.notes), (
        page.notes
    )


def test_matching_schedule_and_stats_produce_no_staleness_note() -> None:
    page = _assemble(played_weeks=2, weekly_stats=_weekly_stats(2))
    assert not any("behind" in note.lower() for note in page.notes), page.notes


# --- totals -------------------------------------------------------------------------------------


def test_the_ytd_total_is_labelled_as_the_whole_roster_not_as_starters() -> None:
    """`starter_ytd` attributed a player's WHOLE-SEASON points to his CURRENT slot: someone
    benched for weeks 1-5 and started since contributed all of it to "starters YTD". That is
    neither what the starting lineups scored nor the team's points-for, and the standings page
    reports a different, also-plausible number for the same team.

    Rest-of-season is forward-looking, so attributing it to the current slot IS meaningful and
    stays a starters-only figure.
    """
    page = _assemble()
    every_ytd = sum(float(row.cells[3].text) for row in page.rows if row.cells[3].text != "—")
    assert page.roster_ytd == pytest.approx(every_ytd)
    assert page.starter_ros > 0.0


# --- the crash the last batch of fixes wrote ---------------------------------------------------


def test_a_player_recorded_at_two_positions_does_not_take_the_page_down() -> None:
    """`actual_season_total` groups by (gsis_id, position), so a player reclassified mid-season
    comes back as TWO rows. The previous fix collapsed them -- but inside the presenter, one
    step AFTER the remaining-points subtraction had already done
    `ytd.set_index("gsis_id")["actual_total"]` on the duplicate index.

    `Series.map` on a non-unique index raises `InvalidIndexError`, which is not in the route's
    caught tuple, so `/team` returned a traceback rather than an empty state. Collapsing at the
    pipeline boundary is what makes every later step safe, and the check belongs here because
    this is where the frame is built.
    """
    weekly = _weekly_stats(2)
    # The same player, credited at a second position from week 2 -- exactly what a
    # reclassification looks like coming out of nflverse.
    reclassified = weekly[weekly["week"] == 2].copy()
    reclassified["position"] = "WR"
    doubled = pd.concat([weekly, reclassified], ignore_index=True)

    page = _assemble(weekly_stats=doubled)
    assert page.rows, "the page renders rather than raising"
    ytd = next(i for i, c in enumerate(page.columns) if c.key == "ytd_points")
    assert all(row.cells[ytd].text != "—" for row in page.rows if row.cells[ytd].text)


def test_his_two_rows_are_summed_into_one_season() -> None:
    weekly = _weekly_stats(2)
    reclassified = weekly[weekly["week"] == 2].copy()
    reclassified["position"] = "WR"
    run = _run(weekly_stats=pd.concat([weekly, reclassified], ignore_index=True))
    assert not run.ytd["gsis_id"].duplicated().any(), "one row per player leaves the pipeline"


# --- nothing vanishes silently ------------------------------------------------------------------


def test_a_player_the_id_map_cannot_resolve_is_listed_and_named() -> None:
    """The mirror of `test_players_the_pool_cannot_project_still_appear`, one filter earlier.

    `espn_to_gsis` returns NA for any ESPN id the crosswalk does not hold -- a just-signed
    player, a defense -- and the previous version dropped those rows with no note, so a 15-man
    roster rendered as 14 and nothing said why.
    """
    payload = espn_payload(played_weeks=2)
    my_team = next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)
    my_team["roster"]["entries"].append(
        {
            "lineupSlotId": 20,
            "playerId": 888_001,  # in no id_map
            "playerPoolEntry": {
                "player": {
                    "id": 888_001,
                    "fullName": "Just Signed",
                    "defaultPositionId": 2,
                    "proTeamId": 1,
                }
            },
        }
    )
    page = _assemble(payload=payload)
    player = next(i for i, c in enumerate(page.columns) if c.key == "player")
    assert any("Just Signed" in row.cells[player].text for row in page.rows), "still listed"
    assert any("id_map" in note for note in page.notes), page.notes


def test_my_own_empty_roster_is_explained_not_rendered_as_a_headed_table() -> None:
    """The league-wide `rosters.empty` check passes as soon as ANYONE has drafted, so mid-draft
    my own empty roster produced a table with a header row and no body and no message."""
    payload = espn_payload(played_weeks=2)
    my_team = next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)
    my_team["roster"]["entries"] = []
    with pytest.raises(ProjectionInputError, match="no players"):
        _run(payload=payload)


# --- the wholesale-clamp alarm reaches this page too --------------------------------------------


def test_a_provider_already_reporting_rest_of_season_is_called_out() -> None:
    """The assumption that cannot be verified until Week 1: if a provider's "season total" is
    already rest-of-season, subtracting actuals double-counts and drives the pool to zero.

    `rest_of_season_pool` has raised this alarm since it was written, and the team page --
    which used its own copy of the subtraction -- did not. Sharing `remaining_totals` is what
    gives it the same alarm.
    """
    pool = _pool()
    # Everyone has already outscored their whole projection: the shape of the inverted
    # assumption.
    pool["season_mean_fpts"] = 1.0
    run = _run(pool=pool)
    assert run.diagnostics.looks_like_double_counting
    assert any("REST-OF-SEASON" in note.upper() for note in run.notes), run.notes
