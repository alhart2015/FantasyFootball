"""View model for the My Team page.

Year-to-date points and rank beside rest-of-season projection and rank, per player. Like the
standings view, **no Flask import belongs here** — every rule below is checked by calling a
function, and `tests/test_web/test_app.py` parses this module's imports to keep it that way.

**This module presents; it does not assemble.** Parsing the ESPN payload, deriving the week,
scoring `weekly_stats` and rewriting the pool to remaining points all live in
`midseason.my_team`, which is the layer `project_league_standings` occupies for the other page.
An earlier version did both here, and the seam between the two halves is where the defects
were: a frame the presenter cleaned was read by the pipeline one step earlier, and a docstring
here described the pipeline the *other* page uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projections.midseason.my_team import MyTeamRun
from projections.midseason.standings import regular_season_complete
from projections.rankings import rank_within_position
from projections.schemas import RosterSlot, display_str
from projections.web.views.columns import (
    TEAM_COLUMNS,
    Cell,
    CellValue,
    Column,
    column_intensities,
    require_every_key,
)

#: Slots that do not start. `RosterSlot` values rather than the strings they wrap, per
#: CLAUDE.md -- `parse_rosters` produces these FROM `ESPN_LINEUP_SLOTS`, which is keyed on the
#: enum, so what makes the comparison safe is that the producer and the check read one source.
#: An unrecognised ESPN slot id becomes `""` there, which is in neither set.
_BENCH_SLOTS = frozenset({RosterSlot.BENCH, RosterSlot.IR})

#: The column keys `_row_values` fills.
#:
#: Checked against the registry at import, so a column nothing fills is caught whether or not a
#: roster happens to be non-empty -- and checked against a REAL row in `build_team_page`, once,
#: because on its own this is a literal compared to another literal and would not notice
#: `_row_values` drifting away from both.
_ROW_KEYS: frozenset[str] = frozenset(
    {"slot", "player", "position", "ytd_points", "ytd_rank", "ros_points", "ros_rank"}
)
require_every_key(_ROW_KEYS, TEAM_COLUMNS, source="the assembled player row")


@dataclass(frozen=True)
class _Row:
    """One player before rendering: who he is, whether he starts, and his raw column values.

    The two bookkeeping fields are FIELDS, not entries in `values`. Folding them into the
    column dict widened its type to `CellValue | bool` and bought four coercion helpers whose
    only job was narrowing it back, most of whose branches were unreachable.
    """

    gsis_id: str
    is_starter: bool
    values: dict[str, CellValue]


@dataclass(frozen=True)
class PlayerRow:
    gsis_id: str
    #: Starters and bench are visually separated; a bench player outscoring a starter is the
    #: single most actionable thing this page can show.
    is_starter: bool
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class TeamPage:
    team_name: str
    season: int
    week: int
    reg_weeks: int
    columns: tuple[Column, ...]
    rows: tuple[PlayerRow, ...]
    #: Season points across the WHOLE roster. Deliberately not a starters-only figure:
    #: `is_starter` reads today's slot while these points span the season, so attributing them
    #: to the current lineup would credit a player benched for weeks 1-5 with all of them. That
    #: is neither what the starting lineups scored nor the team's points-for, which the
    #: standings page reports as PF.
    roster_ytd: float
    #: Rest-of-season IS forward-looking, so attributing it to the current starters is
    #: meaningful and this stays a starters-only figure.
    starter_ros: float
    message: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.message is not None

    @property
    def regular_season_complete(self) -> bool:
        return regular_season_complete(self.week, self.reg_weeks)


def build_team_page(run: MyTeamRun, *, season: int) -> TeamPage:
    """`MyTeamRun` -> the rendered page model. Presentation only.

    **Ranks are computed over the whole league pool, not over this roster.** "RB 4" means
    fourth-best running back in the league; ranking within a 13-man roster would produce a
    number that looks the same and means nothing.

    **Both ranks are computed over the SAME universe** -- the projection pool. They sit in
    adjacent columns under near-identical tooltips, which invites reading "YTD 8, ROS 22" as a
    fall of fourteen places, a subtraction that only means anything if both numbers count the
    same players. `run.ytd` is NFL-wide, straight off `weekly_stats`: every practice-squad
    running back who took a carry is in it, and none of them are in the pool.

    POINTS still come from the full frame. A kicker is not in the pool, so he has no rank, but
    he really did score those points and dropping them to make a rank comparable would be
    fixing the wrong number.
    """
    pool_universe = set(run.ros["gsis_id"].astype(str)) if not run.ros.empty else set()
    ytd_by_id = run.ytd.set_index("gsis_id")
    in_universe = run.ytd[run.ytd["gsis_id"].astype(str).isin(pool_universe)]
    ytd_rank_by_id = _with_rank(in_universe, "actual_total", "ytd_rank", ascending=False).set_index(
        "gsis_id"
    )
    ros_by_id = _with_rank(run.ros, "season_mean_fpts", "ros_rank", ascending=False).set_index(
        "gsis_id"
    )

    rows = [
        _row_values(player, ytd_by_id, ytd_rank_by_id, ros_by_id)
        for _, player in run.roster.iterrows()
    ]
    if rows:
        # Against the row the function actually built, not against `_ROW_KEYS`. The import-time
        # check compares two literals to each other and would miss a rename inside
        # `_row_values` -- which then raises a bare KeyError per request, and a KeyError is not
        # in the route's caught tuple, so it is a 500 rather than an empty state.
        require_every_key(rows[0].values, TEAM_COLUMNS, source="the assembled player row")

    roster_ytd = sum(_number(row.values["ytd_points"]) for row in rows)
    starter_ros = sum(_number(row.values["ros_points"]) for row in rows if row.is_starter)

    # Starters first, then bench, each by rest-of-season projection. A bench player above a
    # starter is the page's most actionable signal, and burying it under ESPN's slot ordering
    # would hide it.
    #
    # Sorted on the VALUE, not the formatted string (at precision=1, 150.04 and 149.96 both
    # render "150.0"), and keyed on `is None` rather than `or`: `remaining_totals` clamps at
    # zero, so a player who has outscored his projection has a REAL 0.0, and `or` would file
    # him with the players who have no projection at all.
    rows.sort(key=lambda row: (not row.is_starter, _sort_key(row.values["ros_points"])))

    scales = column_intensities([row.values for row in rows], TEAM_COLUMNS)
    rendered = tuple(
        PlayerRow(
            gsis_id=row.gsis_id,
            is_starter=row.is_starter,
            cells=tuple(
                Cell(
                    text=column.format(row.values[column.key]),
                    intensity=scales[column.key][i],
                    numeric=column.numeric,
                    is_label=column.is_label,
                )
                for column in TEAM_COLUMNS
            ),
        )
        for i, row in enumerate(rows)
    )

    return TeamPage(
        team_name=run.team_name,
        season=season,
        week=run.week,
        reg_weeks=run.reg_weeks,
        columns=TEAM_COLUMNS,
        rows=rendered,
        roster_ytd=roster_ytd,
        starter_ros=starter_ros,
        notes=run.notes + _blank_column_notes(run, ros_by_id),
    )


def empty_team_page(message: str, *, season: int) -> TeamPage:
    """No roster, or no team selected. Carries the reason rather than an empty table."""
    return TeamPage(
        team_name="",
        season=season,
        week=0,
        reg_weeks=0,
        columns=TEAM_COLUMNS,
        rows=(),
        roster_ytd=0.0,
        starter_ros=0.0,
        message=message,
    )


def _row_values(
    player: pd.Series,
    ytd_by_id: pd.DataFrame,
    ytd_rank_by_id: pd.DataFrame,
    ros_by_id: pd.DataFrame,
) -> _Row:
    """One player before rendering.

    `gsis_id` may be empty: `build_my_team` deliberately keeps roster entries the id_map cannot
    resolve rather than deleting them, so they render as a name and a row of em dashes.
    """
    gsis = display_str(player.get("gsis_id"))
    slot = display_str(player.get("lineup_slot"))
    return _Row(
        gsis_id=gsis,
        # An unrecognised ESPN slot id becomes "" in `parse_rosters`, and "" is in neither slot
        # set -- so an unknown slot used to count as a STARTER and inflate the starters-only
        # total. Unknown reads as bench: crediting a player we cannot place to the starting
        # lineup is the more wrong of the two guesses.
        is_starter=bool(slot) and slot not in _BENCH_SLOTS,
        values={
            "slot": slot or None,
            "player": display_str(player.get("player"))
            or gsis
            or display_str(player.get("player_id")),
            "position": display_str(player.get("pos")),
            "ytd_points": _lookup(ytd_by_id, gsis, "actual_total"),
            "ytd_rank": _lookup_rank(ytd_rank_by_id, gsis, "ytd_rank"),
            "ros_points": _lookup(ros_by_id, gsis, "season_mean_fpts"),
            "ros_rank": _lookup_rank(ros_by_id, gsis, "ros_rank"),
        },
    )


def _with_rank(frame: pd.DataFrame, by: str, name: str, *, ascending: bool) -> pd.DataFrame:
    """Rank within position across the whole frame, tolerating an empty one.

    An empty `weekly_stats` is the normal preseason state, not an error -- `rank_within_position`
    would still work, but building the column on an empty frame needs the dtype set explicitly
    or the merge downstream sees `object`.
    """
    out = frame.copy()
    if out.empty:
        out[name] = pd.Series(dtype="int64")
        return out
    out[name] = rank_within_position(out, by, ascending=ascending)
    return out


def _lookup(frame: pd.DataFrame, gsis: str, column: str) -> float | None:
    """The player's value, or None when he is absent from the frame.

    None rather than 0.0 on purpose: a player with no weekly stats has not scored zero, he has
    not played, and the column renders those differently (an em dash against "0.0").
    """
    if not gsis or gsis not in frame.index:
        return None
    value = frame.at[gsis, column]
    return None if pd.isna(value) else float(value)


def _lookup_rank(frame: pd.DataFrame, gsis: str, column: str) -> int | None:
    if not gsis or gsis not in frame.index:
        return None
    value = frame.at[gsis, column]
    return None if pd.isna(value) else int(value)


def _number(value: CellValue) -> float:
    """A value's contribution to a total. `is not None` rather than `or`: a genuine 0.0 and
    "did not play" are kept distinct everywhere else on this page."""
    return float(value) if isinstance(value, int | float) else 0.0


def _sort_key(value: CellValue) -> float:
    """Descending by projection, with "no projection" last -- and a real 0.0 ahead of it."""
    return float("inf") if not isinstance(value, int | float) else -float(value)


def _blank_column_notes(run: MyTeamRun, ros_by_id: pd.DataFrame) -> tuple[str, ...]:
    """What the page had to leave blank, said out loud.

    A roster full of em dashes looks like a broken page. Saying "no weekly stats for this
    season yet" makes it obviously the expected preseason state instead.
    """
    notes: list[str] = []
    if run.ytd.empty:
        notes.append(
            "No weekly stats for this season yet, so year-to-date columns are empty. They "
            "fill in once Week 1 has been played."
        )
    unprojected: list[str] = []
    for _, player in run.roster.iterrows():
        gsis = display_str(player.get("gsis_id"))
        # Players the id_map could not resolve are skipped: `build_my_team` already names them,
        # under the reason that actually applies. Counting them here too listed the same player
        # twice, the second time miscategorising a just-signed WR as someone the pool does not
        # cover.
        if gsis and gsis not in ros_by_id.index:
            unprojected.append(display_str(player.get("player")) or gsis)
    if unprojected:
        shown = ", ".join(unprojected[:5])
        notes.append(
            f"{len(unprojected)} rostered players have no projection ({shown}) — kickers, "
            "defenses, and anyone the pool does not cover."
        )
    return tuple(notes)
