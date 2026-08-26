"""View model for the My Team page.

Year-to-date points and rank beside rest-of-season projection and rank, per player. Like the
standings view, **no Flask import belongs here** — every rule below is checked by calling a
function.

Unlike the standings view, this one assembles rather than presents: nothing in the repo
previously joined a roster to its actuals and its projection. The assembly is still pure, so
the I/O (reading `weekly_stats`, pulling the roster) stays in the route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from projections.rankings import rank_within_position
from projections.web.views.columns import TEAM_COLUMNS, CellValue, Column
from projections.web.views.standings_view import Cell

#: Bench players are ranked and scored like anyone else, but the slot is worth showing as-is.
_BENCH_SLOTS = frozenset({"BENCH", "IR"})


@dataclass(frozen=True)
class PlayerRow:
    gsis_id: str
    cells: tuple[Cell, ...]
    #: Starters and bench are visually separated; a bench player outscoring a starter is the
    #: single most actionable thing this page can show.
    is_starter: bool = True


@dataclass(frozen=True)
class TeamPage:
    team_name: str
    season: int
    week: int
    columns: tuple[Column, ...]
    rows: tuple[PlayerRow, ...]
    #: Totals across starters only — the bench does not score.
    starter_ytd: float
    starter_ros: float
    message: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.message is not None


def build_team_page(
    roster: pd.DataFrame,
    ytd: pd.DataFrame,
    ros: pd.DataFrame,
    *,
    team_name: str,
    season: int,
    week: int,
) -> TeamPage:
    """Roster + year-to-date actuals + rest-of-season projections -> the page model.

    `roster` is `parse_rosters`-shaped for one team, already carrying `gsis_id` (the route
    resolves ESPN ids through the id_map, exactly as the standings pipeline does).

    `ytd` is `actual_season_total`-shaped: `gsis_id`, `position`, `actual_total`. `ros` is the
    projection pool: `gsis_id`, `position`, `season_mean_fpts`.

    **Ranks are computed over the whole league pool, not over this roster.** "RB 4" means
    fourth-best running back in the league; ranking within a 13-man roster would produce a
    number that looks the same and means nothing.
    """
    ytd_ranked = _with_rank(ytd, "actual_total", "ytd_rank", ascending=False)
    ros_ranked = _with_rank(ros, "season_mean_fpts", "ros_rank", ascending=False)

    ytd_by_id = ytd_ranked.set_index("gsis_id")
    ros_by_id = ros_ranked.set_index("gsis_id")

    rows: list[PlayerRow] = []
    starter_ytd = starter_ros = 0.0
    for _, player in roster.iterrows():
        gsis = str(player["gsis_id"])
        slot = str(player.get("lineup_slot", "") or "")
        is_starter = slot.upper() not in _BENCH_SLOTS

        ytd_points = _lookup(ytd_by_id, gsis, "actual_total")
        ros_points = _lookup(ros_by_id, gsis, "season_mean_fpts")
        if is_starter:
            starter_ytd += ytd_points or 0.0
            starter_ros += ros_points or 0.0

        values: Mapping[str, CellValue] = {
            "slot": slot or "—",
            "player": str(player.get("player", "") or gsis),
            "position": str(player.get("pos", "") or ""),
            "ytd_points": ytd_points,
            "ytd_rank": _lookup_rank(ytd_by_id, gsis, "ytd_rank"),
            "ros_points": ros_points,
            "ros_rank": _lookup_rank(ros_by_id, gsis, "ros_rank"),
        }
        rows.append(
            PlayerRow(
                gsis_id=gsis,
                is_starter=is_starter,
                cells=tuple(
                    Cell(text=column.format(values[column.key]), numeric=column.numeric)
                    for column in TEAM_COLUMNS
                ),
            )
        )

    # Starters first, then bench, each by rest-of-season projection. A bench player above a
    # starter in the same position block is the page's most actionable signal, and burying it
    # under ESPN's slot ordering would hide it.
    rows.sort(key=lambda row: (not row.is_starter, _sort_key(row)))

    return TeamPage(
        team_name=team_name,
        season=season,
        week=week,
        columns=TEAM_COLUMNS,
        rows=tuple(rows),
        starter_ytd=starter_ytd,
        starter_ros=starter_ros,
        notes=_notes(roster, ytd_by_id, ros_by_id),
    )


def empty_team_page(message: str, *, season: int) -> TeamPage:
    """No roster, or no team selected. Carries the reason rather than an empty table."""
    return TeamPage(
        team_name="",
        season=season,
        week=0,
        columns=TEAM_COLUMNS,
        rows=(),
        starter_ytd=0.0,
        starter_ros=0.0,
        message=message,
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
    if gsis not in frame.index:
        return None
    value = frame.at[gsis, column]
    return None if pd.isna(value) else float(value)


def _lookup_rank(frame: pd.DataFrame, gsis: str, column: str) -> int | None:
    if gsis not in frame.index:
        return None
    value = frame.at[gsis, column]
    return None if pd.isna(value) else int(value)


def _sort_key(row: PlayerRow) -> float:
    """Rest-of-season points for ordering, with an unprojected player sorting last rather than
    first — the cell holds an em dash, not a number."""
    ros_index = next(i for i, column in enumerate(TEAM_COLUMNS) if column.key == "ros_points")
    text = row.cells[ros_index].text
    try:
        return -float(text)
    except ValueError:
        return float("inf")


def _notes(roster: pd.DataFrame, ytd: pd.DataFrame, ros: pd.DataFrame) -> tuple[str, ...]:
    """What the page had to leave blank, said out loud.

    A roster full of em dashes looks like a broken page. Saying "no weekly stats for this
    season yet" makes it obviously the expected preseason state instead.
    """
    notes: list[str] = []
    if ytd.empty:
        notes.append(
            "No weekly stats for this season yet, so year-to-date columns are empty. They "
            "fill in once Week 1 has been played."
        )
    unprojected = [
        str(player.get("player", "") or player["gsis_id"])
        for _, player in roster.iterrows()
        if str(player["gsis_id"]) not in ros.index
    ]
    if unprojected:
        shown = ", ".join(unprojected[:5])
        notes.append(
            f"{len(unprojected)} rostered players have no projection ({shown}) — kickers, "
            "defenses, and anyone the pool does not cover."
        )
    return tuple(notes)
