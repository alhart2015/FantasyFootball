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
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import espn_to_gsis, parse_rosters, parse_schedule, parse_teams
from projections.midseason.standings import ProjectionInputError, first_unplayed_week
from projections.rankings import rank_within_position
from projections.schemas import _PYARROW_STR
from projections.scoring.actuals import actual_season_total
from projections.web.views.columns import TEAM_COLUMNS, CellValue, Column
from projections.web.views.standings_view import Cell

#: Bench players are ranked and scored like anyone else, but the slot is worth showing as-is.
_BENCH_SLOTS = frozenset({"BENCH", "IR"})


@dataclass(frozen=True)
class PlayerRow:
    gsis_id: str
    #: Starters and bench are visually separated; a bench player outscoring a starter is the
    #: single most actionable thing this page can show.
    is_starter: bool = True
    #: Rest-of-season points, carried so the sort reads the VALUE rather than parsing it back
    #: out of the formatted cell. None sorts last -- his cell holds an em dash, not a number.
    sort_value: float | None = None
    #: The raw values behind the cells, kept so the colour scale can range over the column.
    values: Mapping[str, CellValue] = field(default_factory=dict)
    cells: tuple[Cell, ...] = ()


@dataclass(frozen=True)
class TeamPage:
    team_name: str
    season: int
    week: int
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
    roster_ytd = starter_ros = 0.0
    for _, player in roster.iterrows():
        gsis = str(player["gsis_id"])
        slot = str(player.get("lineup_slot", "") or "")
        is_starter = slot.upper() not in _BENCH_SLOTS

        ytd_points = _lookup(ytd_by_id, gsis, "actual_total")
        ros_points = _lookup(ros_by_id, gsis, "season_mean_fpts")
        # `is not None` rather than `or`: a genuine 0.0 and "did not play" are kept distinct
        # everywhere else on this page, and this was the one line that erased the difference.
        if ytd_points is not None:
            roster_ytd += ytd_points
        if is_starter and ros_points is not None:
            starter_ros += ros_points

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
                sort_value=ros_points,
                values=values,
            )
        )

    # Starters first, then bench, each by rest-of-season projection. A bench player above a
    # starter in the same position block is the page's most actionable signal, and burying it
    # under ESPN's slot ordering would hide it. Sorted on the VALUE, not on the formatted
    # string: at precision=1, 150.04 and 149.96 both render "150.0" and would tie.
    rows.sort(key=lambda row: (not row.is_starter, -(row.sort_value or float("-inf"))))

    # Colour every directional column against the ROSTER's own range, so a reader can see at a
    # glance which of his players are carrying the team. Ranks invert, since rank 1 is best.
    intensities = {
        column.key: _intensity([r.values.get(column.key) for r in rows], column.sense)
        for column in TEAM_COLUMNS
        if column.sense != "neutral"
    }
    rendered = tuple(
        PlayerRow(
            gsis_id=row.gsis_id,
            is_starter=row.is_starter,
            sort_value=row.sort_value,
            values=row.values,
            cells=tuple(
                Cell(
                    text=column.format(row.values[column.key]),
                    intensity=intensities.get(column.key, [None] * len(rows))[i],
                    numeric=column.numeric,
                )
                for column in TEAM_COLUMNS
            ),
        )
        for i, row in enumerate(rows)
    )

    return TeamPage(
        team_name=team_name,
        season=season,
        week=week,
        columns=TEAM_COLUMNS,
        rows=rendered,
        roster_ytd=roster_ytd,
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
        roster_ytd=0.0,
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


def _intensity(values: list[CellValue], sense: str) -> list[float | None]:
    """Signed [-1, 1] position within the column's own range, or all-None when tied.

    Same rule as the standings scale: a column where every value is identical carries no
    colour rather than every cell painted at the midpoint, and a missing value is uncoloured
    rather than treated as zero.
    """
    numbers = [v for v in values if isinstance(v, int | float)]
    if not numbers or max(numbers) == min(numbers):
        return [None] * len(values)
    low, high = min(numbers), max(numbers)
    out: list[float | None] = []
    for value in values:
        if not isinstance(value, int | float):
            out.append(None)
            continue
        scaled = 2 * ((float(value) - low) / (high - low)) - 1
        out.append(-scaled if sense == "lower-better" else scaled)
    return out


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


def remaining_points(pool: pd.DataFrame, ytd: pd.DataFrame) -> pd.DataFrame:
    """Points a player still has to score: his season projection minus what he already has.

    **This is a remaining TOTAL, and that is what the column claims to show.** Note it is not
    what `rest_of_season_pool` returns -- that one converts to a full-season-equivalent PACE,
    because the variance model divides by a fixed `SEASON_GAMES` to get a per-game mean. A pace
    is the right input for the simulator and the wrong thing to print under "projected points
    for the rest of the season", where a reader wants the number of points still coming.

    Both sides are our own numbers under the league ruleset -- the pool's `season_mean_fpts` is
    consensus projections scored by it, `actual_total` is weekly stats scored by it -- so the
    subtraction is like-for-like.

    Clamped at zero: a player who has already outscored his projection has nothing negative
    left to give, and a negative cell would read as a deduction.
    """
    scored = ytd.set_index("gsis_id")["actual_total"] if not ytd.empty else None
    out = pool.copy()
    if scored is None:
        return out
    already = out["gsis_id"].astype(str).map(scored).fillna(0.0)
    out["season_mean_fpts"] = (out["season_mean_fpts"] - already).clip(lower=0.0)
    return out


def assemble_team_page(
    payload: Mapping[str, Any],
    pool: pd.DataFrame,
    id_map: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    league_config: LeagueConfig,
    *,
    my_team_id: int,
    season: int,
) -> TeamPage:
    """The whole My Team pipeline, with the I/O lifted out so it can be tested.

    Every input is already-fetched data. The route does the reading; this does the assembly.
    That split is the point: pass 1 of the review found eight defects in this logic while it
    lived inline in a route, reachable only through an HTTP request that first made a live ESPN
    call. It was the only code on the branch without a test.

    The steps, and what each one is careful about:

    1. **The week comes from the schedule**, via `first_unplayed_week` -- the same authority the
       standings page uses. Deriving it from `max(weekly_stats.week)` describes nfl_data_py's
       ingest lag rather than the league, so the two pages disagreed on the same day.
    2. **The roster is filtered by `team_id`**, not by pool membership. Filtering by the pool
       deleted kickers and defenses before they could be shown, and made the note written to
       name them unreachable.
    3. **Rest-of-season is actually rest-of-season.** The pool is a full-season projection;
       `rest_of_season_pool` converts it over the games that remain, exactly as
       `project_league_standings` does, so the two pages agree about a player.
    4. A stats partition behind the schedule is reported rather than silently producing a
       partial season total under a later week's header.
    """
    teams = parse_teams(dict(payload))
    if my_team_id not in set(teams["team_id"]):
        raise ProjectionInputError(
            f"team {my_team_id} is not a team in this league. Valid ids: "
            f"{sorted(teams['team_id'])}."
        )

    calendar = LeagueCalendar.from_espn_settings(
        (payload.get("settings", {}) or {}).get("scheduleSettings", {}) or {}
    )
    schedule = parse_schedule(dict(payload), teams)
    week = first_unplayed_week(schedule, calendar) if not schedule.empty else 1

    rosters = parse_rosters(dict(payload))
    if rosters.empty:
        return empty_team_page("No rosters yet — the draft has not happened.", season=season)
    roster = rosters[rosters["team_id"] == my_team_id].assign(
        gsis_id=espn_to_gsis(rosters, id_map).astype(_PYARROW_STR)
    )
    roster = roster[roster["gsis_id"].notna()]

    ytd = (
        actual_season_total(weekly_stats, league_config.ruleset)
        if not weekly_stats.empty
        else _empty_ytd()
    )
    ros = remaining_points(pool, ytd)

    page = build_team_page(
        roster,
        ytd,
        ros,
        team_name=_team_name(teams, my_team_id),
        season=season,
        week=week,
    )
    stale = _staleness_note(weekly_stats, week)
    return replace(page, notes=page.notes + stale) if stale else page


def _empty_ytd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.Series(dtype=_PYARROW_STR),
            "position": pd.Series(dtype=_PYARROW_STR),
            "actual_total": pd.Series(dtype="float64"),
        }
    )


def _staleness_note(weekly_stats: pd.DataFrame, week: int) -> tuple[str, ...]:
    """Say so when the stats partition is behind the schedule.

    The dangerous case is stale-but-present: a PARTIAL season total under a header claiming a
    later week, which reads as a real number. An empty partition is already explained
    elsewhere; this covers the half that is not empty and not current.
    """
    if weekly_stats.empty:
        return ()
    latest = int(weekly_stats["week"].max())
    expected = week - 1
    if latest >= expected:
        return ()
    return (
        f"Year-to-date columns are stale: the stats partition ends at week {latest} but "
        f"{expected} weeks have been played. They are behind by "
        f"{expected - latest} week(s) until the weekly-stats ingest is re-run.",
    )


def _team_name(teams: pd.DataFrame, my_team_id: int) -> str:
    match = teams[teams["team_id"] == my_team_id]
    return str(match.iloc[0]["team_name"]) if not match.empty else f"Team {my_team_id}"
