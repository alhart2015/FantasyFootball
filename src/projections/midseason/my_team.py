"""One team's roster, joined to what it has scored and what it still projects to score.

The domain half of the My Team page. It exists because the assembly kept ending up in the
wrong place: first inline in a Flask route (where nothing could test it, and eight defects
lived), then inside the view model (which is meant to present, not to parse ESPN payloads,
derive a week, run the scoring layer and rewrite a projection pool). The standings page has
had this layer since it was written -- `project_league_standings` -- and its equivalent lives
here, not in `web/`.

**What this module is careful about**, each learned the hard way:

1. **The week comes from the schedule**, via `first_unplayed_week` -- the same authority the
   standings page uses. Deriving it from `max(weekly_stats.week)` describes nfl_data_py's
   ingest lag rather than the league, so the two pages disagreed on the same day.
2. **The roster is filtered by `team_id`**, never by pool membership or by id_map resolution.
   Both of those quietly delete players -- kickers and defenses are never in the pool, and a
   just-signed player is not yet in the id_map -- and a roster that renders 13 men as 11 with
   no message is worse than one that renders an em dash.
3. **`ytd` is collapsed to one row per player before anything reads it.**
   `actual_season_total` groups by `(gsis_id, position)`, so a mid-season reclassification
   yields two rows, and `set_index("gsis_id")` on that is a non-unique index: `.map` raises
   `InvalidIndexError` and `.at` becomes invalid scalar access.
4. **Rest-of-season is a remaining TOTAL**, via `remaining_totals` -- not
   `rest_of_season_pool`, which returns a full-season-equivalent pace for the variance model.
   See that function's docstring for why the two exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import espn_to_gsis, parse_rosters, parse_schedule, parse_teams
from projections.midseason.rest_of_season import RosDiagnostics, remaining_totals
from projections.midseason.standings import ProjectionInputError, first_unplayed_week
from projections.schemas import _PYARROW_STR
from projections.scoring.actuals import actual_season_total


@dataclass(frozen=True)
class MyTeamRun:
    """Everything the My Team page needs, with no rendering decisions taken yet."""

    team_name: str
    #: `parse_rosters` shape for one team, plus `gsis_id`. **Unresolved players are still
    #: here**, carrying NA, so the page can list them rather than pretend they do not exist.
    roster: pd.DataFrame
    #: One row per player: `gsis_id`, `position`, `actual_total`.
    ytd: pd.DataFrame
    #: The pool with `season_mean_fpts` rewritten to remaining points.
    ros: pd.DataFrame
    week: int
    reg_weeks: int
    diagnostics: RosDiagnostics
    #: Non-fatal problems worth telling the reader about.
    notes: tuple[str, ...] = ()

    @property
    def regular_season_complete(self) -> bool:
        """`first_unplayed_week` returns `reg_weeks + 1` once every week has been played, so a
        header printing the week unconditionally reads "week 15" in a 14-week league."""
        return self.reg_weeks > 0 and self.week > self.reg_weeks


def build_my_team(
    payload: Mapping[str, Any],
    pool: pd.DataFrame,
    id_map: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    league_config: LeagueConfig,
    *,
    my_team_id: int,
    season: int,
) -> MyTeamRun:
    """Already-fetched data in, a `MyTeamRun` out. No I/O, no Flask, no formatting.

    Raises `ProjectionInputError` when `my_team_id` is not a team in this league, or when the
    team exists but has no roster -- both are states a page should explain rather than render
    as an empty table.
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

    roster, unresolved = _my_roster(dict(payload), id_map, my_team_id)

    ytd = (
        _one_row_per_player(actual_season_total(weekly_stats, league_config.ruleset))
        if not weekly_stats.empty
        else _empty_ytd()
    )
    scored = (
        ytd.set_index("gsis_id")["actual_total"].astype(float).to_dict() if not ytd.empty else {}
    )
    ros, diagnostics = remaining_totals(pool, scored)

    notes = _notes(unresolved, weekly_stats, week, diagnostics)
    return MyTeamRun(
        team_name=_team_name(teams, my_team_id),
        roster=roster,
        ytd=ytd,
        ros=ros,
        week=week,
        reg_weeks=calendar.reg_weeks,
        diagnostics=diagnostics,
        notes=notes,
    )


def _my_roster(
    payload: dict[str, Any], id_map: pd.DataFrame, my_team_id: int
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """My roster with `gsis_id` attached, plus the names it could not resolve.

    **Unresolved rows are kept.** `espn_to_gsis` returns NA for any ESPN id the crosswalk does
    not hold -- a just-signed player, a defense -- and dropping them makes a 15-man roster
    render as 14 with no explanation. The page lists them with em dashes and the note names
    them, which is the same contract the pool filter already follows.
    """
    rosters = parse_rosters(payload)
    if rosters.empty:
        raise ProjectionInputError("No rosters yet — the draft has not happened.")
    resolved = rosters.assign(gsis_id=espn_to_gsis(rosters, id_map).astype(_PYARROW_STR))
    roster = resolved[resolved["team_id"] == my_team_id]
    if roster.empty:
        # Checked AFTER filtering to my team, not before. The league-wide check passes as soon
        # as anyone has drafted, so mid-draft my own empty roster used to render as a headed
        # table with no rows and no message.
        raise ProjectionInputError(
            f"team {my_team_id} has no players on its roster yet — nothing to show until it "
            "makes a pick."
        )
    unresolved = tuple(
        _text(player.get("player")) or str(player["player_id"])
        for _, player in roster.iterrows()
        if pd.isna(player["gsis_id"])
    )
    return roster, unresolved


def _one_row_per_player(ytd: pd.DataFrame) -> pd.DataFrame:
    """Collapse `actual_season_total`'s (gsis_id, position) rows to one per player.

    That function groups by BOTH, so a player whose nflverse position is not constant across
    the season -- a positional reclassification, a QB/TE hybrid -- yields two rows. Left alone
    they make `set_index("gsis_id")` non-unique, which raises `InvalidIndexError` from `.map`,
    turns `frame.at[gsis, col]` into invalid scalar access, splits his season total in two, and
    counts him twice inside his position group so every player below him drops a rank.

    Done HERE, at the boundary, rather than inside the presenter: the previous version
    collapsed late, so the remaining-points subtraction upstream still saw the duplicate index
    and took the whole page down with a 500.

    His points are summed; his position is the one he scored most at, which is the one a reader
    would call him.
    """
    if ytd.empty or not ytd["gsis_id"].duplicated().any():
        return ytd
    ordered = ytd.sort_values("actual_total", ascending=False)
    return ordered.groupby("gsis_id", as_index=False).agg(
        position=("position", "first"), actual_total=("actual_total", "sum")
    )


def _empty_ytd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.Series(dtype=_PYARROW_STR),
            "position": pd.Series(dtype=_PYARROW_STR),
            "actual_total": pd.Series(dtype="float64"),
        }
    )


def _notes(
    unresolved: tuple[str, ...],
    weekly_stats: pd.DataFrame,
    week: int,
    diagnostics: RosDiagnostics,
) -> tuple[str, ...]:
    """Everything the page had to paper over, said out loud."""
    notes: list[str] = []
    if unresolved:
        shown = ", ".join(unresolved[:5])
        notes.append(
            f"{len(unresolved)} rostered players are not in the id_map ({shown}), so they have "
            "no year-to-date or projected figures. Re-run the id_map build to pick up recent "
            "signings."
        )
    notes.extend(_staleness_note(weekly_stats, week))
    warning = diagnostics.warning()
    if warning:
        # The wholesale-clamp alarm. Before this, the standings page raised it and the team
        # page -- reading the same pool through its own copy of the subtraction -- did not.
        notes.append(warning)
    return tuple(notes)


def _staleness_note(weekly_stats: pd.DataFrame, week: int) -> tuple[str, ...]:
    """Say so when the stats partition is behind the schedule.

    The dangerous case is stale-but-present: a PARTIAL season total under a header claiming a
    later week, which reads as a real number. An empty partition is explained by the page's own
    empty-YTD note; this covers the half that is not empty and not current.
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


def _text(value: object) -> str:
    """A display string, tolerating pandas NA.

    `x or y` evaluates `bool(x)`, and `bool(pd.NA)` raises -- so the idiom crashes the caller
    rather than blanking one field. Name columns here are pyarrow-backed nullable strings,
    where NA is admissible.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value)
