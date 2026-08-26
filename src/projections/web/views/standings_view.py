"""View model for the standings page.

Pure functions in, frozen dataclasses out. **No Flask import belongs in this module** — that
is what lets every rule below be tested by calling a function, with no app, request, or
browser. `tests/test_web/test_app.py` checks the source for it.

The page computes nothing: `project_league_standings` already returns every figure, so the
work here is presentation — ordering, formatting, the colour scale, and the empty states.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projections.midseason.standings import StandingsRun, regular_season_complete
from projections.web.views.columns import (
    STANDINGS_COLUMNS,
    Cell,
    CellValue,
    Column,
    column_intensities,
    require_every_key,
)


@dataclass(frozen=True)
class TeamRow:
    team_id: int
    cells: tuple[Cell, ...]
    #: Highlight this row as the viewer's team.
    is_mine: bool = False


@dataclass(frozen=True)
class RemainingGame:
    week: int
    opponent: str
    at_home: bool
    win_pct: float


@dataclass(frozen=True)
class StandingsPage:
    """Everything the standings template needs. A template reads this and nothing else."""

    league_name: str
    season: int
    week: int
    reg_weeks: int
    weeks_remaining: int
    n_matchups_played: int
    columns: tuple[Column, ...]
    rows: tuple[TeamRow, ...]
    my_games: tuple[RemainingGame, ...]
    #: Set when the page cannot be built. Rendered instead of the table.
    message: str | None = None
    #: Non-fatal warnings worth surfacing (a stale pool, dropped players).
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.message is not None

    @property
    def regular_season_complete(self) -> bool:
        return regular_season_complete(self.week, self.reg_weeks)


def _record(row: pd.Series) -> str:
    """`6-1` or `6-1-1`. The tie is only shown when there is one, because "6-1-0" reads as
    though a tie were expected and absent."""
    base = f"{int(row['wins'])}-{int(row['losses'])}"
    return f"{base}-{int(row['ties'])}" if int(row["ties"]) else base


def build_standings_page(
    run: StandingsRun,
    *,
    season: int,
    my_team_id: int | None,
) -> StandingsPage:
    """`StandingsRun` -> the rendered page model.

    Rows arrive already ordered by `build_standings` (projected wins, then projected points),
    so the `#` column is position in that order rather than a re-sort here — two places
    deciding the order is two places to disagree.
    """
    frame = run.standings.reset_index(drop=True)
    display = frame.assign(
        rank=range(1, len(frame) + 1),
        record=[_record(row) for _, row in frame.iterrows()],
    )

    require_every_key(display.columns, STANDINGS_COLUMNS, source="the standings frame")

    # `require_every_key` above raises when a registry key is absent, so the per-column
    # membership test the previous version carried here could no longer be false.
    # One walk of the frame, and the team id travels WITH its values rather than in a parallel
    # list kept in step by a positional index. The intensity scales are still keyed by position
    # because they are computed per column across all rows, which is what they are.
    rendered_rows = [
        (
            int(row["team_id"]),
            {column.key: _cell_value(row, column) for column in STANDINGS_COLUMNS},
        )
        for _, row in display.iterrows()
    ]
    scales = column_intensities([values for _, values in rendered_rows], STANDINGS_COLUMNS)

    rows = tuple(
        TeamRow(
            team_id=team_id,
            is_mine=my_team_id is not None and team_id == my_team_id,
            cells=tuple(
                Cell(
                    text=column.format(values[column.key]),
                    intensity=scales[column.key][i],
                    numeric=column.numeric,
                    is_label=column.is_label,
                )
                for column in STANDINGS_COLUMNS
            ),
        )
        for i, (team_id, values) in enumerate(rendered_rows)
    )

    notes: list[str] = []
    warning = run.diagnostics.warning()
    if warning:
        notes.append(warning)
    if run.n_players_dropped:
        notes.append(
            f"{run.n_players_dropped} rostered players are outside the projection pool "
            "(kickers, defenses, or no projection) and were skipped."
        )

    return StandingsPage(
        league_name=run.league_name,
        season=season,
        week=run.snapshot_week,
        reg_weeks=run.calendar.reg_weeks,
        weeks_remaining=run.weeks_remaining,
        n_matchups_played=run.n_matchups_played,
        columns=STANDINGS_COLUMNS,
        rows=rows,
        my_games=_my_games(run.odds, my_team_id),
        notes=tuple(notes),
    )


def empty_standings_page(message: str, *, season: int) -> StandingsPage:
    """The page when there is nothing to project — no schedule, no rosters, or before the
    season starts. Carries the reason rather than rendering an empty table, which would read
    as "everyone is 0-0" instead of "there is no data"."""
    return StandingsPage(
        league_name="",
        season=season,
        week=0,
        reg_weeks=0,
        weeks_remaining=0,
        n_matchups_played=0,
        columns=STANDINGS_COLUMNS,
        rows=(),
        my_games=(),
        message=message,
    )


def _cell_value(row: pd.Series, column: Column) -> CellValue:
    value = row.get(column.key)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        # Not `display_str`: an absent NUMBER renders as an em dash, which is a different fact
        # from an absent name rendering as blank, and the two must not converge.
        return None
    if isinstance(value, str):
        return value
    return float(value) if column.precision is not None or column.percent else int(value)


def _my_games(odds: pd.DataFrame, my_team_id: int | None) -> tuple[RemainingGame, ...]:
    """My remaining fixtures, with the probability stated from my side rather than the home
    team's — `home_win_pct` on an away row is my chance of LOSING, and a table that does not
    flip it is quietly reporting the opposite of what it says."""
    if my_team_id is None or odds.empty:
        return ()
    mine = odds[(odds["home_team_id"] == my_team_id) | (odds["away_team_id"] == my_team_id)]
    games: list[RemainingGame] = []
    for _, row in mine.iterrows():
        at_home = int(row["home_team_id"]) == my_team_id
        games.append(
            RemainingGame(
                week=int(row["week"]),
                opponent=str(row["away_team"] if at_home else row["home_team"]),
                at_home=at_home,
                win_pct=float(row["home_win_pct"] if at_home else 1.0 - row["home_win_pct"]),
            )
        )
    return tuple(games)
