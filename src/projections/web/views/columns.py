"""Column definitions for every dashboard table, declared once.

The model repo restates its stat categories in at least four places -- a JS `Set`, literal
lists inside two Jinja partials, and an enum -- so adding a category means editing templates,
and the four copies can disagree without anything failing. `tests/test_web/test_columns.py`
pins that no template here re-declares a column.

A template iterates `COLUMNS[...]` and asks each spec how to render its own value. It never
knows what the columns *are*.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

#: How a cell's value should be read as good or bad. `"neutral"` means no colour at all --
#: which is different from "colour it in the middle", and is the right answer for an identifier
#: or a count that has no direction.
Sense = Literal["higher-better", "lower-better", "neutral"]

#: What a cell can hold. `None` is a real value here -- it means "no number", which is
#: distinct from zero and renders as an em dash.
CellValue = float | int | str | None


@dataclass(frozen=True)
class Cell:
    """One rendered table cell: its text, and how strongly it reads as good or bad.

    Lives here rather than in a page's view model because both pages render one. It was
    declared in `standings_view` and imported across into `team_view`, which made the standings
    page a dependency of the team page for a type neither owns -- and put the shared rendering
    contract in the module least likely to be read when changing the other table.

    `intensity` is a signed share in [-1, 1] which the template emits as a CSS custom property
    so the stylesheet does the colour mixing. `None` means no colour at all -- used for columns
    with no direction, and for a column where every row is tied.
    """

    text: str
    intensity: float | None = None
    numeric: bool = True
    #: The cell naming the row's subject. Styled by class, not by position.
    is_label: bool = False


@dataclass(frozen=True)
class Column:
    """One table column: what it is called, how to format it, and which way is good."""

    #: Attribute name on the row object the template iterates.
    key: str
    #: Column header.
    label: str
    #: Longer text for a `title=` tooltip. Empty means no tooltip.
    help: str = ""
    sense: Sense = "neutral"
    #: Decimal places for a float. None renders the value as-is (strings, ints, pre-formatted).
    precision: int | None = None
    #: Render as a percentage: 0.575 -> "57.5%".
    percent: bool = False
    #: Right-align. Numbers read better right-aligned; names do not.
    numeric: bool = True
    #: Marks the column that names the thing each row is about. The stylesheet targets this
    #: rather than `nth-child`, which hard-coded an ordering only this module owns -- reorder a
    #: table and the accent would have highlighted the wrong cell, silently.
    is_label: bool = False

    def format(self, value: CellValue) -> str:
        """The value as it should appear in a cell.

        `None` renders as an em dash rather than "None" or an empty cell: a missing number and
        a zero are different facts, and a blank cell reads as the latter.

        A non-numeric value reaching a `percent` or `precision` column is a wiring mistake --
        the registry says that column holds a number -- so it raises rather than rendering
        something plausible. `str` is the escape hatch for a genuinely textual column, which
        declares neither.
        """
        if value is None:
            return "—"
        if self.percent or self.precision is not None:
            if isinstance(value, str):
                raise TypeError(
                    f"column {self.key!r} is numeric but got the string {value!r}; a textual "
                    "column should declare neither `percent` nor `precision`"
                )
            if self.percent:
                return f"{float(value) * 100:.1f}%"
            return f"{float(value):.{self.precision}f}"
        return str(value)


#: Standings table. Every column here already exists on `ProjectedStandingsSchema`, so this is
#: a presentation layer over that contract rather than a second declaration of it.
STANDINGS_COLUMNS: tuple[Column, ...] = (
    Column(key="rank", label="#", numeric=True),
    Column(key="team_name", label="Team", numeric=False, is_label=True),
    Column(key="record", label="Rec", help="Wins-losses-ties from played weeks", numeric=False),
    Column(
        key="points_for",
        label="PF",
        help="Points scored so far",
        sense="higher-better",
        precision=1,
    ),
    Column(
        key="projected_wins",
        label="Proj W",
        help=(
            "Mean final wins over the simulations. A tie counts half a win, as ESPN seeds, "
            "so this is not the record plus games remaining."
        ),
        sense="higher-better",
        precision=1,
    ),
    Column(
        key="projected_points_for",
        label="Proj PF",
        help="Mean final points scored — banked plus simulated",
        sense="higher-better",
        precision=0,
    ),
    Column(
        key="make_playoffs_pct",
        label="Playoff",
        help="Share of simulations finishing inside the playoff field",
        sense="higher-better",
        percent=True,
    ),
    Column(
        key="bye_pct",
        label="Bye",
        help="Share of simulations earning a first-round bye",
        sense="higher-better",
        percent=True,
    ),
    Column(
        key="champ_pct",
        label="Title",
        help="Share of simulations won outright",
        sense="higher-better",
        percent=True,
    ),
)

#: My Team table. `ytd_*` come from our own scoring of `weekly_stats` under the league
#: ruleset; `ros_*` from the rest-of-season projection. Ranks are within position.
TEAM_COLUMNS: tuple[Column, ...] = (
    Column(key="slot", label="Slot", numeric=False),
    Column(key="player", label="Player", numeric=False, is_label=True),
    Column(key="position", label="Pos", numeric=False),
    Column(
        key="ytd_points",
        label="YTD",
        help="Fantasy points scored so far, under this league's scoring",
        sense="higher-better",
        precision=1,
    ),
    Column(
        key="ytd_rank",
        label="YTD rk",
        help="Rank at his position, by points scored so far",
        sense="lower-better",
    ),
    Column(
        key="ros_points",
        label="ROS",
        help="Projected points for the rest of the season",
        sense="higher-better",
        precision=1,
    ),
    Column(
        key="ros_rank",
        label="ROS rk",
        help="Rank at his position, by projected rest-of-season points",
        sense="lower-better",
    ),
)


def require_every_key(
    available: Iterable[str], columns: tuple[Column, ...], *, source: str
) -> None:
    """Every column must have somewhere to read its value from.

    Without this a column key that no longer exists on the frame -- renamed upstream, dropped
    by a schema's `strict="filter"` -- renders as an em dash in every row. That is the same
    glyph this page uses for "he has not played", so a whole column of missing DATA is
    indistinguishable from a whole column of missing PLAYERS, and neither the tests nor the
    page say anything.
    """
    missing = [column.key for column in columns if column.key not in set(available)]
    if missing:
        raise KeyError(
            f"{source} has no column for {missing}; every entry in the column registry must "
            "have a value to read, or its cells silently render as em dashes"
        )


COLUMNS: Mapping[str, tuple[Column, ...]] = {
    "standings": STANDINGS_COLUMNS,
    "team": TEAM_COLUMNS,
}
