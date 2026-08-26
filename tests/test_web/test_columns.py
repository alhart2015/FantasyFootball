"""Column definitions, and the invariant that templates never re-declare them.

The last test here is the point of the module. The model repo restates its stat categories in
a JS `Set`, two Jinja literal lists, and an enum, so adding a category means editing templates
and the copies can silently disagree. That failure cannot be caught by rendering a page — the
page renders fine, it is just missing a column — so it is checked against the template source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from projections.web.views.columns import (
    COLUMNS,
    STANDINGS_COLUMNS,
    TEAM_COLUMNS,
    Column,
    require_every_key,
)

_TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "projections" / "web" / "templates"


def test_a_missing_value_renders_as_a_dash_not_a_blank() -> None:
    """A blank cell reads as zero. "No number here" and "zero" are different facts, and the
    difference matters on a page whose whole job is telling you how you are doing."""
    assert Column(key="x", label="X", precision=1).format(None) == "—"


def test_percentages_render_from_a_fraction() -> None:
    assert Column(key="x", label="X", percent=True).format(0.575) == "57.5%"
    assert Column(key="x", label="X", percent=True).format(1.0) == "100.0%"
    assert Column(key="x", label="X", percent=True).format(0.0) == "0.0%"


def test_precision_is_honoured() -> None:
    assert Column(key="x", label="X", precision=0).format(1234.56) == "1235"
    assert Column(key="x", label="X", precision=1).format(1234.56) == "1234.6"


def test_a_column_with_no_precision_passes_the_value_through() -> None:
    """Identifiers, slots and pre-formatted records must not be coerced to floats."""
    assert Column(key="x", label="X").format("6-1-1") == "6-1-1"
    assert Column(key="x", label="X").format(3) == "3"


@pytest.mark.parametrize("table", sorted(COLUMNS))
def test_every_column_key_is_unique_within_its_table(table: str) -> None:
    keys = [column.key for column in COLUMNS[table]]
    assert len(keys) == len(set(keys)), f"duplicate key in {table}: {keys}"


@pytest.mark.parametrize("table", sorted(COLUMNS))
def test_every_column_has_a_label(table: str) -> None:
    assert all(column.label for column in COLUMNS[table])


def test_rank_columns_are_lower_better() -> None:
    """Rank 1 is the best rank. A colour scale that reads it as "higher is better" would paint
    the best player at every position as the worst."""
    ranks = [c for c in TEAM_COLUMNS if c.key.endswith("_rank")]
    assert ranks, "the team table should have rank columns"
    assert all(column.sense == "lower-better" for column in ranks)


def test_probability_columns_are_percentages() -> None:
    for key in ("make_playoffs_pct", "bye_pct", "champ_pct"):
        column = next(c for c in STANDINGS_COLUMNS if c.key == key)
        assert column.percent, f"{key} holds a fraction and must render as a percentage"


def test_projected_wins_explains_that_a_tie_counts_half() -> None:
    """`projected_wins` is a mean of credited wins, so it is fractional whether or not anyone
    tied, and it is NOT the record plus games remaining. A reader differencing the two columns
    gets a number that means nothing, so the column says so itself."""
    column = next(c for c in STANDINGS_COLUMNS if c.key == "projected_wins")
    assert "tie" in column.help.lower()


# --- the invariant ---------------------------------------------------------------------------


def test_no_template_re_declares_a_column() -> None:
    """Templates iterate `COLUMNS[...]`; they never spell out what the columns are.

    A Jinja literal list of column keys renders perfectly well and simply omits whatever the
    registry gained, so nothing fails — which is exactly why this is checked at the source
    rather than by rendering. The model repo has four copies of its category list for want of
    this test.
    """
    every_key = {column.key for columns in COLUMNS.values() for column in columns}
    offenders: dict[str, list[str]] = {}
    for template in _TEMPLATES.rglob("*.html"):
        text = template.read_text(encoding="utf-8")
        # A literal `"key"` or `'key'` inside a template is a re-declaration; `column.key` and
        # `row.key` attribute access are not.
        quoted = [key for key in every_key if f'"{key}"' in text or f"'{key}'" in text]
        if quoted:
            offenders[template.name] = sorted(quoted)
    assert not offenders, (
        "templates must iterate COLUMNS rather than naming columns: "
        f"{offenders}. Adding a column should not mean editing a template."
    )


def test_no_template_hardcodes_a_column_header() -> None:
    """The other half of the same invariant, and the half a key-grep misses.

    A template writing `<th>Playoff</th>` re-declares the column just as surely as one writing
    `"champ_pct"`, and nothing above would have seen it: the label is the human string, not the
    key. The registry is then no longer the single place a header can be changed, which is the
    whole reason it exists.

    Literal `<th>`s are still allowed for tables the registry does not drive -- the remaining-
    games table is hand-written because its rows are fixtures, not rows of a projection frame.
    What is not allowed is a literal header that duplicates a registry LABEL.
    """
    labels = {column.label.strip().lower() for columns in COLUMNS.values() for column in columns}
    offenders: dict[str, list[str]] = {}
    for template in _TEMPLATES.rglob("*.html"):
        text = template.read_text(encoding="utf-8")
        literal = [
            header.strip()
            for header in re.findall(r"<th[^>]*>([^<{]*?)</th>", text)
            if header.strip().lower() in labels
        ]
        if literal:
            offenders[template.name] = sorted(literal)
    assert not offenders, (
        f"these headers duplicate a registry label: {offenders}. Render them from "
        "`page.columns` so the label lives in one place."
    )


def test_the_shared_table_macros_are_the_only_copy() -> None:
    """Both pages draw the same table. Each used to carry its own copy of the scroll wrapper
    with its accessibility attributes, the header loop, and the four-line whitespace-controlled
    cell loop that emits the colour-scale property.

    Two copies of an a11y contract is one copy that goes stale without anything failing, so the
    markup lives in `_table.html` and the pages call it.
    """
    offenders: dict[str, list[str]] = {}
    for template in _TEMPLATES.rglob("*.html"):
        if template.name.startswith("_"):
            continue
        text = template.read_text(encoding="utf-8")
        # Every marker, not just the last one to match -- a dict keyed on the filename inside
        # the comprehension reported one offender per file and hid the rest.
        found = [marker for marker in ("table-scroll", "cell.intensity") if marker in text]
        if found:
            offenders[template.name] = found
    assert not offenders, (
        f"{offenders} re-implements shared table markup; call the macros in _table.html. "
        "This holds for hand-written tables too -- the remaining-games table has literal "
        "headers because no registry drives it, and still gets its wrapper from `t.scroll`."
    )


def test_a_string_in_a_numeric_column_is_a_loud_wiring_error() -> None:
    """The registry says which columns hold numbers. A string arriving at one means a view
    model handed over the wrong field, and `float("6-1-1")` would otherwise surface as a
    ValueError from deep inside a format call with no mention of which column."""
    with pytest.raises(TypeError, match="ytd_points"):
        Column(key="ytd_points", label="YTD", precision=1).format("6-1-1")


def test_exactly_one_label_column_per_table() -> None:
    """The stylesheet accents the cell naming each row's subject. It used to find it with
    `nth-child(2)`, which hard-coded an ordering this module owns -- reorder a table and the
    accent silently moved to the wrong cell, the same failure `test_no_template_re_declares_a_
    column` exists to prevent, expressed in CSS where that test does not look."""
    for table, columns in COLUMNS.items():
        labels = [c for c in columns if c.is_label]
        assert len(labels) == 1, f"{table} should have exactly one label column, got {labels}"


def test_no_stylesheet_rule_targets_a_column_by_position() -> None:
    """`nth-child(2)` on a data cell hard-codes which column holds the name -- an ordering
    `views/columns.py` owns, so reordering a table would silently accent the wrong cell.

    All the positional spellings are checked, not just `nth-child`: `nth-of-type`, `last-child`
    and the `td + td` chain say the same thing in different words, and a rule that bans one
    spelling of a mistake invites the others.

    `:first-child` is the deliberate exception, and it is allowlisted by its exact rule rather
    than by pattern. It is positional because what it means is positional: the LEADING column
    is dimmed whatever it holds -- a rank on one table, a slot on the other -- so it says "this
    is the gutter", not "this is the rank".
    """
    css = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "projections"
        / "web"
        / "static"
        / "season.css"
    ).read_text(encoding="utf-8")
    allowed = {".data-table td:first-child { color: var(--muted); }"}
    # Anchored to `td`, because the position of a ROW is not the position of a COLUMN --
    # `tr:last-child td { border-bottom: none }` is about the bottom of the table and has
    # nothing to do with which column holds what.
    positional = re.compile(r"td:(?:nth-child|nth-of-type|first-child|last-child)|td\s*\+\s*td")
    offenders = [
        line.strip()
        for line in css.splitlines()
        if positional.search(line) and line.strip() not in allowed
    ]
    assert not offenders, f"style these by semantic class, not position: {offenders}"


def test_a_column_with_nowhere_to_read_from_is_an_error_not_an_em_dash() -> None:
    """A column key the row does not carry renders an em dash in EVERY row -- the same glyph
    the page uses for "he has not played". A whole column of missing DATA then looks exactly
    like a whole column of missing PLAYERS, and neither the page nor the tests say anything.

    The realistic cause is a rename upstream, or a pandera schema's `strict="filter"` dropping
    a column the registry still asks for.
    """
    with pytest.raises(KeyError, match="champ_pct"):
        require_every_key(
            [c.key for c in STANDINGS_COLUMNS if c.key != "champ_pct"],
            STANDINGS_COLUMNS,
            source="a frame missing a column",
        )


def test_every_key_present_passes_quietly() -> None:
    require_every_key([c.key for c in TEAM_COLUMNS], TEAM_COLUMNS, source="the team row")
