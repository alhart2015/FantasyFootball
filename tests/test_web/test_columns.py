"""Column definitions, and the invariant that templates never re-declare them.

The last test here is the point of the module. The model repo restates its stat categories in
a JS `Set`, two Jinja literal lists, and an enum, so adding a category means editing templates
and the copies can silently disagree. That failure cannot be caught by rendering a page — the
page renders fine, it is just missing a column — so it is checked against the template source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.web.views.columns import COLUMNS, STANDINGS_COLUMNS, TEAM_COLUMNS, Column

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
    """`nth-child` and `first-child` on a data cell encode column order. The one exception is
    kept deliberately: the leading column is a rank or slot in both tables and is dimmed as a
    group, not as a specific field."""
    css = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "projections"
        / "web"
        / "static"
        / "season.css"
    ).read_text(encoding="utf-8")
    offenders = [line.strip() for line in css.splitlines() if "nth-child" in line and "td" in line]
    assert not offenders, f"style these by semantic class, not position: {offenders}"
