"""`rank_within_position` — the shared positional ranker.

Promoted out of `draft/snake_cheat_sheet.py` so the season dashboard can rank players by
year-to-date and rest-of-season points without importing a `_private` name across a package
boundary. It had no direct tests there; it does now, because the thing that makes it worth
having as its own function is a behaviour (integer ties) that is easy to regress into
`Series.rank()` and hard to notice.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.rankings import rank_within_position
from projections.schemas import _PYARROW_STR


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Columns declared explicitly so an empty `rows` still yields a correctly shaped frame --
    `pd.DataFrame([])` has no columns at all, which would test the fixture rather than the
    function."""
    frame = pd.DataFrame(
        {
            "gsis_id": pd.Series([r[0] for r in rows], dtype=_PYARROW_STR),
            "position": pd.Series([r[1] for r in rows], dtype=_PYARROW_STR),
            "points": pd.Series([r[2] for r in rows], dtype="float64"),
        }
    )
    return frame


def test_ranks_restart_at_one_for_each_position() -> None:
    frame = _frame(
        [
            ("00-0000001", "RB", 300.0),
            ("00-0000002", "RB", 200.0),
            ("00-0000003", "WR", 250.0),
            ("00-0000004", "WR", 150.0),
        ]
    )
    frame["rank"] = rank_within_position(frame, "points", ascending=False)
    by_id = frame.set_index("gsis_id")["rank"]
    assert by_id["00-0000001"] == 1
    assert by_id["00-0000002"] == 2
    assert by_id["00-0000003"] == 1, "WR ranking is independent of RB"
    assert by_id["00-0000004"] == 2


def test_a_tie_gets_consecutive_integers_not_a_half() -> None:
    """The reason this is not `Series.rank()`.

    Its default `method="average"` returns 1.5 for a two-way tie for first. In a table where
    every other rank is a whole number, "RB 1.5" reads as a bug rather than as a tie -- and a
    reader cannot tell it apart from a genuine rendering fault.
    """
    frame = _frame(
        [("00-0000001", "RB", 200.0), ("00-0000002", "RB", 200.0), ("00-0000003", "RB", 100.0)]
    )
    frame["rank"] = rank_within_position(frame, "points", ascending=False)
    assert sorted(frame["rank"]) == [1, 2, 3]
    assert frame["rank"].dtype.kind == "i", "ranks must be integers, never 1.5"


def test_a_tie_breaks_deterministically_on_gsis_id() -> None:
    """Two runs over the same data must produce the same table. Without the tie-break the
    order would follow whatever order the rows arrived in."""
    rows = [("00-0000002", "RB", 200.0), ("00-0000001", "RB", 200.0)]
    first = _frame(rows)
    first["rank"] = rank_within_position(first, "points", ascending=False)
    second = _frame(list(reversed(rows)))
    second["rank"] = rank_within_position(second, "points", ascending=False)

    assert first.set_index("gsis_id")["rank"].to_dict() == (
        second.set_index("gsis_id")["rank"].to_dict()
    )
    assert first.set_index("gsis_id")["rank"]["00-0000001"] == 1, "lower gsis_id wins the tie"


def test_ascending_ranks_the_smallest_first() -> None:
    """ADP is ranked ascending (pick 1 is best); points descending. Both are real callers."""
    frame = _frame(
        [("00-0000001", "RB", 40.0), ("00-0000002", "RB", 10.0), ("00-0000003", "RB", 25.0)]
    )
    frame["rank"] = rank_within_position(frame, "points", ascending=True)
    by_id = frame.set_index("gsis_id")["rank"]
    assert by_id["00-0000002"] == 1
    assert by_id["00-0000003"] == 2
    assert by_id["00-0000001"] == 3


def test_an_unranked_player_sorts_behind_every_ranked_one() -> None:
    """A player with no projection must not lead the position. pandas sorts NaN last by
    default, which is the behaviour wanted here -- pinned because a future `na_position`
    change would silently promote every unprojected player to the top of his position."""
    frame = _frame(
        [
            ("00-0000001", "RB", 100.0),
            ("00-0000002", "RB", float("nan")),
            ("00-0000003", "RB", 50.0),
        ]
    )
    frame["rank"] = rank_within_position(frame, "points", ascending=False)
    by_id = frame.set_index("gsis_id")["rank"]
    assert by_id["00-0000001"] == 1
    assert by_id["00-0000003"] == 2
    assert by_id["00-0000002"] == 3


def test_a_missing_column_is_named() -> None:
    frame = _frame([("00-0000001", "RB", 100.0)]).drop(columns=["position"])
    with pytest.raises(KeyError, match="position"):
        rank_within_position(frame, "points", ascending=False)


def test_an_empty_frame_returns_an_empty_rank() -> None:
    frame = _frame([])
    assert rank_within_position(frame, "points", ascending=False).empty
