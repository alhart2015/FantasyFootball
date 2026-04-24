"""Position enum tests."""

from __future__ import annotations

import pytest

from projections.schemas import Position


def test_all_skill_positions_present() -> None:
    assert {p.value for p in Position} >= {"QB", "RB", "WR", "TE", "K", "DST"}


def test_position_is_string_enum() -> None:
    # str(Position.QB) should be usable in pandas filters.
    assert Position.QB.value == "QB"
    assert Position("QB") is Position.QB


def test_unknown_position_raises() -> None:
    with pytest.raises(ValueError):
        Position("FB")
