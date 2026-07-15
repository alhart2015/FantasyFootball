from collections import Counter
from collections.abc import Mapping

from projections.draft.assistant.auction.nomination import (
    NominationContext,
    drain_max,
    drain_off_position,
)
from projections.schemas import Position


def _ctx(
    hero_positions: dict[Position, int],
    value_by_id: Mapping[str, float],
    position_by_id: Mapping[str, Position],
    position_minimums: Mapping[Position, int],
) -> NominationContext:
    return NominationContext(
        hero_positions=Counter(hero_positions),
        value_by_id=value_by_id,
        position_by_id=position_by_id,
        position_minimums=position_minimums,
    )


def test_drain_max_returns_the_priciest_candidate() -> None:
    ctx = _ctx(
        {},
        {"a": 10.0, "b": 30.0, "c": 20.0},
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
    )
    assert drain_max(["a", "b", "c"], ctx) == "b"  # 30.0 is the max value


def test_drain_off_position_drains_a_filled_position_not_the_priciest() -> None:
    # Hero has filled RB (2 >= min 2); WR is unfilled (0 < 3). 'b' (WR, $30) is priciest overall,
    # but the off-position pick is 'c' (RB, $20) — drain the slot the hero is done with.
    ctx = _ctx(
        {Position.RB: 2},
        {"a": 10.0, "b": 30.0, "c": 20.0},
        {"a": Position.WR, "b": Position.WR, "c": Position.RB},
        {Position.RB: 2, Position.WR: 3},
    )
    assert drain_off_position(["a", "b", "c"], ctx) == "c"


def test_drain_off_position_falls_back_to_drain_max_when_nothing_filled() -> None:
    # Hero has filled no position -> no off-position candidate -> fall back to the priciest overall.
    ctx = _ctx(
        {},
        {"a": 10.0, "b": 30.0, "c": 20.0},
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
    )
    assert drain_off_position(["a", "b", "c"], ctx) == "b"  # == drain_max
