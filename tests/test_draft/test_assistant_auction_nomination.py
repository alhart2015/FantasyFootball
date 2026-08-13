from collections import Counter
from collections.abc import Mapping

from projections.draft.assistant.auction.nomination import (
    NominationContext,
    drain_max,
    drain_off_position,
    drain_value_gap,
    drain_value_gap_off_position,
)
from projections.schemas import Position


def _ctx(
    hero_positions: dict[Position, int],
    value_by_id: Mapping[str, float],
    position_by_id: Mapping[str, Position],
    position_minimums: Mapping[Position, int],
    hero_value_by_id: Mapping[str, float] | None = None,
) -> NominationContext:
    # Default = the room agrees with us exactly (the model market), which is what the price-ranked
    # heuristics' tests want; the gap-ranked tests pass a genuinely disagreeing board.
    return NominationContext(
        hero_positions=Counter(hero_positions),
        value_by_id=value_by_id,
        position_by_id=position_by_id,
        position_minimums=position_minimums,
        hero_value_by_id=dict(value_by_id) if hero_value_by_id is None else hero_value_by_id,
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


# --- gap-ranked (Slice 2b) -------------------------------------------------------------------
# Every case below is built so the max-GAP player is NOT the max-PRICE player, so a heuristic that
# silently degraded to `drain_max` would fail rather than coincidentally pass.

_ROOM = {"a": 10.0, "b": 30.0, "c": 25.0}  # what the room pays
_OURS = {"a": 2.0, "b": 28.0, "c": 5.0}  # what we think they're worth
# gaps: a +8, b +2, c +20  ->  max gap is 'c', max price is 'b'


def test_drain_value_gap_picks_the_biggest_overpay_not_the_priciest() -> None:
    ctx = _ctx(
        {},
        _ROOM,
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
        hero_value_by_id=_OURS,
    )
    assert drain_max(["a", "b", "c"], ctx) == "b"  # the priciest...
    assert drain_value_gap(["a", "b", "c"], ctx) == "c"  # ...is not the biggest overpay


def test_drain_value_gap_is_a_no_op_signal_when_the_room_shares_our_board() -> None:
    # The model market: bot_dollars IS auction_dollars, so every gap is 0 and the heuristic has no
    # signal to act on. Documents why this family is ESPN-only rather than leaving it to be found.
    ctx = _ctx(
        {},
        _ROOM,
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
    )
    assert all(ctx.value_by_id[g] - ctx.hero_value_by_id[g] == 0.0 for g in ("a", "b", "c"))


def test_drain_value_gap_off_position_restricts_to_a_filled_position() -> None:
    # Hero filled RB. 'c' (QB) is the biggest overpay overall but the hero still needs a QB;
    # among RBs the overpay is 'a' (+8) over 'b' (+2), even though 'b' is the pricier RB.
    ctx = _ctx(
        {Position.RB: 2},
        _ROOM,
        {"a": Position.RB, "b": Position.RB, "c": Position.QB},
        {Position.RB: 2, Position.QB: 1},
        hero_value_by_id=_OURS,
    )
    assert drain_value_gap_off_position(["a", "b", "c"], ctx) == "a"


def test_drain_value_gap_off_position_falls_back_when_nothing_filled() -> None:
    ctx = _ctx(
        {},
        _ROOM,
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
        hero_value_by_id=_OURS,
    )
    assert drain_value_gap_off_position(["a", "b", "c"], ctx) == "c"  # == drain_value_gap
