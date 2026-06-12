from projections.draft.backtest.lineup import weekly_lineup_points
from projections.schemas import Position, RosterSlot

SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 1,
    RosterSlot.WR: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 2,
}


def _player(pos: Position, proj: float | None, actual: float | None) -> dict[str, object]:
    return {"position": pos.value, "projected": proj, "actual": actual}


def test_starts_highest_projection_scores_actual() -> None:
    roster = [
        _player(Position.RB, proj=20.0, actual=5.0),  # RB slot -> actual 5
        _player(Position.RB, proj=10.0, actual=30.0),  # FLEX -> actual 30
        _player(Position.RB, proj=1.0, actual=99.0),  # benched (proj too low) -> 0
        _player(Position.QB, proj=15.0, actual=12.0),  # QB slot -> 12
        _player(Position.WR, proj=8.0, actual=8.0),  # WR slot -> 8
    ]
    # by projection: QB(12) + RB#1[proj20]->5 + WR[8] + FLEX=RB#2[proj10]->30 = 55
    assert weekly_lineup_points(roster, SLOTS) == 55.0


def test_player_with_no_projection_is_unstartable() -> None:
    roster = [_player(Position.QB, proj=None, actual=40.0)]
    assert weekly_lineup_points(roster, SLOTS) == 0.0


def test_started_but_no_actual_scores_zero() -> None:
    roster = [_player(Position.QB, proj=20.0, actual=None)]
    assert weekly_lineup_points(roster, SLOTS) == 0.0


def test_unfilled_slot_scores_zero() -> None:
    roster = [_player(Position.QB, proj=20.0, actual=18.0)]
    assert weekly_lineup_points(roster, SLOTS) == 18.0


def test_score_by_projected_sums_projected_of_started() -> None:
    """score_by='projected' starts the same lineup (by projection) but sums projected points.

    This is the "who drafted better under shared beliefs" metric: no actual outcome,
    just the projected value of the lineup a rational manager sets.
    """
    roster = [
        _player(Position.RB, proj=20.0, actual=5.0),  # RB slot -> proj 20
        _player(Position.RB, proj=10.0, actual=30.0),  # FLEX -> proj 10
        _player(Position.RB, proj=1.0, actual=99.0),  # benched (proj too low) -> 0
        _player(Position.QB, proj=15.0, actual=12.0),  # QB slot -> proj 15
        _player(Position.WR, proj=8.0, actual=8.0),  # WR slot -> proj 8
    ]
    # same lineup as the actual test; summed by projection: 15 + 20 + 8 + 10 = 53
    assert weekly_lineup_points(roster, SLOTS, score_by="projected") == 53.0
