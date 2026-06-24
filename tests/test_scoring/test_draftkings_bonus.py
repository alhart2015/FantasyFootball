import pytest

from projections.scoring import dk_actuals_bonus


@pytest.mark.parametrize(
    "pass_yd, rush_yd, rec_yd, expected",
    [
        (299, 99, 99, 0.0),
        (300, 0, 0, 3.0),
        (0, 100, 0, 3.0),
        (0, 0, 100, 3.0),
        (300, 100, 0, 6.0),
        (350, 120, 110, 9.0),
    ],
)
def test_dk_actuals_bonus_thresholds(
    pass_yd: float, rush_yd: float, rec_yd: float, expected: float
) -> None:
    assert (
        dk_actuals_bonus(passing_yards=pass_yd, rushing_yards=rush_yd, receiving_yards=rec_yd)
        == expected
    )
