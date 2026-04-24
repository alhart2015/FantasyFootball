"""Scoring tests -- every rule x every preset."""

from __future__ import annotations

import pytest

from projections.schemas import Ruleset
from projections.scoring import StatLine, score


def _zero_line(**overrides: float) -> StatLine:
    base = dict(
        passing_yards=0.0,
        passing_tds=0,
        interceptions=0,
        passing_2pt_conversions=0,
        rushing_yards=0.0,
        rushing_tds=0,
        rushing_2pt_conversions=0,
        receptions=0,
        receiving_yards=0.0,
        receiving_tds=0,
        receiving_2pt_conversions=0,
        fumbles_lost=0,
        return_tds=0,
    )
    base.update(overrides)
    return StatLine(**base)  # type: ignore[arg-type]


def test_passing_yards() -> None:
    line = _zero_line(passing_yards=300.0)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(12.0)


def test_passing_yards_partial() -> None:
    line = _zero_line(passing_yards=275.0)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(11.0)


def test_passing_td_and_int() -> None:
    line = _zero_line(passing_tds=2, interceptions=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(2 * 4 + 1 * -2)


def test_rushing() -> None:
    line = _zero_line(rushing_yards=120.0, rushing_tds=2)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(12.0 + 12.0)


def test_receiving_full_ppr() -> None:
    line = _zero_line(receptions=8, receiving_yards=100.0, receiving_tds=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(8 + 10 + 6)


def test_receiving_half_ppr() -> None:
    line = _zero_line(receptions=8, receiving_yards=100.0, receiving_tds=1)
    assert score(line, Ruleset.espn_half()) == pytest.approx(4 + 10 + 6)


def test_receiving_standard() -> None:
    line = _zero_line(receptions=8, receiving_yards=100.0, receiving_tds=1)
    assert score(line, Ruleset.standard()) == pytest.approx(0 + 10 + 6)


def test_fumble_and_two_pt_and_return_td() -> None:
    line = _zero_line(fumbles_lost=1, rushing_2pt_conversions=1, return_tds=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(-2 + 2 + 6)


def test_jefferson_real_line() -> None:
    # 9 rec, 110 rec yds, 1 rec TD => 9 + 11 + 6 = 26 in PPR.
    line = _zero_line(receptions=9, receiving_yards=110.0, receiving_tds=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(26.0)


def test_negative_yards_dock_below_zero() -> None:
    line = _zero_line(rushing_yards=-5.0)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(-0.5)
