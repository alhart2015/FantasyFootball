from __future__ import annotations

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, expected_points, score


def test_fractional_counts_are_scored() -> None:
    r = Ruleset()  # ESPN_PPR
    line = {"receptions": 105.0, "receiving_yards": 1335.0, "receiving_tds": 8.4}
    # 105*1 + 1335/10 + 8.4*6 = 105 + 133.5 + 50.4 = 288.9
    assert expected_points(line, r) == 288.9


def test_absent_keys_treated_as_zero() -> None:
    r = Ruleset()
    assert expected_points({"rushing_yards": 100.0}, r) == 10.0
    assert expected_points({}, r) == 0.0


def test_equivalent_to_score_on_integer_lines() -> None:
    r = Ruleset()
    fields = {
        "passing_yards": 4000.0,
        "passing_tds": 30.0,
        "interceptions": 10.0,
        "rushing_yards": 200.0,
        "rushing_tds": 2.0,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "fumbles_lost": 3.0,
    }
    line = StatLine(
        passing_yards=4000.0,
        passing_tds=30,
        interceptions=10,
        rushing_yards=200.0,
        rushing_tds=2,
        fumbles_lost=3,
    )
    assert expected_points(fields, r) == score(line, r)


def test_half_ppr_ruleset_applies() -> None:
    r = Ruleset.espn_half()
    assert expected_points({"receptions": 100.0}, r) == 50.0
