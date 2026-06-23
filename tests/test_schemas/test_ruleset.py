"""Ruleset tests — point values for each scoring component, and named presets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from projections.schemas import _RULESET_NAME_VALUES, Ruleset
from projections.scoring.score import StatLine, score


def test_default_ruleset_is_espn_ppr() -> None:
    r = Ruleset()
    # ESPN standard PPR defaults.
    assert r.passing_yds_per_pt == 25.0
    assert r.passing_td_pts == 4.0
    assert r.interception_pts == -2.0
    assert r.rushing_yds_per_pt == 10.0
    assert r.rushing_td_pts == 6.0
    assert r.receiving_yds_per_pt == 10.0
    assert r.receiving_td_pts == 6.0
    assert r.reception_pts == 1.0  # full PPR
    assert r.fumble_lost_pts == -2.0
    assert r.two_pt_pts == 2.0
    assert r.return_td_pts == 6.0


def test_espn_half_preset() -> None:
    r = Ruleset.espn_half()
    assert r.reception_pts == 0.5
    assert r.passing_td_pts == 4.0


def test_standard_preset() -> None:
    r = Ruleset.standard()
    assert r.reception_pts == 0.0
    assert r.passing_td_pts == 4.0


def test_ruleset_is_immutable() -> None:
    r = Ruleset()
    with pytest.raises(ValidationError):  # pydantic raises ValidationError on frozen
        r.reception_pts = 0.5


def test_ruleset_has_name() -> None:
    assert Ruleset().name == "ESPN_PPR"
    assert Ruleset.espn_half().name == "ESPN_HALF"
    assert Ruleset.standard().name == "STANDARD"


def test_ruleset_custom_name_allowed() -> None:
    r = Ruleset(name="MY_LEAGUE", reception_pts=0.5, passing_td_pts=6.0)
    assert r.name == "MY_LEAGUE"
    assert r.passing_td_pts == 6.0


def test_draftkings_preset_values() -> None:
    dk = Ruleset.draftkings()
    assert dk.name == "DRAFTKINGS"
    assert dk.interception_pts == -1.0  # DK is -1, ESPN is -2
    assert dk.fumble_lost_pts == -1.0
    assert dk.reception_pts == 1.0  # full PPR
    assert dk.passing_yds_per_pt == 25.0
    assert dk.rushing_yds_per_pt == 10.0


def test_draftkings_scores_known_line() -> None:
    dk = Ruleset.draftkings()
    # 300 pass yd, 2 pass TD, 1 INT, 50 rush yd, 5 rec, 80 rec yd, 1 fum
    line = StatLine(
        passing_yards=300,
        passing_tds=2,
        interceptions=1,
        rushing_yards=50,
        receptions=5,
        receiving_yards=80,
        fumbles_lost=1,
    )
    # 300/25 + 2*4 + 1*-1 + 50/10 + 5*1 + 80/10 + 1*-1 = 12 + 8 -1 +5 +5 +8 -1
    assert score(line, dk) == 36.0


def test_draftkings_in_allowlist() -> None:
    assert "DRAFTKINGS" in _RULESET_NAME_VALUES
