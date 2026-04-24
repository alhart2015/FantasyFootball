"""Ruleset tests — point values for each scoring component, and named presets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from projections.schemas import Ruleset


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
