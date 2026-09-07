"""Start/sit: the two-source weekly blend."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.midseason.start_sit import blend_weekly_points
from projections.schemas import Ruleset

_HALF = Ruleset.espn_half()


def _espn(**by_id: dict[str, float]) -> pd.DataFrame:
    """`espn_weekly_statlines` shape: espn_id + the nine fields, absent ones explicitly 0.0."""
    from projections.ingest.external_projections import WEEKLY_BLEND_FIELDS

    rows = [
        {
            "espn_id": eid,
            "position": "WR",
            **{f: float(line.get(f, 0.0)) for f in WEEKLY_BLEND_FIELDS},
        }
        for eid, line in by_id.items()
    ]
    return pd.DataFrame(rows, columns=["espn_id", "position", *WEEKLY_BLEND_FIELDS])


def _sleeper(**by_id: dict[str, float]) -> pd.DataFrame:
    """`parse_sleeper_weekly` shape: sleeper_id + the nine fields, absent ones pd.NA."""
    from projections.ingest.external_projections import WEEKLY_BLEND_FIELDS

    rows = [
        {
            "sleeper_id": sid,
            "position": "WR",
            **{f: line.get(f, pd.NA) for f in WEEKLY_BLEND_FIELDS},
        }
        for sid, line in by_id.items()
    ]
    return pd.DataFrame(rows, columns=["sleeper_id", "position", *WEEKLY_BLEND_FIELDS])


def _id_map(pairs: dict[str, str]) -> pd.DataFrame:
    """{sleeper_id: espn_id}."""
    return pd.DataFrame(
        {
            "gsis_id": [f"00-{i:07d}" for i in range(len(pairs))],
            "espn_id": list(pairs.values()),
            "sleeper_id": list(pairs.keys()),
        }
    )


def test_blends_per_stat_then_scores_once() -> None:
    espn = _espn(**{"1": {"receiving_yards": 100.0, "receptions": 4.0, "receiving_tds": 1.0}})
    sleeper = _sleeper(**{"s1": {"receiving_yards": 50.0, "receptions": 2.0, "receiving_tds": 0.0}})

    out = blend_weekly_points(espn, sleeper, _id_map({"s1": "1"}), weight_espn=0.5, ruleset=_HALF)

    got = out["1"]
    # half-PPR: 0.1/yd, 0.5/rec, 6/TD
    assert got.espn == pytest.approx(18.0)  # 10 + 2 + 6
    assert got.sleeper == pytest.approx(6.0)  # 5 + 1 + 0
    # blended line is rec_yd 75, rec 3, td 0.5 -> 7.5 + 1.5 + 3
    assert got.points == pytest.approx(12.0)
    assert got.sources == "both"


def test_weight_one_reproduces_espn_and_weight_zero_reproduces_sleeper() -> None:
    espn = _espn(**{"1": {"receiving_yards": 100.0, "receptions": 4.0}})
    sleeper = _sleeper(**{"s1": {"receiving_yards": 50.0, "receptions": 2.0}})
    ids = _id_map({"s1": "1"})

    all_espn = blend_weekly_points(espn, sleeper, ids, weight_espn=1.0, ruleset=_HALF)["1"]
    all_slp = blend_weekly_points(espn, sleeper, ids, weight_espn=0.0, ruleset=_HALF)["1"]

    assert all_espn.points == pytest.approx(all_espn.espn)
    assert all_slp.points == pytest.approx(all_slp.sleeper)


def test_a_field_only_one_source_reports_is_not_treated_as_zero() -> None:
    """The whole reason the blend lives in stat space rather than points space.

    Sleeper omits receptions. Stat space takes ESPN's 4 receptions at FULL weight and blends
    only the yards; points space would average a total containing those receptions against
    one that silently excludes them, reading low by half of ESPN's reception points.
    """
    espn = _espn(**{"1": {"receiving_yards": 100.0, "receptions": 4.0}})
    sleeper = _sleeper(**{"s1": {"receiving_yards": 50.0}})  # no receptions key -> pd.NA

    got = blend_weekly_points(espn, sleeper, _id_map({"s1": "1"}), weight_espn=0.5, ruleset=_HALF)[
        "1"
    ]

    # yards blend to 75 (7.5 pts); receptions stay 4.0 from ESPN alone (2.0 pts)
    assert got.points == pytest.approx(9.5)
    # the points-space answer, which this must NOT equal
    assert got.espn is not None and got.sleeper is not None
    assert got.points != pytest.approx((got.espn + got.sleeper) / 2)


def test_espn_only_player_is_priced_by_espn_and_says_so() -> None:
    """K and D/ST always land here: `parse_sleeper_weekly` filters to QB/RB/WR/TE."""
    espn = _espn(**{"1": {"receiving_yards": 70.0}})
    got = blend_weekly_points(espn, _sleeper(), _id_map({}), weight_espn=0.5, ruleset=_HALF)["1"]

    assert got.sources == "espn"
    assert got.sleeper is None
    assert got.points == pytest.approx(7.0)
    assert got.espn == pytest.approx(7.0)


def test_sleeper_only_player_is_priced_by_sleeper_and_says_so() -> None:
    got = blend_weekly_points(
        _espn(),
        _sleeper(**{"s1": {"receiving_yards": 40.0}}),
        _id_map({"s1": "1"}),
        weight_espn=0.5,
        ruleset=_HALF,
    )["1"]

    assert got.sources == "sleeper"
    assert got.espn is None
    assert got.points == pytest.approx(4.0)


def test_a_player_neither_source_prices_is_absent_not_zero() -> None:
    """Absence is unstartability downstream. A 0.0 would fill a slot nobody else can."""
    out = blend_weekly_points(
        _espn(**{"1": {"receiving_yards": 70.0}}),
        _sleeper(),
        _id_map({}),
        weight_espn=0.5,
        ruleset=_HALF,
    )
    assert "2" not in out


def test_a_sleeper_player_missing_from_the_id_map_falls_back_to_espn() -> None:
    """The just-signed player `waivers` deliberately avoids gsis for. He must not vanish."""
    espn = _espn(**{"1": {"receiving_yards": 100.0}})
    sleeper = _sleeper(**{"s99": {"receiving_yards": 50.0}})  # not in the crosswalk

    got = blend_weekly_points(espn, sleeper, _id_map({"s1": "1"}), weight_espn=0.5, ruleset=_HALF)[
        "1"
    ]

    assert got.sources == "espn"
    assert got.points == pytest.approx(10.0)


def test_weight_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight_espn"):
        blend_weekly_points(_espn(), _sleeper(), _id_map({}), weight_espn=1.5, ruleset=_HALF)
