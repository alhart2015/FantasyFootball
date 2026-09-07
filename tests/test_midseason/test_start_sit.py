"""Start/sit: the two-source weekly blend."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.midseason.start_sit import LineupRow, blend_weekly_points
from projections.schemas import InjuryStatus, RosterSlot, Ruleset

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


# --- the injury rule --------------------------------------------------------------------


def test_healthy_player_keeps_his_points() -> None:
    from projections.midseason.start_sit import startable_points

    assert startable_points(12.0, InjuryStatus.ACTIVE) == pytest.approx(12.0)


def test_questionable_takes_the_measured_haircut() -> None:
    """0.86 is measured, not chosen. Read the constant so a refit cannot silently diverge."""
    from projections.midseason.injuries import WEEKLY_MULTIPLIER
    from projections.midseason.start_sit import startable_points

    expected = 12.0 * WEEKLY_MULTIPLIER[InjuryStatus.QUESTIONABLE]
    assert startable_points(12.0, InjuryStatus.QUESTIONABLE) == pytest.approx(expected)


def test_doubtful_is_reduced_but_still_startable() -> None:
    """0.04, not None.

    `choose_starters` reads None as "cannot fill this slot at all". If the only alternative is
    a bye-week player -- who really is None -- starting the doubtful player is correct, and a
    None here would leave the slot empty instead.
    """
    from projections.midseason.start_sit import startable_points

    got = startable_points(12.0, InjuryStatus.DOUBTFUL)
    assert got is not None
    assert 0.0 < got < 1.0


def test_out_is_zero_not_none_so_a_forced_slot_can_still_be_filled() -> None:
    from projections.midseason.start_sit import startable_points

    assert startable_points(12.0, InjuryStatus.OUT) == pytest.approx(0.0)
    assert startable_points(12.0, InjuryStatus.SUSPENSION) == pytest.approx(0.0)


def test_injury_reserve_is_structurally_unstartable() -> None:
    """The one None. IR is a roster slot ESPN will not let you start out of, not a projection."""
    from projections.midseason.start_sit import startable_points

    assert startable_points(12.0, InjuryStatus.INJURY_RESERVE) is None


def test_no_projection_stays_no_projection() -> None:
    from projections.midseason.start_sit import startable_points

    assert startable_points(None, InjuryStatus.ACTIVE) is None
    assert startable_points(None, InjuryStatus.QUESTIONABLE) is None


def test_the_blend_is_not_told_the_source_already_priced_injuries() -> None:
    """The guard that keeps an Out player from reading at half of Sleeper's projection.

    ESPN zeroes Out; Sleeper's behaviour is unmeasured, so the blend carries ~half of
    Sleeper's number. `source_is_injury_aware=True` would leave that half standing.
    """
    from projections.midseason.injuries import weekly_multiplier
    from projections.midseason.start_sit import startable_points

    blended_out = 6.0  # half of Sleeper's 12, ESPN having zeroed its half
    assert startable_points(blended_out, InjuryStatus.OUT) == pytest.approx(0.0)
    # the wrong call, pinned so it cannot be reintroduced as a "simplification"
    aware = blended_out * weekly_multiplier(InjuryStatus.OUT, source_is_injury_aware=True)
    assert aware == pytest.approx(6.0)


# --- slot labels ------------------------------------------------------------------------


def test_the_solver_reports_the_slot_it_filled() -> None:
    from projections.draft.roster_eligibility import choose_starters_with_slots

    slots = {
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.FLEX: 1,
        RosterSlot.DST: 1,
    }
    players = [
        ("QB", 25.0),
        ("RB", 17.0),
        ("RB", 14.0),
        ("WR", 16.0),
        ("WR", 13.0),
        ("TE", 10.0),
        ("DST", 7.0),
        ("RB", 9.0),
    ]
    filled = choose_starters_with_slots(
        players, slots, value=lambda p: p[1], position=lambda p: p[0]
    )
    assert [slot for _, slot in filled] == [
        RosterSlot.QB,
        RosterSlot.RB,
        RosterSlot.RB,
        RosterSlot.WR,
        RosterSlot.WR,
        RosterSlot.TE,
        RosterSlot.DST,
        RosterSlot.FLEX,
    ]


def test_an_unfillable_slot_does_not_shift_the_labels_after_it() -> None:
    """The bug that reached a live run: a FLEX running back printed as the D/ST.

    The greedy SKIPS a slot it cannot fill, so the indices it returns are dense while the slot
    order is not. Reconstructing the mapping by zipping the two shifted every pick after the
    hole up by one. Here the D/ST slot has no eligible player -- exactly the live case, where
    neither source ships a defense stat line this layer can score.
    """
    from projections.draft.roster_eligibility import choose_starters_with_slots

    slots = {RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.DST: 1, RosterSlot.FLEX: 1}
    players = [("QB", 20.0), ("RB", 15.0), ("RB", 9.0)]  # no DST on the roster at all
    filled = choose_starters_with_slots(
        players, slots, value=lambda p: p[1], position=lambda p: p[0]
    )
    assert filled == [(0, RosterSlot.QB), (1, RosterSlot.RB), (2, RosterSlot.FLEX)]


def test_super_flex_is_filled_after_flex() -> None:
    from projections.draft.roster_eligibility import choose_starters_with_slots

    slots = {RosterSlot.QB: 1, RosterSlot.SUPER_FLEX: 1, RosterSlot.FLEX: 1}
    players = [("QB", 25.0), ("QB", 18.0), ("RB", 12.0)]
    filled = choose_starters_with_slots(
        players, slots, value=lambda p: p[1], position=lambda p: p[0]
    )
    assert [slot for _, slot in filled] == [
        RosterSlot.QB,
        RosterSlot.FLEX,
        RosterSlot.SUPER_FLEX,
    ]


def test_bench_and_ir_are_never_filled() -> None:
    """They carry counts in `roster_slots` and the greedy never walks them."""
    from projections.draft.roster_eligibility import choose_starters_with_slots

    filled = choose_starters_with_slots(
        [("QB", 20.0), ("RB", 9.0)],
        {RosterSlot.QB: 1, RosterSlot.BENCH: 6, RosterSlot.IR: 2},
        value=lambda p: p[1],
        position=lambda p: p[0],
    )
    assert filled == [(0, RosterSlot.QB)]


# --- the lineup currently set -----------------------------------------------------------


def _roster(*rows: tuple[int, str, str, str]) -> pd.DataFrame:
    """(player_id, player, pos, lineup_slot) in `parse_rosters` shape."""
    return pd.DataFrame(
        [
            {"player_id": pid, "player": name, "pos": pos, "lineup_slot": slot}
            for pid, name, pos, slot in rows
        ]
    )


def test_current_starters_exclude_bench_ir_and_unknown_slots() -> None:
    from projections.midseason.start_sit import current_starter_ids

    roster = _roster(
        (1, "Starter", "RB", RosterSlot.RB.value),
        (2, "Flexed", "WR", RosterSlot.FLEX.value),
        (3, "Benched", "WR", RosterSlot.BENCH.value),
        (4, "Injured", "TE", RosterSlot.IR.value),
        (5, "Unplaceable", "QB", ""),  # an ESPN slot id we do not recognise
    )
    assert current_starter_ids(roster) == {1, 2}


# --- current vs optimal -----------------------------------------------------------------


def _row(pid: int, pos: str, points: float | None, slot: RosterSlot | None) -> LineupRow:
    return LineupRow(
        player_id=pid,
        name=f"p{pid}",
        position=pos,
        status=InjuryStatus.ACTIVE,
        current_slot=slot,
        points=points,
        blend=None,
    )


_SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 2,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 5,
}


def test_an_already_optimal_lineup_produces_no_swaps() -> None:
    """A success, not a failure. In a settled week this is the usual answer."""
    from projections.midseason.start_sit import build_swaps

    rows = [
        _row(1, "QB", 20.0, RosterSlot.QB),
        _row(2, "RB", 15.0, RosterSlot.RB),
        _row(3, "RB", 12.0, RosterSlot.RB),
        _row(4, "WR", 14.0, RosterSlot.WR),
        _row(5, "WR", 11.0, RosterSlot.WR),
        _row(6, "TE", 8.0, RosterSlot.TE),
        _row(7, "RB", 10.0, RosterSlot.FLEX),
        _row(8, "WR", 4.0, None),
    ]
    assert build_swaps(rows, _SLOTS, params=None) == []


def test_the_gain_is_the_lineup_gain_not_the_pairwise_one() -> None:
    """The cascade the waiver work already paid to learn.

    WR c beats the WR in the lineup by 0.2. Starting him pushes that WR into the FLEX, which
    pushes the FLEX RB out entirely: the lineup gains 1.2, six times the pairwise number a
    human eye would compute against the man he appears to replace.
    """
    from projections.midseason.start_sit import build_swaps

    rows = [
        _row(1, "QB", 20.0, RosterSlot.QB),
        _row(2, "RB", 15.0, RosterSlot.RB),
        _row(3, "RB", 12.0, RosterSlot.RB),
        _row(4, "WR", 14.0, RosterSlot.WR),
        _row(5, "WR", 11.0, RosterSlot.WR),
        _row(6, "TE", 8.0, RosterSlot.TE),
        _row(7, "RB", 10.0, RosterSlot.FLEX),
        _row(8, "WR", 11.2, None),  # on the bench, 0.2 better than the started WR
    ]
    swaps = build_swaps(rows, _SLOTS, params=None)

    assert [s.start.player_id for s in swaps] == [8]
    assert [s.sit.player_id for s in swaps] == [7]
    assert swaps[0].gain == pytest.approx(1.2)


def test_swap_gains_sum_to_the_total_lineup_gain() -> None:
    """Two lineups, one set difference: the per-swap gains are a partition of the total."""
    from projections.midseason.start_sit import build_swaps, lineup_total, optimal_starters

    rows = [
        _row(1, "QB", 20.0, RosterSlot.QB),
        _row(2, "RB", 9.0, RosterSlot.RB),
        _row(3, "RB", 8.0, RosterSlot.RB),
        _row(4, "WR", 14.0, RosterSlot.WR),
        _row(5, "WR", 6.0, RosterSlot.WR),
        _row(6, "TE", 8.0, RosterSlot.TE),
        _row(7, "RB", 5.0, RosterSlot.FLEX),
        _row(8, "WR", 13.0, None),
        _row(9, "RB", 12.0, None),
    ]
    swaps = build_swaps(rows, _SLOTS, params=None)
    chosen, _ = optimal_starters(rows, _SLOTS)
    current = [r for r in rows if r.current_slot is not None]

    total = lineup_total(rows, chosen) - lineup_total(current, range(len(current)))
    assert sum(s.gain for s in swaps) == pytest.approx(total)


def test_a_bye_week_starter_is_swapped_out_for_anyone_startable() -> None:
    from projections.midseason.start_sit import build_swaps

    rows = [
        _row(1, "QB", 20.0, RosterSlot.QB),
        _row(2, "RB", 15.0, RosterSlot.RB),
        _row(3, "RB", 12.0, RosterSlot.RB),
        _row(4, "WR", None, RosterSlot.WR),  # bye: no projection from either source
        _row(5, "WR", 11.0, RosterSlot.WR),
        _row(6, "TE", 8.0, RosterSlot.TE),
        _row(7, "RB", 10.0, RosterSlot.FLEX),
        _row(8, "WR", 2.0, None),
    ]
    swaps = build_swaps(rows, _SLOTS, params=None)
    assert [s.start.player_id for s in swaps] == [8]
    assert [s.sit.player_id for s in swaps] == [4]


# --- P(right) ---------------------------------------------------------------------------


def test_p_right_is_a_coin_flip_between_identical_players() -> None:
    import numpy as np

    from projections.draft.assistant.performance_variance import VarianceParams
    from projections.midseason.start_sit import p_right

    params = VarianceParams.load()
    got = p_right(
        _row(1, "RB", 12.0, None),
        _row(2, "RB", 12.0, None),
        params,
        n_sims=20_000,
        rng=np.random.default_rng(0),
    )
    assert got == pytest.approx(0.5, abs=0.02)


def test_p_right_is_confident_when_the_edge_is_large() -> None:
    import numpy as np

    from projections.draft.assistant.performance_variance import VarianceParams
    from projections.midseason.start_sit import p_right

    params = VarianceParams.load()
    got = p_right(
        _row(1, "RB", 22.0, None),
        _row(2, "RB", 3.0, None),
        params,
        n_sims=20_000,
        rng=np.random.default_rng(0),
    )
    assert got > 0.90


def test_p_right_never_falls_as_the_edge_widens() -> None:
    import numpy as np

    from projections.draft.assistant.performance_variance import VarianceParams
    from projections.midseason.start_sit import p_right

    params = VarianceParams.load()
    probs = [
        p_right(
            _row(1, "RB", start, None),
            _row(2, "RB", 10.0, None),
            params,
            n_sims=20_000,
            rng=np.random.default_rng(7),
        )
        for start in (10.0, 12.0, 15.0, 20.0)
    ]
    assert probs == sorted(probs)


def test_p_right_is_reproducible_for_a_given_seed() -> None:
    """It is printed to a digit. A number that moves between runs is not a recommendation."""
    import numpy as np

    from projections.draft.assistant.performance_variance import VarianceParams
    from projections.midseason.start_sit import p_right

    params = VarianceParams.load()
    args = (_row(1, "RB", 13.0, None), _row(2, "WR", 11.0, None), params)
    first = p_right(*args, n_sims=5_000, rng=np.random.default_rng(3))
    second = p_right(*args, n_sims=5_000, rng=np.random.default_rng(3))
    assert first == second


def test_p_right_against_an_unstartable_player_is_certain() -> None:
    """Nothing beats nothing. A bye-week or IR counterpart scores no points at all."""
    import numpy as np

    from projections.draft.assistant.performance_variance import VarianceParams
    from projections.midseason.start_sit import p_right

    params = VarianceParams.load()
    got = p_right(
        _row(1, "RB", 8.0, None),
        _row(2, "RB", None, None),
        params,
        n_sims=2_000,
        rng=np.random.default_rng(0),
    )
    assert got == pytest.approx(1.0)


# --- end to end, through the real parsers -----------------------------------------------


def _espn_payload(*players: tuple[int, int, dict[str, float]]) -> dict[str, object]:
    """(espn_id, defaultPositionId, {stat_id: value}) -> a kona_player_info weekly payload."""
    return {
        "players": [
            {
                "player": {
                    "id": pid,
                    "defaultPositionId": pos_id,
                    "stats": [
                        {
                            "scoringPeriodId": 2,
                            "statSourceId": 1,
                            "statSplitTypeId": 1,
                            "stats": stats,
                        }
                    ],
                }
            }
            for pid, pos_id, stats in players
        ]
    }


def _sleeper_payload(*players: tuple[str, str, dict[str, float]]) -> list[dict[str, object]]:
    """(sleeper_id, position, {sleeper_stat: value}) -> the weekly endpoint's shape."""
    return [
        {
            "player_id": sid,
            "player": {"first_name": f"P{sid}", "last_name": "X", "position": pos},
            "stats": stats,
        }
        for sid, pos, stats in players
    ]


_E2E_SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 1,
    RosterSlot.WR: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 3,
}


def test_end_to_end_from_raw_payloads_through_the_real_parsers() -> None:
    """The seam between ingest and logic is where the web UI's defects lived. Test through it."""
    from projections.draft.backtest.espn_weekly import espn_weekly_statlines
    from projections.ingest.sleeper_weekly_projections import parse_sleeper_weekly
    from projections.midseason.start_sit import recommend_start_sit

    # ESPN stat ids: 3 pass_yd, 4 pass_td, 24 rush_yd, 42 rec_yd, 53 rec
    espn = espn_weekly_statlines(
        _espn_payload(
            (10, 1, {"3": 250.0, "4": 2.0}),  # QB
            (11, 2, {"24": 80.0}),  # RB, started
            (12, 3, {"42": 90.0, "53": 6.0}),  # WR, started
            (13, 2, {"24": 30.0}),  # RB, in the FLEX
            (14, 3, {"42": 120.0, "53": 8.0}),  # WR, benched and clearly better
        ),
        week=2,
    )
    sleeper = parse_sleeper_weekly(
        _sleeper_payload(
            ("s10", "QB", {"pass_yd": 250.0, "pass_td": 2.0}),
            ("s11", "RB", {"rush_yd": 80.0}),
            ("s12", "WR", {"rec_yd": 90.0, "rec": 6.0}),
            ("s13", "RB", {"rush_yd": 30.0}),
            ("s14", "WR", {"rec_yd": 120.0, "rec": 8.0}),
        ),
        season=2026,
        week=2,
    )
    id_map = pd.DataFrame(
        {
            "gsis_id": [f"00-000000{i}" for i in range(5)],
            "espn_id": ["10", "11", "12", "13", "14"],
            "sleeper_id": ["s10", "s11", "s12", "s13", "s14"],
        }
    )
    roster = pd.DataFrame(
        [
            {
                "player_id": 10,
                "player": "Passer",
                "pos": "QB",
                "lineup_slot": "QB",
                "injury_status": "",
            },
            {
                "player_id": 11,
                "player": "Back",
                "pos": "RB",
                "lineup_slot": "RB",
                "injury_status": "",
            },
            {
                "player_id": 12,
                "player": "Wideout",
                "pos": "WR",
                "lineup_slot": "WR",
                "injury_status": "",
            },
            {
                "player_id": 13,
                "player": "Flexed",
                "pos": "RB",
                "lineup_slot": "FLEX",
                "injury_status": "",
            },
            {
                "player_id": 14,
                "player": "Benched",
                "pos": "WR",
                "lineup_slot": RosterSlot.BENCH.value,
                "injury_status": "",
            },
        ]
    )

    run = recommend_start_sit(
        roster,
        espn,
        sleeper,
        id_map,
        _E2E_SLOTS,
        _HALF,
        team_name="Mine",
        week=2,
        params=None,
    )

    assert run.week == 2
    assert len(run.rows) == 5
    # the benched WR is better than the flexed RB, and the FLEX admits him
    assert [s.start.player_id for s in run.swaps] == [14]
    assert [s.sit.player_id for s in run.swaps] == [13]
    assert run.gain > 0
    assert sum(s.gain for s in run.swaps) == pytest.approx(run.gain)
    # both sources agreed on every player, so nothing to warn about
    assert run.notes == ()


def test_a_player_only_espn_prices_still_starts() -> None:
    """A D/ST or a just-signed back. Sleeper cannot see him; he must not vanish or read zero."""
    from projections.draft.backtest.espn_weekly import espn_weekly_statlines
    from projections.ingest.sleeper_weekly_projections import parse_sleeper_weekly
    from projections.midseason.start_sit import recommend_start_sit

    espn = espn_weekly_statlines(
        _espn_payload((10, 1, {"3": 250.0}), (11, 2, {"24": 80.0})), week=2
    )
    sleeper = parse_sleeper_weekly(
        _sleeper_payload(("s10", "QB", {"pass_yd": 250.0})), season=2026, week=2
    )
    id_map = pd.DataFrame({"gsis_id": ["00-0000000"], "espn_id": ["10"], "sleeper_id": ["s10"]})
    roster = pd.DataFrame(
        [
            {
                "player_id": 10,
                "player": "Passer",
                "pos": "QB",
                "lineup_slot": "QB",
                "injury_status": "",
            },
            {
                "player_id": 11,
                "player": "Lone",
                "pos": "RB",
                "lineup_slot": "RB",
                "injury_status": "",
            },
        ]
    )

    run = recommend_start_sit(
        roster,
        espn,
        sleeper,
        id_map,
        {RosterSlot.QB: 1, RosterSlot.RB: 1},
        _HALF,
        team_name="Mine",
        week=2,
        params=None,
    )

    lone = next(row for row in run.rows if row.player_id == 11)
    assert lone.blend is not None
    assert lone.blend.sources == "espn"
    assert lone.points == pytest.approx(8.0)
    assert 11 in {run.rows[i].player_id for i in run.starters}
    assert any("espn only" in note and "Lone" in note for note in run.notes)


def test_an_injured_starter_is_benched_for_a_healthy_backup() -> None:
    from projections.draft.backtest.espn_weekly import espn_weekly_statlines
    from projections.ingest.sleeper_weekly_projections import parse_sleeper_weekly
    from projections.midseason.start_sit import recommend_start_sit

    espn = espn_weekly_statlines(
        _espn_payload((11, 2, {"24": 100.0}), (12, 2, {"24": 90.0})), week=2
    )
    sleeper = parse_sleeper_weekly(
        _sleeper_payload(("s11", "RB", {"rush_yd": 100.0}), ("s12", "RB", {"rush_yd": 90.0})),
        season=2026,
        week=2,
    )
    id_map = pd.DataFrame(
        {
            "gsis_id": ["00-0000000", "00-0000001"],
            "espn_id": ["11", "12"],
            "sleeper_id": ["s11", "s12"],
        }
    )
    roster = pd.DataFrame(
        [
            # started, better on paper, but OUT -> multiplier 0.0
            {
                "player_id": 11,
                "player": "Hurt",
                "pos": "RB",
                "lineup_slot": "RB",
                "injury_status": "OUT",
            },
            {
                "player_id": 12,
                "player": "Fit",
                "pos": "RB",
                "lineup_slot": RosterSlot.BENCH.value,
                "injury_status": "",
            },
        ]
    )

    run = recommend_start_sit(
        roster,
        espn,
        sleeper,
        id_map,
        {RosterSlot.RB: 1},
        _HALF,
        team_name="Mine",
        week=2,
        params=None,
    )

    assert [s.start.player_id for s in run.swaps] == [12]
    assert [s.sit.player_id for s in run.swaps] == [11]
    hurt = next(row for row in run.rows if row.player_id == 11)
    assert hurt.points == pytest.approx(0.0)  # zero, not None: he could fill a forced slot


def test_an_unplaceable_lineup_slot_reads_as_bench_rather_than_raising() -> None:
    """`parse_rosters` should never emit one, so this is about which way to fail if it does.

    Reading him as a STARTER credits an unplaceable player to the current lineup and
    understates the gain from fixing it -- the same trap the My Team page documents. An
    earlier cut of this module raised ValueError here instead, which at least was loud.
    """
    from projections.midseason.start_sit import current_starter_ids, starting_slot

    assert starting_slot("BE") is None  # an ESPN abbreviation, not a RosterSlot value
    assert starting_slot("") is None
    assert starting_slot(None) is None
    assert starting_slot(RosterSlot.BENCH.value) is None
    assert starting_slot(RosterSlot.IR.value) is None
    assert starting_slot(RosterSlot.FLEX.value) is RosterSlot.FLEX

    roster = _roster((1, "Odd", "RB", "BE"), (2, "Real", "RB", RosterSlot.RB.value))
    assert current_starter_ids(roster) == {2}
