"""Tests for the Monte-Carlo season valuer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.season_value import expected_season_points
from projections.schemas import _PYARROW_STR, RosterSlot


def _roster(players: list[tuple[str, str, float]]) -> pd.DataFrame:
    """players = [(gsis_id, position, season_mean_fpts), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([p[0] for p in players], dtype=_PYARROW_STR),
            "position": pd.array([p[1] for p in players], dtype=_PYARROW_STR),
            "season_mean_fpts": [p[2] for p in players],
        }
    )


def _avail(p: dict[str, float], bye: dict[str, int] | None = None) -> PlayerAvailability:
    return PlayerAvailability(p=p, bye=bye or {})


def test_closed_form_single_slot() -> None:
    # 1 RB, p=0.5, no bye, 2 weeks. per_game = 170/17 = 10. E = 2 * 0.5 * 10 = 10.
    roster = _roster([("00-0000001", "RB", 170.0)])
    avail = _avail({"00-0000001": 0.5})
    val = expected_season_points(
        roster,
        {RosterSlot.RB: 1},
        avail,
        n_sims=20000,
        rng=np.random.default_rng(0),
        weeks=range(1, 3),
    )
    assert abs(val - 10.0) < 0.3  # MC tolerance


def test_closed_form_two_player_backup() -> None:
    # The core insurance math: {RB:1}, starter S=200 (p=0.6) + backup B=120 (p=0.7),
    # no bye, 1 week. The best AVAILABLE RB starts, so
    #   E[week] = [p_s*p_b*max(S,B) + p_s*(1-p_b)*S + (1-p_s)*p_b*B] / 17.
    # (Pins the max(S,B) fill-in term -- a wrong insurance term, e.g. S+B, fails this.)
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "RB", 120.0)])
    avail = _avail({"00-0000001": 0.6, "00-0000002": 0.7})
    expected = (0.6 * 0.7 * 200 + 0.6 * 0.3 * 200 + 0.4 * 0.7 * 120) / 17  # = 9.035...
    val = expected_season_points(
        roster,
        {RosterSlot.RB: 1},
        avail,
        n_sims=40000,
        rng=np.random.default_rng(0),
        weeks=range(1, 2),
    )
    assert abs(val - expected) < 0.2


def test_reduces_to_starters_when_always_available() -> None:
    # p=1.0, no byes, 17 weeks -> equals optimal_lineup_points exactly (17 * season/17).
    roster = _roster(
        [("00-0000001", "RB", 200.0), ("00-0000002", "WR", 180.0), ("00-0000003", "RB", 120.0)]
    )
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000001": 1.0, "00-0000002": 1.0, "00-0000003": 1.0})
    val = expected_season_points(
        roster, slots, avail, n_sims=50, rng=np.random.default_rng(0), weeks=range(1, 18)
    )
    assert val == optimal_lineup_points(roster, slots)


def test_depth_is_rewarded_over_qb_hoarding() -> None:
    # Two rosters, same total projection, same starters. One adds a 3rd RB (real depth),
    # the other a spare QB (useless beyond the 1 QB slot). Depth must score higher.
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.FLEX: 1}
    base = [("00-0000001", "QB", 300.0), ("00-0000002", "RB", 200.0), ("00-0000003", "RB", 190.0)]
    depth = _roster([*base, ("00-0000004", "RB", 150.0)])
    hoard = _roster([*base, ("00-0000005", "QB", 150.0)])
    p = {f"00-000000{i}": 0.8 for i in range(1, 6)}
    val_depth = expected_season_points(
        depth, slots, _avail(p), n_sims=3000, rng=np.random.default_rng(1), weeks=range(1, 18)
    )
    val_hoard = expected_season_points(
        hoard, slots, _avail(p), n_sims=3000, rng=np.random.default_rng(1), weeks=range(1, 18)
    )
    assert val_depth > val_hoard


def test_determinism() -> None:
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "RB", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000001": 0.7, "00-0000002": 0.7})
    a = expected_season_points(
        roster, slots, avail, n_sims=200, rng=np.random.default_rng(5), weeks=range(1, 18)
    )
    b = expected_season_points(
        roster, slots, avail, n_sims=200, rng=np.random.default_rng(5), weeks=range(1, 18)
    )
    assert a == b


def test_bye_costs_points_and_factorization_matches_bruteforce() -> None:
    # One RB on bye in week 7; the factorized result must match a brute-force per-week MC.
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = _avail({"00-0000001": 0.85, "00-0000002": 0.85}, bye={"00-0000001": 7})

    fact = expected_season_points(
        roster, slots, avail, n_sims=8000, rng=np.random.default_rng(0), weeks=range(1, 18)
    )

    # Brute force: simulate every week independently.
    rng = np.random.default_rng(0)
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([avail.p_week(g) for g in gsis])
    weeks = list(range(1, 18))
    acc = 0.0
    n_sims = 8000
    for _ in range(n_sims):
        season_pts = 0.0
        for w in weeks:
            forced = np.array([avail.bye_week(g) == w for g in gsis])
            mask = (rng.random(len(roster)) < p_arr) & ~forced
            sub = roster.iloc[np.flatnonzero(mask)]
            season_pts += optimal_lineup_points(sub, slots) / 17.0
        acc += season_pts
    brute = acc / n_sims

    assert abs(fact - brute) / brute < 0.02  # within 2% (MC noise, same expectation)
