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
        n_sims=5000,
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
        n_sims=6000,
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
    # One RB on bye in week 3; the factorized result must match a brute-force per-week MC.
    # A short week range (with the bye inside it) keeps the brute-force reference cheap.
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = _avail({"00-0000001": 0.85, "00-0000002": 0.85}, bye={"00-0000001": 3})
    weeks = range(1, 6)
    n_sims = 3000

    fact = expected_season_points(
        roster, slots, avail, n_sims=n_sims, rng=np.random.default_rng(0), weeks=weeks
    )

    # Brute force: simulate every week independently.
    rng = np.random.default_rng(0)
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([avail.p_week(g) for g in gsis])
    acc = 0.0
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


def test_crn_matches_expected_season_points_no_bye_exact() -> None:
    # With identity column mapping, a no-bye roster, and the same seed, the CRN
    # kernel is BIT-IDENTICAL to expected_season_points (one rng.random((n_sims,n))
    # equals n_sims successive rng.random(n) draws). Guards column alignment + that
    # CRN reuses the same kernel — i.e. changes variance, not the mean.
    from projections.draft.assistant.season_value import expected_season_points_crn

    roster = _roster(
        [("00-0000001", "RB", 200.0), ("00-0000002", "WR", 180.0), ("00-0000003", "RB", 120.0)]
    )
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000001": 0.7, "00-0000002": 0.8, "00-0000003": 0.6})
    n_sims = 200
    col_of = {"00-0000001": 0, "00-0000002": 1, "00-0000003": 2}  # roster order

    draws = np.random.default_rng(11).random((n_sims, 3))
    crn = expected_season_points_crn(
        roster, slots, avail, draws=draws, col_of=col_of, weeks=range(1, 18)
    )
    esp = expected_season_points(
        roster, slots, avail, n_sims=n_sims, rng=np.random.default_rng(11), weeks=range(1, 18)
    )
    assert crn == esp


def test_crn_matches_expected_season_points_with_bye_in_expectation() -> None:
    # With a bye, CRN reuses the shared matrix across weeks while expected_season_points
    # draws fresh per week, so they are NOT bit-equal — but equal IN EXPECTATION.
    # Regression guard for the bye handling of the CRN kernel (spec §4, finding #2).
    from projections.draft.assistant.season_value import expected_season_points_crn

    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = _avail({"00-0000001": 0.85, "00-0000002": 0.85}, bye={"00-0000001": 3})
    n_sims = 4000
    col_of = {"00-0000001": 0, "00-0000002": 1}

    draws = np.random.default_rng(3).random((n_sims, 2))
    crn = expected_season_points_crn(
        roster, slots, avail, draws=draws, col_of=col_of, weeks=range(1, 6)
    )
    esp = expected_season_points(
        roster, slots, avail, n_sims=n_sims, rng=np.random.default_rng(7), weeks=range(1, 6)
    )
    assert abs(crn - esp) / esp < 0.02  # same expectation, independent MC noise


def test_crn_column_selection_is_by_gsis_not_position() -> None:
    # The kernel must pull each player's OWN column by gsis, so the SAME per-player
    # draws placed at DIFFERENT universe columns give an identical value. We build the
    # two players' draws once, then scatter them into two differently-ordered wide
    # universes; an exact match pins by-gsis selection (a by-position bug would diverge).
    from projections.draft.assistant.season_value import expected_season_points_crn

    roster = _roster([("00-0000002", "RB", 200.0), ("00-0000004", "RB", 120.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000002": 0.7, "00-0000004": 0.6})
    player_draws = np.random.default_rng(5).random((300, 2))  # cols: [p2, p4]

    # Universe A: [filler, p2, filler, p4]; Universe B reverses the ordering.
    universe_a = ["00-0000001", "00-0000002", "00-0000003", "00-0000004"]
    draws_a = np.zeros((300, 4))
    draws_a[:, 1], draws_a[:, 3] = player_draws[:, 0], player_draws[:, 1]
    col_a = {g: i for i, g in enumerate(universe_a)}

    universe_b = ["00-0000004", "00-0000003", "00-0000002", "00-0000001"]
    draws_b = np.zeros((300, 4))
    draws_b[:, 2], draws_b[:, 0] = player_draws[:, 0], player_draws[:, 1]
    col_b = {g: i for i, g in enumerate(universe_b)}

    val_a = expected_season_points_crn(roster, slots, avail, draws=draws_a, col_of=col_a)
    val_b = expected_season_points_crn(roster, slots, avail, draws=draws_b, col_of=col_b)
    assert val_a == val_b  # same per-player draws → identical value, regardless of column
    assert val_a > 0.0  # a real roster does not hit the empty short-circuit


def test_marginal_matches_closed_form_insurance() -> None:
    # Base = one risky starter S(p=0.6). Candidate backup B(120, p=0.7), {RB:1}, 1 week.
    # Marginal = points B adds = the insurance term only: (1-p_s)*p_b*B / 17.
    from projections.draft.assistant.season_value import marginal_season_values

    base = _roster([("00-0000001", "RB", 200.0)])
    cands = _roster([("00-0000002", "RB", 120.0)])
    avail = _avail({"00-0000001": 0.6, "00-0000002": 0.7})
    expected = 0.4 * 0.7 * 120 / 17  # ≈ 1.976
    out = marginal_season_values(
        base,
        cands,
        {RosterSlot.RB: 1},
        avail,
        n_sims=8000,
        rng=np.random.default_rng(0),
        weeks=range(1, 2),
    )
    assert abs(out["00-0000002"] - expected) < 0.1


def test_marginal_is_low_variance_under_crn() -> None:
    # The whole point of CRN: a marginal computed under shared draws is far tighter
    # across seeds than a naive marginal from independent draws, because the base
    # roster's (shared) availability variance cancels in V(base+c) - V(base) instead
    # of being carried twice (spec §3.3, §5.2). With a 3-starter base, that shared
    # variance dominates, so CRN's spread is several times smaller.
    from projections.draft.assistant.season_value import marginal_season_values

    base = _roster(
        [("00-0000001", "RB", 200.0), ("00-0000002", "RB", 180.0), ("00-0000003", "WR", 170.0)]
    )
    cands = _roster([("00-0000004", "RB", 150.0)])
    full = pd.concat([base, cands], ignore_index=True)
    avail = _avail(
        {f"00-000000{i}": p for i, p in zip(range(1, 5), (0.6, 0.6, 0.6, 0.7), strict=True)}
    )
    slots = {RosterSlot.RB: 2, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    week = range(1, 2)
    n_sims = 100
    seeds = 14

    crn = [
        marginal_season_values(
            base, cands, slots, avail, n_sims=n_sims, rng=np.random.default_rng(s), weeks=week
        )["00-0000004"]
        for s in range(seeds)
    ]
    # Naive marginal: score base and base+candidate with INDEPENDENT rngs (no CRN).
    indep = []
    for s in range(seeds):
        b = expected_season_points(
            base, slots, avail, n_sims=n_sims, rng=np.random.default_rng(100 + s), weeks=week
        )
        c = expected_season_points(
            full, slots, avail, n_sims=n_sims, rng=np.random.default_rng(900 + s), weeks=week
        )
        indep.append(c - b)

    assert all(v > 0.0 for v in crn)  # adding a useful backup always helps
    # True std ratio is ~0.3 (CRN cancels the 3 shared starters' variance); 0.75 leaves
    # ample margin for sampling noise at this seed count while still pinning the benefit.
    assert float(np.std(crn)) < 0.75 * float(np.std(indep))


def test_marginal_empty_base_is_solo_value() -> None:
    # With an empty base roster (first pick), the marginal is the candidate's own
    # expected season points (positive).
    from projections.draft.assistant.season_value import marginal_season_values

    base = _roster([])
    cands = _roster([("00-0000002", "RB", 180.0)])
    avail = _avail({"00-0000002": 0.9})
    out = marginal_season_values(
        base, cands, {RosterSlot.RB: 1}, avail, n_sims=300, rng=np.random.default_rng(0)
    )
    assert out["00-0000002"] > 0.0
