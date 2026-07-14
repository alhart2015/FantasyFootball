import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.auction.market import (
    AggressiveBot,
    BalancedBot,
    BotArchetype,
    PatientValueBot,
    SeatView,
    assign_bot_archetypes,
    bot_max_bid,
    resolve_bids,
    resolve_unbid,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import Position, RosterSlot, Ruleset


def _config(min_bid: int = 1) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=min_bid,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {"in_pool": [True], "auction_dollars": [40], "bot_dollars": [40]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )


def _player() -> pd.Series:
    return pd.Series({"gsis_id": "00-0000001", "position": "RB", "season_mean_fpts": 200.0})


def test_bot_centers_on_baseline_with_zero_jitter() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3),
        _player(),
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 40


def test_bot_floors_at_min_bid() -> None:
    base = pd.DataFrame(
        {"in_pool": [False], "auction_dollars": [0], "bot_dollars": [0]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )
    bid = bot_max_bid(
        SeatView(open_slots=3),
        _player(),
        base,
        _config(min_bid=2),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 2


def test_full_seat_abstains() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=0),
        _player(),
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.5,
    )
    assert bid == 0


def test_resolve_second_price_plus_min_bid() -> None:
    winner, price = resolve_bids({0: 40, 1: 25, 2: 10}, min_bid=1, rng=np.random.default_rng(0))
    assert winner == 0
    assert price == 26  # second-highest (25) + min_bid (1)


def test_resolve_caps_at_winner_max() -> None:
    winner, price = resolve_bids({0: 5, 1: 4}, min_bid=3, rng=np.random.default_rng(0))
    assert winner == 0
    assert price == min(5, 4 + 3)  # == 5, never above the winner's own ceiling


def test_resolve_lone_bidder_pays_min_bid() -> None:
    assert resolve_bids({2: 80}, min_bid=1, rng=np.random.default_rng(0)) == (2, 1)


def test_resolve_ties_broken_at_random_not_by_seat_index() -> None:
    # Ties must NOT be systematically awarded to the lowest seat index. A lowest-index rule dumped
    # every $1 tie on seat 0 (the seat-1 artifact: ~4 junk min-bid players/draft, ~0.10 win%). Over
    # many rng draws both tied seats win, and a top tie clears at the tied bid.
    winners = {
        resolve_bids({3: 20, 1: 20}, min_bid=1, rng=np.random.default_rng(s))[0]
        for s in range(50)
    }
    assert winners == {1, 3}
    _, price = resolve_bids({3: 20, 1: 20}, min_bid=1, rng=np.random.default_rng(0))
    assert price == 20


def test_bot_abstains_when_position_not_eligible() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3, eligible_positions=frozenset({Position.WR})),  # RB not eligible
        _player(),  # position "RB"
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 0


def test_bot_bids_when_position_eligible() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3, eligible_positions=frozenset({Position.RB})),
        _player(),  # position "RB"
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 40  # baseline, unchanged when eligible


# ---------------------------------------------------------------------------
# Task 2: bot archetype tests
# ---------------------------------------------------------------------------


def _tiered_baseline() -> pd.DataFrame:
    # 10 in-pool players, values descending: ranks 0=stud(>=0.10*10=1 -> rank 0 only),
    # ranks 5..9 = scrub (>= (1-0.50)*10 = 5), ranks 1..4 = mid.
    ids = [f"00-000000{i}" for i in range(10)]
    dollars = [60, 50, 40, 30, 25, 20, 15, 10, 5, 2]
    return pd.DataFrame(
        {"in_pool": [True] * 10, "auction_dollars": dollars, "bot_dollars": dollars},
        index=pd.Index(ids, name="gsis_id"),
    )


def _p(gid: str, pos: str = "RB") -> pd.Series:
    return pd.Series({"gsis_id": gid, "position": pos, "season_mean_fpts": 150.0})


def test_aggressive_matches_legacy_bot_max_bid() -> None:
    bl, cfg = _tiered_baseline(), _config()
    agg = AggressiveBot().max_bid(
        SeatView(open_slots=3, budget=100),
        _p("00-0000000"),
        bl,
        cfg,
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    legacy = bot_max_bid(
        SeatView(open_slots=3),
        _p("00-0000000"),
        bl,
        cfg,
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert agg == legacy == 60


def test_patient_underbids_a_stud() -> None:
    bid = PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=100),
        _p("00-0000000"),
        _tiered_baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 30  # value 60 (stud) * understud 0.5 == 30, below market


def test_patient_pays_premium_for_midtier_with_reserve() -> None:
    bid = PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=100),
        _p("00-0000002"),
        _tiered_baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == round(40 * 1.35)  # value 40 (mid) * (1+0.35) == 54, above market


def test_patient_midtier_without_reserve_bids_min() -> None:
    bid = PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=3),
        _p("00-0000002"),
        _tiered_baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == _config().min_bid  # budget (3) not > min_bid*open_slots (3) -> no reserve


def test_patient_scrub_and_ineligible() -> None:
    cfg = _config()
    assert (
        PatientValueBot().max_bid(
            SeatView(open_slots=3, budget=100),
            _p("00-0000008"),
            _tiered_baseline(),
            cfg,
            np.random.default_rng(0),
            price_jitter=0.0,
        )
        == cfg.min_bid
    )  # value 5 -> scrub -> min_bid
    assert (
        PatientValueBot().max_bid(
            SeatView(open_slots=3, budget=100, eligible_positions=frozenset({Position.WR})),
            _p("00-0000000", "RB"),
            _tiered_baseline(),
            cfg,
            np.random.default_rng(0),
            price_jitter=0.0,
        )
        == 0
    )  # RB not eligible -> abstain


def test_balanced_caps_at_pace_ceiling() -> None:
    bid = BalancedBot(pace=2.0).max_bid(
        SeatView(open_slots=3, budget=20),
        _p("00-0000000"),
        _tiered_baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 13  # min(value 60, 2*(20/3)=13.33) -> 13, paced (won't blow the bank)


def test_assign_bot_archetypes_round_robins() -> None:
    mix: list[BotArchetype] = [AggressiveBot(), PatientValueBot(), BalancedBot()]
    out = assign_bot_archetypes(5, mix)
    assert [type(a).__name__ for a in out] == [
        "AggressiveBot",
        "PatientValueBot",
        "BalancedBot",
        "AggressiveBot",
        "PatientValueBot",
    ]


def test_resolve_unbid_awards_nominator_when_eligible() -> None:
    # Nobody bid; the nominator can roster the nominee's position -> nominator takes it at min_bid.
    seat_eligible = {0: frozenset({Position.RB}), 1: frozenset({Position.RB, Position.WR})}
    assert resolve_unbid(Position.RB, 1, [0, 1], seat_eligible, 1) == (1, 1)


def test_resolve_unbid_falls_to_lowest_index_eligible_seat() -> None:
    # Nominator (seat 1) cannot roster the WR nominee; the lowest-index open seat that can wins it.
    seat_eligible = {
        0: frozenset({Position.RB}),
        1: frozenset({Position.RB}),
        2: frozenset({Position.WR}),
        3: frozenset({Position.WR}),
    }
    assert resolve_unbid(Position.WR, 1, [0, 1, 2, 3], seat_eligible, 1) == (2, 1)


def test_resolve_unbid_asserts_when_no_seat_eligible() -> None:
    # The room-union rule guarantees an eligible seat on the non-forced path; a pathological call
    # where none can roster the nominee asserts rather than silently mis-awarding.
    seat_eligible = {0: frozenset({Position.RB}), 1: frozenset({Position.RB})}
    with pytest.raises(AssertionError):
        resolve_unbid(Position.WR, 1, [0, 1], seat_eligible, 1)
