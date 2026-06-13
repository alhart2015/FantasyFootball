"""Tests for the draft strategies."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.pick_timing import my_next_pick
from projections.draft.assistant.season_value import marginal_season_values
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
    _finalize,
)
from projections.draft.assistant.survival import LogisticSurvival, expected_best_by_position
from projections.draft.league_config import LeagueConfig
from projections.schemas import (
    _PYARROW_STR,
    GsisId,
    Position,
    RecommendationSchema,
    RosterSlot,
    Ruleset,
)


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _pool() -> pd.DataFrame:
    # rb1 scarce (low survival), wr1 highest VORP but safe.
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000010", "00-0000011", "00-0000020", "00-0000021"],
                dtype=_PYARROW_STR,
            ),
            "position": pd.array(["RB", "RB", "WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 240.0, 252.0, 230.0],
            "vorp": [50.0, 40.0, 52.0, 30.0],
            "replacement_fpts": [200.0, 200.0, 200.0, 200.0],
            "consensus_adp": pd.array([5.0, 6.0, 7.0, 8.0], dtype=pd.Float64Dtype()),
        }
    )


def _state(
    current_pick: int = 7,
    rounds: int = 9,
    my_roster: tuple[Position, ...] = (),
) -> DraftState:
    """Build a state standing at `current_pick` (materialize filler picks so the
    derived current_pick is correct). Fillers use a 9-prefix so they never
    collide with the pool's 00-0000xxx ids. Empty roster → RB and WR both
    eligible & start-needed unless `my_roster` overrides.
    """
    fillers = tuple(GsisId(f"00-9{i:06d}") for i in range(current_pick - 1))
    return DraftState(my_slot=7, n_teams=12, rounds=rounds, picks=fillers, my_roster=my_roster)


class _FakeSurvival:
    """Deterministic survival lookup keyed by adp, for hand-computed expectations."""

    _P: ClassVar[dict[float, float]] = {5.0: 0.1, 6.0: 0.9, 7.0: 0.95, 8.0: 0.9}

    def p_available(self, adp: float, at_pick: int) -> float:
        # at_pick ignored: the fake table is keyed by ADP only, so expectations
        # remain deterministic and hand-computable.
        return self._P[adp]


def test_both_satisfy_protocol() -> None:
    assert isinstance(RawVorpStrategy(), DraftStrategy)
    assert isinstance(NowOrNeverStrategy(_FakeSurvival()), DraftStrategy)


def test_raw_vorp_orders_by_vorp_and_nulls_p_available() -> None:
    rec = RawVorpStrategy().recommend(_state(), _pool(), _config())
    RecommendationSchema.validate(rec)
    assert list(rec["gsis_id"]) == [
        "00-0000020",  # wr1 52
        "00-0000010",  # rb1 50
        "00-0000011",  # rb2 40
        "00-0000021",  # wr2 30
    ]
    assert rec["p_available_next"].isna().all()


def test_now_or_never_reorders_cross_position() -> None:
    rec = NowOrNeverStrategy(_FakeSurvival()).recommend(_state(), _pool(), _config())
    RecommendationSchema.validate(rec)
    # E[best RB survivor] = 50*.1 + 40*.9*.9 = 37.4 → rb1 score 12.6, rb2 2.6
    # E[best WR survivor] = 52*.95 + 30*.9*.05 = 50.75 → wr1 score 1.25, wr2 -20.75
    assert list(rec["gsis_id"]) == [
        "00-0000010",  # rb1 12.6  (jumps wr1 — the reorder)
        "00-0000011",  # rb2 2.6
        "00-0000020",  # wr1 1.25
        "00-0000021",  # wr2 -20.75
    ]
    assert rec.loc[rec["gsis_id"] == "00-0000010", "score"].iloc[0] == 12.6


def test_within_position_order_is_vorp() -> None:
    rec = NowOrNeverStrategy(_FakeSurvival()).recommend(_state(), _pool(), _config())
    rb = rec[rec["position"] == "RB"]
    assert list(rb["gsis_id"]) == ["00-0000010", "00-0000011"]  # vorp desc


def test_last_pick_fallback_equals_raw_vorp() -> None:
    # rounds=1 → my only pick is pick 7, no next pick.
    last = _state(current_pick=7, rounds=1)
    non = NowOrNeverStrategy(_FakeSurvival()).recommend(last, _pool(), _config())
    raw = RawVorpStrategy().recommend(last, _pool(), _config())
    assert list(non["gsis_id"]) == list(raw["gsis_id"])
    assert non["p_available_next"].isna().all()


def test_roster_eligible_filter_drops_filled_position() -> None:
    # Fill both RB slots + FLEX with RBs → RB only benchable, WR still starts.
    state = DraftState(
        my_slot=7,
        n_teams=12,
        rounds=9,
        picks=(),
        my_roster=(Position.RB, Position.RB, Position.RB),
    )
    rec = RawVorpStrategy().recommend(state, _pool(), _config())
    # RB still rosterable (bench), but WR fills a starting slot → WR tier first.
    assert bool(rec.iloc[0]["fills_starting_slot"]) is True
    assert rec.iloc[0]["position"] == "WR"


def test_equal_score_tie_break_is_gsis_id() -> None:
    # Two WRs, identical vorp → identical raw-vorp score; rank by gsis_id asc.
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000031", "00-0000030"], dtype=_PYARROW_STR),
            "position": pd.array(["WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [240.0, 240.0],
            "vorp": [40.0, 40.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([10.0, 11.0], dtype=pd.Float64Dtype()),
        }
    )
    rec = RawVorpStrategy().recommend(_state(), pool, _config())
    assert list(rec["gsis_id"]) == ["00-0000030", "00-0000031"]  # gsis asc
    assert list(rec["rank"]) == [1, 2]


def test_now_or_never_null_adp_p_available_is_null() -> None:
    # A null-ADP player still ranks, but its displayed p_available_next is null
    # (spec §3.5 output contract). Uses the real survival model (handles NaN).
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000040", "00-0000041"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [240.0, 230.0],
            "vorp": [40.0, 30.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([5.0, pd.NA], dtype=pd.Float64Dtype()),
        }
    )
    rec = NowOrNeverStrategy(LogisticSurvival(sigma=8.0)).recommend(_state(), pool, _config())
    RecommendationSchema.validate(rec)
    by_id = rec.set_index("gsis_id")["p_available_next"]
    assert pd.isna(by_id["00-0000041"])  # null ADP → null p_available_next
    assert pd.notna(by_id["00-0000040"])  # has ADP → populated


def test_drafted_player_excluded_from_recommendations() -> None:
    # A player already in state.picks must not reappear in the recommendation.
    pool = _pool()
    state = DraftState(
        my_slot=7,
        n_teams=12,
        rounds=9,
        picks=(GsisId("00-0000010"),),  # rb1 already drafted (by someone)
        my_roster=(),
    )
    rec = RawVorpStrategy().recommend(state, pool, _config())
    assert "00-0000010" not in set(rec["gsis_id"])
    assert "00-0000020" in set(rec["gsis_id"])  # undrafted players remain


def test_missing_consensus_adp_degrades_gracefully() -> None:
    # A pool WITHOUT a consensus_adp column must not raise — it should degrade to
    # all-null and still produce a valid recommendation.
    pool = _pool().drop(columns=["consensus_adp"])
    rec = NowOrNeverStrategy(LogisticSurvival(sigma=8.0)).recommend(_state(), pool, _config())
    RecommendationSchema.validate(rec)
    assert rec["consensus_adp"].isna().all()
    assert rec["p_available_next"].isna().all()  # no ADP → null display


def test_finalize_starting_tier_toggle_changes_order() -> None:
    # Discriminating fixture: the RB FILLS a starting slot but has the LOWER score;
    # the WR is a non-filler with the HIGHER score. The two branches must DISAGREE —
    # tier-on lifts the low-score filler to the top, tier-off ranks purely by score.
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000050", "00-0000051"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "vorp": [10.0, 20.0],
            "consensus_adp": pd.array([3.0, 4.0], dtype=pd.Float64Dtype()),
            "score": [1.0, 9.0],  # RB (the filler) is the LOWER score
        }
    )
    elig = {Position.RB: True, Position.WR: False}  # RB fills a slot, WR does not

    def order(*, tier: bool) -> list[str]:
        p_na: pd.Series[float] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
        out = _finalize(df, elig, p_na, starting_need_tier=tier)
        assert set(out.columns) >= {"fills_starting_slot"}  # emitted regardless of the tier
        return list(out["gsis_id"])

    # Tier on: the starting-slot filler bubbles up despite its lower score.
    assert order(tier=True) == ["00-0000050", "00-0000051"]
    # Tier off: pure score order — the higher-score non-filler wins.
    assert order(tier=False) == ["00-0000051", "00-0000050"]


def test_finalize_fails_loud_on_position_outside_eligibility() -> None:
    # Invariant: _eligible_subset filters to elig's keyset before _finalize, so a
    # position absent from elig must never reach here. If it does, fail loud
    # rather than coerce a NaN to True and silently mislabel fills_starting_slot.
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000099"], dtype=_PYARROW_STR),
            "position": pd.array(["QB"], dtype=_PYARROW_STR),
            "vorp": [10.0],
            "consensus_adp": pd.array([3.0], dtype=pd.Float64Dtype()),
            "score": [10.0],
        }
    )
    p_na: pd.Series[float] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
    with pytest.raises(KeyError, match="eligibility keyset"):
        _finalize(df, {Position.RB: True}, p_na)


# ---------------------------------------------------------------------------
# SeasonValueStrategy helpers
# ---------------------------------------------------------------------------


def _depth_pool() -> pd.DataFrame:
    # My roster will be {WR_a safe, WR_b safe, RB_a RISKY}. Candidates: WR_c (high VORP,
    # saturated WR room) vs RB_b (insurance for the risky RB). season_value must take
    # the insurance; now_or_never takes the higher VORP.
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000201", "00-0000202", "00-0000301", "00-0000203", "00-0000302"],
                dtype=_PYARROW_STR,
            ),
            "position": pd.array(["WR", "WR", "RB", "WR", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [200.0, 195.0, 200.0, 190.0, 185.0],
            "vorp": [60.0, 50.0, 58.0, 55.0, 45.0],
            "replacement_fpts": [140.0, 140.0, 140.0, 140.0, 140.0],
            "consensus_adp": pd.array([3.0, 4.0, 5.0, 20.0, 20.0], dtype=pd.Float64Dtype()),
        }
    )


def _depth_state() -> DraftState:
    # 4 teams, my_slot=1 → my picks at #1, #8, #9. Place WR_a, WR_b, RB_a there;
    # fillers (not in pool) elsewhere. current_pick = 10.
    picks = (
        GsisId("00-0000201"),  # #1 mine (WR_a)
        GsisId("00-9000002"),
        GsisId("00-9000003"),
        GsisId("00-9000004"),
        GsisId("00-9000005"),
        GsisId("00-9000006"),
        GsisId("00-9000007"),
        GsisId("00-0000202"),  # #8 mine (WR_b)
        GsisId("00-0000301"),  # #9 mine (RB_a)
    )
    return DraftState(
        my_slot=1,
        n_teams=4,
        rounds=5,  # == roster_size (RB+WR+FLEX+2*BENCH)
        picks=picks,
        my_roster=(Position.WR, Position.WR, Position.RB),
    )


def _depth_config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=4,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 2},
        ruleset=Ruleset.espn_ppr(),
    )


def _depth_avail() -> PlayerAvailability:
    return PlayerAvailability(
        p={
            "00-0000201": 0.97,  # WR_a (safe)
            "00-0000202": 0.97,  # WR_b (safe)
            "00-0000301": 0.40,  # RB_a (risky starter — depth at RB pays off)
            "00-0000203": 0.97,  # WR_c (candidate, redundant WR)
            "00-0000302": 0.95,  # RB_b (candidate, RB insurance)
        },
        bye={},
    )


def test_season_value_satisfies_protocol() -> None:
    strat = SeasonValueStrategy(_depth_avail(), n_sims=200, base_seed=0)
    assert isinstance(strat, DraftStrategy)


def test_season_value_rejects_degenerate_config() -> None:
    # n_sims < 1 → empty MC draw matrix → nan marginals → silent VORP fallback.
    # top_k < 1 → no candidate ever scored. Both must fail loud at construction.
    with pytest.raises(ValueError, match="n_sims"):
        SeasonValueStrategy(_depth_avail(), n_sims=0, base_seed=0)
    with pytest.raises(ValueError, match="top_k"):
        SeasonValueStrategy(_depth_avail(), n_sims=10, base_seed=0, top_k=0)


def test_season_value_takes_insurance_where_now_or_never_takes_vorp() -> None:
    state, pool, config = _depth_state(), _depth_pool(), _depth_config()
    season = SeasonValueStrategy(_depth_avail(), n_sims=4000, base_seed=0).recommend(
        state, pool, config
    )
    RecommendationSchema.validate(season)
    assert season.iloc[0]["gsis_id"] == "00-0000302"  # RB_b, the RB insurance pick

    non = NowOrNeverStrategy(LogisticSurvival(sigma=8.0)).recommend(state, pool, config)
    assert non.iloc[0]["gsis_id"] == "00-0000203"  # WR_c, the higher-VORP redundant WR


def test_season_value_is_deterministic() -> None:
    state, pool, config = _depth_state(), _depth_pool(), _depth_config()
    a = SeasonValueStrategy(_depth_avail(), n_sims=300, base_seed=7).recommend(state, pool, config)
    b = SeasonValueStrategy(_depth_avail(), n_sims=300, base_seed=7).recommend(state, pool, config)
    assert list(a["gsis_id"]) == list(b["gsis_id"])
    assert list(a["score"]) == list(b["score"])


def test_season_value_pruning_keeps_argmax_but_prunes_tail() -> None:
    # Three WR candidates at one position so top_k ACTUALLY prunes. top_k=1 evaluates
    # only the top-VORP WR (the other two fall to the 0.0 pruned tail); top_k=3 evaluates
    # all. The #1 pick (argmax) is identical — within a position the best add is the
    # highest-VORP one (monotonic in points) — while the tail differs, proving pruning
    # is genuinely active (not a degenerate same==same).
    ids = ["00-0000401", "00-0000402", "00-0000403"]
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(["WR", "WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [210.0, 195.0, 180.0],
            "vorp": [70.0, 55.0, 40.0],
            "replacement_fpts": [140.0, 140.0, 140.0],
            "consensus_adp": pd.array([3.0, 4.0, 5.0], dtype=pd.Float64Dtype()),
        }
    )
    avail = PlayerAvailability(p=dict.fromkeys(ids, 0.95), bye={})
    config = LeagueConfig(
        name="t",
        n_teams=4,
        roster_slots={RosterSlot.WR: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 3},
        ruleset=Ruleset.espn_ppr(),
    )
    state = DraftState(my_slot=1, n_teams=4, rounds=5, picks=(), my_roster=())

    k1 = SeasonValueStrategy(avail, n_sims=300, base_seed=2, top_k=1).recommend(state, pool, config)
    k3 = SeasonValueStrategy(avail, n_sims=300, base_seed=2, top_k=3).recommend(state, pool, config)

    # Argmax invariant: the highest-VORP WR wins under both top_k.
    assert k1.iloc[0]["gsis_id"] == k3.iloc[0]["gsis_id"] == "00-0000401"
    # Pruning is active: the lowest WR is pruned (score 0.0) at top_k=1 but evaluated
    # to a real positive marginal at top_k=3.
    k1_score = dict(zip(k1["gsis_id"], k1["score"], strict=True))
    k3_score = dict(zip(k3["gsis_id"], k3["score"], strict=True))
    assert k1_score["00-0000403"] == 0.0
    assert k3_score["00-0000403"] > 0.0


def test_season_value_warns_on_roster_player_missing_from_pool() -> None:
    # A rostered id absent from the pool is dropped from the valued base, with a warning.
    pool = _depth_pool()
    pool = pool[pool["gsis_id"] != "00-0000301"].copy()  # drop rostered RB_a from the pool
    state, config = _depth_state(), _depth_config()
    with pytest.warns(UserWarning, match="absent from the VORP pool"):
        rec = SeasonValueStrategy(_depth_avail(), n_sims=200, base_seed=0).recommend(
            state, pool, config
        )
    RecommendationSchema.validate(rec)


def test_season_value_empty_eligible_returns_valid_empty_frame() -> None:
    # Every pool player is already drafted → no eligible candidates. recommend must
    # return a valid, empty RecommendationSchema frame, not raise (spec §4 edge).
    pool = _depth_pool()
    state = DraftState(
        my_slot=1,
        n_teams=4,
        rounds=5,
        picks=tuple(GsisId(g) for g in pool["gsis_id"].astype(str)),  # all drafted
        my_roster=(),
    )
    rec = SeasonValueStrategy(_depth_avail(), n_sims=50, base_seed=0).recommend(
        state, pool, _depth_config()
    )
    RecommendationSchema.validate(rec)
    assert len(rec) == 0


def _flat_availability(pool: pd.DataFrame) -> PlayerAvailability:
    # Every player available every week, no byes -> deterministic, MC-stable marginals.
    # PlayerAvailability is the (p, bye) dataclass from availability.py.
    return PlayerAvailability(p={str(g): 0.95 for g in pool["gsis_id"]}, bye={})


def _timing(pool: pd.DataFrame, sigma: float = 8.0) -> SeasonValueTimingStrategy:
    return SeasonValueTimingStrategy(
        _flat_availability(pool), n_sims=20, base_seed=0, survival=LogisticSurvival(sigma=sigma)
    )


def test_timing_validates_construction() -> None:
    av = _flat_availability(_pool())
    with pytest.raises(ValueError):
        SeasonValueTimingStrategy(av, n_sims=0, base_seed=0, survival=LogisticSurvival(sigma=8.0))
    with pytest.raises(ValueError):
        SeasonValueTimingStrategy(
            av, n_sims=20, base_seed=0, survival=LogisticSurvival(sigma=8.0), top_k=0
        )


def test_timing_is_deterministic() -> None:
    pool, state, cfg = _pool(), _state(), _config()
    r1 = _timing(pool).recommend(state, pool, cfg)
    r2 = _timing(pool).recommend(state, pool, cfg)
    pd.testing.assert_frame_equal(r1, r2)


def test_timing_score_equals_marginal_minus_opp_cost() -> None:
    # score must be exactly marginal - E[best surviving marginal at pos], no fudge factor.
    pool, state, cfg = _pool(), _state(), _config()
    rec = _timing(pool).recommend(state, pool, cfg)

    df = pool[~pool["gsis_id"].isin(state.drafted_ids)].copy()
    pruned = (
        df.sort_values(["position", "vorp"], ascending=[True, False])
        .groupby("position", sort=False)
        .head(8)
    )
    rng = np.random.default_rng([0, state.current_pick])
    base = pool.loc[pool["gsis_id"].isin([str(g) for g in state.my_pick_ids])]
    marg = marginal_season_values(
        base[["gsis_id", "position", "season_mean_fpts"]],
        pruned[["gsis_id", "position", "season_mean_fpts"]],
        cfg.roster_slots,
        _flat_availability(pool),
        n_sims=20,
        rng=rng,
    )
    nxt = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
    assert nxt is not None
    surv = LogisticSurvival(sigma=8.0)
    pos = np.array([str(r.position) for r in rec.itertuples(index=False)])
    m = np.array([float(marg.get(str(r.gsis_id), 0.0)) for r in rec.itertuples(index=False)])
    p = np.array(
        [
            surv.p_available(
                float(r.consensus_adp) if pd.notna(r.consensus_adp) else float("nan"), nxt
            )
            for r in rec.itertuples(index=False)
        ]
    )
    gid = np.array([str(r.gsis_id) for r in rec.itertuples(index=False)])
    opp = expected_best_by_position(pos, m, p, gid)
    for i, r in enumerate(rec.itertuples(index=False)):
        expected = round(m[i] - opp[str(r.position)], 10)
        assert abs(float(r.score) - expected) < 1e-9


def test_timing_promotes_scarce_position_over_safer_higher_marginal() -> None:
    # rb1 scarce (adp 1 -> ~0 survival) -> opp_cost[RB] ~ 0 -> keeps ~full marginal.
    # wr1 highest marginal but safe (adp 200) -> opp_cost[WR] ~ its own marginal -> score ~ 0.
    # season_value ranks wr1 #1 (highest marginal); timing flips rb1 to the top.
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000010", "00-0000011", "00-0000020", "00-0000021"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["RB", "RB", "WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [220.0, 100.0, 350.0, 150.0],
            "vorp": [50.0, 20.0, 80.0, 30.0],
            "replacement_fpts": [170.0, 80.0, 270.0, 120.0],
            "consensus_adp": pd.array([1.0, 90.0, 400.0, 400.0], dtype=pd.Float64Dtype()),
        }
    )
    state, cfg = _state(), _config()
    sv = SeasonValueStrategy(_flat_availability(pool), n_sims=20, base_seed=0)
    sv_top = sv.recommend(state, pool, cfg).iloc[0]
    timing_top = _timing(pool).recommend(state, pool, cfg).iloc[0]
    assert sv_top["gsis_id"] == "00-0000020"  # season_value: highest marginal (wr1)
    assert timing_top["gsis_id"] == "00-0000010"  # timing: scarce rb1 promoted


def test_timing_last_pick_fallback_equals_season_value() -> None:
    # Seat 7's final pick in a 2-round/12-team draft (pick 18) -> no next pick ->
    # timing falls back to ranking by raw marginal, identical to season_value.
    pool, cfg = _pool(), _config()
    state = _state(current_pick=18, rounds=2)
    sv = SeasonValueStrategy(_flat_availability(pool), n_sims=20, base_seed=0)
    pd.testing.assert_frame_equal(
        _timing(pool).recommend(state, pool, cfg), sv.recommend(state, pool, cfg)
    )


def test_timing_prunes_to_top_k_and_zeros_the_tail() -> None:
    # 10 RBs, top_k=2: only the top-2 by VORP are MC-evaluated; the other 8 get
    # marginal 0 (cosmetic tail) -> identical score (0 - opp_cost[RB]) and never the pick.
    n = 10
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array([f"00-00000{i:02d}" for i in range(n)], dtype=_PYARROW_STR),
            "position": pd.array(["RB"] * n, dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0 - i * 5 for i in range(n)],
            "vorp": [50.0 - i * 5 for i in range(n)],
            "replacement_fpts": [200.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )
    state, cfg = _state(), _config()
    strat = SeasonValueTimingStrategy(
        _flat_availability(pool),
        n_sims=20,
        base_seed=0,
        survival=LogisticSurvival(sigma=8.0),
        top_k=2,
    )
    rec = strat.recommend(state, pool, cfg)
    evaluated = {"00-0000000", "00-0000001"}
    assert rec.iloc[0]["gsis_id"] in evaluated  # the pick is an evaluated candidate
    tail = rec[~rec["gsis_id"].isin(evaluated)]["score"]
    assert tail.nunique() == 1  # the 8 pruned-out share the cosmetic 0-marginal score


def test_timing_null_adp_treated_as_surviving() -> None:
    # WR alone at its position with null ADP -> p=1 (certain to survive). With
    # self-inclusion, opp_cost[WR] == its own marginal -> score == 0; display p is null.
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 250.0],
            "vorp": [50.0, 50.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([5.0, pd.NA], dtype=pd.Float64Dtype()),
        }
    )
    state, cfg = _state(), _config()
    rec = _timing(pool).recommend(state, pool, cfg)
    wr = rec[rec["gsis_id"] == "00-0000020"].iloc[0]
    assert pd.isna(wr["p_available_next"])  # null ADP -> null display
    assert abs(float(wr["score"])) < 1e-9  # p=1 surviving + self-inclusion -> opp == marginal


def test_timing_satisfies_protocol() -> None:
    assert isinstance(_timing(_pool()), DraftStrategy)
