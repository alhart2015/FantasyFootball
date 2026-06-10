"""Tests for the draft strategies."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    _finalize,
)
from projections.draft.assistant.survival import LogisticSurvival
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
