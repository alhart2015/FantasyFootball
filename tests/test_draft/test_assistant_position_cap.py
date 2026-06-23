"""Tests for PositionCapStrategy (hard per-position roster caps) + season_value_qb_cap wiring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import (
    MC_STRATEGY_KEYS,
    STRATEGY_KEYS,
    PositionCapStrategy,
    RawVorpStrategy,
    build_position_targeted,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, RecommendationSchema


def _rec(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a RecommendationSchema-valid frame from (gsis_id, position, score) rows."""
    df = pd.DataFrame(
        {
            "gsis_id": pd.array([g for g, _, _ in rows], dtype=_PYARROW_STR),
            "position": pd.array([p for _, p, _ in rows], dtype=_PYARROW_STR),
            "vorp": [s for _, _, s in rows],
            "consensus_adp": pd.array([1.0] * len(rows), dtype=pd.Float64Dtype()),
            "p_available_next": pd.array([pd.NA] * len(rows), dtype=pd.Float64Dtype()),
            "fills_starting_slot": [True] * len(rows),
            "score": [s for _, _, s in rows],
            "rank": pd.array(range(1, len(rows) + 1), dtype=pd.Int64Dtype()),
        }
    )
    return RecommendationSchema.validate(df)


@dataclass(frozen=True)
class _Stub:
    frame: pd.DataFrame

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        return self.frame


def _state(my_roster: tuple[Position, ...]) -> DraftState:
    return DraftState(my_slot=1, n_teams=16, rounds=13, picks=(), my_roster=my_roster)


_REC = _rec([("00-0000001", "QB", 10.0), ("00-0000002", "RB", 9.0), ("00-0000003", "QB", 8.0)])


def test_passthrough_below_cap() -> None:
    strat = PositionCapStrategy(_Stub(_REC), {Position.QB: 2})
    out = strat.recommend(_state((Position.QB,)), pool=None, config=None)  # type: ignore[arg-type]
    assert out.equals(_REC)  # 1 QB rostered, cap=2 not reached -> untouched


def test_drops_position_at_cap_and_redensifies_rank() -> None:
    strat = PositionCapStrategy(_Stub(_REC), {Position.QB: 2})
    out = strat.recommend(_state((Position.QB, Position.QB)), pool=None, config=None)  # type: ignore[arg-type]
    assert "QB" not in set(out["position"])  # both QBs dropped
    assert out.iloc[0]["gsis_id"] == "00-0000002"  # top remaining pick is the RB
    assert list(out["rank"]) == [1]  # rank re-densified


def test_never_strands_the_draft() -> None:
    # At the cap with ONLY capped-position candidates left -> return inner rec, don't empty it.
    only_qb = _rec([("00-0000001", "QB", 10.0), ("00-0000002", "QB", 9.0)])
    strat = PositionCapStrategy(_Stub(only_qb), {Position.QB: 2})
    out = strat.recommend(_state((Position.QB, Position.QB)), pool=None, config=None)  # type: ignore[arg-type]
    assert out.equals(only_qb)


def test_uncapped_position_unaffected() -> None:
    strat = PositionCapStrategy(_Stub(_REC), {Position.QB: 2})
    # 5 RBs rostered, no RB cap set -> nothing dropped.
    out = strat.recommend(_state((Position.RB,) * 5), pool=None, config=None)  # type: ignore[arg-type]
    assert out.equals(_REC)


def test_build_position_targeted_caps() -> None:
    capped = build_position_targeted(RawVorpStrategy())
    assert capped.caps == {Position.QB: 2, Position.TE: 2, Position.RB: 6, Position.WR: 5}


def test_targeted_keys_registered_and_mc_gating() -> None:
    # Both targeted strategies are public keys; only the season_value one needs availability.
    assert "now_or_never_targeted" in STRATEGY_KEYS
    assert "season_value_targeted" in STRATEGY_KEYS
    assert "season_value_targeted" in MC_STRATEGY_KEYS
    assert "now_or_never_targeted" not in MC_STRATEGY_KEYS  # wraps the analytic nn_floored
