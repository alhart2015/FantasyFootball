"""Tests for the LiveDraftSession controller and its helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.assistant.live import LiveDraftSession, build_session_strategy
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _league() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _id_map() -> pd.DataFrame:
    ids = [f"00-000{i:04d}" for i in range(1, 40)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(
                ["RB", "WR", "QB", "TE"] * 9 + ["RB", "WR", "QB"], dtype=_PYARROW_STR
            ),
            "full_name": pd.array([f"P{i}" for i in range(1, 40)], dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * 39, dtype=_PYARROW_STR),
        }
    )


def _pool() -> pd.DataFrame:
    ids = [f"00-000{i:04d}" for i in range(1, 40)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(
                ["RB", "WR", "QB", "TE"] * 9 + ["RB", "WR", "QB"], dtype=_PYARROW_STR
            ),
            "season_mean_fpts": [300.0 - i for i in range(39)],
            "vorp": [150.0 - i for i in range(39)],
            "replacement_fpts": [100.0] * 39,
            "consensus_adp": pd.array([float(i + 1) for i in range(39)], dtype=pd.Float64Dtype()),
        }
    )


class _FakeStrategy:
    """A DraftStrategy that returns the eligible pool sorted by vorp (no MC)."""

    def recommend(self, state: DraftState, pool: pd.DataFrame, config) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)].copy()
        avail = avail.sort_values("vorp", ascending=False).reset_index(drop=True)
        avail["p_available_next"] = pd.array([pd.NA] * len(avail), dtype=pd.Float64Dtype())
        avail["fills_starting_slot"] = True
        avail["score"] = avail["vorp"]
        avail["rank"] = pd.array(range(1, len(avail) + 1), dtype=pd.Int64Dtype())
        return avail


def _session(picks: list[str] | None = None, mode: str = "copilot") -> LiveDraftSession:
    return LiveDraftSession(
        league=_league(),
        my_slot=7,
        id_map=_id_map(),
        pool=_pool(),
        strategy=_FakeStrategy(),
        strategy_name="fake",
        mode=mode,  # type: ignore[arg-type]
        adp_jitter=0.0,  # pure ADP order → deterministic bot picks in tests
        base_seed=0,
        picks=list(picks or []),
    )


def test_build_session_strategy_analytic_types() -> None:
    league = _league()
    assert isinstance(
        build_session_strategy(
            "raw_vorp", league=league, sigma=None, availability=None, n_sims=1, base_seed=0
        ),
        RawVorpStrategy,
    )
    assert isinstance(
        build_session_strategy(
            "now_or_never", league=league, sigma=None, availability=None, n_sims=1, base_seed=0
        ),
        NowOrNeverStrategy,
    )


def test_build_session_strategy_mc_requires_availability() -> None:
    league = _league()
    with pytest.raises(ValueError, match="availability"):
        build_session_strategy(
            "season_value", league=league, sigma=None, availability=None, n_sims=1, base_seed=0
        )


def test_build_session_strategy_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        build_session_strategy(
            "nope", league=_league(), sigma=None, availability=None, n_sims=1, base_seed=0
        )


def test_current_pick_and_my_pick_progression() -> None:
    s = _session()
    assert s.current_pick == 1
    assert not s.is_my_pick  # slot 1 on the clock, I'm slot 7
    s.picks = [f"00-000{i:04d}" for i in range(1, 7)]  # 6 picks made → pick 7 is mine
    assert s.current_pick == 7
    assert s.is_my_pick
    assert s.round_and_slot() == (1, 7)


def test_next_pick_number_snakes() -> None:
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, 7)])  # standing at pick 7 (mine)
    # 12 teams, slot 7: next pick after #7 is #18 (snake).
    assert s.next_pick_number == 18


def test_is_complete_when_roster_full() -> None:
    s = _session()
    total = s.league.n_teams * s.league.roster_size
    s.picks = [f"00-000{i:04d}" for i in range(1, total + 1)]
    assert s.is_complete
