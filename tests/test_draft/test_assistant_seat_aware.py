"""Tests for SeatAwareStrategy dispatch (timing at wing/mid, sv_var at the turn)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import SeatAwareStrategy
from projections.draft.league_config import LeagueConfig


@dataclass(frozen=True)
class _Stub:
    """Records nothing; returns a one-row frame tagged with its label so the
    dispatcher's choice is observable in the output."""

    label: str

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        return pd.DataFrame({"who": [self.label]})


def _state(my_slot: int, n_teams: int = 16) -> DraftState:
    return DraftState(my_slot=my_slot, n_teams=n_teams, rounds=13, picks=(), my_roster=())


_POOL = pd.DataFrame({"gsis_id": ["00-0000001"]})


def _strat(turn_band: int = 2) -> SeatAwareStrategy:
    return SeatAwareStrategy(timing=_Stub("timing"), turn=_Stub("turn"), turn_band=turn_band)


@pytest.mark.parametrize("slot", [1, 8, 14])
def test_non_turn_seats_use_timing(slot: int) -> None:
    out = _strat().recommend(_state(slot), _POOL, config=None)  # type: ignore[arg-type]
    assert out["who"].iloc[0] == "timing"


@pytest.mark.parametrize("slot", [15, 16])
def test_turn_seats_use_turn(slot: int) -> None:
    out = _strat().recommend(_state(slot), _POOL, config=None)  # type: ignore[arg-type]
    assert out["who"].iloc[0] == "turn"


def test_turn_band_one_only_last_seat() -> None:
    strat = _strat(turn_band=1)
    assert strat.recommend(_state(16), _POOL, config=None).iloc[0]["who"] == "turn"  # type: ignore[arg-type]
    assert strat.recommend(_state(15), _POOL, config=None).iloc[0]["who"] == "timing"  # type: ignore[arg-type]


def test_invalid_turn_band_raises() -> None:
    with pytest.raises(ValueError, match="turn_band"):
        SeatAwareStrategy(timing=_Stub("t"), turn=_Stub("u"), turn_band=0)
