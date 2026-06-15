"""Tests for the LiveDraftSession controller and its helpers."""

from __future__ import annotations

import pytest

from projections.draft.assistant.live import build_session_strategy
from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _league() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.WR: 2,
            RosterSlot.FLEX: 1, RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def test_build_session_strategy_analytic_types() -> None:
    league = _league()
    assert isinstance(
        build_session_strategy("raw_vorp", league=league, sigma=None,
                               availability=None, n_sims=1, base_seed=0),
        RawVorpStrategy,
    )
    assert isinstance(
        build_session_strategy("now_or_never", league=league, sigma=None,
                               availability=None, n_sims=1, base_seed=0),
        NowOrNeverStrategy,
    )


def test_build_session_strategy_mc_requires_availability() -> None:
    league = _league()
    with pytest.raises(ValueError, match="availability"):
        build_session_strategy("season_value", league=league, sigma=None,
                               availability=None, n_sims=1, base_seed=0)


def test_build_session_strategy_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        build_session_strategy("nope", league=_league(), sigma=None,
                               availability=None, n_sims=1, base_seed=0)
