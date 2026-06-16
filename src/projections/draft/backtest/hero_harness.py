"""Hero-vs-bots strategy evaluation.

Runs each strategy as the SOLE hero (one seat) vs a noisy-ADP bot field, scored on the
real-outcome H2H season, swept across all seats with common random numbers across
strategies. The deployment-realistic counterpart to the mixed-field harness (harness.py).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.strategy import _DEFAULT_FLOOR, _DEFAULT_FLOOR_WEIGHT
from projections.draft.backtest.draft_field import hero_seat_layout
from projections.draft.backtest.harness import _build_strategy
from projections.draft.backtest.league import Calendar, LeagueResult, simulate_league
from projections.draft.league_config import LeagueConfig

_MC_KEYS = frozenset({"season_value", "season_value_var", "season_value_timing"})


def simulate_hero_cell(
    *,
    strategy_key: str,
    hero_seat: int,
    seed: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability | None,
    proj_lookup: Mapping[tuple[str, int], float],
    actual_lookup: Mapping[tuple[str, int], float],
    calendar: Calendar,
    jitter: float = 8.0,
    strategy_n_sims: int = 50,
    base_seed: int = 0,
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
) -> tuple[LeagueResult, LeagueResult]:
    """Simulate one (strategy, seat, seed) cell; return the hero seat's (actual, projected).

    The league seed is ``base_seed + seed`` -- independent of strategy and seat, so every
    strategy at a given (seat, seed) faces the identical schedule + bot draws (CRN).
    """
    if strategy_key in _MC_KEYS and availability is None:
        raise ValueError(f"strategy {strategy_key!r} requires availability data (None given)")
    layout = hero_seat_layout(hero_seat=hero_seat, hero_label=strategy_key, n_teams=config.n_teams)
    hero = _build_strategy(
        strategy_key,
        availability=availability,  # type: ignore[arg-type]
        n_teams=config.n_teams,
        strategy_n_sims=strategy_n_sims,
        base_seed=base_seed,
        floor=floor,
        floor_weight=floor_weight,
    )
    seat_strategies = {s: (hero if label != "bot" else None) for s, label in layout.items()}
    outcome = simulate_league(
        base_seed + seed,
        seat_strategies=seat_strategies,
        strategy_labels=layout,
        pool=pool,
        config=config,
        proj_lookup=proj_lookup,
        actual_lookup=actual_lookup,
        calendar=calendar,
        jitter=jitter,
    )
    (a,) = [r for r in outcome.actual if r.seat == hero_seat]
    (p,) = [r for r in outcome.projected if r.seat == hero_seat]
    return a, p
