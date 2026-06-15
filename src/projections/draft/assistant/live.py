"""LiveDraftSession — the live draft board's controller (testable; Streamlit-free).

Holds the mutable draft truth (ordered picks + league + data) and delegates every
decision to existing engine functions. scripts/draft_board.py is a thin view over it.
"""

from __future__ import annotations

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.league_config import LeagueConfig

# Strategy names the board's dropdown offers (season_value_var is in STRATEGY_KEYS
# but excluded — its A/B showed no draft benefit; see the spec §2 / memory).
BOARD_STRATEGIES: tuple[str, ...] = (
    "now_or_never",
    "raw_vorp",
    "season_value",
    "season_value_timing",
)


def build_session_strategy(
    name: str,
    *,
    league: LeagueConfig,
    sigma: float | None,
    availability: PlayerAvailability | None,
    n_sims: int,
    base_seed: int,
) -> DraftStrategy:
    """Map a strategy name (+ live params) to a DraftStrategy.

    Shared by the sidebar dropdown and the resume path. MC strategies
    (`season_value*`) require a non-null `availability` and fail loud otherwise.
    """
    if name == "raw_vorp":
        return RawVorpStrategy()
    if name == "now_or_never":
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return NowOrNeverStrategy(LogisticSurvival(sigma=spread))
    if name in ("season_value", "season_value_var", "season_value_timing"):
        if availability is None:
            raise ValueError(f"strategy {name!r} requires availability data (None given)")
        if name == "season_value":
            return SeasonValueStrategy(availability, n_sims=n_sims, base_seed=base_seed)
        if name == "season_value_var":
            return SeasonValueStrategy(
                availability, n_sims=n_sims, base_seed=base_seed, risk_aware=True
            )
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return SeasonValueTimingStrategy(
            availability,
            n_sims=n_sims,
            base_seed=base_seed,
            survival=LogisticSurvival(sigma=spread),
        )
    raise ValueError(f"unknown strategy {name!r}")
