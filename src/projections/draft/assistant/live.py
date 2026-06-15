"""LiveDraftSession — the live draft board's controller (testable; Streamlit-free).

Holds the mutable draft truth (ordered picks + league + data) and delegates every
decision to existing engine functions. scripts/draft_board.py is a thin view over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.pick_timing import my_next_pick, slot_for
from projections.draft.assistant.state import DraftState, build_draft_state
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, validate_gsis_id

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


@dataclass
class LiveDraftSession:
    """Mutable, Streamlit-free controller for one live/mock snake draft."""

    league: LeagueConfig
    my_slot: int
    id_map: pd.DataFrame
    pool: pd.DataFrame
    strategy: DraftStrategy
    strategy_name: str
    mode: Literal["copilot", "mock"] = "copilot"
    adp_jitter: float = 8.0
    base_seed: int = 0
    n_sims: int = 300
    sigma: float | None = None
    season: int = 2026
    picks: list[GsisId] = field(default_factory=list)
    # Persistence-only paths (defaults keep core tests path-free).
    league_config_path: Path = field(default=Path("."))
    vorp_path: Path = field(default=Path("."))
    id_map_path: Path = field(default=Path("."))
    data_root: Path = field(default=Path("data"))

    def state(self) -> DraftState:
        """Rebuild the immutable engine snapshot from current picks (cheap; O(picks))."""
        return build_draft_state(
            self.picks, my_slot=self.my_slot, league=self.league, id_map=self.id_map
        )

    @property
    def current_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.league.n_teams * self.league.roster_size

    @property
    def on_clock_slot(self) -> int:
        return slot_for(self.current_pick, self.league.n_teams)

    @property
    def is_my_pick(self) -> bool:
        return not self.is_complete and self.on_clock_slot == self.my_slot

    @property
    def next_pick_number(self) -> int | None:
        return my_next_pick(
            self.current_pick, self.my_slot, self.league.n_teams, self.league.roster_size
        )

    def round_and_slot(self) -> tuple[int, int]:
        rnd = (self.current_pick - 1) // self.league.n_teams + 1
        return rnd, self.on_clock_slot

    def available_pool(self) -> pd.DataFrame:
        drafted = self.state().drafted_ids
        return self.pool[~self.pool["gsis_id"].isin(drafted)].reset_index(drop=True)

    def record_pick(self, gsis_id: str) -> None:
        gid = validate_gsis_id(str(gsis_id))
        if gid in self.state().drafted_ids:
            raise ValueError(f"{gid} already drafted")
        if gid not in set(self.id_map["gsis_id"]):
            raise ValueError(f"{gid} absent from id_map (cannot resolve position)")
        self.picks.append(gid)

    def undo(self) -> GsisId | None:
        return self.picks.pop() if self.picks else None

    def recommendation(self) -> pd.DataFrame:
        return self.strategy.recommend(self.state(), self.pool, self.league)

    def suggested_pick(self) -> GsisId | None:
        avail = self.available_pool()
        if avail.empty:
            return None
        # Deterministic per board state → stable across Streamlit reruns, reproducible
        # in mock mode. (Re-deriving the seed each call is intentional; no stored RNG.)
        rng = np.random.default_rng([self.base_seed, self.current_pick])
        return bot_pick(avail, rng, adp_jitter=self.adp_jitter)
