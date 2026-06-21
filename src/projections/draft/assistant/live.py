"""LiveDraftSession — the live draft board's controller (testable; Streamlit-free).

Holds the mutable draft truth (ordered picks + league + data) and delegates every
decision to existing engine functions. scripts/draft_board.py is a thin view over it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import SeatProjection, project_draft
from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.pick_timing import my_next_pick, slot_for
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.state import DraftState, build_draft_state
from projections.draft.assistant.strategy import (
    _DEFAULT_FLOOR,
    _DEFAULT_FLOOR_WEIGHT,
    MC_STRATEGY_KEYS,
    DraftStrategy,
    NowOrNeverFlooredStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
    build_seat_aware,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import allocate_roster_slots
from projections.schemas import GsisId, Position, RosterSlot, validate_gsis_id

# MC strategies that require an availability load. Aliases the single source of truth in
# strategy.py (shared with hero_harness._MC_KEYS) so the gate can't drift across modules;
# re-exported here under the name build_session_strategy/the CLI/the board already import.
MC_STRATEGIES: frozenset[str] = MC_STRATEGY_KEYS

# Strategy names the board's dropdown offers (season_value_var is in STRATEGY_KEYS
# but excluded — its A/B showed no draft benefit; see the spec §2 / memory).
# seat_aware routes to season_value_timing (wing/mid) or season_value_var (turn) by the
# hero's slot — the post-availability-fix per-seat frontier winner (Test 14).
BOARD_STRATEGIES: tuple[str, ...] = (
    "now_or_never",
    "now_or_never_floored",
    "raw_vorp",
    "season_value",
    "season_value_timing",
    "seat_aware",
)


def build_session_strategy(
    name: str,
    *,
    league: LeagueConfig,
    sigma: float | None,
    availability: PlayerAvailability | None,
    n_sims: int,
    base_seed: int,
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
) -> DraftStrategy:
    """Map a strategy name (+ live params) to a DraftStrategy.

    Shared by the sidebar dropdown and the resume path. MC strategies
    (`season_value*`) require a non-null `availability` and fail loud otherwise.
    `floor`/`floor_weight` apply only to `now_or_never_floored` (analytic; no availability).
    """
    if name == "raw_vorp":
        return RawVorpStrategy()
    if name == "now_or_never":
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return NowOrNeverStrategy(LogisticSurvival(sigma=spread))
    if name == "now_or_never_floored":
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return NowOrNeverFlooredStrategy(
            LogisticSurvival(sigma=spread), floor=floor, floor_weight=floor_weight
        )
    if name in MC_STRATEGIES:
        if availability is None:
            raise ValueError(f"strategy {name!r} requires availability data (None given)")
        if name == "season_value":
            return SeasonValueStrategy(availability, n_sims=n_sims, base_seed=base_seed)
        if name == "season_value_var":
            return SeasonValueStrategy(
                availability, n_sims=n_sims, base_seed=base_seed, risk_aware=True
            )
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        if name == "seat_aware":
            return build_seat_aware(
                availability,
                n_sims=n_sims,
                base_seed=base_seed,
                survival=LogisticSurvival(sigma=spread),
            )
        return SeasonValueTimingStrategy(
            availability,
            n_sims=n_sims,
            base_seed=base_seed,
            survival=LogisticSurvival(sigma=spread),
        )
    raise ValueError(f"unknown strategy {name!r}")


@dataclass
class RosterView:
    """My current roster: filled slots + remaining open starting slots."""

    filled: pd.DataFrame  # columns: slot, gsis_id, full_name, position
    open_slots: dict[RosterSlot, int]


def attach_names(df: pd.DataFrame, names: Mapping[str, str]) -> pd.DataFrame:
    """Return a copy of `df` with a `full_name` column from a prebuilt name map.

    `names` is a `gsis_id → full_name` mapping (see `LiveDraftSession.player_names`),
    passed in so callers don't rebuild it from the full id_map on every call.
    """
    out = df.copy()
    out.insert(1, "full_name", [names.get(g, "—") for g in out["gsis_id"]])
    return out


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
    # Memo of the last-built DraftState, keyed on the picks tuple (state() is hit
    # ~4-5x per render and twice per bot pick; rebuilding scans the whole id_map).
    _state_cache: tuple[tuple[GsisId, ...], DraftState] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @cached_property
    def player_names(self) -> dict[str, str]:
        """gsis_id -> full_name: id_map names overlaid with the pool's own full_name.

        The consensus VORP pool carries full_name for players absent from id_map
        (placeholder-gsis rookies); those names win so every drafted/available player
        resolves. A pool player with a null name falls back to id_map, then '—'.
        """
        names: dict[str, str] = dict(
            zip(self.id_map["gsis_id"], self.id_map["full_name"], strict=False)
        )
        if "full_name" in self.pool.columns:
            for gid, nm in zip(self.pool["gsis_id"], self.pool["full_name"], strict=False):
                if pd.notna(nm):
                    names[str(gid)] = str(nm)
        return names

    @cached_property
    def _id_map_ids(self) -> frozenset[str]:
        """gsis_ids present in id_map — the players whose position build_draft_state can
        resolve. record_pick's my-pick guard checks this directly (player_names is no
        longer id_map-only, so it can't stand in for id_map membership)."""
        return frozenset(str(g) for g in self.id_map["gsis_id"])

    def name(self, gsis_id: str) -> str:
        """Display name for a gsis_id ('—' if absent)."""
        return self.player_names.get(gsis_id, "—")

    def state(self) -> DraftState:
        """The immutable engine snapshot for the current picks (memoized per picks tuple)."""
        key = tuple(self.picks)
        if self._state_cache is None or self._state_cache[0] != key:
            built = build_draft_state(
                self.picks, my_slot=self.my_slot, league=self.league, id_map=self.id_map
            )
            self._state_cache = (key, built)
        return self._state_cache[1]

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

    @property
    def pick_in_round(self) -> int:
        """1-based pick position within the current round (1..n_teams), counting up every round
        regardless of snake direction. For the 'Round.Pick' display label, where '2.01' is the
        first pick of round 2 (the team that drafted last in round 1), not the on-clock slot."""
        return (self.current_pick - 1) % self.league.n_teams + 1

    def available_pool(self) -> pd.DataFrame:
        drafted = self.state().drafted_ids
        return self.pool[~self.pool["gsis_id"].isin(drafted)].reset_index(drop=True)

    def record_pick(self, gsis_id: str) -> None:
        gid = validate_gsis_id(str(gsis_id))
        if gid in self.state().drafted_ids:
            raise ValueError(f"{gid} already drafted")
        # Only *my* picks need an id_map position (roster accounting via build_draft_state).
        # player_names is now broader than id_map (it includes pool-only rookies), so it can
        # no longer stand in for id_map membership; _id_map_ids is the authoritative check.
        # An opponent pick just leaves the available pool, so an off-id_map opponent pick
        # — e.g. a placeholder-gsis rookie carried in the VORP pool but not yet in id_map —
        # is fine; without this, mock_advance's bot picks would crash on such a player.
        if self.on_clock_slot == self.my_slot and gid not in self._id_map_ids:
            raise ValueError(f"{gid} absent from id_map (cannot resolve position for my roster)")
        self.picks.append(gid)

    def undo(self) -> GsisId | None:
        return self.picks.pop() if self.picks else None

    def recommendation(self) -> pd.DataFrame:
        return self.strategy.recommend(self.state(), self.pool, self.league)

    def suggested_pick(self) -> GsisId | None:
        avail = self.available_pool()
        if avail.empty:
            return None
        if "consensus_adp" not in avail.columns:
            # A non-consensus VORP pool (the column is Optional in VorpTableSchema) has no
            # market signal; back-fill all-NA so bot_pick treats everyone as +inf (ties
            # break on gsis_id) instead of raising KeyError. Mirrors the strategy path.
            avail = avail.assign(
                consensus_adp=pd.array([pd.NA] * len(avail), dtype=pd.Float64Dtype())
            )
        # Deterministic per board state → stable across Streamlit reruns, reproducible
        # in mock mode. (Re-deriving the seed each call is intentional; no stored RNG.)
        rng = np.random.default_rng([self.base_seed, self.current_pick])
        return bot_pick(avail, rng, adp_jitter=self.adp_jitter)

    def my_roster_view(self) -> RosterView:
        state = self.state()
        placements, open_, _ = allocate_roster_slots(
            zip(state.my_pick_ids, state.my_roster, strict=False), self.league.roster_slots
        )
        rows = [
            {
                "slot": slot.value,
                "gsis_id": gid,
                "full_name": self.name(gid),
                "position": pos.value,
            }
            for gid, pos, slot in placements
        ]
        filled = pd.DataFrame(rows, columns=["slot", "gsis_id", "full_name", "position"])
        open_slots: dict[RosterSlot, int] = {
            s: c for s, c in open_.items() if c > 0 and s != RosterSlot.BENCH
        }
        return RosterView(filled=filled, open_slots=open_slots)

    def best_available_by_position(self, top: int) -> dict[Position, pd.DataFrame]:
        avail = self.available_pool()
        out: dict[Position, pd.DataFrame] = {}
        for pos in Position:
            sub = avail[avail["position"] == pos.value].sort_values("vorp", ascending=False)
            if not sub.empty:
                out[pos] = sub.head(top).reset_index(drop=True)
        return out

    def available_for_pick(
        self, position: Position | None = None, query: str = "", top: int = 60
    ) -> pd.DataFrame:
        """Name-attached available players for the picker pane.

        Filters to `position` (None = all positions), then to rows whose resolved name
        contains `query` (case-insensitive substring; "" = no filter), sorts by vorp
        descending, and caps to `top`. The cap is applied LAST — after the position and
        query filters and the sort — so a position selection or a search can reach a deep
        player an unfiltered cross-position top-N would hide. Names use the same
        pool-over-id_map source as `player_names` (so rookies match what's displayed).
        """
        avail = self.available_pool()
        if position is not None:
            avail = avail[avail["position"] == position.value]
        # Drop a pre-existing full_name column (pool-sourced) so attach_names can insert the
        # canonical resolved name (pool-over-id_map) without a duplicate-column error.
        if "full_name" in avail.columns:
            avail = avail.drop(columns=["full_name"])
        named = attach_names(avail, self.player_names)
        if query:
            named = named[named["full_name"].str.contains(query, case=False, na=False, regex=False)]
        return named.sort_values("vorp", ascending=False).head(top).reset_index(drop=True)

    def mock_advance_to_my_pick(self) -> list[GsisId]:
        if self.mode != "mock":
            raise RuntimeError("mock_advance_to_my_pick is only valid in mock mode")
        made: list[GsisId] = []
        while not self.is_complete and not self.is_my_pick:
            gid = self.suggested_pick()
            if gid is None:
                break
            self.record_pick(gid)
            made.append(gid)
        return made

    def roster_scorecard(self) -> float:
        mine = self.pool[self.pool["gsis_id"].isin(self.state().my_pick_ids)]
        return optimal_lineup_points(mine, self.league.roster_slots)

    def project_league_outcomes(
        self,
        *,
        n_sims: int = 2000,
        seed: int = 0,
        availability: PlayerAvailability | None = None,
        params: VarianceParams | None = None,
        data_root: Path = Path("data"),
    ) -> dict[int, SeatProjection]:
        """Projected-vs-projected per-seat season metrics for the COMPLETED draft.

        Reconstructs every team from the snake pick order, then runs the variance-model league
        sim (injury + performance draws, optimal lineup both sides, the fixed top-6/top-2-bye
        bracket). `availability`/`params` default to the store + the fitted variance config;
        tests inject them to stay hermetic. Raises if the draft is not complete.
        """
        if not self.is_complete:
            raise ValueError("draft must be complete to project league outcomes")
        pool = attach_is_rookie(self.pool, season=self.season, data_root=data_root)
        if availability is None:
            from projections.draft.assistant.availability_loader import load_store_availability

            availability = load_store_availability(pool, season=self.season, data_root=data_root)
        if params is None:
            params = VarianceParams.load()
        n_teams = self.league.n_teams
        rosters = {
            slot: [str(p) for i, p in enumerate(self.picks) if slot_for(i + 1, n_teams) == slot]
            for slot in range(1, n_teams + 1)
        }
        return project_draft(
            rosters=rosters,
            pool=pool,
            availability=availability,
            params=params,
            league_config=self.league,
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
        )

    def to_state_dict(self) -> dict[str, object]:
        """CLI-compatible superset: load_draft_state reads the required keys; the rest
        (mode/strategy/data paths) drive one-click resume."""
        return {
            "league_config": str(self.league_config_path),
            "my_slot": self.my_slot,
            "picks": list(self.picks),
            "mode": self.mode,
            "adp_jitter": self.adp_jitter,
            "strategy_name": self.strategy_name,
            "n_sims": self.n_sims,
            "sigma": self.sigma,
            "season": self.season,
            "vorp_table": str(self.vorp_path),
            "id_map": str(self.id_map_path),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        id_map: pd.DataFrame,
        pool: pd.DataFrame,
        data_root: Path = Path("data"),
    ) -> LiveDraftSession:
        """Rebuild a session from a saved state dict; strategy via build_session_strategy
        (MC strategies load availability from data_root + saved season)."""
        from projections.draft.assistant.availability_loader import load_store_availability

        data = json.loads(path.read_text())
        league = LeagueConfig.model_validate_json(Path(data["league_config"]).read_text())
        name = str(data["strategy_name"])
        n_sims = int(data.get("n_sims", 300))
        season = int(data.get("season", 2026))
        availability = None
        if name in MC_STRATEGIES:
            availability = load_store_availability(pool, season=season, data_root=data_root)
        strategy = build_session_strategy(
            name,
            league=league,
            sigma=data.get("sigma"),
            availability=availability,
            n_sims=n_sims,
            base_seed=0,
        )
        return cls(
            league=league,
            my_slot=int(data["my_slot"]),
            id_map=id_map,
            pool=pool,
            strategy=strategy,
            strategy_name=name,
            mode=data.get("mode", "copilot"),
            adp_jitter=float(data.get("adp_jitter", 8.0)),
            n_sims=n_sims,
            sigma=data.get("sigma"),
            season=season,
            picks=[validate_gsis_id(str(p)) for p in data["picks"]],
            league_config_path=Path(data["league_config"]),
            vorp_path=Path(data.get("vorp_table", ".")),
            id_map_path=Path(data.get("id_map", ".")),
        )
