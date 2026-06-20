"""Draft strategies: the substitution seam + three concrete implementations.

`RawVorpStrategy` is the best-available control. `NowOrNeverStrategy` is the
analytic opportunity-cost strategy (spec §3.5): rank by value locked in over the
expected best survivor at the same position by my next pick. `SeasonValueStrategy`
is the depth-aware strategy (spec §3.2): rank by the marginal expected season points
a pick adds to the current roster, under common random numbers. All share `_finalize`,
which filters to roster-eligible positions, tags the starting-need tier, and applies
the deterministic final ordering (the season-marginal strategy opts out of the tier
as a sort key — its score already values open slots).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.pick_timing import my_next_pick
from projections.draft.assistant.season_value import (
    marginal_season_values,
    marginal_season_values_var,
)
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.survival import SurvivalModel, expected_best_by_position
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import eligible_positions
from projections.schemas import _PYARROW_STR, Position, RecommendationSchema

# Single source of truth for the valid strategy identifiers used by the assistant
# CLI, the backtest harness, and any other caller that needs to enumerate strategies.
STRATEGY_KEYS = (
    "now_or_never",
    "now_or_never_floored",
    "season_value",
    "season_value_var",
    "season_value_timing",
    "seat_aware",
    "raw_vorp",
)

# PROVISIONAL defaults for the now_or_never_floored knobs — a mid-grid starting point,
# replaced by the A/B winner (spec 2026-06-16 §8). Imported by build_session_strategy,
# the harness registry, and both CLIs so there is ONE literal to update.
_DEFAULT_FLOOR = 40.0
_DEFAULT_FLOOR_WEIGHT = 1.0


@runtime_checkable
class DraftStrategy(Protocol):
    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        """Rank the available pool; return a RecommendationSchema frame."""
        ...


def _eligible_subset(
    state: DraftState, pool: pd.DataFrame, config: LeagueConfig
) -> tuple[pd.DataFrame, dict[Position, bool]]:
    """Drop already-drafted + roster-ineligible rows. Returns (subset, eligibility)."""
    elig = eligible_positions(config.roster_slots, list(state.my_roster))
    eligible_values = {pos.value for pos in elig}
    subset = pool[
        ~pool["gsis_id"].isin(state.drafted_ids) & pool["position"].isin(eligible_values)
    ].copy()
    # consensus_adp is optional on VorpTableSchema (absent on the weekly path).
    # The assistant is consensus-only, but guard so a missing column degrades to
    # all-null (everything survives) instead of a KeyError in _finalize.
    if "consensus_adp" not in subset.columns:
        subset["consensus_adp"] = pd.array([pd.NA] * len(subset), dtype=pd.Float64Dtype())
    return subset, elig


def _finalize(
    df: pd.DataFrame,
    elig: dict[Position, bool],
    p_available: pd.Series[float],
    *,
    starting_need_tier: bool = True,
) -> pd.DataFrame:
    """Attach the starting-need tier, order deterministically, validate.

    `df` must already carry `score`. `p_available` is index-aligned (Float64,
    null where unknown).

    `starting_need_tier`: when True (default), `fills_starting_slot` is the primary
    sort key so players filling an unmet starting slot bubble above bench-only adds.
    Pass False when `score` already encodes roster-construction value (e.g. the
    season-marginal strategy, whose score is the expected season points a pick adds)
    and the tier promotion would double-count starting-slot value. `fills_starting_slot`
    is still computed and emitted either way — only its use as a sort key is gated.
    """
    out = df.copy()
    # `score` is a difference of float sums (vorp - E[best survivor]); strip the
    # IEEE float dust so hand-computable spec values are exact and ordering is
    # stable across accumulation order. 10 decimals is far below any meaningful
    # VORP delta.
    out["score"] = out["score"].astype(float).round(10)
    fills_by_value = {pos.value: fills for pos, fills in elig.items()}
    fills = out["position"].map(fills_by_value)
    # _eligible_subset already filtered to elig's keyset, so every position maps.
    # Fail loud if that invariant is ever broken — a NaN here would otherwise
    # coerce to True via astype(bool) and silently float a player to the top.
    if fills.isna().any():
        missing = sorted(set(out.loc[fills.isna(), "position"]))
        raise KeyError(f"position(s) outside the eligibility keyset reached _finalize: {missing}")
    out["fills_starting_slot"] = fills.astype(bool)
    out["p_available_next"] = p_available.astype(pd.Float64Dtype())
    out["consensus_adp"] = out["consensus_adp"].astype(pd.Float64Dtype())
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    out["position"] = out["position"].astype(_PYARROW_STR)
    # `fills_starting_slot` is just prepended as the primary key when the tier is on.
    tier = ["fills_starting_slot"] if starting_need_tier else []
    sort_cols = [*tier, "score", "vorp", "gsis_id"]
    ascending = [*([False] * len(tier)), False, False, True]
    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    out["rank"] = pd.array(range(1, len(out) + 1), dtype=pd.Int64Dtype())
    cols = list(RecommendationSchema.to_schema().columns)
    return RecommendationSchema.validate(out[cols])


def _raw_vorp_result(df: pd.DataFrame, elig: dict[Position, bool]) -> pd.DataFrame:
    """Score = VORP, no timing signal (null p_available). The control and the
    last-pick fallback share this."""
    df["score"] = df["vorp"].astype(float)
    p_na: pd.Series[float] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
    return _finalize(df, elig, p_na)


def _validate_mc_params(n_sims: int, top_k: int) -> None:
    """Fail loud on degenerate Monte-Carlo parameters.

    n_sims < 1 produces an empty draw matrix (nan marginals -> silent VORP fallback).
    top_k < 1 means no candidate is ever MC-evaluated. Both must be rejected at
    construction time by any strategy that uses season marginals.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1; got {n_sims}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}")


@lru_cache(maxsize=1)
def _variance_params() -> VarianceParams:
    """Load the fitted performance-variance params once (cached for the process)."""
    return VarianceParams.load()


def _rookie_map(pool: pd.DataFrame) -> dict[str, bool]:
    """gsis_id -> is_rookie from the pool; empty (all-veteran) when the column is absent."""
    if "is_rookie" not in pool.columns:
        return {}
    return dict(zip(pool["gsis_id"].astype(str), pool["is_rookie"].astype(bool), strict=False))


def _season_marginals(
    state: DraftState,
    df: pd.DataFrame,
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability,
    *,
    n_sims: int,
    base_seed: int,
    top_k: int,
    risk_aware: bool,
) -> pd.DataFrame:
    """Return df with ``score`` = each candidate's marginal expected season points.

    ``risk_aware=False`` uses the deterministic per-game MC (the original ``season_value``);
    ``risk_aware=True`` uses the performance-variance model (sampled weekly points + per-player
    ``is_rookie``). Evaluated (top-k-by-VORP-per-position) candidates carry their real marginal;
    pruned-out rows get 0.0 (cosmetic tail). Shared by the SeasonValue* strategies.
    """
    my_ids = {str(g) for g in state.my_pick_ids}
    pool_ids = pool["gsis_id"].astype(str)
    base_roster = pool.loc[
        pool_ids.isin(my_ids), ["gsis_id", "position", "season_mean_fpts"]
    ].copy()
    missing = sorted(my_ids - set(base_roster["gsis_id"].astype(str)))
    if missing:
        warnings.warn(
            f"{len(missing)} rostered player(s) absent from the VORP pool; "
            f"excluded from season valuation: {missing}",
            # 3, not 2: this helper adds a frame, so the warning points at the
            # strategy's caller (the CLI/harness), preserving the pre-extraction behavior.
            stacklevel=3,
        )
    pruned = (
        df.sort_values(["position", "vorp"], ascending=[True, False])
        .groupby("position", sort=False)
        .head(top_k)
    )
    cand = pruned[["gsis_id", "position", "season_mean_fpts"]].copy()
    rng = np.random.default_rng([base_seed, state.current_pick])
    if risk_aware:
        rookie_map = _rookie_map(pool)
        base_roster["is_rookie"] = (
            base_roster["gsis_id"].astype(str).map(rookie_map).fillna(False).astype(bool)
        )
        cand["is_rookie"] = cand["gsis_id"].astype(str).map(rookie_map).fillna(False).astype(bool)
        marginals = marginal_season_values_var(
            base_roster,
            cand,
            config.roster_slots,
            availability,
            _variance_params(),
            n_sims=n_sims,
            rng=rng,
        )
    else:
        marginals = marginal_season_values(
            base_roster, cand, config.roster_slots, availability, n_sims=n_sims, rng=rng
        )
    out = df.copy()
    out["score"] = out["gsis_id"].astype(str).map(marginals).fillna(0.0).astype(float)
    return out


@dataclass(frozen=True)
class RawVorpStrategy:
    """Best available by VORP (roster-eligible), no timing. The control."""

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        return _raw_vorp_result(df, elig)


@dataclass(frozen=True)
class NowOrNeverStrategy:
    """Opportunity-cost strategy: value over the expected best survivor (spec §3.5)."""

    survival: SurvivalModel

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
        if next_pick is None:
            # Last-pick fallback → raw VORP, null p_available.
            return _raw_vorp_result(df, elig)

        # Internal survival prob per row (1.0 for null ADP); displayed value is
        # null where ADP is null.
        adp = df["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)

        # E[best survivor at each position], shared with the season-value timing
        # strategy. Same lexsort + sequential accumulation as before -> bit-identical.
        pos = df["position"].to_numpy()
        vorp = df["vorp"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = df["gsis_id"].to_numpy()
        e_best = expected_best_by_position(pos, vorp, p, gsis)

        # score = vorp - E[best survivor at position], reusing the numpy arrays.
        df["score"] = vorp - np.array([e_best[pos_i] for pos_i in pos], dtype=float)
        return _finalize(df, elig, display_p)


@dataclass(frozen=True)
class NowOrNeverFlooredStrategy:
    """now_or_never plus an absolute quality floor (spec 2026-06-16).

    score = vorp - E[best survivor at position by my next pick]
            - floor_weight * max(0, floor - vorp)

    The hinge demotes sub-``floor`` players so the dynamic-scarcity term can no longer
    float a best-of-a-bad-tier player over a better one elsewhere. ``floor_weight == 0``
    reproduces ``NowOrNeverStrategy`` exactly. The ~8-line score prelude is duplicated from
    ``NowOrNeverStrategy`` *deliberately* — the spec keeps ``now_or_never`` byte-identical as
    the A/B control, so we copy rather than extract-and-share (which would edit the control).
    ``floor`` / ``floor_weight`` defaults are PROVISIONAL — set from the A/B.
    """

    survival: SurvivalModel
    floor: float = _DEFAULT_FLOOR
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT

    def __post_init__(self) -> None:
        if not math.isfinite(self.floor) or not math.isfinite(self.floor_weight):
            raise ValueError(
                f"floor and floor_weight must be finite; got floor={self.floor}, "
                f"floor_weight={self.floor_weight}"
            )
        if self.floor_weight < 0:
            raise ValueError(f"floor_weight must be >= 0; got {self.floor_weight}")

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
        if next_pick is None:
            # Last pick → raw VORP, floor not applied (matches now_or_never's fallback).
            return _raw_vorp_result(df, elig)

        adp = df["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)

        pos = df["position"].to_numpy()
        vorp = df["vorp"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = df["gsis_id"].to_numpy()
        e_best = expected_best_by_position(pos, vorp, p, gsis)

        penalty = self.floor_weight * np.maximum(0.0, self.floor - vorp)
        df["score"] = vorp - np.array([e_best[pos_i] for pos_i in pos], dtype=float) - penalty
        return _finalize(df, elig, display_p)


@dataclass(frozen=True)
class SeasonValueStrategy:
    """Depth-aware: rank by marginal expected season points (spec §3.2).

    Scores each candidate by V(my_roster + candidate) - V(my_roster) under common
    random numbers, prunes to top_k-by-VORP per position, ranks purely by that
    marginal (no fills_starting_slot tier — the season metric already values open
    slots). Holds the MC config like NowOrNeverStrategy holds a SurvivalModel.
    """

    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    top_k: int = 8
    risk_aware: bool = False  # True = "season_value_var": sampled weekly points (variance model)

    def __post_init__(self) -> None:
        # Fail loud at construction so neither CLI can build a degenerate strategy:
        # n_sims < 1 makes the MC mean a nan (empty draw matrix), which would silently
        # collapse every marginal to nan -> 0.0 and rank purely by VORP.
        _validate_mc_params(self.n_sims, self.top_k)

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        out = _season_marginals(
            state,
            df,
            pool,
            config,
            self.availability,
            n_sims=self.n_sims,
            base_seed=self.base_seed,
            top_k=self.top_k,
            risk_aware=self.risk_aware,
        )
        p_na: pd.Series[float] = pd.Series(pd.NA, index=out.index, dtype=pd.Float64Dtype())
        return _finalize(out, elig, p_na, starting_need_tier=False)


@dataclass(frozen=True)
class SeasonValueTimingStrategy:
    """Depth-aware + pick-timing: season_value's marginal minus the opportunity cost
    of waiting, in season-value units.

    score = marginal_season_value(c) - E[best surviving marginal at pos(c) by my next pick].
    Same per-pick cost as SeasonValueStrategy (one marginal MC); the timing term reuses
    the already-computed marginals + the ADP survival model (no extra MC). Last pick ->
    rank by raw marginal (today's season_value), mirroring nn's raw-VORP fallback.
    """

    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    survival: SurvivalModel
    top_k: int = 8
    risk_aware: bool = False  # True = variance-model marginals under the timing layer

    def __post_init__(self) -> None:
        _validate_mc_params(self.n_sims, self.top_k)

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        out = _season_marginals(
            state,
            df,
            pool,
            config,
            self.availability,
            n_sims=self.n_sims,
            base_seed=self.base_seed,
            top_k=self.top_k,
            risk_aware=self.risk_aware,
        )

        next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
        if next_pick is None:
            # Last pick: no timing signal -> rank by raw marginal (today's season_value).
            p_na: pd.Series[float] = pd.Series(pd.NA, index=out.index, dtype=pd.Float64Dtype())
            return _finalize(out, elig, p_na, starting_need_tier=False)

        adp = out["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)

        # opp_cost[pos] = E[best surviving marginal at pos]. Computed over `out`: the tail's
        # marginal is 0 and sorts last (by value desc), so it contributes nothing -- equivalent
        # to computing over the pruned set only.
        pos = out["position"].to_numpy()
        marg = out["score"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = out["gsis_id"].to_numpy()
        opp = expected_best_by_position(pos, marg, p, gsis)
        out["score"] = marg - np.array([opp[pos_i] for pos_i in pos], dtype=float)
        return _finalize(out, elig, display_p, starting_need_tier=False)


@dataclass(frozen=True)
class SeatAwareStrategy:
    """Route to the empirically-best sub-strategy for the hero's drawn draft slot.

    The post-availability-fix bake-off (Test 14, `reports/draft_strategy_tests.md`) shows
    a per-seat Pareto frontier: the pick-timing layer (`season_value_timing`) wins at the
    wing/mid, where the wait to your next pick is long, but *hurts* at the turn (the last
    `turn_band` seats), where picks come back-to-back and the timing term adds noise rather
    than signal -- there the pure variance-aware marginal (`season_value`/risk_aware, i.e.
    `season_value_var`) wins. The hero's slot is fixed and known at draft time, so a single
    deployable strategy can route to the right policy per slot. It strictly beats every
    fixed baseline on the pooled multi-year win% while matching the per-seat best at each
    seat. This only routes -- it adds no new scoring term.
    """

    timing: DraftStrategy
    turn: DraftStrategy
    turn_band: int = 2

    def __post_init__(self) -> None:
        if self.turn_band < 1:
            raise ValueError(f"turn_band must be >= 1; got {self.turn_band}")

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        is_turn = state.my_slot > state.n_teams - self.turn_band
        sub = self.turn if is_turn else self.timing
        return sub.recommend(state, pool, config)
