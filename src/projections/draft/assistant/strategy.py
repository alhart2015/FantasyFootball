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

import warnings
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.pick_timing import my_next_pick
from projections.draft.assistant.season_value import marginal_season_values
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.survival import SurvivalModel, expected_best_by_position
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import eligible_positions
from projections.schemas import _PYARROW_STR, Position, RecommendationSchema

# Single source of truth for the valid strategy identifiers used by the assistant
# CLI, the backtest harness, and any other caller that needs to enumerate strategies.
STRATEGY_KEYS = ("now_or_never", "season_value", "season_value_timing", "raw_vorp")


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
) -> pd.DataFrame:
    """Return df with ``score`` = each candidate's marginal expected season points.

    Evaluated (top-k-by-VORP-per-position) candidates carry their real marginal;
    pruned-out rows get 0.0 (cosmetic tail). Shared by SeasonValueStrategy and
    SeasonValueTimingStrategy.
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
    rng = np.random.default_rng([base_seed, state.current_pick])
    marginals = marginal_season_values(
        base_roster,
        pruned[["gsis_id", "position", "season_mean_fpts"]],
        config.roster_slots,
        availability,
        n_sims=n_sims,
        rng=rng,
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
