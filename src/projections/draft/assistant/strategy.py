"""Draft strategies: the substitution seam + two concrete implementations.

`RawVorpStrategy` is the best-available control. `NowOrNeverStrategy` is the
analytic opportunity-cost strategy (spec §3.5): rank by value locked in over the
expected best survivor at the same position by my next pick. Both share
`_finalize`, which filters to roster-eligible positions, tags the scale-free
starting-need tier, and applies the deterministic final ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from projections.draft.assistant.pick_timing import my_next_pick
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.survival import SurvivalModel
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import eligible_positions
from projections.schemas import _PYARROW_STR, Position, RecommendationSchema


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
    df: pd.DataFrame, elig: dict[Position, bool], p_available: pd.Series[float]
) -> pd.DataFrame:
    """Attach the starting-need tier, order deterministically, validate.

    `df` must already carry `score`. `p_available` is index-aligned (Float64,
    null where unknown).
    """
    out = df.copy()
    # `score` is a difference of float sums (vorp - E[best survivor]); strip the
    # IEEE float dust so hand-computable spec values are exact and ordering is
    # stable across accumulation order. 10 decimals is far below any meaningful
    # VORP delta.
    out["score"] = out["score"].astype(float).round(10)
    out["fills_starting_slot"] = out["position"].map(lambda value: elig[Position(value)])
    out["p_available_next"] = p_available.astype(pd.Float64Dtype())
    out["consensus_adp"] = out["consensus_adp"].astype(pd.Float64Dtype())
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    out["position"] = out["position"].astype(_PYARROW_STR)
    out = out.sort_values(
        ["fills_starting_slot", "score", "vorp", "gsis_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["rank"] = pd.array(range(1, len(out) + 1), dtype=pd.Int64Dtype())
    cols = list(RecommendationSchema.to_schema().columns)
    return RecommendationSchema.validate(out[cols])


@dataclass(frozen=True)
class RawVorpStrategy:
    """Best available by VORP (roster-eligible), no timing. The control."""

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        df["score"] = df["vorp"].astype(float)
        p_na: pd.Series[float] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
        return _finalize(df, elig, p_na)


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
            df["score"] = df["vorp"].astype(float)
            p_na: pd.Series[float] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
            return _finalize(df, elig, p_na)

        # Internal survival prob per row (1.0 for null ADP); displayed value is
        # null where ADP is null.
        adp = df["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)

        df = df.assign(_p=internal_p)
        e_best: dict[str, float] = {}
        for position, sub in df.groupby("position"):
            sub = sub.sort_values(["vorp", "gsis_id"], ascending=[False, True])
            expected = 0.0
            prob_all_better_gone = 1.0
            for vorp_i, p_i in zip(sub["vorp"], sub["_p"], strict=True):
                expected += float(vorp_i) * p_i * prob_all_better_gone
                prob_all_better_gone *= 1.0 - p_i
            e_best[str(position)] = expected

        df["score"] = df["vorp"].astype(float) - df["position"].map(e_best).astype(float)
        return _finalize(df, elig, display_p)
