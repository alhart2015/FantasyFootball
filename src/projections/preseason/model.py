"""Preseason model interface + naive baseline.

The PreseasonModel Protocol matches the existing Distribution Protocol
pattern (runtime_checkable, attribute-based). NaivePreseasonModel is the
v1.0 baseline; v1.5+ trained models implement the same Protocol.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from projections.schemas import Ruleset

logger = logging.getLogger(__name__)

# v1 constant: projected games played per rostered player. Historical median
# across 2018-2025 rostered seasons. v2 ships per-player injury priors.
_PROJECTED_GAMES_PLAYED = 16

# v1 constant: UDFA rookies (not in draft_picks for their target season) are
# imputed at this overall pick before running through the rookie GLM.
_UDFA_IMPUTED_PICK = 300


@runtime_checkable
class PreseasonModel(Protocol):
    """Per-player season-total distribution model contract.

    v1.0 NaivePreseasonModel returns degenerate point-mass distributions.
    v1.5+ trained models implementing this Protocol may return real
    distributions (mean != p10 != p50 != p90).
    """

    model_id: str

    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None: ...

    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> PreseasonModel: ...


class NaivePreseasonModel:
    """Three-branch naive baseline (v1.0):

    Branch 1 — veterans with prior-1 season stats: `prior_1_per_game * 16`.
    Branch 2 — veterans missing prior-1: fall back to prior_2, then prior_3.
                Drop with warning if all three missing.
    Branch 3 — rookies: per-(position, stat) Gamma GLM on `log(pick + 1)`,
                fitted on rookie-year season totals during `fit()`.

    Degenerate distribution: mean == p10 == p50 == p90 for every output.
    """

    model_id: str = "naive-preseason-v1"

    def __init__(self) -> None:
        # Per-(position, stat) GLM coefficients populated by fit().
        # Each value is (intercept, slope) for `log(season_total) ~ β₀ + β₁ · log(pick + 1)`.
        self._rookie_glm: dict[tuple[str, str], tuple[float, float]] = {}

    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None:
        """Fit per-(position, stat) rookie-year GLMs. Implemented in Task 13."""
        raise NotImplementedError("fit not yet implemented (Task 13)")

    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        """Predict per-stat season-total degenerate distributions per player.

        Implemented across Tasks 11 (veteran branch), 12 (fallback), 14 (rookies),
        15 (canonical scoring layer integration).
        """
        raise NotImplementedError("predict not yet implemented (Task 11)")

    def save(self, path: Path) -> None:
        """Persist rookie GLM coefficients via joblib. Implemented in Task 16."""
        raise NotImplementedError("save not yet implemented (Task 16)")

    @classmethod
    def load(cls, path: Path) -> NaivePreseasonModel:
        """Load a saved NaivePreseasonModel. Implemented in Task 16."""
        raise NotImplementedError("load not yet implemented (Task 16)")
