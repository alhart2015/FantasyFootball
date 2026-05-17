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

from projections.schemas import PreseasonFeaturesSchema, PreseasonProjectionSchema, Ruleset

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
        features = PreseasonFeaturesSchema.validate(features)
        per_stat_predictions = self._predict_per_stat(features)
        fpts_mean = self._stub_fpts_from_stats(per_stat_predictions, ruleset)

        out = features[["gsis_id", "season", "position", "team"]].copy()
        out["ruleset"] = ruleset.name
        out["model_id"] = self.model_id

        # Degenerate distribution: mean == p10 == p50 == p90.
        for col, vals in per_stat_predictions.items():
            for q in ("mean", "p10", "p50", "p90"):
                out[f"{col}_{q}"] = vals
        for q in ("mean", "p10", "p50", "p90"):
            out[f"season_total_fpts_{q}"] = fpts_mean

        out = PreseasonProjectionSchema.validate(out)
        return out

    def _predict_per_stat(self, features: pd.DataFrame) -> dict[str, pd.Series]:
        """Veteran branch (Task 11): prior_1_per_game * 16 per stat.

        Tasks 12 (prior-2/3 fallback) + 14 (rookie GLM) add layered branches.
        Returns a dict mapping `<stat>_season_total` -> Series indexed like features.
        """
        stats = (
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_yards",
            "rushing_tds",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        )
        result: dict[str, pd.Series] = {}
        for stat in stats:
            col_in = f"prior_1_season_per_game_{stat}"
            col_out = f"{stat}_season_total"
            if col_in not in features.columns:
                continue
            result[col_out] = (features[col_in] * _PROJECTED_GAMES_PLAYED).astype("float32")
        return result

    def _stub_fpts_from_stats(self, per_stat: dict[str, pd.Series], ruleset: Ruleset) -> pd.Series:
        """Placeholder fpts computation -- replaced by `_compute_fpts_from_stats`
        in Task 15 once `projections.scoring.scoring_coefficients` is wired up.

        Currently applies the ruleset's coefficients directly to each per-stat
        season total. Returns a float32 Series clipped at 0.
        """
        # Pick any non-empty series to set the index -- degenerate fpts is
        # zero-indexed if per_stat is empty (no veteran branch fired).
        if not per_stat:
            return pd.Series([], dtype="float32")
        sample = next(iter(per_stat.values()))
        fpts = pd.Series(0.0, index=sample.index, dtype="float64")
        for col, vals in per_stat.items():
            stat = col.replace("_season_total", "")
            v = vals.fillna(0).astype("float64")
            if stat == "passing_yards":
                fpts = fpts + v / ruleset.passing_yds_per_pt
            elif stat == "passing_tds":
                fpts = fpts + v * ruleset.passing_td_pts
            elif stat == "passing_interceptions":
                fpts = fpts + v * ruleset.interception_pts
            elif stat == "rushing_yards":
                fpts = fpts + v / ruleset.rushing_yds_per_pt
            elif stat == "rushing_tds":
                fpts = fpts + v * ruleset.rushing_td_pts
            elif stat == "receptions":
                fpts = fpts + v * ruleset.reception_pts
            elif stat == "receiving_yards":
                fpts = fpts + v / ruleset.receiving_yds_per_pt
            elif stat == "receiving_tds":
                fpts = fpts + v * ruleset.receiving_td_pts
        return fpts.clip(lower=0).astype("float32")

    def save(self, path: Path) -> None:
        """Persist rookie GLM coefficients via joblib. Implemented in Task 16."""
        raise NotImplementedError("save not yet implemented (Task 16)")

    @classmethod
    def load(cls, path: Path) -> NaivePreseasonModel:
        """Load a saved NaivePreseasonModel. Implemented in Task 16."""
        raise NotImplementedError("load not yet implemented (Task 16)")
