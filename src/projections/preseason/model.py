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

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor

from projections.schemas import (
    PreseasonFeaturesSchema,
    PreseasonProjectionSchema,
    Ruleset,
    Stat,
)
from projections.scoring import scoring_coefficients

logger = logging.getLogger(__name__)

# v1 constant: projected games played per rostered player. Historical median
# across 2018-2025 rostered seasons. v2 ships per-player injury priors.
_PROJECTED_GAMES_PLAYED = 16

# v1 constant: UDFA rookies (not in draft_picks for their target season) are
# imputed at this overall pick before running through the rookie GLM.
_UDFA_IMPUTED_PICK = 300

# Schema-stat-name -> Stat enum, for scoring-layer coefficient lookup.
_STAT_BY_SCHEMA_NAME: dict[str, Stat] = {
    "passing_yards": Stat.PASSING_YARDS,
    "passing_tds": Stat.PASSING_TDS,
    "passing_interceptions": Stat.INTERCEPTIONS,
    "rushing_yards": Stat.RUSHING_YARDS,
    "rushing_tds": Stat.RUSHING_TDS,
    "receptions": Stat.RECEPTIONS,
    "receiving_yards": Stat.RECEIVING_YARDS,
    "receiving_tds": Stat.RECEIVING_TDS,
}

# Ordered tuple of schema stat names, derived from the mapping above. Both
# naive model implementations iterate over this set; the dict-key order is
# the single source of truth.
_SCORABLE_SCHEMA_STATS: tuple[str, ...] = tuple(_STAT_BY_SCHEMA_NAME)


def _compute_fpts_from_stats(per_stat: dict[str, pd.Series], ruleset: Ruleset) -> pd.Series:
    """Linear combination of per-stat season totals using the canonical
    scoring coefficient map. Returns a float32 Series clipped at 0."""
    if not per_stat:
        return pd.Series([], dtype="float32")
    coef_map = scoring_coefficients(ruleset)
    sample = next(iter(per_stat.values()))
    fpts = pd.Series(0.0, index=sample.index, dtype="float64")
    for col, vals in per_stat.items():
        stat_name = col.replace("_season_total", "")
        stat = _STAT_BY_SCHEMA_NAME[stat_name]
        coef = coef_map.get(stat, 0.0)
        fpts = fpts + vals.fillna(0).astype("float64") * coef
    return fpts.clip(lower=0).astype("float32")


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
        """Fit per-(position, stat) Gamma GLMs on rookie-year season totals.

        For each rookie (player with a draft_picks row in season S and a
        weekly_stats row in season S), aggregate to season total and fit
        `log(stat + epsilon) ~ β₀ + β₁ * log(pick + 1)` per (position, stat).

        `id_map` is currently unused — kept in the signature for v1.5+
        models that may consume it (e.g., for cross-platform IDs).
        """
        del id_map  # reserved for future model variants

        from projections.preseason.features import _STATS_BY_POSITION, _schema_stat_name

        # Identify rookies: (gsis_id, rookie_season, pick) from draft_picks.
        rookies = draft_picks.dropna(subset=["pick"]).copy()
        rookie_season = rookies[["gsis_id", "season", "pick"]].rename(
            columns={"season": "rookie_season"}
        )

        # Aggregate weekly_stats to per-player-per-season totals.
        season_totals = weekly_stats.groupby(["gsis_id", "season", "position"], as_index=False).agg(
            games_played=("week", "count"),
            passing_yards=("passing_yards", "sum"),
            passing_tds=("passing_tds", "sum"),
            interceptions=("interceptions", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rushing_tds", "sum"),
            receptions=("receptions", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("receiving_tds", "sum"),
        )

        # Join — keep only rookie-year rows.
        rookie_year_totals = season_totals.merge(
            rookie_season,
            left_on=["gsis_id", "season"],
            right_on=["gsis_id", "rookie_season"],
            how="inner",
        )

        self._rookie_glm.clear()
        for position, stats in _STATS_BY_POSITION.items():
            pos_rows = rookie_year_totals.loc[
                rookie_year_totals["position"] == position.value
            ].copy()
            if pos_rows.empty:
                logger.warning(
                    "NaivePreseasonModel.fit: no rookie training data for position=%s; "
                    "skipping. Rookies at this position will fall back to zero.",
                    position.value,
                )
                continue
            # sklearn convention: capital X for the design matrix.
            X = np.log(pos_rows["pick"].astype(float).to_numpy() + 1).reshape(-1, 1)  # noqa: N806
            for stat in stats:
                schema_stat = _schema_stat_name(stat)
                # Gamma family requires strictly positive y. Add epsilon for zeros.
                y_raw = pos_rows[stat.value].astype(float).to_numpy()
                y = np.maximum(y_raw, 0.01)
                try:
                    reg = GammaRegressor(alpha=0.0, fit_intercept=True, max_iter=200)
                    reg.fit(X, y)
                    intercept = float(reg.intercept_)
                    slope = float(reg.coef_[0])
                except Exception as e:
                    # sklearn raises diverse error types from GLM fits
                    # (ValueError, ConvergenceWarning-as-error, etc.). Fallback
                    # path is well-defined: degenerate intercept-only model.
                    logger.warning(
                        "GammaRegressor failed for (%s, %s): %s. Falling back to log-mean-only.",
                        position.value,
                        schema_stat,
                        e,
                    )
                    intercept = float(np.log(y.mean()))
                    slope = 0.0
                self._rookie_glm[(position.value, schema_stat)] = (intercept, slope)

    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        features = PreseasonFeaturesSchema.validate(features)
        per_stat_predictions, retained = self._predict_per_stat(features)
        fpts_mean = _compute_fpts_from_stats(per_stat_predictions, ruleset)

        out = retained[["gsis_id", "season", "position", "team"]].copy()
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

    def _predict_per_stat(
        self, features: pd.DataFrame
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """Per-stat predictions:
        - Veterans: prior_1 -> prior_2 -> prior_3 fallback (Task 12).
        - Rookies: GLM overlay on log(draft_pick_overall + 1). UDFAs imputed to pick=300.

        Veterans with no_prior_3_seasons are dropped with a WARNING.
        """
        stats = _SCORABLE_SCHEMA_STATS

        # Effective-prior tier per row (veterans).
        gp1 = features["prior_1_season_games_played"].fillna(0)
        gp2 = features["prior_2_season_games_played"].fillna(0)
        gp3 = features["prior_3_season_games_played"].fillna(0)
        effective_prior = pd.Series(0, index=features.index, dtype="int8")
        effective_prior = effective_prior.mask(gp3 > 0, 3)
        effective_prior = effective_prior.mask(gp2 > 0, 2)
        effective_prior = effective_prior.mask(gp1 > 0, 1)

        is_rookie = features["is_rookie"].astype(bool)
        is_vet = ~is_rookie
        drop_mask = (effective_prior == 0) & is_vet
        if drop_mask.any():
            dropped_ids = features.loc[drop_mask, "gsis_id"].tolist()
            logger.warning(
                "NaivePreseasonModel: dropping %d veteran(s) with no_prior_3_seasons: %s",
                len(dropped_ids),
                dropped_ids[:5],
            )
        retained = features.loc[~drop_mask].copy()
        effective_prior = effective_prior.loc[retained.index]
        is_rookie = is_rookie.loc[retained.index]

        # UDFA imputation: draft_pick_overall = 300 if missing.
        pick = retained["draft_pick_overall"].fillna(_UDFA_IMPUTED_PICK).astype(float)
        log_pick = np.log(pick + 1)

        result: dict[str, pd.Series] = {}
        for stat in stats:
            chosen = pd.Series(float("nan"), index=retained.index, dtype="float64")

            # Veteran branch — multi-tier fallback.
            for tier in (1, 2, 3):
                tier_mask = (effective_prior == tier) & ~is_rookie
                if not tier_mask.any():
                    continue
                col = f"prior_{tier}_season_per_game_{stat}"
                if col not in retained.columns:
                    continue
                chosen.loc[tier_mask] = (
                    retained.loc[tier_mask, col].astype("float64") * _PROJECTED_GAMES_PLAYED
                )

            # Rookie branch — GLM overlay.
            if is_rookie.any():
                for position in retained.loc[is_rookie, "position"].unique():
                    pos_mask = is_rookie & (retained["position"] == position)
                    if not pos_mask.any():
                        continue
                    key = (position, stat)
                    if key not in self._rookie_glm:
                        # No GLM fitted for this (position, stat) cell — fall to zero.
                        chosen.loc[pos_mask] = 0.0
                        continue
                    intercept, slope = self._rookie_glm[key]
                    chosen.loc[pos_mask] = np.exp(intercept + slope * log_pick.loc[pos_mask])

            result[f"{stat}_season_total"] = chosen.fillna(0.0).clip(lower=0.0).astype("float32")

        return result, retained

    def save(self, path: Path) -> None:
        """Persist rookie GLM coefficients via joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model_id": self.model_id, "rookie_glm": self._rookie_glm},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> NaivePreseasonModel:
        state = joblib.load(path)
        if state.get("model_id") != cls.model_id:
            raise ValueError(
                f"Artifact model_id={state.get('model_id')!r} does not match "
                f"NaivePreseasonModel.model_id={cls.model_id!r}"
            )
        m = cls()
        m._rookie_glm = state["rookie_glm"]
        return m


class NaivePriorOnlyModel:
    """Strictly-simpler baseline for v1.5+ benchmarking.

    Veterans with `prior_1_season_games_played > 0` only. Per-stat prediction
    is `prior_1_season_per_game_<stat> * 16`. No fallback to prior_2/prior_3.
    No rookie branch. Players missing prior-1 (rookies, IR-comeback veterans)
    are dropped from the output with a single aggregate WARNING.

    This is the floor that `NaivePreseasonModel` (and any future trained
    model) is gated against. Keep it deliberately simple — the diff between
    this and `NaivePreseasonModel` measures the value of the rookie GLM and
    fallback chain.
    """

    model_id: str = "naive-prior-only-v1"

    def __init__(self) -> None:
        pass  # No fitted state; this model has no parameters.

    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None:
        """No-op: this model has no parameters. Accepts the same signature as
        the Protocol so it's swap-in compatible with NaivePreseasonModel."""
        del weekly_stats, draft_picks, id_map  # reserved for Protocol shape

    def predict_season_distribution(
        self,
        features: pd.DataFrame,
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        features = PreseasonFeaturesSchema.validate(features)

        # Veterans with prior_1 games only. Drop everyone else.
        gp1 = features["prior_1_season_games_played"].fillna(0)
        keep_mask = gp1 > 0
        n_dropped = int((~keep_mask).sum())
        if n_dropped:
            logger.warning(
                "NaivePriorOnlyModel: dropping %d player(s) with no prior_1 history "
                "(rookies + comeback vets); this model only projects strict veterans.",
                n_dropped,
            )
        retained = features.loc[keep_mask].copy()

        per_stat: dict[str, pd.Series] = {}
        for stat in _SCORABLE_SCHEMA_STATS:
            col_in = f"prior_1_season_per_game_{stat}"
            col_out = f"{stat}_season_total"
            if col_in not in retained.columns:
                continue
            per_stat[col_out] = (
                (retained[col_in].fillna(0).astype("float64") * _PROJECTED_GAMES_PLAYED)
                .clip(lower=0)
                .astype("float32")
            )

        fpts = _compute_fpts_from_stats(per_stat, ruleset)

        out = retained[["gsis_id", "season", "position", "team"]].copy()
        out["ruleset"] = ruleset.name
        out["model_id"] = self.model_id
        for col, vals in per_stat.items():
            for q in ("mean", "p10", "p50", "p90"):
                out[f"{col}_{q}"] = vals
        for q in ("mean", "p10", "p50", "p90"):
            out[f"season_total_fpts_{q}"] = fpts

        out = PreseasonProjectionSchema.validate(out)
        return out

    def save(self, path: Path) -> None:
        """No fitted state to persist; create the directory + write an empty
        sentinel file so external callers can still expect a path to exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_id": self.model_id}, path)

    @classmethod
    def load(cls, path: Path) -> NaivePriorOnlyModel:
        state = joblib.load(path)
        if state.get("model_id") != cls.model_id:
            raise ValueError(
                f"Artifact model_id={state.get('model_id')!r} does not match "
                f"NaivePriorOnlyModel.model_id={cls.model_id!r}"
            )
        return cls()
