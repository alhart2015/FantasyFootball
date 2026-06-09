"""Hybrid LightGBM with NB-2 for count stats — Plan 5c.

Subclass of LightGBMTunedModel. For zero-inflated count stats (the 13 cells
Plan 3e routes through NB-2 in Ridge — passing_tds / rushing_tds /
receiving_tds / interceptions / fumbles_lost, intersected with each
position's target_stats), trains one lgb.LGBMRegressor with
``objective="poisson"``, reads predicted mu directly from
``regressor.predict(x)`` (lgb's poisson predict returns mean in original
scale, already exponentiated), and fits NB-2 dispersion on training
residuals via ``nb_dispersion_from_residuals``. Predict-time distribution
per count stat: ``ParametricNegativeBinomial(mu, dispersion)``.

For yards / receptions stats: 5-quantile sub-models exactly as
LightGBMTunedModel does today. Predict-time distribution:
``QuantileDistribution`` (unchanged).

Per-row ``ProjectionWeeklySchema.family`` is set to
``DistributionFamily.MIXED``; per-stat families remain encoded
individually inside the params blob via the codec's existing per-stat
dispatch.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from projections.distributions import ParametricNegativeBinomial, pack_per_stat_params
from projections.distributions.parametric import (
    _NB_MU_FLOOR,
    nb_dispersion_from_residuals,
)
from projections.distributions.quantile import QuantileDistribution
from projections.models.base import compute_code_hash
from projections.models.lightgbm import (
    _QB_FEATURE_COLUMNS,
    _QB_NON_NEGATIVE,
    _QB_TARGET_STATS,
    _RB_FEATURE_COLUMNS,
    _RB_NON_NEGATIVE,
    _RB_TARGET_STATS,
    _TE_FEATURE_COLUMNS,
    _TE_NON_NEGATIVE,
    _TE_TARGET_STATS,
    _WR_FEATURE_COLUMNS,
    _WR_NON_NEGATIVE,
    _WR_TARGET_STATS,
    QUANTILE_GRID,
    _filter_features,
    _LightGBMConfig,
)
from projections.models.lightgbm_tuned import (
    _TUNED_PARAMS_PATH,
    LightGBMTunedModel,
)
from projections.schemas import (
    _PYARROW_STR,
    DATETIME_UNIT,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
from projections.scoring.score_distribution import (
    derive_row_seed,
    score_distribution,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

# TODO #33c schema-swap: drop per-game implied_team_total + spread from the
# lgb-nb feature list for QB + WR; add the four Vegas team-context cols
# (preseason_* + season_avg_*). The schemas keep all cols so other model
# classes (BaselineModel via hardcoded list; lgb / lgb-tuned via schema
# derivation) are unaffected -- this override is lgb-nb-specific by design.
_VEGAS_SWAP_REPLACE: Final[frozenset[str]] = frozenset({"implied_team_total", "spread"})
_VEGAS_SWAP_ADD: Final[tuple[str, ...]] = (
    "preseason_implied_team_total",
    "preseason_spread",
    "season_avg_implied_team_total",
    "season_avg_spread",
)


def _swap_for(cols: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the Vegas-swap per-game cols and append the 4 Vegas team-context cols.

    Idempotent on the 'add' side: filters out any cols already in
    ``_VEGAS_SWAP_ADD`` before appending (so a future schema bump that adds
    the new cols a second time doesn't duplicate them in the lgb-nb list).
    """
    swap_add_set = set(_VEGAS_SWAP_ADD)
    kept = tuple(c for c in cols if c not in _VEGAS_SWAP_REPLACE and c not in swap_add_set)
    return kept + _VEGAS_SWAP_ADD


_QB_FEATURE_COLUMNS_NB: Final[tuple[str, ...]] = _swap_for(_filter_features(_QB_FEATURE_COLUMNS))
_WR_FEATURE_COLUMNS_NB: Final[tuple[str, ...]] = _swap_for(_filter_features(_WR_FEATURE_COLUMNS))

# Stats Plan 3e routes through NB-2 in Ridge's _<POS>_DIST_FAMILIES.
# Per-position intersection with target_stats yields 13 cells: QB 4
# (passing_tds, interceptions, rushing_tds, fumbles_lost); RB / TE / WR
# 3 each (receiving_tds, rushing_tds, fumbles_lost).
COUNT_STATS_FOR_NB: Final[frozenset[Stat]] = frozenset(
    {
        Stat.PASSING_TDS,
        Stat.RUSHING_TDS,
        Stat.RECEIVING_TDS,
        Stat.INTERCEPTIONS,
        Stat.FUMBLES_LOST,
    }
)


def _code_hash_files_nb(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into the NB model's model_id.

    Mirrors LightGBMTunedModel's set + adds lightgbm_nb.py. parametric.py is
    already in the parent's set (it owns the QuantileDistribution codec
    that the parent uses); after Plan 5c Phase 0 it also owns
    nb_dispersion_from_residuals + ParametricNegativeBinomial, so the
    same path covers both reasons.
    """
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
        src / "models" / "lightgbm_nb.py",
        src / "models" / "lightgbm_tuned.py",
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "vegas_team_context_features.py",
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
        _TUNED_PARAMS_PATH,
    )


class LightGBMNbModel(LightGBMTunedModel):
    """LightGBM with NB-2 for count stats and QuantileDistribution for yards stats.

    Inherits from LightGBMTunedModel: tuned-params loader, _hyperparams_for(stat)
    hook, joblib save/load, feature/weekly_stats join. Overrides fit and
    predict_distribution to branch per stat between count (NB-2) and yards
    (5-quantile) paths. Overrides code_hash and model_id to reflect the
    lightgbm-nb: prefix and the lightgbm_nb.py file in the hash.
    """

    def __init__(
        self,
        *,
        config: _LightGBMConfig,
        tuned_params_path: Path = _TUNED_PARAMS_PATH,
    ) -> None:
        super().__init__(config=config, tuned_params_path=tuned_params_path)
        self._count_models: dict[Stat, lgb.Booster] = {}
        self._count_dispersions: dict[Stat, float] = {}
        self._count_best_iters: dict[Stat, int] = {}

    @property
    def code_hash(self) -> str:
        return compute_code_hash(_code_hash_files_nb(self._config.position))

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm-nb:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train per-stat sub-models with hybrid count/yards routing.

        For stat in COUNT_STATS_FOR_NB: train one lgb.LGBMRegressor with
        objective="poisson" + tuned hyperparameters; fit NB-2 dispersion
        on training residuals via nb_dispersion_from_residuals.

        Otherwise: 5 quantile sub-models, identical to LightGBMTunedModel.
        """
        # Validate features against the position schema (mirrors parent).
        features = self._config.feature_schema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
        weekly_stats = weekly_stats[weekly_stats["position"] == self._config.position.value].copy()

        target_cols = [s.value for s in self._config.target_stats]
        joined = features.merge(
            weekly_stats[["gsis_id", "season", "week", *target_cols]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError("Empty training set after feature/weekly_stats join")

        seasons = sorted(joined["season"].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"Need >=2 training seasons for early-stopping validation slice; got {len(seasons)}"
            )

        val_season = seasons[-1]
        train_mask = joined["season"] != val_season
        val_mask = joined["season"] == val_season

        feat_cols = list(self._config.feature_columns)
        x_train = joined.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
        x_val = joined.loc[val_mask, feat_cols].to_numpy(dtype=np.float64)

        for stat in self._config.target_stats:
            stat_params = self._hyperparams_for(stat)
            y_train = joined.loc[train_mask, stat.value].to_numpy(dtype=np.float64)
            y_val = joined.loc[val_mask, stat.value].to_numpy(dtype=np.float64)

            if stat in COUNT_STATS_FOR_NB:
                # Single poisson regressor; predict() returns mu in original scale.
                regressor = lgb.LGBMRegressor(
                    objective="poisson",
                    **stat_params,
                )
                regressor.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                best_iter = int(regressor.best_iteration_ or 0)
                if best_iter == 0:
                    warnings.warn(
                        f"LightGBMNbModel.fit: best_iter=0 for "
                        f"{self._config.position.value}/{stat.value} (poisson); "
                        "early stopping fired immediately. Sub-model will "
                        "predict at constant baseline.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._count_models[stat] = regressor.booster_
                self._count_best_iters[stat] = best_iter
                # Fit NB-2 dispersion on training residuals. lgb.LGBMRegressor's
                # poisson predict() already returns mu (the mean) in original
                # scale -- not log-mu. Floor matches the predict-time clip.
                mu_hat_train = np.maximum(
                    np.asarray(regressor.predict(x_train), dtype=np.float64),
                    _NB_MU_FLOOR,
                )
                dispersion = nb_dispersion_from_residuals(mu_hat=mu_hat_train, actual=y_train)
                self._count_dispersions[stat] = dispersion
            else:
                # Inherited quantile-stat behavior: 5 sub-models.
                self._sub_models[stat] = {}
                for q in QUANTILE_GRID:
                    regressor = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        **stat_params,
                    )
                    regressor.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    best_iter = int(regressor.best_iteration_ or 0)
                    if best_iter == 0:
                        warnings.warn(
                            f"LightGBMNbModel.fit: best_iter=0 for "
                            f"{self._config.position.value}/{stat.value}/q={q}; "
                            "early stopping fired immediately. Sub-model will "
                            "predict at constant baseline.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    self._sub_models[stat][q] = regressor.booster_
                    self._best_iters[(stat, q)] = best_iter

        self._train_start = int(seasons[0])
        self._train_end = int(seasons[-1])
        self._is_fitted = True

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-row composite fantasy-points distribution with hybrid families."""
        if not self._is_fitted:
            raise RuntimeError("predict_distribution requires fit() first")

        feat_cols = list(self._config.feature_columns)
        actual_cols = set(features.columns)
        missing = set(feat_cols) - actual_cols
        if missing:
            raise ValueError(
                f"Feature columns differ from training: missing={sorted(missing)}; "
                f"expected feature_columns={feat_cols}"
            )

        features = self._config.feature_schema.validate(features)
        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        x = features[feat_cols].to_numpy(dtype=np.float64)
        n_rows = x.shape[0]
        quant_arr = np.array(QUANTILE_GRID, dtype=np.float64)

        # Per-stat predictions: count stats get mu_hat; yards stats get sorted/clipped quantiles.
        per_stat_count_mu: dict[Stat, np.ndarray[Any, np.dtype[np.float64]]] = {}
        per_stat_quantile_pred: dict[Stat, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for stat in self._config.target_stats:
            if stat in COUNT_STATS_FOR_NB:
                # LightGBM poisson predict() returns mu (mean) in original
                # scale, already exponentiated -- no np.exp needed here.
                mu_hat = np.maximum(
                    np.asarray(self._count_models[stat].predict(x), dtype=np.float64),
                    _NB_MU_FLOOR,
                )
                per_stat_count_mu[stat] = mu_hat
            else:
                preds_per_q = np.column_stack(
                    [self._sub_models[stat][q].predict(x) for q in QUANTILE_GRID]
                ).astype(np.float64)
                preds_per_q.sort(axis=1)
                if stat in self._config.non_negative_stats:
                    np.maximum(preds_per_q, 0.0, out=preds_per_q)
                per_stat_quantile_pred[stat] = preds_per_q

        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)
        gsis_id_col = features["gsis_id"].to_numpy()
        season_col = features["season"].to_numpy()
        week_col = features["week"].to_numpy()
        team_col = features["team"].to_numpy()
        opponent_col = features["opponent"].to_numpy()

        for row_idx in range(n_rows):
            per_stat_dists: dict[Stat, Any] = {}
            for stat in self._config.target_stats:
                if stat in COUNT_STATS_FOR_NB:
                    per_stat_dists[stat] = ParametricNegativeBinomial(
                        mean=float(per_stat_count_mu[stat][row_idx]),
                        dispersion=self._count_dispersions[stat],
                    )
                else:
                    per_stat_dists[stat] = QuantileDistribution(
                        quantiles=quant_arr,
                        values=per_stat_quantile_pred[stat][row_idx],
                    )

            seed = derive_row_seed(
                gsis_id=str(gsis_id_col[row_idx]),
                season=int(season_col[row_idx]),
                week=int(week_col[row_idx]),
                ruleset_name=ruleset.name,
            )
            composite = score_distribution(per_stat_dists, ruleset, seed=seed)

            out_rows.append(
                {
                    "gsis_id": str(gsis_id_col[row_idx]),
                    "season": int(season_col[row_idx]),
                    "week": int(week_col[row_idx]),
                    "position": self._config.position.value,
                    "team": str(team_col[row_idx]),
                    "opponent": str(opponent_col[row_idx]),
                    "ruleset": ruleset.name,
                    "family": DistributionFamily.MIXED.value,
                    "params": pack_per_stat_params(per_stat_dists),
                    "mean": composite.mean(),
                    "p10": composite.quantile(0.10),
                    "p50": composite.quantile(0.50),
                    "p90": composite.quantile(0.90),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit(DATETIME_UNIT),
                }
            )
        out = pd.DataFrame(out_rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        out["position"] = out["position"].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)


def qb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_QB_FEATURE_COLUMNS_NB,  # TODO #33c Vegas swap
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )


def rb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
        )
    )


def te_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
        )
    )


def wr_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_WR_FEATURE_COLUMNS_NB,  # TODO #33c Vegas swap
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )
