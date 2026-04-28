"""Optuna-driven hyperparameter tuning for LightGBMModel — Plan 5b.

Runs 24 per-(position, stat) studies. Each study:
  - sampler: TPESampler(seed=<--seed>)
  - pruner:  MedianPruner(n_startup_trials=10, n_warmup_steps=20)
  - trials:  configured by --trials (default 50)
  - objective: sum of 5 pinball losses on the 2023 trial-scorer slice

Train slice = season in [2018..2022]; early-stop val = season == 2022;
trial scorer = season == 2023.

Tuned params are written to --out (default data/tuned_params/lightgbm.json).
Optuna studies persist at --studies-db (default
data/tuned_params/optuna_studies.db); resumable across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, cast

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.models.lightgbm import QUANTILE_GRID, LightGBMModel
from projections.schemas import Position, Stat, WeeklyStatsSchema
from projections.store import read_partition

# Search window: most-recent fold's training data. Train 2018-2022;
# 2022 is the early-stopping val; 2023 is the trial scorer.
_TRAIN_SEASONS: Final[range] = range(2018, 2023)  # 2018..2022 inclusive
_EARLY_STOP_VAL_SEASON: Final[int] = 2022
_TRIAL_SCORER_SEASON: Final[int] = 2023

_FIXED_PARAMS: Final[dict[str, Any]] = {
    "n_estimators": 4000,  # raised from default 2000; early stopping picks actual count
    "subsample_freq": 1,
    "verbose": -1,
    "random_state": 42,
}
_DEFAULT_TRIALS: Final[int] = 50
_DEFAULT_SEED: Final[int] = 42


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one set of LightGBM hyperparameters for a trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss at level alpha."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def _load_join_for_position(
    position: Position,
    *,
    seasons: Iterable[int],
    data_root: Path,
    features_root: Path,
) -> pd.DataFrame:
    """Load features + weekly_stats for the given seasons; inner-join."""
    season_list = list(seasons)
    feat_frames = [read_features(position, s, features_root=features_root) for s in season_list]
    feat = pd.concat(feat_frames, ignore_index=True)
    feat = POSITION_DISPATCH[position].feature_schema.validate(feat)

    ws_frames = [read_partition(data_root, "weekly_stats", season=s) for s in season_list]
    ws = WeeklyStatsSchema.validate(pd.concat(ws_frames, ignore_index=True))
    ws = ws[ws["position"] == position.value].copy()

    # POSITION_DISPATCH.factories returns the Model Protocol; narrow to
    # LightGBMModel to read concrete attributes (target_stats / feature_columns).
    lgbm_model = cast(LightGBMModel, POSITION_DISPATCH[position].factories["lightgbm"]())
    target_cols = [s.value for s in lgbm_model.target_stats]
    joined = feat.merge(
        ws[["gsis_id", "season", "week", *target_cols]],
        on=["gsis_id", "season", "week"],
        how="inner",
        validate="one_to_one",
    )
    return joined


def _objective(
    trial: optuna.Trial,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_score: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Trial objective: sum of 5 pinball losses on the trial-scorer slice."""
    params = _sample_params(trial)
    total = 0.0
    for q in QUANTILE_GRID:
        regressor = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            **_FIXED_PARAMS,
            **params,
        )
        regressor.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                LightGBMPruningCallback(trial, metric="quantile", valid_name="valid_0"),
            ],
        )
        y_pred_score = regressor.predict(x_score)
        total += _pinball_loss(y_score, y_pred_score, q)
    return total


def _run_one_study(
    position: Position,
    stat: Stat,
    *,
    joined: pd.DataFrame,
    feat_cols: Sequence[str],
    n_trials: int,
    seed: int,
    studies_db: Path | None,
) -> dict[str, float]:
    """Run one (position, stat) Optuna study; return best_params (8 axes)."""
    train_mask = joined["season"].isin(list(_TRAIN_SEASONS))
    val_mask = joined["season"] == _EARLY_STOP_VAL_SEASON
    score_mask = joined["season"] == _TRIAL_SCORER_SEASON

    if not train_mask.any() or not val_mask.any() or not score_mask.any():
        raise ValueError(
            f"Insufficient data for {position.value}/{stat.value}: "
            f"train_rows={train_mask.sum()}, val_rows={val_mask.sum()}, "
            f"score_rows={score_mask.sum()}"
        )

    x_all = joined[list(feat_cols)].to_numpy(dtype=np.float64)
    y_all = joined[stat.value].to_numpy(dtype=np.float64)

    # The early-stop val (2022) is part of the train mask used by
    # LightGBMRegressor.fit's eval_set. The trial scorer (2023) is held out.
    # Train rows passed to fit() are 2018-2021; val (2022) goes to eval_set
    # for early stopping; the trained model is then scored on 2023.
    inner_train_mask = joined["season"].isin([2018, 2019, 2020, 2021])

    x_train = x_all[inner_train_mask.to_numpy()]
    y_train = y_all[inner_train_mask.to_numpy()]
    x_val = x_all[val_mask.to_numpy()]
    y_val = y_all[val_mask.to_numpy()]
    x_score = x_all[score_mask.to_numpy()]
    y_score = y_all[score_mask.to_numpy()]

    storage_url = f"sqlite:///{studies_db}" if studies_db is not None else None
    study_name = f"lightgbm:{position.value.lower()}:{stat.value}:v1"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: _objective(
            t,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_score=x_score,
            y_score=y_score,
        ),
        n_trials=n_trials,
        catch=(Exception,),
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"Study {study_name} produced 0 completed trials "
            f"(all pruned or failed). Check pruner / data."
        )
    return dict(study.best_params)


def run_studies(
    *,
    positions: Sequence[Position],
    stats_per_position: dict[Position, Sequence[Stat]],
    n_trials: int,
    seed: int,
    data_root: Path,
    features_root: Path,
    studies_db: Path | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run studies for every (position, stat) pair; return tuned-params dict."""
    seasons = [*_TRAIN_SEASONS, _TRIAL_SCORER_SEASON]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for position in positions:
        joined = _load_join_for_position(
            position, seasons=seasons, data_root=data_root, features_root=features_root
        )
        # cast() because factories return the Model Protocol; the concrete
        # LightGBMModel carries feature_columns on its (private) _config.
        # Mirrors tests/test_models/test_lightgbm.py which reads the same field.
        lgbm_model = cast(LightGBMModel, POSITION_DISPATCH[position].factories["lightgbm"]())
        feat_cols = list(lgbm_model._config.feature_columns)
        pos_key = position.value.lower()
        out.setdefault(pos_key, {})
        for stat in stats_per_position[position]:
            print(f"[tune] running study {pos_key}/{stat.value} ({n_trials} trials)…", flush=True)
            best = _run_one_study(
                position,
                stat,
                joined=joined,
                feat_cols=feat_cols,
                n_trials=n_trials,
                seed=seed,
                studies_db=studies_db,
            )
            out[pos_key][stat.value] = best
            print(f"[tune] {pos_key}/{stat.value} best_params: {best}", flush=True)
    return out


def _all_positions() -> tuple[Position, ...]:
    return (Position.QB, Position.RB, Position.TE, Position.WR)


def _stats_for(position: Position) -> tuple[Stat, ...]:
    # cast() because factories return the Model Protocol; the concrete
    # LightGBMModel exposes target_stats.
    model = cast(LightGBMModel, POSITION_DISPATCH[position].factories["lightgbm"]())
    return model.target_stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position",
        choices=["qb", "rb", "te", "wr", "all"],
        default="all",
        help="Position to tune; 'all' runs every position.",
    )
    parser.add_argument(
        "--stat",
        default=None,
        help=(
            "Single stat to tune (e.g. 'receiving_yards'). "
            "If omitted, every target stat for the selected position(s) is tuned."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=_DEFAULT_TRIALS,
        help=f"Trials per study (default {_DEFAULT_TRIALS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"TPE sampler seed (default {_DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tuned_params/lightgbm.json"),
        help="Output JSON path for tuned params.",
    )
    parser.add_argument(
        "--studies-db",
        type=Path,
        default=Path("data/tuned_params/optuna_studies.db"),
        help="SQLite path for Optuna study persistence; resumable across runs.",
    )
    parser.add_argument(
        "--in-memory-storage",
        action="store_true",
        help="Use in-memory study storage (overrides --studies-db); not resumable.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Raw data root.")
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="Feature cache root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all studies but do not overwrite --out.",
    )
    args = parser.parse_args(argv)

    if args.position == "all":
        positions = list(_all_positions())
    else:
        positions = [Position(args.position.upper())]

    stats_per_position: dict[Position, Sequence[Stat]] = {}
    for p in positions:
        if args.stat is None:
            stats_per_position[p] = _stats_for(p)
        else:
            try:
                stats_per_position[p] = (Stat(args.stat),)
            except ValueError:
                print(f"unknown stat: {args.stat}", file=sys.stderr)
                return 2

    studies_db: Path | None
    if args.in_memory_storage:
        studies_db = None
    else:
        studies_db = args.studies_db
        studies_db.parent.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    tuned = run_studies(
        positions=positions,
        stats_per_position=stats_per_position,
        n_trials=args.trials,
        seed=args.seed,
        data_root=args.data_root,
        features_root=args.features_root,
        studies_db=studies_db,
    )

    if args.dry_run:
        print(f"[tune] --dry-run; not writing {args.out}")
        print(json.dumps(tuned, indent=2, sort_keys=True))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n")
    print(f"[tune] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
