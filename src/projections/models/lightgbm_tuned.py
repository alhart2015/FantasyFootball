"""Tuned LightGBM model — Plan 5b.

Subclass of LightGBMModel. Reuses fit / predict_distribution / save / load
unchanged. Two responsibilities:
  1. Override `_hyperparams_for(stat)` to load tuned hyperparameters from
     data/tuned_params/lightgbm.json and merge into LGBM_DEFAULTS.
  2. Override `code_hash` and `model_id` to use the lightgbm-tuned: prefix
     and include the tuned-params JSON in the hash so JSON edits invalidate
     artifacts and force snapshot regeneration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

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
    LGBM_DEFAULTS,
    LightGBMModel,
    _filter_features,
    _LightGBMConfig,
)
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Stat,
    TeFeaturesSchema,
    WrFeaturesSchema,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_TUNED_PARAMS_PATH: Final[Path] = _PROJECT_ROOT / "data" / "tuned_params" / "lightgbm.json"

# Subset of LGBM_DEFAULTS keys that Optuna is allowed to tune. Any other key
# in the tuned-params JSON is rejected by the validator on load.
_TUNED_AXES: Final[frozenset[str]] = frozenset(
    {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
)

_EXPECTED_POSITIONS: Final[frozenset[str]] = frozenset({"qb", "rb", "te", "wr"})


@lru_cache(maxsize=4)
def _load_tuned_params(
    path: Path,
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    """Load + validate the tuned-params JSON. Cached by path.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        json.JSONDecodeError: ``path`` is not valid JSON.
        ValueError: top-level position keys are not exactly {qb,rb,te,wr}, or
            any (position, stat) entry contains an unknown axis key.
    """
    with path.open() as f:
        raw: dict[str, dict[str, dict[str, float]]] = json.load(f)
    if set(raw.keys()) != _EXPECTED_POSITIONS:
        raise ValueError(
            f"tuned-params JSON {path}: top-level keys must be "
            f"{sorted(_EXPECTED_POSITIONS)}; got {sorted(raw.keys())}"
        )
    for pos_key, stat_map in raw.items():
        for stat_key, axis_map in stat_map.items():
            extras = set(axis_map.keys()) - _TUNED_AXES
            if extras:
                raise ValueError(
                    f"tuned-params JSON {path}: position={pos_key} "
                    f"stat={stat_key} has unknown tuned-axis keys: "
                    f"{sorted(extras)}; allowed: {sorted(_TUNED_AXES)}"
                )
    return raw


def _code_hash_files_tuned(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into the tuned model's model_id.

    Mirrors `lightgbm.py`'s _code_hash_files but adds the tuned-params JSON
    so that JSON edits invalidate cached artifacts and force snapshot
    regeneration.
    """
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
        src / "models" / "lightgbm_tuned.py",
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
        _TUNED_PARAMS_PATH,
    )


class LightGBMTunedModel(LightGBMModel):
    """LightGBM with Optuna-tuned per-(position, stat) hyperparameters.

    Loads tuned params from ``data/tuned_params/lightgbm.json`` at
    construction. Overrides only ``_hyperparams_for(stat)``, ``code_hash``,
    and ``model_id``; everything else (fit, predict_distribution, save,
    load) inherits unchanged.
    """

    def __init__(
        self,
        *,
        config: _LightGBMConfig,
        tuned_params_path: Path = _TUNED_PARAMS_PATH,
    ) -> None:
        super().__init__(config=config)
        self._tuned_params_path = tuned_params_path
        self._tuned = _load_tuned_params(tuned_params_path)

    def _hyperparams_for(self, stat: Stat) -> dict[str, Any]:
        pos_key = self._config.position.value.lower()
        try:
            tuned_axes = self._tuned[pos_key][stat.value]
        except KeyError as e:
            raise KeyError(
                f"tuned-params JSON {self._tuned_params_path} missing entry "
                f"for position={pos_key} stat={stat.value}"
            ) from e
        merged = dict(LGBM_DEFAULTS)
        merged.update(tuned_axes)
        return merged

    @property
    def code_hash(self) -> str:
        return compute_code_hash(_code_hash_files_tuned(self._config.position))

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm-tuned:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )


def qb_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )


def rb_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
        )
    )


def te_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
        )
    )


def wr_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )
