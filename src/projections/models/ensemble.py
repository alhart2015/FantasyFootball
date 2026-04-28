"""EnsembleModel — Plan 6 Model D.

Per-(position, stat) weighted mixture of Model A (BaselineModel) and Model
C-NB (LightGBMNbModel). Weights are constant per (position, stat); per-row
distributions are MixtureDistribution(F_a, F_b, w[stat]).

Phase 2 scaffolding: weights default to 0.5 per stat. Phase 3 wires the
pinball-loss optimizer into fit().

Per-row schema:
    family = DistributionFamily.MIXED
    params = pack_per_stat_params({stat: MixtureDistribution(...) for stat in target_stats})
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd

from projections.distributions import (
    MixtureDistribution,
    pack_per_stat_params,
    unpack_per_stat_params,
)
from projections.distributions.base import Distribution
from projections.models.base import compute_code_hash
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
from projections.models.lightgbm import (
    _QB_TARGET_STATS,
    _RB_TARGET_STATS,
    _TE_TARGET_STATS,
    _WR_TARGET_STATS,
)
from projections.models.lightgbm_nb import (
    LightGBMNbModel,
    qb_lightgbm_nb,
    rb_lightgbm_nb,
    te_lightgbm_nb,
    wr_lightgbm_nb,
)
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    Stat,
)
from projections.scoring.score_distribution import (
    derive_row_seed,
    score_distribution,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_DEFAULT_WEIGHTS_DIR: Final[Path] = _PROJECT_ROOT / "data" / "ensemble_weights"


def _code_hash_files_ensemble() -> tuple[Path, ...]:
    """Source files whose content is hashed into EnsembleModel.code_hash."""
    src = _PROJECT_ROOT / "src" / "projections"
    return (
        src / "models" / "ensemble.py",
        src / "distributions" / "mixture.py",
        src / "distributions" / "codec.py",
    )


@dataclass(slots=True)
class _EnsembleConfig:
    position: Position
    target_stats: tuple[Stat, ...]
    child_a_factory: Callable[[], BaselineModel]
    child_b_factory: Callable[[], LightGBMNbModel]
    weights_dir: Path = field(default=_DEFAULT_WEIGHTS_DIR)


class EnsembleModel:
    """Per-(position, stat) weighted mixture of Model A and Model C-NB."""

    _config: _EnsembleConfig
    _child_a: BaselineModel | None
    _child_b: LightGBMNbModel | None
    _weights: dict[Stat, float]
    _train_start: int | None
    _train_end: int | None
    _calibration_year: int | None
    _is_fitted: bool

    def __init__(self, *, config: _EnsembleConfig) -> None:
        self._config = config
        self._child_a = None
        self._child_b = None
        self._weights = {}
        self._train_start = None
        self._train_end = None
        self._calibration_year = None
        self._is_fitted = False

    @property
    def position(self) -> Position:
        return self._config.position

    @property
    def target_stats(self) -> tuple[Stat, ...]:
        return self._config.target_stats

    @property
    def code_hash(self) -> str:
        """SHA-256 first 8 hex of source files + child code-hashes + weights."""
        files_hash = compute_code_hash(_code_hash_files_ensemble())
        if not self._is_fitted:
            return files_hash
        assert self._child_a is not None and self._child_b is not None
        # BaselineModel.code_hash is `str | None` on the dataclass; fit()
        # populates it. _is_fitted is True here so both child code-hashes
        # are guaranteed non-None.
        child_a_hash = self._child_a.code_hash
        child_b_hash = self._child_b.code_hash
        assert child_a_hash is not None and child_b_hash is not None
        h = hashlib.sha256()
        h.update(files_hash.encode("utf-8"))
        h.update(child_a_hash.encode("utf-8"))
        h.update(child_b_hash.encode("utf-8"))
        weights_canonical = json.dumps(
            {s.value: round(w, 6) for s, w in sorted(self._weights.items())},
            sort_keys=True,
        )
        h.update(weights_canonical.encode("utf-8"))
        return h.hexdigest()[:8]

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"ensemble:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Phase 2 simplified fit: train both children on full span; static
        weights = 0.5. Phase 3 replaces this with the 4-stage weight-fitting
        flow."""
        seasons = sorted(int(s) for s in features["season"].unique())
        if len(seasons) < 2:
            raise ValueError(f"EnsembleModel.fit needs >=2 training seasons; got {len(seasons)}")

        self._child_a = self._config.child_a_factory()
        self._child_a.fit(features, weekly_stats)
        self._child_b = self._config.child_b_factory()
        self._child_b.fit(features, weekly_stats)

        self._weights = {stat: 0.5 for stat in self._config.target_stats}

        self._train_start = seasons[0]
        self._train_end = seasons[-1]
        self._calibration_year = seasons[-1]
        self._is_fitted = True

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-row composite fantasy-points distribution as the
        weighted mixture of A and C-NB per stat."""
        if not self._is_fitted or self._child_a is None or self._child_b is None:
            raise RuntimeError("predict_distribution requires fit() first")

        pred_a = self._child_a.predict_distribution(features, ruleset)
        pred_b = self._child_b.predict_distribution(features, ruleset)

        keys = ["gsis_id", "season", "week"]
        pred_a_idx = pred_a.set_index(keys, drop=False)
        pred_b_idx = pred_b.set_index(keys, drop=False)
        if not pred_a_idx.index.equals(pred_b_idx.index):
            raise RuntimeError(
                "child predictions misaligned — both children should predict on the same features"
            )

        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)

        for row_idx in range(len(pred_a_idx)):
            row_a = pred_a_idx.iloc[row_idx]
            row_b = pred_b_idx.iloc[row_idx]

            per_stat_a = unpack_per_stat_params(bytes(row_a["params"]))
            per_stat_b = unpack_per_stat_params(bytes(row_b["params"]))

            per_stat_dists: dict[Stat, Distribution] = {}
            for stat in self._config.target_stats:
                per_stat_dists[stat] = MixtureDistribution(
                    component_a=per_stat_a[stat],
                    component_b=per_stat_b[stat],
                    weight=self._weights[stat],
                )

            seed = derive_row_seed(
                gsis_id=str(row_a["gsis_id"]),
                season=int(row_a["season"]),
                week=int(row_a["week"]),
                ruleset_name=ruleset.name,
            )
            composite = score_distribution(per_stat_dists, ruleset, seed=seed)

            out_rows.append(
                {
                    "gsis_id": str(row_a["gsis_id"]),
                    "season": int(row_a["season"]),
                    "week": int(row_a["week"]),
                    "position": self._config.position.value,
                    "team": str(row_a["team"]),
                    "opponent": str(row_a["opponent"]),
                    "ruleset": ruleset.name,
                    "family": DistributionFamily.MIXED.value,
                    "params": pack_per_stat_params(per_stat_dists),
                    "mean": composite.mean(),
                    "p10": composite.quantile(0.10),
                    "p50": composite.quantile(0.50),
                    "p90": composite.quantile(0.90),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit("us"),
                }
            )

        out = pd.DataFrame(out_rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        out["position"] = out["position"].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "child_a": self._child_a,
                "child_b": self._child_b,
                "weights": {s.value: w for s, w in self._weights.items()},
                "train_start": self._train_start,
                "train_end": self._train_end,
                "calibration_year": self._calibration_year,
                "config_position": self._config.position.value,
                "config_target_stats": [s.value for s in self._config.target_stats],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> EnsembleModel:
        data = joblib.load(path)
        position = Position(data["config_position"])
        target_stats = tuple(Stat(s) for s in data["config_target_stats"])
        # Factories aren't serialized; we wire trivial passthroughs that return
        # the loaded children. Callers that re-fit must use the original
        # factory-based construction path.
        loaded_a: BaselineModel = data["child_a"]
        loaded_b: LightGBMNbModel = data["child_b"]
        config = _EnsembleConfig(
            position=position,
            target_stats=target_stats,
            child_a_factory=lambda: loaded_a,
            child_b_factory=lambda: loaded_b,
        )
        instance = cls(config=config)
        instance._child_a = loaded_a
        instance._child_b = loaded_b
        instance._weights = {Stat(k): float(v) for k, v in data["weights"].items()}
        instance._train_start = int(data["train_start"])
        instance._train_end = int(data["train_end"])
        instance._calibration_year = int(data["calibration_year"])
        instance._is_fitted = True
        return instance


def qb_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            child_a_factory=qb_baseline,
            child_b_factory=qb_lightgbm_nb,
        )
    )


def rb_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            child_a_factory=rb_baseline,
            child_b_factory=rb_lightgbm_nb,
        )
    )


def te_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            child_a_factory=te_baseline,
            child_b_factory=te_lightgbm_nb,
        )
    )


def wr_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            child_a_factory=wr_baseline,
            child_b_factory=wr_lightgbm_nb,
        )
    )
