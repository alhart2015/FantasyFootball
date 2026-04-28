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
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from projections.distributions import (
    MixtureDistribution,
    pack_per_stat_params,
    unpack_per_stat_params,
)
from projections.distributions.base import Distribution
from projections.distributions.mixture import (
    _bracket_for_components,
    _quantile_with_bracket,
)
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
_QUANTILES_FOR_FIT: Final[tuple[float, float]] = (0.10, 0.90)
_WEIGHT_BOUNDS: Final[tuple[float, float]] = (0.001, 0.999)


def _pinball(actual: float, q_pred: float, q: float) -> float:
    """Standard quantile pinball loss.

    pinball(y, q_pred, q) = max(q * (y - q_pred), (q - 1) * (y - q_pred))

    Equivalently: q * (y - q_pred) when y >= q_pred (under-estimate),
                  (1 - q) * (q_pred - y) when y < q_pred (over-estimate).
    """
    diff = actual - q_pred
    return float(max(q * diff, (q - 1.0) * diff))


def _fit_weight_for_stat(
    *,
    components_a: Sequence[Distribution],
    components_b: Sequence[Distribution],
    actuals: NDArray[np.float64],
) -> float:
    """Fit one scalar weight w in (0.001, 0.999) minimizing summed pinball
    loss at q in {0.10, 0.90} on the per-row mixture distribution.

    components_a[i], components_b[i], actuals[i] correspond to the same row
    on the calibration year. len(components_a) == len(components_b) ==
    len(actuals) is required.

    Returns 0.5 with RuntimeWarning on zero-length input. Raises ValueError
    on length mismatch.

    Uses scipy.optimize.minimize_scalar (bounded brent). Falls back to a
    coarse 11-point grid search if scipy fails (non-finite loss, optimizer
    failure). If the grid also produces only non-finite losses, returns
    0.5 with RuntimeWarning.
    """
    n = len(actuals)
    if not (n == len(components_a) == len(components_b)):
        raise ValueError(
            f"length mismatch: components_a={len(components_a)}, "
            f"components_b={len(components_b)}, actuals={n}"
        )
    if n == 0:
        warnings.warn(
            "_fit_weight_for_stat received zero-length input; returning 0.5 default",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.5

    # Pre-compute per-row brackets — invariant in w. Saves ~4 component-quantile
    # calls per row per outer brent iteration; with ~50 outer iterations x N
    # rows x 2 quantiles, this is the dominant cost in the loss closure.
    brackets: list[tuple[float, float]] = [
        _bracket_for_components(a, b) for a, b in zip(components_a, components_b, strict=True)
    ]

    def loss(w: float) -> float:
        total = 0.0
        for a_dist, b_dist, actual, (lo, hi) in zip(
            components_a, components_b, actuals, brackets, strict=True
        ):
            mix = MixtureDistribution(component_a=a_dist, component_b=b_dist, weight=w)
            for q in _QUANTILES_FOR_FIT:
                q_pred = _quantile_with_bracket(mix, q, lo, hi)
                total += _pinball(float(actual), q_pred, q)
        return total

    try:
        result = minimize_scalar(
            loss,
            method="bounded",
            bounds=_WEIGHT_BOUNDS,
            options={"xatol": 1e-3},
        )
        if result.success and np.isfinite(result.fun):
            return float(np.clip(result.x, *_WEIGHT_BOUNDS))
    except (ValueError, OverflowError) as exc:
        warnings.warn(
            f"_fit_weight_for_stat: scipy optimization failed ({exc!r}); falling back to grid",
            RuntimeWarning,
            stacklevel=2,
        )

    # Grid-search fallback.
    grid = np.linspace(_WEIGHT_BOUNDS[0], _WEIGHT_BOUNDS[1], 11)
    losses: list[float] = []
    for w in grid:
        try:
            losses.append(loss(float(w)))
        except (ValueError, OverflowError):
            losses.append(float("inf"))
    losses_arr = np.asarray(losses)
    if not np.any(np.isfinite(losses_arr)):
        warnings.warn(
            "_fit_weight_for_stat: all grid points returned non-finite loss; returning 0.5 default",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.5
    return float(grid[int(np.argmin(losses_arr))])


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
    # Forward-compat: directory where the Phase 3 pinball-loss optimizer
    # will persist per-(position, stat) weight blobs. Unused in Phase 2
    # (static 0.5 weights); wired now to avoid a config-shape change in
    # the next phase.
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
        # unused in Phase 2; Phase 3 weight optimizer uses it as the Y-1 calibration year.
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
        weighted mixture of A and C-NB per stat.

        Per-row codec round-trip note: each row's params blob from each
        child is unpacked back into a {Stat -> Distribution} dict, wrapped
        in MixtureDistribution per stat, and re-packed for output. This
        double pack/unpack is the only Distribution-Protocol-clean way to
        consume two children that respect the Model contract; bypassing it
        (e.g., by exposing children's internal stat-dist structures
        directly) would require new protocol surface and isn't worth it
        for the predict-time cost.
        """
        if not self._is_fitted or self._child_a is None or self._child_b is None:
            raise RuntimeError("predict_distribution requires fit() first")

        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        pred_a = self._child_a.predict_distribution(features, ruleset)
        pred_b = self._child_b.predict_distribution(features, ruleset)

        keys = ["gsis_id", "season", "week"]
        pred_a_idx = pred_a.set_index(keys, drop=False)
        pred_b_idx = pred_b.set_index(keys, drop=False)
        if not pred_a_idx.index.equals(pred_b_idx.index):
            raise RuntimeError(
                "child predictions misaligned — both children should predict on the same features"
            )

        # Extract column arrays once outside the loop. Both children predicted
        # on the same features, so {gsis_id, season, week, team, opponent} are
        # identical across pred_a and pred_b — read from pred_a only. params
        # blobs differ per child (each child's own per-stat distribution
        # encoding) so we read both. Mirrors LightGBMNbModel.predict_distribution.
        gsis_id_col = pred_a_idx["gsis_id"].to_numpy()
        season_col = pred_a_idx["season"].to_numpy()
        week_col = pred_a_idx["week"].to_numpy()
        team_col = pred_a_idx["team"].to_numpy()
        opponent_col = pred_a_idx["opponent"].to_numpy()
        params_a_col = pred_a_idx["params"].to_numpy()
        params_b_col = pred_b_idx["params"].to_numpy()

        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)
        n_rows = len(pred_a_idx)

        for row_idx in range(n_rows):
            per_stat_a = unpack_per_stat_params(bytes(params_a_col[row_idx]))
            per_stat_b = unpack_per_stat_params(bytes(params_b_col[row_idx]))

            per_stat_dists: dict[Stat, Distribution] = {}
            for stat in self._config.target_stats:
                per_stat_dists[stat] = MixtureDistribution(
                    component_a=per_stat_a[stat],
                    component_b=per_stat_b[stat],
                    weight=self._weights[stat],
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
