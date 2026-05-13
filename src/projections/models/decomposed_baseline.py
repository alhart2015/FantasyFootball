"""DecomposedBaselineModel — per-stat volume x efficiency decomposition.

Subclass of ``BaselineModel`` with a constructor argument
``decomposed_stats: Mapping[Stat, DecompositionSpec]`` mapping composite stats
to their (volume, efficiency) decomposition recipe. Stats absent from
``decomposed_stats`` fall through to the inherited direct-RidgeCV path.

Per-row prediction for decomposed stats samples volume x efficiency factors
with within-row coherent sampling (a single shared volume draw flows into
every decomposed stat with the same ``volume_stat``). Persistence uses
``QuantileDistribution`` summaries via the existing codec branch — no codec
edits required.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.models.baseline import BaselineModel
from projections.schemas import Stat, WeeklyStatsSchema


@dataclass(frozen=True, slots=True)
class DecompositionSpec:
    """Per-stat decomposition recipe.

    Attributes:
        volume_stat: the volume axis (e.g., ``Stat.TARGETS``). Multiple
            decomposed stats sharing the same ``volume_stat`` get a shared
            per-row volume draw at predict time.
        efficiency_label: human-readable label of the efficiency factor
            (e.g., ``"catch_rate"``, ``"yards_per_target"``). Used in logs
            and diagnostics.
        efficiency_clip_hi: upper bound for sample-time efficiency clipping.
            ``1.0`` for ratio efficiency factors (catch_rate, td_rate);
            ``float("inf")`` for unbounded efficiency factors (yards_per_target).
    """

    volume_stat: Stat
    efficiency_label: str
    efficiency_clip_hi: float


@dataclass
class DecomposedBaselineModel(BaselineModel):
    """BaselineModel + per-stat decomposition.

    See module docstring for the architectural overview.

    Per-stat decomposition opt-in via ``decomposed_stats``. Stats not in the
    mapping fall through to the inherited direct-RidgeCV path. Stats in the
    mapping are predicted as ``mu_volume * mu_efficiency`` at the mean level
    and as ``volume_samples * efficiency_samples`` at the distribution level
    (with the volume samples shared across all decomposed stats with the same
    ``volume_stat``, baking within-row cross-stat correlation into the per-row
    sample arrays).
    """

    decomposed_stats: Mapping[Stat, DecompositionSpec] = field(default_factory=dict)
    volume_ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    efficiency_ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    volume_variance: dict[Stat, float] = field(default_factory=dict)
    efficiency_variance: dict[Stat, float] = field(default_factory=dict)

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train direct ridges (parent) + decomposition sub-models.

        Parent ``BaselineModel.fit`` handles schema validation, the
        ``(gsis_id, season, week)`` inner-join, NaN-drop, median-imputation, and
        per-stat direct ridges. We extend it with the volume + efficiency
        sub-models for each entry in ``decomposed_stats``.
        """
        # Run parent fit first — populates self.ridges, self.feature_means,
        # self.variance_params, self.train_seasons, self.code_hash.
        super().fit(features, weekly_stats)

        if not self.decomposed_stats:
            return

        # Re-build the same train frame parent used. We need the joined truth
        # rows (post-NaN-drop) to fit volume + efficiency ridges against.
        features_validated = self.feature_schema.validate(features)
        weekly_validated = WeeklyStatsSchema.validate(weekly_stats)
        ws = weekly_validated[weekly_validated["position"] == self.position.value].copy()
        truth_cols = (
            ["gsis_id", "season", "week"]
            + [s.value for s in self.target_stats]
            + [spec.volume_stat.value for spec in self.decomposed_stats.values()]
        )
        # Dedupe in case a volume_stat is already in target_stats (it is for
        # WR receptions/yards/tds — volume_stat is TARGETS which is NOT in
        # _WR_TARGET_STATS, but defensive dedupe keeps the schema-select fast).
        truth_cols = list(dict.fromkeys(truth_cols))
        joined = features_validated.merge(
            ws[truth_cols],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        feature_frame = self._x_frame_with_bool_coercion(joined)
        # Apply the same NaN-drop the parent applied. ``self.feature_means``
        # is populated by parent fit on the pre-dropna frame, so post-dropna
        # alignment is implicit.
        keep_mask = feature_frame.notna().all(axis=1)
        feature_frame = feature_frame.loc[keep_mask]
        joined_kept = joined.loc[feature_frame.index]
        x_train = feature_frame.to_numpy(dtype=np.float64)

        alphas = np.logspace(-3, 3, 13)

        # Fit shared volume sub-models (one per unique volume_stat).
        unique_volume_stats = {spec.volume_stat for spec in self.decomposed_stats.values()}
        for volume_stat in unique_volume_stats:
            y_vol = joined_kept[volume_stat.value].to_numpy(dtype=np.float64)
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(x_train, y_vol)
            self.volume_ridges[volume_stat] = ridge
            mu_vol: np.ndarray = ridge.predict(x_train).astype(np.float64)
            residuals_vol = y_vol - mu_vol
            self.volume_variance[volume_stat] = max(float(residuals_vol.std()), 1e-6)

        # Fit per-stat efficiency sub-models on rows with volume_stat > 0.
        for composite_stat, spec in self.decomposed_stats.items():
            vol_col = joined_kept[spec.volume_stat.value].to_numpy(dtype=np.float64)
            mask = vol_col > 0
            if not mask.any():
                raise ValueError(
                    f"Cannot fit efficiency factor for {composite_stat.value}: "
                    f"no training rows with {spec.volume_stat.value} > 0."
                )
            x_pos = x_train[mask]
            vol_pos = vol_col[mask]
            num_pos = joined_kept.loc[mask, composite_stat.value].to_numpy(dtype=np.float64)
            ratio = num_pos / vol_pos
            ridge_eff = RidgeCV(alphas=alphas)
            ridge_eff.fit(x_pos, ratio)
            self.efficiency_ridges[composite_stat] = ridge_eff
            mu_eff: np.ndarray = ridge_eff.predict(x_pos).astype(np.float64)
            residuals_eff = ratio - mu_eff
            self.efficiency_variance[composite_stat] = max(float(residuals_eff.std()), 1e-6)

    @property
    def model_id(self) -> str:
        """Stable identifier of the form
        ``"decomposed-baseline:<position>:<8-char-code-hash>:<train-start>-<train-end>"``.

        Mirrors ``BaselineModel.model_id`` except for the class-name prefix.
        """
        if self.code_hash is None or self.train_seasons is None:
            raise RuntimeError("model_id is undefined for unfitted models")
        return (
            f"decomposed-baseline:{self.position.value.lower()}:{self.code_hash}"
            f":{self.train_seasons[0]}-{self.train_seasons[1]}"
        )
