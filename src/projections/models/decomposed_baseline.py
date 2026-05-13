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
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.distributions import Distribution, FrozenSampledDistribution, QuantileDistribution
from projections.models.baseline import _RIDGE_ALPHA_GRID, BaselineModel
from projections.schemas import Stat, WeeklyStatsSchema
from projections.scoring.score_distribution import derive_row_seed

_N_SAMPLES: Final[int] = 10_000

# Persisted quantile grid: 19 knots at q in {0.05, 0.10, ..., 0.95}. Per-row
# persistence cost is 19 floats per decomposed stat — small relative to the
# existing per-stat parametric encoding. QuantileDistribution recomposes via
# linear interpolation between knots.
_PERSISTED_QUANTILES: Final[np.ndarray] = np.arange(0.05, 0.96, 0.05)


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

        alphas = _RIDGE_ALPHA_GRID

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

    def _persistable_dists_for_packing(
        self, stat_dists: Mapping[Stat, Distribution]
    ) -> Mapping[Stat, Distribution]:
        """Convert FrozenSampledDistribution entries (decomposed stats) into
        QuantileDistribution summaries for persistence via the existing
        QUANTILE codec branch.

        The cross-stat correlation baked into the FrozenSampledDistribution's
        sample array is lost at the persistence boundary — see spec §3.1.5 +
        §5 risk #4. This is acceptable for v1 (no post-hoc re-scoring from
        persisted params blobs). The scoring step has already consumed the
        live in-memory FrozenSampledDistributions via score_distribution
        upstream; this conversion is downstream of that.

        Downstream consumers that re-score from the persisted blob
        (e.g., ``aggregate_to_season`` in projections.aggregation.season)
        get per-stat QuantileDistribution samples drawn independently; the
        cross-stat correlation from the shared volume axis is not recoverable.
        Season-level variance for decomposed stats may therefore be slightly
        under-estimated relative to the live-weekly variance.
        """
        out: dict[Stat, Distribution] = {}
        for stat, dist in stat_dists.items():
            if isinstance(dist, FrozenSampledDistribution):
                values = np.quantile(dist.samples, _PERSISTED_QUANTILES).astype(np.float64)
                # QuantileDistribution requires values to be non-decreasing
                # (validated in its __init__); np.quantile already produces
                # sorted output for ascending quantiles.
                out[stat] = QuantileDistribution(
                    quantiles=_PERSISTED_QUANTILES.copy(),
                    values=values,
                )
            else:
                out[stat] = dist
        return out

    def build_stat_distributions(self, features: pd.DataFrame) -> list[dict[Stat, Distribution]]:
        """Per-row dicts of {Stat -> Distribution}.

        Non-decomposed stats: inherited parametric path (Normal/Gamma/NB per
        ``dist_families``).
        Decomposed stats: per-row FrozenSampledDistribution with 10_000 samples.

        Within-row cross-stat coherence: all decomposed stats sharing a
        ``volume_stat`` reuse the same per-row volume draw, baking element-wise
        correlation into their composed sample arrays. ``score_distribution``
        consumes the FrozenSampledDistributions via ``.sample(n=10_000)``;
        the ``n == len`` branch returns the underlying arrays verbatim,
        preserving the correlation through scoring.
        """
        # Parent path emits parametric distributions for all target_stats --
        # including the decomposed ones, which we will overwrite below.
        per_row_parametric = super().build_stat_distributions(features)

        if not self.decomposed_stats:
            return per_row_parametric

        # Feature matrix (impute with train medians; bool -> int8).
        # self.feature_means is guaranteed non-None here: super().build_stat_distributions
        # (called above) already raises if the model is not fitted.
        x_frame = self._x_frame_with_bool_coercion(features)
        assert self.feature_means is not None, "feature_means is None after parent guard"
        x_frame = x_frame.fillna(self.feature_means)
        x = x_frame.to_numpy(dtype=np.float64)

        # Vectorized volume + efficiency predictions.
        unique_volume_stats = {spec.volume_stat for spec in self.decomposed_stats.values()}
        per_volume_mu: dict[Stat, np.ndarray] = {}
        for vs in unique_volume_stats:
            mu_v: np.ndarray = self.volume_ridges[vs].predict(x).astype(np.float64)
            per_volume_mu[vs] = mu_v

        per_decomposed_mu_eff: dict[Stat, np.ndarray] = {}
        for cs in self.decomposed_stats:
            mu_e: np.ndarray = self.efficiency_ridges[cs].predict(x).astype(np.float64)
            per_decomposed_mu_eff[cs] = mu_e

        # Sort volume_stats by Stat.value for stable seed assignment regardless
        # of decomposed_stats insertion order.
        sorted_volume_stats = sorted(per_volume_mu.keys(), key=lambda s: s.value)
        n_vol = len(sorted_volume_stats)
        # Sort decomposed_stats by composite_stat.value for stable seed assignment.
        sorted_decomposed = sorted(self.decomposed_stats.items(), key=lambda kv: kv[0].value)

        # Per-row coherent sampling.
        features_iter = features.reset_index(drop=True)
        for i in range(len(features_iter)):
            feat_row = features_iter.iloc[i]
            row_seed = derive_row_seed(
                gsis_id=str(feat_row["gsis_id"]),
                season=int(feat_row["season"]),
                week=int(feat_row["week"]),
                ruleset_name="__decomp_build__",
            )
            # Per-row volume draws. Each unique volume_stat gets a distinct
            # seed offset (sorted-index order) so multi-volume-stat configs
            # don't produce artificially-correlated volume draws.
            # Seeds: row_seed + 0, row_seed + 1, ..., row_seed + n_vol-1.
            vol_samples: dict[Stat, np.ndarray] = {}
            for vol_idx, vs in enumerate(sorted_volume_stats):
                sigma_v = self.volume_variance[vs]
                rng_v = np.random.default_rng(row_seed + vol_idx)
                vs_raw = rng_v.normal(
                    loc=float(per_volume_mu[vs][i]), scale=sigma_v, size=_N_SAMPLES
                )
                vol_samples[vs] = np.maximum(vs_raw, 0.0)

            # Per-stat efficiency draws + composition. Efficiency seeds start
            # at row_seed + n_vol to avoid overlap with the volume seed range.
            # Seeds: row_seed + n_vol + 0, ..., row_seed + n_vol + K-1.
            for eff_idx, (composite_stat, spec) in enumerate(sorted_decomposed):
                sigma_e = self.efficiency_variance[composite_stat]
                rng_e = np.random.default_rng(row_seed + n_vol + eff_idx)
                eff_raw = rng_e.normal(
                    loc=float(per_decomposed_mu_eff[composite_stat][i]),
                    scale=sigma_e,
                    size=_N_SAMPLES,
                )
                eff_samples = np.clip(eff_raw, 0.0, spec.efficiency_clip_hi)
                composed = vol_samples[spec.volume_stat] * eff_samples
                # Replace the parametric entry with the live FrozenSampled.
                per_row_parametric[i][composite_stat] = FrozenSampledDistribution(samples=composed)

        return per_row_parametric
