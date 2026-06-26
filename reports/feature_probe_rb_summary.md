# RB feature-lift signal probe — G1 verdict (#55)

**Date:** 2026-06-25. Eval years 2021–2024, RB only. Probe:
`scripts/probe_feature_signal` (augment mode, per-stat Ridge Δ-CV-RMSE,
`effect_size_floor=0.05`). Plan: `docs/superpowers/plans/2026-06-25-rb-trajectory-vegas-features.md`.

## Measured per-column coverage (over 7580 RB baseline rows, 2021–2024)

| family | column | non-null frac |
|---|---|---:|
| trajectory | age | 0.969 |
| trajectory | is_rookie | 0.969 |
| trajectory | volume_trend_l4_minus_prior_l4 | **0.601** |
| trajectory | snap_pct_change_l4_vs_prior_l4 | 0.745 |
| vegas | preseason_implied_team_total | 1.000 |
| vegas | preseason_spread | 1.000 |
| vegas | season_avg_implied_team_total | 0.942 |
| vegas | season_avg_spread | 0.942 |

Non-feature columns (`draft_year_inferred`, `position`) were stripped before probing.
The trajectory family was **split** into a clean core (age/is_rookie) and the sparse
trend pair, probed separately at coverage thresholds 0.90 and 0.55 respectively; Vegas at
0.90. (Mean-imputing the trend pair's ~40%/26% NaN biases the Ridge screen *toward* NULL,
so a SIGNAL there is conservative — real, not an imputation artifact.)

## Per-family Phase-1 verdicts

| candidate | columns | threshold | Phase-1 cells | family verdict |
|---|---|---:|---|:---:|
| augment_rb_traj_core | age, is_rookie | 0.90 | 0/30 SIGNAL, 29 NULL, 1 REGRESSION | **NULL** |
| augment_rb_traj_trend | volume_trend_l4_minus_prior_l4, snap_pct_change_l4_vs_prior_l4 | 0.55 | 3/30 SIGNAL (rushing_yards 2022, 2024, pooled) | **SIGNAL** |
| augment_rb_vegas | preseason_*, season_avg_* | 0.90 | 0/30 SIGNAL, 30 NULL | **NULL** |

### The signaling cells — `rushing_yards` Δ-RMSE (negative = improvement)

| year | point | 95% CI | n_paired | verdict |
|---|---:|---|---:|:---:|
| 2021 | −0.030 | [−0.255, +0.191] | 857 | NULL |
| 2022 | −0.325 | [−0.549, −0.089] | 597 | **SIGNAL** |
| 2023 | −0.167 | [−0.363, +0.025] | 765 | NULL |
| 2024 | −0.275 | [−0.480, −0.082] | 884 | **SIGNAL** |
| **pooled** | **−0.173** | **[−0.284, −0.060]** | 3103 | **SIGNAL** |

Direction is consistent across all four years (all improvements); two years and the pooled
estimate are CI-separated from 0. Robust, not a one-year fluke. No other stat (rushing_tds,
receptions, receiving_yards/tds, fumbles_lost) signals — the effect is specifically the
volume/snap **trend** features improving **rushing-yards** prediction, which is mechanistically
sensible (recent carry-volume trend captures committee/role shifts the static L4 means miss).

## G1 verdict → **SIGNAL** (trajectory trend) → Phase 2

- **Proceed to Phase 2** integrating **only** `volume_trend_l4_minus_prior_l4` and
  `snap_pct_change_l4_vs_prior_l4` (the signaling columns).
- **Drop** `age`/`is_rookie` (NULL) and the entire **Vegas** family (NULL) — shipping them
  would be dead columns (spec §6/§8 "no dead columns").
- This refines the plan's assumption (all 4 trajectory cols): the data says the lift, if any,
  is the trend pair. Phase 2's dual-run adoption gate (G2) remains the ship decision.

(Contradicts the spec §2 prior that both families likely NULL — trajectory's trend signal is real.)

## G2 verdict (Phase 2) — **DO_NOT_ADOPT** → STOP, integration reverted

After integrating the two trend columns into `RbFeaturesSchema` + `build_rb_features` +
`_RB_FEATURE_COLUMNS`, the model bake-off on the augmented cache (`data/features_rb_aug`,
`reports/rb_model_bakeoff_augmented.md`) vs the old baseline (Phase 0: rmse 6.5661, mae 4.9870,
spearman 0.9694):

| augmented pooled | baseline | ensemble | lightgbm-nb |
|---|---:|---:|---:|
| composite_rmse | 6.5630 | 6.5729 | 6.6043 |
| composite_mae | 4.9594 | 5.0910 | 5.1516 |
| spearman_topN | 0.9709 | 0.9694 | 0.9680 |

**Ship decision (baseline):** the probe Phase-2 composite paired ΔRMSE = −0.0044, 95% CI
[−0.0156, +0.0061] **brackets zero** → fails the adoption-gate `hi_95 < 0` criterion
(DO_NOT_ADOPT). The absolute bake-off agrees — augmented baseline improves composite RMSE by
only −0.0031 (0.05%, sub-noise, fantasy-irrelevant); mae −0.028, spearman +0.0015 are likewise
negligible. (Methodology: for the baseline = Ridge class, the probe's Phase-2 composite is the
adoption-gate computation; combined with the absolute bake-off it is the G2 ship decision. The
production `BaselineModel.fit` `dropna` path — which the bake-off exercises — additionally drops
~40% of RB training rows for the 40%-NaN trend column, so the production effect is if anything
weaker than the imputed probe.)

**Default decision:** lightgbm-nb (6.6043) and ensemble (6.5729) on the augmented features
**still lose to the old baseline (6.5661)** — no non-baseline class beats baseline, so no
`default_model_class` flip is warranted.

**Mechanism:** the trend columns carry a real *rushing-yards* RMSE improvement (pooled −0.173,
CI [−0.284, −0.060]) but it does **not** survive into composite fantasy points (the other five
RB stats NULL'd, diluting it), and the dropna training-row penalty offsets the rest.

**Action:** revert the integration (commits 44bfb5f, 4fb2f3e, 4bc783c) — no dead columns shipped
(spec §6/§8). RB stays on `baseline`. The DFS STOP verdict (#54/#55) **stands for RB**: even after
exhausting the two unprobed feature families (Vegas NULL; trajectory core NULL; trajectory trend
SIGNAL on rushing_yards but DO_NOT_ADOPT on composite), RB is not liftable with the available signal.
Kept: the CLI passthroughs (`--features-root`/`--position`) and all probe/bake-off evidence.
