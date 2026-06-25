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
