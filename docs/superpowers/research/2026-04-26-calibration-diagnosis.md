# Calibration Diagnosis (Plan 3e Phase 0)

**Date:** 2026-04-26
**Author:** alden + claude
**Inputs:** `data/backtest/run_20260426T212154Z/results.parquet`, `data/diagnostics/calibration_20260426T212154Z/`
**Spec:** `docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md`

## Executive summary

The under-dispersion has three distinct root causes, each concentrated in a different family of cells. **(1) Count stats with heavy zero inflation are catastrophically miscalibrated** — every TD and `fumbles_lost` cell across all four positions reports `coverage_p10p90 = 0.0` because the assumed GAMMA family has zero density at zero, while >90% of actual values are zero; KS rejects the assumed family with p ≈ 0 everywhere. **(2) Continuous "yards"-type stats are mildly under-dispersed and have heavy tails** — `passing_yards`, `rushing_yards`, and `receiving_yards` for the skill positions sit at coverage 0.81–0.93 (close to the 0.80 target on the low side) but residuals are positively skewed with kurtosis 0.08–4.9, and AIC prefers Student-t over Normal by 317–2160 points across all five non-degenerate cells. **(3) Heteroscedasticity is pervasive** — 18 of 24 cells have `heteroscedasticity_ratio > 1.5`, with high-prediction tertiles showing 1.8–2.7× the residual std of low-prediction tertiles, indicating that constant per-stat sigma is the wrong scale model. The recommended Phase 1+ direction is to **(a) replace GAMMA with a zero-inflated count family (negative binomial or ZIP) for all TD / fumbles_lost / receptions stats, (b) replace NORMAL with Student-t for `*_yards` stats, and (c) introduce prediction-magnitude buckets for sigma estimation so that the high-pred tertile doesn't bleed under-dispersion into the low-pred tertile and vice versa.**

## Per-(position, stat) findings

| position | stat | n | coverage_p10p90 | heteroscedasticity_ratio | ks_assumed_pvalue | best_alt_family | aic_delta | recommended_fix |
|---|---|---|---|---|---|---|---|---|
| QB | fumbles_lost | 2676 | 0.000 | 1.319 | 0.000 | neg_binomial | NaN | no_change |
| QB | interceptions | 2676 | 0.305 | 1.229 | 0.000 | neg_binomial | NaN | no_change |
| QB | passing_tds | 2676 | 0.528 | 1.397 | 4.4e-260 | neg_binomial | NaN | no_change |
| QB | passing_yards | 2676 | 0.812 | 0.744 | 0.059 | student_t | -1175.013 | no_change |
| QB | rushing_tds | 2676 | 0.021 | 1.962 | 0.000 | neg_binomial | NaN | variance_bucket |
| QB | rushing_yards | 2676 | 0.848 | 2.337 | 2.3e-32 | student_t | -317.224 | variance_bucket |
| RB | fumbles_lost | 5273 | 0.000 | 2.118 | 0.000 | neg_binomial | NaN | variance_bucket |
| RB | receiving_tds | 5273 | 0.000 | 1.996 | 0.000 | neg_binomial | NaN | variance_bucket |
| RB | receiving_yards | 5273 | 0.888 | 1.778 | 2.2e-86 | none | NaN | no_change |
| RB | receptions | 5273 | 0.597 | 1.657 | 0.000 | neg_binomial | NaN | variance_bucket |
| RB | rushing_tds | 5273 | 0.104 | 2.241 | 0.000 | neg_binomial | NaN | variance_bucket |
| RB | rushing_yards | 5273 | 0.879 | 1.763 | 2.0e-60 | student_t | -1817.207 | variance_bucket |
| TE | fumbles_lost | 4257 | 0.000 | 1.116 | 0.000 | neg_binomial | NaN | no_change |
| TE | receiving_tds | 4257 | 0.020 | 1.678 | 0.000 | neg_binomial | NaN | variance_bucket |
| TE | receiving_yards | 4257 | 0.876 | 2.021 | 6.8e-41 | student_t | -692.860 | variance_bucket |
| TE | receptions | 4257 | 0.700 | 1.930 | 8.5e-48 | neg_binomial | NaN | variance_bucket |
| TE | rushing_tds | 4257 | 0.000 | 0.267 | 0.000 | neg_binomial | NaN | no_change |
| TE | rushing_yards | 4257 | 0.984 | 6.017 | 0.000 | none | NaN | no_change |
| WR | fumbles_lost | 8460 | 0.000 | 1.545 | 0.000 | neg_binomial | NaN | variance_bucket |
| WR | receiving_tds | 8460 | 0.062 | 1.763 | 0.000 | neg_binomial | NaN | variance_bucket |
| WR | receiving_yards | 8460 | 0.864 | 1.794 | 1.3e-60 | student_t | -2160.136 | variance_bucket |
| WR | receptions | 8460 | 0.696 | 1.754 | 2.0e-91 | neg_binomial | NaN | variance_bucket |
| WR | rushing_tds | 8460 | 0.000 | 2.243 | 0.000 | neg_binomial | NaN | variance_bucket |
| WR | rushing_yards | 8460 | 0.928 | 2.675 | 0.000 | none | NaN | no_change |

(Source: `data/diagnostics/calibration_20260426T212154Z/summary.parquet`. `aic_delta = best_alt_aic − assumed_aic`; negative means the alternative wins. `aic_delta` is NaN for cells where the alternative fitter returned `none` — typically because the count fitter produced numerical failure or the assumed family was already GAMMA and only the marginal fits were attempted.)

## Recommended-fix matrix

The CLI's mechanical `recommended_fix` column is dominated by `variance_bucket` (14 cells) and `no_change` (10 cells), but reading the underlying signals tells a more useful story. I group the cells by the *substantive* fix they actually need, overriding the mechanical recommendation per the caveat about conditional-vs-marginal AIC asymmetry:

**Group A — Family swap to a zero-inflated count distribution (10 cells).** Every `*_tds` and `fumbles_lost` cell across all four positions. Symptoms: `coverage_p10p90 ∈ [0.0, 0.53]` (mostly 0.0), `ks_assumed_pvalue ≈ 0` everywhere, `residual_skew > 1.7` and excess kurtosis 4–1008. The mechanical CLI labels half of these `no_change` because no alt fit converged with a positive aic_delta — but the coverage signal alone makes the fix obvious. These are integer-valued, mostly-zero, occasionally-large stats; GAMMA cannot represent a point mass at zero. **Phase 1+ must add a discrete count family (negative binomial preferred for overdispersion; consider zero-inflated NB if the zero-fraction exceeds what NB can reach with a low mean parameter).**

**Group B — Family swap to Student-t (5 cells).** `QB rushing_yards`, `RB rushing_yards`, `TE receiving_yards`, `WR receiving_yards`, and marginally `QB passing_yards`. Symptoms: AIC strongly prefers Student-t (`aic_delta ∈ [−2160, −317]`), KS rejects normal at p < 1e-30 (except passing_yards at p=0.059). Coverage is in a healthy 0.81–0.89 band, so the under-dispersion isn't catastrophic — but the heavy tails are real. `RB receiving_yards` and `WR rushing_yards` would also benefit from Student-t but the marginal alt-fit returned `none` for those (likely a numerical edge case in `_fit_student_t`); the residual_skew of 1.66 and 5.18 respectively, plus excess kurtosis of 4.5 and 45.8, points the same direction.

**Group C — Variance bucketing genuinely needed (most cells).** 18 of 24 cells have `heteroscedasticity_ratio > 1.5`, and the high-pred tertile carries 1.5–2.7× the std of the low-pred tertile across the board. Even the cells that get the family fix in groups A/B will still benefit from prediction-magnitude bucketing. This is independent of the family choice. **Two surprises:** (i) `QB passing_yards` has heteroscedasticity_ratio = **0.74** — high-pred QBs are *more* consistent, not less, so for this single cell bucketing would *help* by raising sigma on the low-pred tail rather than the high-pred tail; (ii) `TE rushing_yards` has heteroscedasticity_ratio = 6.02 but coverage = 0.984, meaning the cell is so degenerate (mean prediction 0.10 yards, mean actual 0.13 yards, residual_std 2.36 with kurtosis 2618) that any parametric fix is essentially modeling noise around zero. We should explicitly carve out a "degenerate-stat" rule: when `mean_pred < 1.0` and the actual support is dominated by zero, route to the zero-inflated count family with a tiny mean parameter and skip variance bucketing.

**Group D — No fix needed (essentially 1 cell).** `QB passing_yards` is the closest thing the diagnostic has to a calibrated cell: coverage 0.812 (right at target), KS p=0.059 (won't reject at α=0.05), residual_std of 83 yards with low skew (0.29) and near-zero excess kurtosis (0.08). The student-t aic_delta of −1175 says the heavy tails would *still* fit better, and the heteroscedasticity ratio of 0.74 says the variance structure is mildly inverted, but neither is a calibration-blocker. Treat it as the validation reference for the rest of the work — if a Phase 1+ change makes passing_yards worse, something is off.

## What this implies for Phase 1+

**Family swaps are the highest-leverage change.** The single biggest source of under-dispersion (and the only source of the `coverage = 0.0` rows) is using GAMMA for stats that are predominantly zero with occasional integer values. Phase 1 should land the discrete count family (`NegativeBinomial` and/or `ZeroInflatedNegativeBinomial`) in the `DistributionFamily` enum + `src/projections/distributions/`, and the model layer should route every `*_tds` and `fumbles_lost` stat to it. Student-t should land in the same phase — it's the obvious right call for the five `*_yards` cells with strong AIC signals, and it's the natural generalization of Normal so the row-parameterization story (loc, scale, df) stays familiar. We will need to estimate `df` either per-stat globally (cheap, conservative) or per-prediction-bucket (couples with the bucketing work below).

**Bucketed sigma estimation should follow.** Once the family is right, the next-largest under-dispersion source is constant per-stat sigma. The diagnostic shows the high-pred tertile carries 1.5–2.7× the variance of the low-pred tertile in essentially every continuous-yardage cell. Phase 2 should introduce a small number of prediction-magnitude buckets (3–5 quantile-based buckets per stat) and estimate sigma within each bucket. This is the spec's `variance_bucket` fix and should net us coverage gains independent of the family choice. It also resolves the `QB passing_yards` inverted-heteroscedasticity case: by estimating sigma per-bucket rather than globally, we stop forcing a single sigma to compromise between consistent high-pred QBs and noisier low-pred QBs.

**Cross-week correlation is plausibly out of 3e's scope.** The diagnostic is strictly per-(position, stat, row); it cannot see whether the season-level coverage gap (which Plan 3d's harness reports independently) is partly driven by within-player serial correlation. The per-row coverage signals here are mostly explained by the family-fit and variance-bucket stories above, so Phase 1+2 should close most of the per-row gap. If the season-level coverage gap remains substantial after Phase 2 lands, the residual is most likely cross-week correlation in the player-time dimension (e.g., a player who underperforms his projection in week N also tends to underperform in weeks N+1, N+2 because of injury, role change, or scheme drift) — and that is genuinely a different modeling problem (joint distribution / copula / hierarchical model), not a marginal-distribution fix. Recommend explicitly scoping Plan 3e to per-row marginal calibration and deferring the season-correlation work to a follow-up plan if the gap persists.

## Caveats

- **Marginal vs conditional AIC asymmetry:** the assumed-family AIC is computed under conditional log-likelihood (`norm.logpdf(actual, loc=pred, scale=sigma)` for NORMAL; `gamma.logpdf(actual, a=alpha, scale=mu/alpha)` for GAMMA). The alternative families (`_fit_student_t`, `_fit_log_normal`, `_fit_neg_binomial`) are fit MARGINALLY on `actual` alone, with `pred` discarded. As long as `pred` carries any signal, the assumed family will tend to win on log-likelihood. The `aic_delta >= 5` threshold for `family_swap` will therefore rarely fire on this diagnostic; coverage_p10p90 and ks_assumed_pvalue are the orthogonal signals that don't suffer this asymmetry. The report-writing pass should override the numeric `recommended_fix` recommendation where coverage and KS p-value disagree.
- **Coverage / KS divergence under row-parameterized fits:** for NORMAL with the current Plan 3d estimator (constant per-stat sigma), coverage and KS p-value are tightly coupled. For GAMMA where scale = mu/alpha is per-row, the two metrics measure different things and may diverge. Plan 3e Phase 1+ row-specific param fits will further decouple them.
- **`assumed_family` column:** the summary frame currently includes both `assumed_family` (the family name from extract_per_stat_residuals) and the AIC/coverage columns. This is informational and not in the spec's section 2.3 schema, but useful context for the report writer.
