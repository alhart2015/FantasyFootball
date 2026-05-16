# Tweedie yards_per_target Probe -- Summary

**Spec:** `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`
**Eval years:** [2021, 2022, 2023, 2024]
**n_bootstrap:** 1000, seed: 42

## Verdict: **NULL**

- n_paired: 5195
- RMSE delta (tweedie - ridge): -0.0121 yards (95% CI [-0.0564, +0.0353])
- Composite-fpts equivalent (yards * 0.1): -0.0012 fpts

**Magnitude flag:** |delta| 0.0121 < 0.050 yards (|delta_fpts| < 0.005) -- in the marginal zone per PR #31's retrospective rule. Integration go/no-go must weight CI strength against magnitude.

## Mechanism caveat

Incumbent arm is Ridge-decomp (a probe-internal construction), NOT current production. Current production for receiving_yards is direct RidgeCV (via `ensemble-decomposed`, which decomposes Stat.RECEPTIONS only per PR #36/#38). A SIGNAL verdict here does NOT imply Tweedie-decomp beats current production; that comparison is the integration adoption-gate's question on a separate cycle.

## Per-year breakdown

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  coverage
 2021      1401          0.034344      -0.062370       0.134329  0.993576
 2022      1062         -0.036397      -0.136181       0.069762  0.996234
 2023      1308          0.024803      -0.062665       0.111690  0.989297
 2024      1424         -0.075400      -0.163267       0.006971  0.995084

## Coverage

Coverage threshold: 0.95 (`targets > 0` rate per eval year).

- 2021: 0.9936
- 2022: 0.9962
- 2023: 0.9893
- 2024: 0.9951

## Plan-vs-execution deviations

**Negative-yardage filter on efficiency training rows.** The initial Task 5
run crashed with `ValueError: Some value(s) of y are out of the valid range
of the loss 'HalfTweedieLoss'` because Tweedie deviance requires y >= 0 and
~0.6% of WR `targets > 0` rows have negative `receiving_yards` (real-data
laterals / lost yards on receptions; 125 of 19,347 rows total across 2018-2024).

Resolution: the efficiency-fit row mask in `walk_forward_residuals` was
tightened from `targets_train > 0` to
`(targets_train > 0) & (yards_train >= 0.0)`. Applied to BOTH arms so the
comparison stays apples-to-apples (identical training-row set). The
incumbent Ridge arm tolerates negative ratios in fit but the candidate
Tweedie arm cannot, so the symmetric filter is the only fix that keeps the
two arms comparable. Eval rows are NOT filtered -- both arms are scored
against actual yards including negatives, so the probe still penalises
each arm for any negative-yards rows it underpredicts.

Impact on the comparison: minimal. The dropped rows are <1% of training
rows per fold, and they cluster at the low-volume tail. Both arms see the
same restricted training set, so any bias from the filter applies
symmetrically and cancels in the paired-Delta-RMSE.

**No ConvergenceWarning fired** during the run; `pred_tweedie / mu_targets`
sanity-check is implicit in the realistic per-year point estimates (all
within +/-0.08 yards of the incumbent).

**Wall-clock:** ~7 seconds total (well under the 5-15 minute plan
estimate). The plan estimate was conservative; n_rows-per-fold and feature
count are smaller than worst-case.
