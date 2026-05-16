# Logit catch_rate Probe — Summary

**Spec:** `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`
**Eval years:** [2021, 2022, 2023, 2024]
**n_bootstrap:** 1000, seed: 42

## Verdict: **NULL**

- n_paired: 5195
- RMSE Δ (logit - ridge): -0.0018 (95% CI [-0.0047, +0.0009])

**Magnitude flag:** |Δ| 0.0018 < 0.005 receptions — in the marginal zone per PR #31's retrospective rule. Integration go/no-go must weight CI strength against magnitude.

## Per-year breakdown

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  coverage
 2021      1401         -0.002073      -0.009143       0.004641  0.993576
 2022      1062         -0.005203      -0.010739       0.000680  0.996234
 2023      1308          0.001107      -0.004746       0.006238  0.989297
 2024      1424         -0.001791      -0.006617       0.002918  0.995084

## Coverage

Coverage threshold: 0.95 (`targets > 0` rate per eval year).

- 2021: 0.9936
- 2022: 0.9962
- 2023: 0.9893
- 2024: 0.9951
