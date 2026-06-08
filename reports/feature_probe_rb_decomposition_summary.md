# RB Rushing + Receiving Decomposition Probe -- Summary

**Spec:** `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`
**Eval years:** [2021, 2022, 2023, 2024]
**n_bootstrap:** 1000, seed: 42

## Per-stat verdicts

| Stat | n_paired | RMSE delta (decomp - direct) | 95% CI | Composite-fpts equiv | Magnitude flag | Verdict |
|---|---:|---:|---|---:|---|:---:|
| rushing_yards | 3291 | -0.0931 | [-0.1915, +0.0058] | -0.0093 |  | **NULL** |
| rushing_tds | 3291 | +0.0010 | [-0.0014, +0.0033] | +0.0062 |  | **NULL** |
| receptions | 3291 | -0.0004 | [-0.0022, +0.0016] | -0.0004 | MARGINAL | **NULL** |
| receiving_yards | 3291 | -0.0344 | [-0.0850, +0.0150] | -0.0034 | MARGINAL | **NULL** |
| receiving_tds | 3291 | -0.0003 | [-0.0012, +0.0006] | -0.0017 | MARGINAL | **NULL** |

## Coverage (eval rows)

Coverage threshold: 0.95 per volume axis per eval year.

### Carries > 0 rate (rushing axis)
- 2021: 0.9657
- 2022: 0.9638
- 2023: 0.9761
- 2024: 0.9720

### Targets > 0 rate (receiving axis)
- 2021: 0.8221 -- BELOW THRESHOLD
- 2022: 0.8110 -- BELOW THRESHOLD
- 2023: 0.8518 -- BELOW THRESHOLD
- 2024: 0.7799 -- BELOW THRESHOLD

## Factor orthogonality (Pearson rho on eval rows where volume > 0)

Spec section 5 risk #2: |rho| > 0.2 across years flags decomposition double-counting risk.

| Stat | 2021 | 2022 | 2023 | 2024 | Max |rho| |
|---|---:|---:|---:|---:|---:|
| rushing_yards | +0.092 | +0.146 | +0.124 | +0.110 | +0.146 |
| rushing_tds | +0.118 | +0.114 | +0.051 | +0.055 | +0.118 |
| receptions | +0.065 | -0.008 | +0.016 | -0.045 | +0.065 |
| receiving_yards | +0.092 | +0.038 | -0.006 | -0.032 | +0.092 |
| receiving_tds | -0.023 | +0.035 | -0.039 | -0.019 | +0.039 |

## Mechanism caveat

This probe tests decomposition with RidgeCV everywhere (the same model class on both arms). Factor-appropriate sub-model classes (Poisson, Gamma / Tweedie, logistic) are separate probe + integration cycles per spec section 1.4 #3. PR #39 / PR #44 closed two of these on WR with NULL verdicts; RB-side factor-class probes remain independent tests if any stat here SIGNALs.
