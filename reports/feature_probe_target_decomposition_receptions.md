# Target Decomposition Probe — receptions

**Verdict:** SIGNAL

## Pooled per-stat verdict

| n_paired | RMSE direct | RMSE decomposed | Delta-RMSE | 95% CI | Verdict |
|---:|---:|---:|---:|---|:---:|
| 8460 | 2.0324 | 2.0282 | -0.0042 | [-0.0079, -0.0004] | **SIGNAL** |

**Expected composite-fpts Delta (rough)**: -0.0042 fpts (stat RMSE Delta x ESPN PPR coefficient +1.0000). Per section 5 risk #1, magnitudes < 0.005 fpts under coverage relaxation should be treated as MARGINAL, not SIGNAL.

## Per-eval-year coverage

| Year | Eval n | Eval (targets > 0) | Train n | Train (targets > 0) |
|---:|---:|---:|---:|---:|
| 2021 | 2109 | 0.988 | 3819 | 0.993 |
| 2022 | 2102 | 0.985 | 5220 | 0.993 |
| 2023 | 2201 | 0.981 | 6282 | 0.994 |
| 2024 | 2048 | 0.984 | 7590 | 0.993 |

## Factor residual correlation (Pearson rho)

Per-eval-year Pearson rho between (predicted-volume residual, predicted-efficiency residual) on rows with targets > 0. |rho| > 0.2 in any year is a documented caveat per section 5 risk #2.

| Year | rho |
|---:|---:|
| 2021 | -0.008 |
| 2022 | +0.031 |
| 2023 | +0.016 |
| 2024 | -0.012 |

_Bootstrap n_resamples = 5000, seed = 54208._
