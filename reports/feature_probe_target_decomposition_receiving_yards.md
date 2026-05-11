# Target Decomposition Probe — receiving_yards

**Verdict:** NULL

## Pooled per-stat verdict

| n_paired | RMSE direct | RMSE decomposed | Delta-RMSE | 95% CI | Verdict |
|---:|---:|---:|---:|---|:---:|
| 8460 | 31.1654 | 31.1600 | -0.0054 | [-0.0601, +0.0492] | **NULL** |

**Expected composite-fpts Delta (rough)**: -0.0005 fpts (stat RMSE Delta x ESPN PPR coefficient +0.1000). Per section 5 risk #1, magnitudes < 0.005 fpts under coverage relaxation should be treated as MARGINAL, not SIGNAL.

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
| 2021 | -0.048 |
| 2022 | +0.024 |
| 2023 | -0.029 |
| 2024 | -0.024 |

_Bootstrap n_resamples = 5000, seed = 54208._
