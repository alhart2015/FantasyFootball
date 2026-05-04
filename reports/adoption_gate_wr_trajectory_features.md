# Adoption Gate Report — WR Trajectory Features

Run: baseline=`data\backtest\run_20260503T014536Z` candidate=`data\backtest\run_20260504T083759Z`. Position: WR. n_bootstrap=1000, seed=42.

Spec §1.3.5 binding cell: `(BaselineModel, WR)` — verdict ADOPT triggers ship.

## Per-(model_class, WR) verdicts

| Model class | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| **baseline** | 8460 | **-0.0371** | **[-0.0567, -0.0172]** | +0.0047 | [+0.0015, +0.0081] | **ADOPT** |
| lightgbm | 8460 | -0.0207 | [-0.0289, -0.0121] | +0.0026 | [+0.0005, +0.0047] | ADOPT |
| lightgbm-tuned | 8460 | +0.0025 | [-0.0056, +0.0106] | +0.0014 | [-0.0003, +0.0032] | DO_NOT_ADOPT |
| lightgbm-nb | 8460 | -0.0171 | [-0.0269, -0.0071] | +0.0020 | [+0.0002, +0.0038] | ADOPT |
| ensemble | 8460 | -0.0242 | [-0.0351, -0.0138] | +0.0019 | [+0.0001, +0.0040] | ADOPT |

## Per-year breakdown — `(BaselineModel, WR)` binding cell

| Year | n_paired | RMSE Δ | RMSE 95% CI | Spearman Δ | Spearman 95% CI |
|---|---:|---:|---|---:|---|
| 2021 | 2109 | -0.0553 | [-0.0940, -0.0179] | +0.0068 | [+0.0001, +0.0133] |
| 2022 | 2102 | -0.0295 | [-0.0728, +0.0079] | +0.0052 | [-0.0015, +0.0119] |
| 2023 | 2201 | -0.0397 | [-0.0767, -0.0039] | +0.0034 | [-0.0026, +0.0093] |
| 2024 | 2048 | -0.0233 | [-0.0571, +0.0120] | +0.0035 | [-0.0030, +0.0095] |
