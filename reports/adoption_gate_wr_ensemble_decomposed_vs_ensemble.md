# Adoption gate report — ensemble-decomposed vs ensemble

Run: `data\backtest\run_task4_wr`
n_bootstrap: 1000, seed: 42

### WR — ensemble-decomposed vs ensemble: **ADOPT**

_RMSE delta -0.004 (95% CI [-0.008, -0.000]); Spearman lo_95 -0.0004 > floor -0.020_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: -0.0038 (95% CI [-0.0079, -0.0002])
- Spearman delta: +0.0002 (95% CI [-0.0004, +0.0007])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109         -0.004517      -0.012363       0.002914              0.000350          -0.000816           0.001606
 2022      2102         -0.005596      -0.014020       0.003807             -0.000577          -0.002265           0.001063
 2023      2201         -0.004575      -0.014484       0.005059              0.000721          -0.000621           0.001997
 2024      2048         -0.000485      -0.002116       0.001209              0.000149          -0.000167           0.000494

Wrote CSV: C:\Users\alden\FantasyFootball\.worktrees\feat-wr-ensemble-decomposed-child\reports\adoption_gate_wr_ensemble_decomposed_vs_ensemble.csv
