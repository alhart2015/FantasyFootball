# Adoption gate report — ensemble-decomposed vs decomposed-baseline

Run: `data\backtest\run_task4_wr`
n_bootstrap: 1000, seed: 42

### WR — ensemble-decomposed vs decomposed-baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.023, +0.009] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: -0.0074 (95% CI [-0.0234, +0.0089])
- Spearman delta: +0.0041 (95% CI [+0.0013, +0.0067])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109         -0.014714      -0.046822       0.016020              0.009591           0.004430           0.014837
 2022      2102         -0.013425      -0.041673       0.012941              0.003399          -0.001479           0.008329
 2023      2201          0.004858      -0.026716       0.035498              0.000863          -0.004044           0.005715
 2024      2048         -0.006717      -0.044280       0.027211              0.002561          -0.003837           0.009202

Wrote CSV: C:\Users\alden\FantasyFootball\.worktrees\feat-wr-ensemble-decomposed-child\reports\adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.csv
