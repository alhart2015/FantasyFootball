# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_lightgbm-nb` candidate=`data\backtest\run_candidate_lightgbm-nb`
n_bootstrap: 1000, seed: 42

### RB — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.006, +0.009] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.0012 (95% CI [-0.0064, +0.0090])
- Spearman delta: -0.0002 (95% CI [-0.0015, +0.0011])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315          0.010899      -0.007571       0.028634             -0.001415          -0.004517           0.001536
 2022      1331          0.013207      -0.000942       0.026320              0.000461          -0.002031           0.003039
 2023      1311         -0.018814      -0.034487      -0.004288              0.000706          -0.001603           0.003102
 2024      1316         -0.001945      -0.015499       0.010951             -0.000402          -0.002544           0.001554

Wrote CSV: reports\adoption_gate_weather_refined_lightgbm-nb_rb.csv
