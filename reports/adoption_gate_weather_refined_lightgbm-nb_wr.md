# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_lightgbm-nb` candidate=`data\backtest\run_candidate_lightgbm-nb`
n_bootstrap: 1000, seed: 42

### WR — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.000, +0.012] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: +0.0060 (95% CI [-0.0001, +0.0119])
- Spearman delta: -0.0016 (95% CI [-0.0029, -0.0002])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109          0.007844      -0.005886       0.020762             -0.002874          -0.005945           0.000296
 2022      2102         -0.006719      -0.021067       0.007604              0.001410          -0.001463           0.004624
 2023      2201          0.008051      -0.001637       0.018374             -0.001979          -0.004246           0.000319
 2024      2048          0.014894       0.003145       0.026188             -0.002957          -0.005419          -0.000442

Wrote CSV: reports\adoption_gate_weather_refined_lightgbm-nb_wr.csv
