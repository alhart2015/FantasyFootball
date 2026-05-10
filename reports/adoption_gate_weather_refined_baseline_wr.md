# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_baseline` candidate=`data\backtest\run_candidate_baseline`
n_bootstrap: 1000, seed: 42

### WR — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.001, +0.023] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: +0.0120 (95% CI [+0.0013, +0.0228])
- Spearman delta: -0.0025 (95% CI [-0.0045, -0.0005])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109          0.014627      -0.009418       0.037520             -0.003054          -0.007735           0.001702
 2022      2102          0.013144      -0.011314       0.036758             -0.002807          -0.007393           0.001939
 2023      2201          0.007520      -0.014127       0.030612             -0.002093          -0.005712           0.001869
 2024      2048          0.012681      -0.004905       0.028865             -0.002139          -0.005614           0.001218

Wrote CSV: reports\adoption_gate_weather_refined_baseline_wr.csv
