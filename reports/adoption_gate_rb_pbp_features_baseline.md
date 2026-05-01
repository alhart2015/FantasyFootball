# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\baseline_run` candidate=`data\backtest\candidate_run`
n_bootstrap: 1000, seed: 42

### RB — _candidate_run vs _baseline_run: **ADOPT**

_RMSE delta -0.012 (95% CI [-0.025, -0.001]); Spearman lo_95 -0.0003 > floor -0.020_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: -0.0124 (95% CI [-0.0255, -0.0006])
- Spearman delta: +0.0021 (95% CI [-0.0003, +0.0044])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315         -0.027646      -0.053331      -0.004928              0.003047          -0.001605           0.007993
 2022      1331         -0.020967      -0.046389       0.004187              0.003935          -0.000784           0.008427
 2023      1311         -0.004460      -0.027410       0.020742             -0.000606          -0.005275           0.003752
 2024      1316          0.004857      -0.018895       0.030045              0.001843          -0.002913           0.006749

Wrote CSV: reports\adoption_gate_rb_pbp_features_baseline.csv
